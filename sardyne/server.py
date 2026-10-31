"""Server HTTP compatibile OpenAI, con FastAPI.

Uso: python -m sardyne.server models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf [--port 8080]

Endpoint: POST /v1/chat/completions (con e senza stream, con strumenti), GET /v1/models, GET /health.
Il modello è uno e la KV cache pure: si genera una richiesta alla volta, le altre aspettano il loro turno.

Strumenti: il server restituisce le chiamate, ma non le esegue: quello spetta al client. Con tool_choice
"required" o una funzione precisa si fa iniziare la risposta da <tool_call>, così il modello deve chiamare.
Il testo di <think> resta in content: decidere se mostrarlo spetta al client.
"""
import argparse
import asyncio
import json
import os
import secrets
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from sardyne.chat import TOOL_OPEN, ChatEngine, ToolCallStream, parse_tool_calls
from sardyne.sampling import Sampler, SamplingParams


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # i campi OpenAI che non usiamo non sono un errore

    model: str = ""
    messages: list[dict] = Field(min_length=1)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    stream: bool = False
    stream_options: dict | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    n: int = 1
    stop: str | list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    # Oltre a OpenAI, come llama.cpp.
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    enable_thinking: bool | None = None


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Formato OpenAI -> quello dei template: testo a parti unito, developer = system."""
    out = []
    for m in messages:
        if "role" not in m:
            raise ValueError("messaggio senza role")
        if m["role"] not in ("system", "developer", "user", "assistant", "tool"):
            raise ValueError(f"role non supportato: {m['role']}")
        m = {**m, "role": "system" if m["role"] == "developer" else m["role"]}
        content = m.get("content")
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            if len(parts) != len(content):
                raise ValueError("sono supportate solo parti di tipo text")
            m["content"] = "".join(parts)
        elif content is None:
            m["content"] = ""
        out.append(m)
    return out


def select_tools(req: ChatRequest) -> tuple[list | None, bool]:
    """(strumenti da mostrare al modello, chiamata obbligatoria)."""
    tools, choice = req.tools or None, req.tool_choice
    if choice is None or choice == "auto":
        return tools, False
    if choice == "none":
        return None, False
    if not tools:
        raise ValueError("tool_choice richiede tools")
    if choice == "required":
        return tools, True
    if isinstance(choice, dict):
        name = choice.get("function", {}).get("name")
        chosen = [t for t in tools if t.get("function", {}).get("name") == name]
        if not chosen:
            raise ValueError(f"tool_choice: funzione {name!r} non presente in tools")
        return chosen, True
    raise ValueError(f"tool_choice non valido: {choice!r}")


class StopFilter:
    """Tronca il testo alla prima stringa di stop, anche se è spezzata tra più pezzi."""

    def __init__(self, stop):
        self.stops = [stop] if isinstance(stop, str) else list(stop or [])
        self._buf = ""
        self.hit = False

    def feed(self, text: str) -> str:
        if not self.stops:
            return text
        self._buf += text
        pos = min((p for p in map(self._buf.find, self.stops) if p >= 0), default=-1)
        if pos >= 0:
            out, self._buf, self.hit = self._buf[:pos], "", True
            return out
        keep = max(map(len, self.stops)) - 1
        cut = max(0, len(self._buf) - keep)
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return out

    def finish(self) -> str:
        rest, self._buf = self._buf, ""
        return rest


def tool_call_json(call: dict) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}


def usage(stats) -> dict:
    return {"prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.completion_tokens,
            "total_tokens": stats.prompt_tokens + stats.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": stats.cached_tokens}}


def error_body(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict:
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


def create_app(engine: ChatEngine, name: str = "local", api_key: str | None = None,
               defaults: SamplingParams | None = None, max_tokens: int = 1024) -> FastAPI:
    app = FastAPI(title="sardyne")
    defaults = defaults or SamplingParams(temperature=0.7)
    turn = asyncio.Lock()
    created = int(time.time())

    @app.exception_handler(RequestValidationError)
    async def bad_request(_, exc):
        return JSONResponse(error_body(str(exc.errors()[0].get("msg", "richiesta non valida"))), status_code=400)

    @app.exception_handler(ValueError)
    async def bad_value(_, exc):
        return JSONResponse(error_body(str(exc)), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_, exc):
        kind = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return JSONResponse(error_body(str(exc.detail), kind), status_code=exc.status_code)

    def check_key(request: Request):
        if api_key is None:
            return
        given = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not secrets.compare_digest(given.encode(), api_key.encode()):
            raise HTTPException(401, "API key non valida")

    def sampler_for(req: ChatRequest) -> Sampler:
        def pick(value, default):
            return default if value is None else value

        return Sampler(SamplingParams(pick(req.temperature, defaults.temperature), pick(req.top_k, defaults.top_k),
                                      pick(req.top_p, defaults.top_p), pick(req.min_p, defaults.min_p),
                                      pick(req.repeat_penalty, defaults.repeat_penalty), defaults.repeat_last_n,
                                      pick(req.seed, defaults.seed)))

    def prepare(req: ChatRequest) -> dict:
        """Argomenti di engine.stream, validati prima di far aspettare il client."""
        if req.n != 1:
            raise ValueError("è supportato solo n=1")
        tools, forced = select_tools(req)
        thinking = False if forced else req.enable_thinking
        return dict(messages=normalize_messages(req.messages), tools=tools, sampler=sampler_for(req),
                    max_tokens=req.max_tokens or req.max_completion_tokens or max_tokens, enable_thinking=thinking,
                    prefix=TOOL_OPEN + "\n" if forced else "")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models(request: Request):
        check_key(request)
        return {"object": "list", "data": [{"id": name, "object": "model", "created": created, "owned_by": "local"}]}

    def complete(args: dict, stop) -> tuple[str, list, str]:
        filt = StopFilter(stop)
        text = []
        gen = engine.stream(**args)
        for piece in gen:
            text.append(filt.feed(piece))
            if filt.hit:
                gen.close()
                break
        text.append(filt.finish())
        reason = "stop" if filt.hit else engine.stats.finish_reason
        content, calls = parse_tool_calls("".join(text)) if args["tools"] else ("".join(text), [])
        return content, calls, "tool_calls" if calls else reason

    async def streaming(req: ChatRequest, args: dict, request: Request, cid: str):
        now = int(time.time())

        def chunk(delta: dict, finish=None, **extra) -> str:
            body = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
            return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

        async with turn:
            gen = engine.stream(**args)
            filt, tools = StopFilter(req.stop), ToolCallStream() if args["tools"] else None
            n_calls = 0

            def events(text: str, last: bool = False):
                """Chunk SSE per un pezzo di testo generato."""
                nonlocal n_calls
                got = tools.feed(text) if tools else [("text", text)]
                if last and tools:
                    got += tools.finish()
                for kind, value in got:
                    if kind == "text" and value:
                        yield chunk({"content": value})
                    elif kind == "tool_call":
                        yield chunk({"tool_calls": [{"index": n_calls, **tool_call_json(value)}]})
                        n_calls += 1

            try:
                yield chunk({"role": "assistant", "content": ""})
                while not filt.hit:
                    piece = await run_in_threadpool(next, gen, None)
                    if piece is None:
                        break
                    if await request.is_disconnected():
                        return
                    for out in events(filt.feed(piece)):
                        yield out
                for out in events(filt.finish(), last=True):
                    yield out
                gen.close()
                reason = "tool_calls" if n_calls else "stop" if filt.hit else engine.stats.finish_reason
                yield chunk({}, reason)
                if (req.stream_options or {}).get("include_usage"):
                    body = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": name, "choices": [],
                            "usage": usage(engine.stats)}
                    yield f"data: {json.dumps(body)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                gen.close()

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest, request: Request):
        check_key(request)
        args = prepare(req)
        # Errori del prompt subito, in JSON: durante lo streaming lo stato HTTP sarebbe già partito.
        n_prompt = len(engine.prompt_ids(args["messages"], args["tools"], args["enable_thinking"]))
        if n_prompt >= engine.ctx.max_seq:
            raise ValueError(f"prompt di {n_prompt} token oltre il contesto di {engine.ctx.max_seq}")
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        if req.stream:
            return StreamingResponse(streaming(req, args, request, cid), media_type="text/event-stream")
        async with turn:
            content, calls, reason = await run_in_threadpool(complete, args, req.stop)
            stats = engine.stats
        message = {"role": "assistant", "content": content or None if calls else content}
        if calls:
            message["tool_calls"] = [tool_call_json(c) for c in calls]
        return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": name,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}], "usage": usage(stats)}

    return app


def arg_parser() -> argparse.ArgumentParser:
    from sardyne.runtime import add_sampling_args

    ap = argparse.ArgumentParser(description="Server compatibile OpenAI")
    ap.add_argument("model")
    ap.add_argument("--tokenizer", help="cartella HF (default: il tokenizer nel file .gguf)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api-key", default=os.environ.get("SARDYNE_API_KEY") or None,
                    help="se indicata, le richieste devono presentarla (default: $SARDYNE_API_KEY)")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1024, help="token massimi per risposta, se il client non li indica")
    ap.add_argument("--threads", type=int, default=0)
    add_sampling_args(ap, temperature=0.7)
    return ap


def main():
    import uvicorn

    from sardyne.runtime import sampling_params

    args = arg_parser().parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not args.api_key:
        print("[attenzione: il server è raggiungibile dalla rete e senza chiave chiunque può usarlo]")

    engine = ChatEngine(args.model, args.max_seq, args.threads, args.tokenizer)
    uvicorn.run(create_app(engine, Path(args.model).stem, args.api_key, sampling_params(args), args.max_tokens),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
