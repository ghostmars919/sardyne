"""Server compatibile OpenAI: unità con un motore finto, poi il client ufficiale su un server vero."""
import json
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from sardyne import capi  # noqa: E402
from sardyne.chat import ChatStats  # noqa: E402
from sardyne.server import StopFilter, create_app, normalize_messages  # noqa: E402

GGUF_06 = Path(__file__).resolve().parent.parent / "models" / "gguf" / "qwen3-0.6b-bf16.gguf"
USER = [{"role": "user", "content": "ciao"}]
TOOL = {"type": "function", "function": {"name": "weather", "description": "Current weather of a city",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                        "required": ["city"]}}}
CALL_TEXT = '<tool_call>\n{"name": "weather", "arguments": {"city": "Rome"}}\n</tool_call>'


class FakeCtx:
    max_seq = 100


class FakeEngine:
    """Restituisce pezzi già scritti: i test del protocollo non dipendono dal modello."""

    def __init__(self, pieces=("Buon", " gior", "no")):
        self.pieces = list(pieces)
        self.ctx = FakeCtx()
        self.stats = ChatStats()
        self.calls = []
        self.closed = 0
        self.gate = None

    def prompt_ids(self, messages, tools=None, enable_thinking=None):
        n = sum(len(m.get("content") or "") for m in messages)
        return [0] * n

    def stream(self, messages, tools=None, sampler=None, max_tokens=1024, enable_thinking=None, prefix=""):
        self.calls.append(dict(messages=messages, tools=tools, sampler=sampler, max_tokens=max_tokens,
                               enable_thinking=enable_thinking, prefix=prefix))
        self.stats = ChatStats(prompt_tokens=7, cached_tokens=2, finish_reason="length")
        try:
            if prefix:
                yield prefix
            for p in self.pieces:
                if self.gate:
                    self.gate.wait(5)
                self.stats.completion_tokens += 1
                yield p
            self.stats.finish_reason = "stop"
        finally:
            self.closed += 1


def client(engine=None, **kw):
    engine = engine or FakeEngine()
    return TestClient(create_app(engine, "prova", **kw), raise_server_exceptions=False), engine


def post(c, **body):
    return c.post("/v1/chat/completions", json={"messages": USER, **body})


def sse(c, **body):
    with c.stream("POST", "/v1/chat/completions", json={"messages": USER, "stream": True, **body}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln[6:] for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(ln) for ln in lines[:-1]]


def test_completion_shape():
    c, _ = client()
    r = post(c).json()
    assert r["object"] == "chat.completion" and r["model"] == "prova"
    assert r["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "Buon giorno"}, "finish_reason": "stop"}]
    assert r["usage"]["prompt_tokens"] == 7 and r["usage"]["completion_tokens"] == 3 and r["usage"]["total_tokens"] == 10
    assert r["usage"]["prompt_tokens_details"]["cached_tokens"] == 2


def test_stream_shape():
    c, _ = client()
    chunks = sse(c)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(x["choices"][0]["delta"].get("content", "") for x in chunks) == "Buon giorno"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(x["object"] == "chat.completion.chunk" for x in chunks)


def test_stream_usage_on_request():
    c, _ = client()
    assert "usage" not in sse(c)[-1]
    last = sse(c, stream_options={"include_usage": True})[-1]
    assert last["choices"] == [] and last["usage"]["completion_tokens"] == 3


def test_models_and_health():
    c, _ = client()
    assert c.get("/v1/models").json()["data"][0]["id"] == "prova"
    assert c.get("/health").json() == {"status": "ok"}


def test_sampling_parameters_reach_the_sampler():
    c, e = client()
    post(c, temperature=0.3, top_p=0.5, seed=9, top_k=4, min_p=0.1, repeat_penalty=1.2, max_tokens=5)
    s = e.calls[0]["sampler"].params
    assert (s.temperature, s.top_p, s.seed, s.top_k, s.min_p, s.repeat_penalty) == (0.3, 0.5, 9, 4, 0.1, 1.2)
    assert e.calls[0]["max_tokens"] == 5


def test_max_completion_tokens_and_default():
    c, e = client(max_tokens=33)
    post(c, max_completion_tokens=8)
    post(c)
    assert [x["max_tokens"] for x in e.calls] == [8, 33]


@pytest.mark.parametrize("body,msg", [(dict(n=2), "n=1"), (dict(temperature=-1), "temperature"), (dict(top_p=2), "top_p"),
                                      (dict(tool_choice="required"), "tools"), (dict(tool_choice="boh", tools=[TOOL]), "tool_choice"),
                                      (dict(tool_choice={"type": "function", "function": {"name": "x"}}, tools=[TOOL]), "x")])
def test_bad_requests_are_openai_errors(body, msg):
    c, e = client()
    r = post(c, **body)
    assert r.status_code == 400 and msg in r.json()["error"]["message"]
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert not e.calls


@pytest.mark.parametrize("messages", [[], [{"content": "x"}], [{"role": "mago", "content": "x"}],
                                      [{"role": "user", "content": [{"type": "image_url"}]}]], ids=["vuoto", "senza-role", "role", "immagine"])
def test_bad_messages(messages):
    c, _ = client()
    r = c.post("/v1/chat/completions", json={"messages": messages})
    assert r.status_code == 400 and "error" in r.json()


@pytest.mark.parametrize("stream", [False, True])
def test_prompt_too_long_is_a_400_even_in_streaming(stream):
    c, _ = client()
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 200}], "stream": stream})
    assert r.status_code == 400 and "contesto" in r.json()["error"]["message"]


def test_normalize_messages():
    got = normalize_messages([{"role": "developer", "content": "a"},
                              {"role": "user", "content": [{"type": "text", "text": "b"}, {"type": "text", "text": "c"}]},
                              {"role": "assistant", "content": None}])
    assert [m["content"] for m in got] == ["a", "bc", ""] and got[0]["role"] == "system"


def test_api_key():
    c, _ = client(api_key="segreto")
    assert post(c).status_code == 401
    assert post(c).json()["error"]["type"] == "authentication_error"
    assert c.get("/v1/models").status_code == 401
    assert c.post("/v1/chat/completions", json={"messages": USER}, headers={"Authorization": "Bearer no"}).status_code == 401
    assert c.post("/v1/chat/completions", json={"messages": USER}, headers={"Authorization": "Bearer segreto"}).status_code == 200


# ---------------------------------------------------------------- strumenti

def test_tool_call_response():
    c, _ = client(FakeEngine([CALL_TEXT]))
    r = post(c, tools=[TOOL]).json()["choices"][0]
    call = r["message"]["tool_calls"][0]
    assert r["finish_reason"] == "tool_calls" and r["message"]["content"] is None
    assert call["type"] == "function" and call["function"]["name"] == "weather" and call["id"].startswith("call_")
    assert json.loads(call["function"]["arguments"]) == {"city": "Rome"}


def test_text_before_the_call_is_kept():
    c, _ = client(FakeEngine(["Controllo.\n", CALL_TEXT]))
    assert post(c, tools=[TOOL]).json()["choices"][0]["message"]["content"] == "Controllo."


def test_tool_call_text_without_tools_stays_text():
    c, _ = client(FakeEngine([CALL_TEXT]))
    r = post(c).json()["choices"][0]
    assert r["message"]["content"] == CALL_TEXT and r["finish_reason"] == "stop"


@pytest.mark.parametrize("size", [1, 4, 1000])
def test_tool_call_streaming(size):
    text = "Ok " + CALL_TEXT + CALL_TEXT.replace("Rome", "Bari")
    c, _ = client(FakeEngine([text[i:i + size] for i in range(0, len(text), size)]))
    chunks = sse(c, tools=[TOOL])
    calls = [t for x in chunks for t in x["choices"][0]["delta"].get("tool_calls", [])]
    assert [t["index"] for t in calls] == [0, 1]
    assert [json.loads(t["function"]["arguments"])["city"] for t in calls] == ["Rome", "Bari"]
    assert "".join(x["choices"][0]["delta"].get("content", "") for x in chunks) == "Ok "
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_hides_the_tools():
    c, e = client()
    post(c, tools=[TOOL], tool_choice="none")
    assert e.calls[0]["tools"] is None and e.calls[0]["prefix"] == ""


def test_tool_choice_required_forces_a_call():
    c, e = client(FakeEngine(['{"name": "weather", "arguments": {"city": "Rome"}}\n</tool_call>']))
    r = post(c, tools=[TOOL], tool_choice="required").json()["choices"][0]
    assert e.calls[0]["prefix"] == "<tool_call>\n" and e.calls[0]["enable_thinking"] is False
    assert r["message"]["tool_calls"][0]["function"]["name"] == "weather"


def test_tool_choice_function_shows_only_that_tool():
    other = {"type": "function", "function": {"name": "altro"}}
    c, e = client()
    post(c, tools=[other, TOOL], tool_choice={"type": "function", "function": {"name": "weather"}})
    assert e.calls[0]["tools"] == [TOOL]


# ---------------------------------------------------------------- stop

@pytest.mark.parametrize("size", [1, 2, 100])
def test_stop_filter(size):
    text = "uno due FINE tre"
    f = StopFilter(["FINE", "xx"])
    out = ""
    for i in range(0, len(text), size):
        out += f.feed(text[i:i + size])
        if f.hit:
            break
    assert out + f.finish() == "uno due " and f.hit


def test_stop_filter_without_a_match_returns_everything():
    f = StopFilter("zz")
    assert f.feed("abc z") + f.feed("d") + f.finish() == "abc zd" and not f.hit


@pytest.mark.parametrize("stream", [False, True])
def test_stop_strings(stream):
    c, e = client(FakeEngine(["uno ", "du", "e tre", " quattro"]))
    if stream:
        chunks = sse(c, stop=["due"])
        text = "".join(x["choices"][0]["delta"].get("content", "") for x in chunks)
        reason = chunks[-1]["choices"][0]["finish_reason"]
    else:
        r = post(c, stop="due").json()["choices"][0]
        text, reason = r["message"]["content"], r["finish_reason"]
    assert text == "uno " and reason == "stop" and e.closed == 1


# ---------------------------------------------------------------- concorrenza e disconnessione

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(engine, **kw):
    import uvicorn

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine, "prova", **kw), port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    return server, thread, f"http://127.0.0.1:{port}"


class SlowEngine(FakeEngine):
    """Controlla che non ci siano mai due generazioni insieme."""

    def __init__(self, n=40, delay=0.01):
        super().__init__(["x"] * n)
        self.delay, self.active, self.most = delay, 0, 0

    def stream(self, *a, **kw):
        self.active += 1
        self.most = max(self.most, self.active)
        try:
            for piece in super().stream(*a, **kw):
                time.sleep(self.delay)
                yield piece
        finally:
            self.active -= 1


def test_requests_run_one_at_a_time():
    e = SlowEngine(n=10)
    server, thread, url = serve(e)
    out = []

    def one(stream):
        with httpx.Client(timeout=30) as h:
            r = h.post(url + "/v1/chat/completions", json={"messages": USER, "stream": stream})
            out.append(r.status_code)

    try:
        ts = [threading.Thread(target=one, args=(i % 2 == 0,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    finally:
        server.should_exit = True
        thread.join(5)
    assert out == [200] * 4 and e.most == 1


def test_client_disconnect_stops_generation_and_frees_the_turn():
    e = SlowEngine(n=500)
    server, thread, url = serve(e)
    try:
        with httpx.Client(timeout=30) as h:
            with h.stream("POST", url + "/v1/chat/completions", json={"messages": USER, "stream": True}) as r:
                for line in r.iter_lines():
                    if '"x"' in line:
                        break
            for _ in range(300):
                if e.active == 0:
                    break
                time.sleep(0.02)
            assert e.active == 0 and e.closed == 1 and e.stats.completion_tokens < 400
            e.pieces = ["ok"]
            assert h.post(url + "/v1/chat/completions", json={"messages": USER}).json()["choices"][0]["message"]["content"] == "ok"
    finally:
        server.should_exit = True
        thread.join(5)


# ---------------------------------------------------------------- client OpenAI e modello vero

@pytest.fixture(scope="module")
def real():
    openai = pytest.importorskip("openai")
    if not capi.available() or not GGUF_06.exists():
        pytest.skip("serve qwen3-0.6b-bf16.gguf")
    from sardyne.chat import ChatEngine

    server, thread, url = serve(ChatEngine(GGUF_06, max_seq=512), api_key="k")
    yield openai.OpenAI(base_url=url + "/v1", api_key="k")
    server.should_exit = True
    thread.join(5)


NO_THINK = {"enable_thinking": False}


def test_openai_client_completion(real):
    r = real.chat.completions.create(model="x", messages=[{"role": "user", "content": "What is the capital of France?"}],
                                     max_tokens=30, temperature=0, extra_body=NO_THINK)
    assert "Paris" in r.choices[0].message.content and r.choices[0].finish_reason == "stop" and r.usage.completion_tokens > 0


def test_openai_client_streaming(real):
    s = real.chat.completions.create(model="x", messages=[{"role": "user", "content": "What is the capital of France?"}],
                                     max_tokens=30, temperature=0, stream=True, stream_options={"include_usage": True},
                                     extra_body=NO_THINK)
    chunks = list(s)
    assert "Paris" in "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert chunks[-1].usage.completion_tokens > 0


def test_openai_client_list_models(real):
    assert real.models.list().data[0].id


def test_openai_client_wrong_key(real):
    import openai

    bad = openai.OpenAI(base_url=str(real.base_url), api_key="sbagliata")
    with pytest.raises(openai.AuthenticationError):
        bad.models.list()


def test_openai_client_bad_request(real):
    import openai

    with pytest.raises(openai.BadRequestError, match="n=1"):
        real.chat.completions.create(model="x", messages=USER, n=2)


@pytest.mark.parametrize("stream", [False, True], ids=["intero", "stream"])
def test_openai_client_tool_roundtrip(real, stream):
    msgs = [{"role": "user", "content": "What is the weather in Rome? Use the tool."}]
    args = dict(model="x", tools=[TOOL], max_tokens=80, temperature=0, extra_body=NO_THINK)
    if stream:
        deltas = [c.choices[0].delta for c in real.chat.completions.create(messages=msgs, stream=True, **args) if c.choices]
        call = next(t for d in deltas for t in (d.tool_calls or []))
    else:
        r = real.chat.completions.create(messages=msgs, **args).choices[0]
        assert r.finish_reason == "tool_calls"
        call = r.message.tool_calls[0]
    assert call.function.name == "weather" and "rome" in json.loads(call.function.arguments)["city"].lower()
    msgs += [{"role": "assistant", "content": None, "tool_calls": [
                 {"id": call.id, "type": "function",
                  "function": {"name": call.function.name, "arguments": call.function.arguments}}]},
             {"role": "tool", "tool_call_id": call.id, "content": "Sunny, 37 degrees"}]
    answer = real.chat.completions.create(messages=msgs, **args).choices[0].message.content
    assert "37" in answer


def test_openai_client_forced_tool_call(real):
    r = real.chat.completions.create(model="x", messages=[{"role": "user", "content": "Tell me about Rome."}], tools=[TOOL],
                                     tool_choice={"type": "function", "function": {"name": "weather"}},
                                     max_tokens=60, temperature=0).choices[0]
    assert r.message.tool_calls[0].function.name == "weather"
