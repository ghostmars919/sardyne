"""Sampling: ogni filtro su distribuzioni scritte a mano, poi la generazione col runtime C."""
import numpy as np
import pytest

from engine.sampling import Sampler, SamplingParams, apply_repeat_penalty, keep_mask, softmax

P = np.array([0.05, 0.4, 0.1, 0.3, 0.15])  # in ordine: 1, 3, 4, 2, 0
LOGITS = np.log(P)


def kept(**kw):
    return np.flatnonzero(keep_mask(P, **kw)).tolist()


def test_no_filter_keeps_everything():
    assert kept() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("k,expected", [(1, [1]), (2, [1, 3]), (3, [1, 3, 4]), (9, [0, 1, 2, 3, 4])])
def test_top_k(k, expected):
    assert kept(top_k=k) == expected


@pytest.mark.parametrize("p,expected", [(0.1, [1]), (0.4, [1]), (0.41, [1, 3]), (0.7, [1, 3]), (0.71, [1, 3, 4]),
                                        (0.95, [1, 2, 3, 4]), (0.96, [0, 1, 2, 3, 4])])
def test_top_p(p, expected):
    assert kept(top_p=p) == expected


@pytest.mark.parametrize("m,expected", [(0.1, [0, 1, 2, 3, 4]), (0.2, [1, 2, 3, 4]), (0.3, [1, 3, 4]), (0.5, [1, 3]),
                                        (0.8, [1]), (1.0, [1])])
def test_min_p(m, expected):
    assert kept(min_p=m) == expected


def test_filters_combine():
    assert kept(top_k=3, top_p=0.5) == [1, 3]
    assert kept(top_k=4, min_p=0.5) == [1, 3]


def test_best_token_always_survives():
    assert kept(top_k=1, top_p=0.001, min_p=1.0) == [1]


def test_repeat_penalty_divides_positive_and_multiplies_negative():
    x = np.array([2.0, -2.0, 1.0])
    np.testing.assert_allclose(apply_repeat_penalty(x, [0, 1], 2.0, 64), [1.0, -4.0, 1.0])


def test_repeat_penalty_only_looks_at_recent_tokens():
    x = np.array([2.0, 2.0])
    np.testing.assert_allclose(apply_repeat_penalty(x, [0, 1], 2.0, 1), [2.0, 1.0])
    assert apply_repeat_penalty(x, [0, 1], 2.0, 0) is x
    assert apply_repeat_penalty(x, [0, 1], 1.0, 64) is x


def test_repeat_penalty_counts_a_token_once():
    x = np.array([8.0, 1.0])
    np.testing.assert_allclose(apply_repeat_penalty(x, [0, 0, 0], 2.0, 64), [4.0, 1.0])


def test_temperature_zero_is_argmax():
    s = Sampler(SamplingParams(temperature=0, seed=1))
    assert all(s(LOGITS) == 1 for _ in range(20))


def test_temperature_zero_sees_the_repeat_penalty():
    s = Sampler(SamplingParams(temperature=0, repeat_penalty=3.0))
    assert s(np.array([3.0, 2.0]), [0]) == 1


def test_seed_gives_the_same_sequence():
    params = SamplingParams(temperature=1.0, seed=42)
    a, b = Sampler(params), Sampler(params)
    seq = [a(LOGITS) for _ in range(100)]
    assert seq == [b(LOGITS) for _ in range(100)]
    assert seq != [Sampler(SamplingParams(temperature=1.0, seed=43))(LOGITS) for _ in range(100)]


def frequencies(n=40000, **kw):
    s = Sampler(SamplingParams(seed=0, **kw))
    return np.bincount([s(LOGITS) for _ in range(n)], minlength=len(P)) / n


def test_temperature_one_follows_the_distribution():
    np.testing.assert_allclose(frequencies(temperature=1.0), P, atol=0.01)


@pytest.mark.parametrize("t", [0.5, 2.0])
def test_temperature_reshapes_the_distribution(t):
    np.testing.assert_allclose(frequencies(temperature=t), softmax(LOGITS / t), atol=0.01)


def test_filtered_tokens_are_never_drawn_and_the_rest_is_renormalized():
    expected = np.array([0, 0.4, 0, 0.3, 0]) / 0.7
    np.testing.assert_allclose(frequencies(temperature=1.0, top_k=2), expected, atol=0.01)


def test_filters_use_probabilities_before_temperature():
    # Con T alta la distribuzione si appiattisce, ma top_p deve tagliare gli stessi token.
    assert (frequencies(n=4000, temperature=50.0, top_p=0.7)[[0, 2, 4]] == 0).all()


def test_huge_logits_do_not_overflow():
    s = Sampler(SamplingParams(temperature=0.1, seed=0))
    assert s(np.array([1e4, 0.0, -1e4])) == 0


@pytest.mark.parametrize("kw", [dict(temperature=-1), dict(top_k=-1), dict(top_p=0), dict(top_p=1.5), dict(min_p=-0.1),
                                dict(min_p=2), dict(repeat_penalty=0), dict(repeat_last_n=-1)])
def test_bad_params(kw):
    with pytest.raises(ValueError, match=next(iter(kw))):
        SamplingParams(**kw)


# ---------------------------------------------------------------- generazione

def tiny_model(tmp_path):
    from test_moe import TOKENS, weights, write_gguf

    from engine.runtime import Context, Model

    path = write_gguf(tmp_path / "m.gguf", weights(8))
    return Context(Model(path), max_seq=48), TOKENS[:4]


def test_generate_default_is_greedy_and_sampler_is_reproducible(tmp_path):
    from engine import capi
    if not capi.available():
        pytest.skip("libengine non compilata")
    from engine.runtime import generate

    ctx, prompt = tiny_model(tmp_path)
    greedy = generate(ctx, prompt, 12)
    assert generate(ctx, prompt, 12, sampler=Sampler()) == greedy
    params = SamplingParams(temperature=1.5, seed=5)
    a = generate(ctx, prompt, 12, sampler=Sampler(params))
    assert a == generate(ctx, prompt, 12, sampler=Sampler(params))
    assert a != greedy


def test_generate_passes_prompt_and_new_tokens_to_the_penalty(tmp_path):
    from engine import capi
    if not capi.available():
        pytest.skip("libengine non compilata")
    from engine.runtime import generate

    ctx, prompt = tiny_model(tmp_path)
    seen = []

    def spy(logits, history):
        seen.append(list(history))
        return 7

    assert generate(ctx, prompt, 3, sampler=spy) == [7, 7, 7]
    assert seen == [prompt, prompt + [7], prompt + [7, 7]]
