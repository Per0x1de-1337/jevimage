"""The shapes a request and its answers take, with no dependency on torch.

This module is deliberately pure Python. An API-only install - someone who just calls a
`jev serve` instance over HTTP - gets the same Answer objects as someone running the
models locally, without installing a deep-learning stack to do it.
"""
from __future__ import annotations

import math
import numbers

__all__ = ["Answer", "check_template", "confidence", "normalise_questions",
           "positive_temperature", "TEMPERATURE_RANGE"]

# The usable range. Probabilities are reported to six decimals, so below the floor every
# distribution is one-hot and above the ceiling every one is uniform to the last digit
# printed - and past about 1e38 the fp32 division underflows to zero, which made the
# ANSWER flip to whichever option was listed first. Both ends are the same failure inf
# is refused for: a well-formed answer carrying no signal.
TEMPERATURE_RANGE = (1e-6, 1e6)


def positive_temperature(temperature):
    """A negative temperature inverts every softmax - the least similar option wins, and
    the answer still reports high confidence - so it must never reach one.

    Here rather than in `core`, so the torch-free client can refuse a bad temperature
    with the same three sentences a local `Jev` gives instead of its own wording.
    """
    # numbers.Real, not (int, float): a numpy scalar is a number, and refusing
    # np.float32(0.5) with "must be a number greater than 0, got 0.5" states a falsehood
    # about the value it prints - while np.float64 sailed through for subclassing float.
    if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
        raise ValueError(f"temperature must be a number greater than 0, "
                         f"got {temperature!r}")
    if not temperature > 0:
        raise ValueError(f"temperature must be greater than 0, got {temperature!r}")
    # inf is "greater than 0" and divides every logit to zero, so every option comes
    # back equally likely and confidence 0 - a well-formed answer carrying no signal,
    # which is worse than an error because nothing downstream can tell.
    if not math.isfinite(temperature):
        raise ValueError(f"temperature must be finite, got {temperature!r}; an infinite "
                         "temperature makes every option equally likely")
    lo, hi = TEMPERATURE_RANGE
    if not lo <= temperature <= hi:
        raise ValueError(f"temperature must be between {lo:g} and {hi:g}, got "
                         f"{temperature!r}; outside that range every option comes back "
                         "equally likely, or exactly one does, whatever the image says")
    return float(temperature)


def check_template(tpl, field: str = "prompt"):
    """A caption template must substitute exactly once.

    Testing for a literal '{}' is not enough: 'a photo of {x}', '{} and {}' and '{ {}'
    all contain a brace, all passed, and all then blew up inside str.format with a raw
    stdlib error naming neither the field nor the question. A trial substitution answers
    the only question that matters - can this template turn one option into one caption.
    """
    probe = "\x00probe\x00"
    try:
        if isinstance(tpl, str) and tpl.format(probe).count(probe) == 1:
            return tpl
    except (IndexError, KeyError, ValueError):
        pass
    # without exactly one placeholder every option gets the same caption, so the answer
    # is a uniform distribution over identical strings and `choice` is whichever came first
    raise ValueError(f"{field} {tpl!r} has no '{{}}' placeholder, so every option "
                     "would get the same caption; write e.g. 'a photo of {}'")


class Answer(dict):
    """One question's answer. A plain dict, so it serialises to JSON unchanged, with
    attribute access because `a.choice` reads better than `a["choice"]`."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(
                f"{self.get('type', 'answer')!r} answers have no {k!r}; "
                f"this one has: {', '.join(self)}") from None

    def __repr__(self):
        if "choice" in self:
            return (f"<{self['type']} {self['choice']!r} "
                    f"p={max(self['probabilities'].values()):.3f}>")
        if "noul" in self:
            return f"<noul {self['noul']:.3f}>"
        if "score" in self:
            return f"<score {self['score']:.2f}/{len(self['legend']) - 1}>"
        return f"<{self.get('type', 'answer')}>"


def confidence(p) -> float:
    """How peaked a distribution is, rescaled so uniform is 0 and certain is 1.

    Raw max-probability is not comparable across questions: 0.5 is lukewarm over two
    options and emphatic over fifty. (max - 1/K) / (1 - 1/K) removes that dependence,
    so one threshold works for every question you ask.
    """
    k = len(p)
    return (max(p) - 1 / k) / (1 - 1 / k) if k > 1 else 1.0


def normalise_questions(questions) -> dict:
    """A dict keyed by id, or a list that gets q1, q2, ... . Both are common at call
    sites; neither should need a wrapper."""
    if isinstance(questions, dict):
        return questions
    if isinstance(questions, (list, tuple)):
        out = {}
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                # same sentence core.plan gives for the dict form, so a malformed
                # question reads the same whichever shape it arrived in - and an
                # AttributeError below would escape as a 500 rather than a 422
                raise ValueError(f"a question must be an object, got {type(q).__name__}")
            qid = q.get("id") or f"q{i + 1}"
            if qid in out:
                # the dict would keep only the last: two questions in, one answer out
                raise ValueError(f"two questions share the id {qid!r}; ids must be unique")
            out[qid] = q
        return out
    raise TypeError("questions must be a {id: question} object or a list of questions")
