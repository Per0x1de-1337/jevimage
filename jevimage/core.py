"""The readout engine: one image encode, many typed questions against it.

Nothing here generates text. An image becomes one unit-norm embedding, and every
question is a cheap read against that same vector - a softmax over option embeddings,
or a matmul against a trained head. That is why N questions about one image cost
barely more than one, and why every answer arrives as a probability distribution
rather than a sentence you have to parse.

Four question types:

    choice   pick one of 2..255 named options          -> choice, probabilities, confidence
    score    place on an ordered 2..10 level rubric     -> score (a float), probabilities
    noul     one yes/no probability ("no-or-yes level") -> noul (0..1)
    head     a head you trained on your own images      -> choice, probabilities, confidence

All four take the same shape: {"type": ..., "instructions": ..., "criteria": ...}.
"""
from __future__ import annotations

import math

import torch

from .answers import (Answer, check_template, confidence,
                      positive_temperature as _positive)

__all__ = ["Answer", "Engine", "confidence", "plan", "MAX_OPTIONS", "MAX_LEVELS",
           "MAX_QUESTIONS"]

MAX_OPTIONS = 255      # options in one choice
MAX_LEVELS = 10        # levels on one score rubric
MAX_QUESTIONS = 64     # questions in one ask()
# Captions are caller-supplied and a server lives for weeks, so the text cache needs a
# ceiling. 50k captions is ~150MB at dim 768, far past any real question set.
MAX_CACHED_TEXTS = 50_000


def _flat(v) -> str:
    """A criteria entry may be a string, a list, or a nested object. Flatten to the one
    caption that gets embedded."""
    if v is None:
        return ""                      # an absent key is absent, not the word "None"
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return ", ".join(_flat(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_flat(x)}" for k, x in v.items())
    return str(v)


def plan(q: dict, heads: dict | None = None):
    """Resolve one question into (kind, texts, labels).

    Pure: it touches no model. A whole request is planned first so every string it
    needs goes through the text encoder in a single batched pass.
    """
    if not isinstance(q, dict):
        raise ValueError(f"a question must be an object, got {type(q).__name__}")
    kind = str(q.get("type") or "choice").lower()
    # Dual encoders were trained on captions, not questions, so criteria are phrased as
    # descriptions ("an indoor scene"). The template wraps each one.
    # A trial substitution, not a search for "{}": every other brace shape reaches
    # str.format() below and escapes as its own exception - {name} as a KeyError the app
    # turns into a 404 whose whole body is the placeholder name, '{} and {}' as an
    # IndexError, a lone brace as CPython's formatter text - and none of those mention
    # `prompt`. '{.__class__}' passed the literal test and captioned every option
    # "<class 'str'>", which is the exact failure the check exists to prevent.
    tpl = check_template(q.get("prompt") or "{}")
    crit = q.get("criteria")
    # `instruction` is a documented alias of `instructions` (docs/questions.md), the same
    # way `name` aliases `head` below. Reading only the plural dropped the singular
    # silently, so a single-caption noul written the documented way failed as if no
    # caption had been given at all.
    instr = _flat(q.get("instructions") or q.get("instruction") or "")

    if kind == "choice":
        if isinstance(crit, (list, tuple)):      # a bare list means label == description
            captions = [_flat(c) for c in crit]
            dupes = {c for c in captions if captions.count(c) > 1}
            if dupes:
                # the dict below would eat them, leaving a distribution over fewer
                # options than the caller listed
                raise ValueError(f"choice criteria repeats {', '.join(sorted(dupes))!r}; "
                                 "each option needs its own caption")
            crit = {c: c for c in captions}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_OPTIONS:
            raise ValueError(f"choice needs a criteria object of 2..{MAX_OPTIONS} options, "
                             f'e.g. {{"indoor": "an indoor scene", "outdoor": "the outdoors"}}')
        labels = list(crit)
        return "choice", [tpl.format(_flat(crit[k]) or k) for k in labels], labels

    if kind == "score":
        if not isinstance(crit, (list, tuple)) or not 2 <= len(crit) <= MAX_LEVELS:
            raise ValueError(f"score needs an ordered criteria array of 2..{MAX_LEVELS} "
                             'levels, low end first, e.g. ["empty", "busy", "packed"]')
        levels = [_flat(c) for c in crit]
        return "score", [tpl.format(c) for c in levels], levels

    if kind == "head":
        name = q.get("head") or q.get("name") or instr
        if not name:
            raise ValueError('head needs a name: {"type": "head", "head": "my_head"}')
        if not isinstance(name, str):
            # the membership test below is a dict lookup, so an unhashable name is a
            # TypeError out of store.Heads rather than an answer about the head
            raise ValueError(f"head name must be a string, got {type(name).__name__}")
        if heads is not None and name not in heads:
            known = ", ".join(sorted(heads)) or "none trained yet"
            raise ValueError(f"unknown head {name!r}; trained heads: {known}")
        return "head", [], name

    if kind == "noul":
        t_txt = _flat(crit.get("true")) if isinstance(crit, dict) else ""
        f_txt = _flat(crit.get("false")) if isinstance(crit, dict) else ""
        if isinstance(crit, dict) and ("true" in crit or "false" in crit) \
                and not (t_txt and f_txt):
            # a half-written pair would otherwise fall through to the sigmoid form and
            # silently discard the side that was filled in
            empty = "true" if not t_txt else "false"
            raise ValueError(f"noul criteria has an empty {empty!r} caption; both sides "
                             "of a pair need text")
        if t_txt and f_txt:
            # Two captions that mean opposite things is a real two-way decision, so it
            # reads like a choice and needs no calibration. Prefer this form.
            return "noul_pair", [tpl.format(t_txt), tpl.format(f_txt)], None
        if not instr:
            raise ValueError('noul needs instructions, or a {"true": ..., "false": ...} '
                             "criteria pair (the pair is better calibrated)")
        return "noul_sigmoid", [tpl.format(t_txt or instr)], None

    raise ValueError(f"unknown question type {kind!r}; expected one of: "
                     "choice, score, noul, head")


def _scale_temperature(encoder, scale):
    try:
        return _positive(1.0 / scale)
    except (ValueError, ZeroDivisionError, TypeError) as e:
        raise ValueError(f"encoder {getattr(encoder, 'name', '?')!r} publishes "
                         f"logit_scale={scale!r}, which reads as temperature "
                         f"1/{scale!r}: {e}") from None


def _finite(obj, path):
    """Refuse a non-finite number rather than emitting one.

    An encoder that emits nan yields nan, which json.dumps turns into invalid JSON and a
    caller turns into an unexplained crash. Naming the field here turns that into an
    error you can act on.
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            # not "(temperature must be > 0)": temperature is validated before it gets
            # anywhere near here, so telling a caller who passed a good one to check it
            # sends them after the one thing that is already right
            raise ValueError(f"{path} is {obj}: the image could not be scored (its "
                             "embedding is not finite)")
        return obj
    if isinstance(obj, dict):
        return {k: _finite(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    return obj


class Engine:
    """Answers questions against embeddings. Holds the encoder, the text cache and
    the trained heads; `jevimage.Jev` wraps it with image loading and training."""

    def __init__(self, encoder, heads: dict | None = None, temperature: float | None = None):
        self.encoder = encoder
        self.heads = heads if heads is not None else {}
        # getattr, not encoder.logit_scale: the documented encoder contract is .name,
        # .dim, .img and .txt, so a duck-typed encoder that publishes no calibration
        # must load rather than die here.
        scale = getattr(encoder, "logit_scale", None)
        # A raw cosine lives in about [0, 0.3] and softmaxes to nearly uniform. SigLIP's
        # learned logit_scale is the temperature it was trained to be read at; 100 is
        # the CLIP-family convention for models that publish no scale.
        # the encoder's scale goes through the same guard the caller's temperature does:
        # a negative logit_scale inverts every softmax, and an encoder that publishes one
        # produced a confident answer on the WRONG option rather than an error
        self.temperature = (_positive(temperature) if temperature is not None
                            else _scale_temperature(encoder, scale) if scale else 0.01)
        self._text: dict[str, torch.Tensor] = {}
        self.stats = {"text_hits": 0, "text_misses": 0}

    # -- text ------------------------------------------------------------
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Unit text embeddings, encoding only the strings not already cached, in one
        batched pass. Cached per string rather than per question, so two questions
        sharing an option pay for it once."""
        if not texts:
            return torch.empty(0, self.encoder.dim)
        missing = [t for t in dict.fromkeys(texts) if t not in self._text]
        self.stats["text_hits"] += len(texts) - len(missing)
        self.stats["text_misses"] += len(missing)
        if missing:
            for t, v in zip(missing, self.encoder.txt(missing)):
                self._text[t] = v
        out = torch.stack([self._text[t] for t in texts])
        while len(self._text) > MAX_CACHED_TEXTS:      # oldest first; dicts keep order
            del self._text[next(iter(self._text))]
        return out

    # -- answering -------------------------------------------------------
    def answer_one(self, emb, kind, texts, labels, temp=None) -> Answer:
        if kind == "head":
            h = self.heads[labels]
            self._check_head(labels, h, emb.shape[-1])
            W = h["W"].to(emb.device, torch.float32)
            b = h["b"].to(emb.device, torch.float32)
            p = torch.softmax((emb.float() @ W + b)[0] / (temp or 1.0), -1).tolist()
            cls = h["classes"]
            return Answer(type="head", head=labels,
                          choice=cls[p.index(max(p))],
                          probabilities={c: round(v, 6) for c, v in zip(cls, p)},
                          confidence=round(confidence(p), 6),
                          trained_on=h.get("n"), accuracy=h.get("cv_accuracy"))

        if kind == "noul_sigmoid":
            scale = getattr(self.encoder, "logit_scale", None)
            bias = getattr(self.encoder, "logit_bias", None)
            sim = float((emb.float() @ self.encode_texts(texts).float().T)[0, 0])
            if scale is None or bias is None:
                # Without a pairwise-trained scale AND bias there is no honest way to
                # turn one cosine into P(yes). Say so instead of returning a number
                # that looks like a probability and is not one.
                raise ValueError(
                    f"encoder {self.encoder.name!r} has no calibrated sigmoid, so a "
                    "single-caption noul cannot produce a probability. Give a "
                    '{"true": ..., "false": ...} criteria pair instead, which works '
                    "on every encoder.")
            noul = float(torch.sigmoid(torch.tensor(sim * scale + bias)))
            return Answer(type="noul", noul=round(noul, 6), basis="sigmoid readout")

        logits = emb.float() @ self.encode_texts(texts).float().T
        p = torch.softmax(logits / (temp or self.temperature), dim=-1)[0].tolist()

        if kind == "noul_pair":
            return Answer(type="noul", noul=round(p[0], 6), basis="contrast pair")

        if kind == "choice":
            return Answer(type="choice",
                          choice=labels[p.index(max(p))],
                          probabilities={k: round(v, 6) for k, v in zip(labels, p)},
                          confidence=round(confidence(p), 6))

        return Answer(type="score",
                      # the expected level index: each level's number times its probability
                      score=round(sum(i * v for i, v in enumerate(p)), 4),
                      probabilities={str(i): round(v, 6) for i, v in enumerate(p)},
                      legend={str(i): c for i, c in enumerate(labels)},
                      confidence=round(confidence(p), 6))

    def _check_head(self, name, head, dim):
        """A head is a matrix in one encoder's embedding space. Running it on another
        encoder's embedding is silently wrong when the widths happen to match, so the
        encoder name is checked, not just the shape."""
        want = head.get("encoder")
        if not want:
            # a record with no provenance cannot be checked, and an unchecked head runs
            # silently wrong wherever the widths happen to match
            raise ValueError(f"head {name!r} records no encoder, so it cannot be checked "
                             "against this one; retrain it")
        if want != self.encoder.name:
            raise ValueError(
                f"head {name!r} was trained on encoder {want!r} but this instance runs "
                f"{self.encoder.name!r}. Load {want!r}, or retrain the head here.")
        if head["W"].shape[0] != dim:
            raise ValueError(f"head {name!r} expects {head['W'].shape[0]}-dim embeddings, "
                             f"this encoder produces {dim}")
        if len(head["classes"]) != head["W"].shape[1]:
            # zip() below would silently drop the extra class, or IndexError on a short one
            raise ValueError(f"head {name!r} lists {len(head['classes'])} classes but its "
                             f"weights have {head['W'].shape[1]} columns; the file is "
                             "damaged, retrain it")

    def answer_all(self, emb, questions: dict, temperature=None) -> dict[str, Answer]:
        if len(questions) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions per request")
        if temperature is not None:
            _positive(temperature)
        # Every logit here is a cosine, which is only a cosine if the embedding is
        # unit-norm. A rescaled one saturates the softmax to exactly 1.0 and 0.0 - the
        # one number a probability-first API must never emit by accident. (nan compares
        # False, so a blank image still reaches _finite, which names the field.)
        if float((emb.float().norm(dim=-1) - 1).abs().max()) > 1e-2:
            raise ValueError(
                f"embeddings must be L2-normalised, got norm "
                f"{float(emb.float().norm(dim=-1).max()):.4g}; wrap your encoder's "
                "output in jevimage.encoders.unit()")
        plans = {qid: plan(q, self.heads) for qid, q in questions.items()}
        # every string the request needs, encoded once, in one batched pass
        self.encode_texts([t for _, texts, _ in plans.values() for t in texts])
        return {qid: Answer(_finite(self.answer_one(emb, *p, temperature), qid))
                for qid, p in plans.items()}
