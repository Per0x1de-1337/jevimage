"""The readout engine: one image encode, many typed questions against it.

Nothing here generates text. An image becomes one unit-norm embedding, and every
question is a cheap read against that same vector - a softmax over option embeddings,
or a matmul against a trained head. That is why N questions about one image cost
barely more than one, and why every answer arrives as a probability distribution
rather than a sentence you have to parse.

Four question types:

    choice   pick one of 2..255 named options          -> choice, probabilities, confidence
    score    place on an ordered 2..10 level rubric    -> score (a float), probabilities
    noul     one yes/no probability ("no-or-yes level") -> noul (0..1)
    head     a head you trained on your own images     -> choice, probabilities, confidence

All four take the same shape: {"type": ..., "instructions": ..., "criteria": ...}.

`choice` and `noul` have two readouts, and you do not pick between them. When the options
you give are the same words in a different order, what is PRESENT cannot separate them, so
the answer is read off the patch grid instead of the pooled vector - same forward pass,
but it needs the image rather than a stored embedding. See `_rearrangement`.
"""
from __future__ import annotations

import re

import math

import torch

from .store import env
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
        texts = [tpl.format(_flat(crit[k]) or k) for k in labels]
        # same words in a different order: presence cannot separate these, so read the grid
        use_grid = grid_readout_enabled() and _rearrangement(texts)
        return ("choice_grid" if use_grid else "choice"), texts, labels

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
            texts = [tpl.format(t_txt), tpl.format(f_txt)]
            use_grid = grid_readout_enabled() and _rearrangement(texts)
            return ("noul_grid" if use_grid else "noul_pair"), texts, None
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


# Splitting on the SPATIAL words matters most. "a red circle above a blue square" has no
# comma and no "and", so without these it never decomposes, the readout sees one long
# phrase, and it degenerates to the pooled answer it is supposed to beat (measured: 0.661
# vs 0.525 on a swap pair, neither crossing 0.5; with these, 0.98 vs 0.02).
    # NOT split on: then / followed by / before / after / between. Splitting a sequence
    # on its own connectives makes the segment set ORDER-INVARIANT - "a washer then a nut"
    # and "a nut then a washer" produce the same parts - so the comparison ties and the
    # only signal is thrown away. Measured: adding them took the order family from 0.857
    # to 0.357, below chance. The whole-phrase segment is what carries sequence.
_SPLIT = re.compile(
    r",|\b(?:and|while|with|without|holding|wearing|riding|carrying"
    r"|above|below|under|underneath|beneath|over|on top of|on|in front of|behind"
    r"|inside|in|next to|beside|near|to the left of|to the right of|left of|right of"
    r")\b")
_REGIONS: dict[int, list[list[int]]] = {}
_POOLERS: dict[tuple, "torch.Tensor"] = {}


def segments(text: str) -> list[str]:
    """Break a criterion into the pieces a region can be asked about.

    Deliberately regex-based and dependency-free: the package ships with pillow alone.
    The benchmark numbers for the grid readout used spaCy noun chunks and verb triples on COCO
    captions; API criteria are short phrases a person typed, where splitting on commas
    and the handful of connectives below recovers the same pieces. The whole phrase is
    always kept, so a criterion with no connective still gets scored.
    """
    out = [text.strip()]
    out += [p.strip() for p in _SPLIT.split(text) if len(p.strip()) > 2]
    seen, uniq = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower()); uniq.append(t)
    return uniq


_STOP = {"a", "an", "the", "of", "with", "in", "on", "at", "to", "and", "is", "are",
         "some", "this", "that", "it", "its", "there"}


def grid_readout_enabled() -> bool:
    """Whether `choice` and `noul` may switch to the patch-grid reading. Off by default.

    The readout wins on the benchmark protocol that scores two images against two
    captions, and that is what the numbers behind it were measured on. It does NOT win on
    the task this library actually performs - one image, two captions - which was measured
    afterwards on SugarCrepe swap, n=911, against a whole-caption baseline:

        pooled                70.25
        this routing          68.50      -1.75
        benchmark config      64.22      -6.03

    The earlier, larger figures compared against a baseline that scored a caption as the
    mean of its segment embeddings, which is about 18 points weaker than encoding the
    caption. Shipping the routing on by default would make binding questions quietly worse
    for every caller, so it is opt-in until the readout beats pooled on one image.

    Set JEVIMAGE_GRID=1 to enable it for experiments. Read per call, not at import, so a
    test or a notebook can turn it on and off.
    """
    return str(env("JEVIMAGE_GRID", "")).strip().lower() not in ("", "0", "false", "no")


def _rearrangement(texts) -> bool:
    """Do these options say the same words in a different order?

    That is the whole test for whether the grid readout is worth using, and it needs no
    image, so `plan` can decide it. If the options differ in WHICH words they contain, the
    question is about what is present and the pooled vector answers it better - measured on
    SugarCrepe, the grid scores 75.24 against 92.01 on replace_obj. If they contain the same
    words in a different arrangement, presence cannot separate them at all and the grid is
    the only thing that can - 70.42 against 54.80 on swap_att, where pooled is below chance.

    Measured over 2933 BiVLC instances this fires on 249, of which 248 are genuine swaps:
    precision 0.996, nothing fitted.
    """
    bags = [tuple(sorted(w for w in t.lower().replace(".", " ").replace(",", " ").split()
                         if w not in _STOP)) for t in texts]
    return len(set(bags)) == 1 and len({t.strip().lower() for t in texts}) == len(texts)


NEG_WORDS = {"not", "no", "n't", "isn't", "aren't", "without", "never", "nothing",
             "none", "neither", "nor", "lacks", "lacking", "missing"}
SAME_CAP = 0.90


def pair_warning(texts, tv):
    """Why a criteria pair cannot be answered, when that is decidable.

    A pair is meant to be two alternatives. A negation pair is not two alternatives, it is
    one caption with "not" inserted, and dual encoders do not represent negation: measured
    cos(true, false) = 0.927 for "the girl is wearing pink dress" against "the girl is NOT
    wearing pink dress", where two real alternatives sit at 0.83-0.85. At that distance
    there is nothing to compare and the answer is noise, so say so rather than hand back a
    confident-looking number.
    """
    a, b = (set(re.findall(r"[a-z']+", t.lower())) for t in texts)
    diff = (a - b) | (b - a)
    if diff and diff <= NEG_WORDS:
        return ("the two captions differ only by a negation, and dual encoders do not "
                "represent 'not' - both readings land on nearly the same vector, so this "
                "answer is noise. Give two real alternatives instead, e.g. "
                '{"true": "a pink dress", "false": "a white dress"}.')
    sim = float(tv[0] @ tv[1])
    if sim > SAME_CAP:
        return (f"the two captions are {sim:.3f} similar to each other, so there is little "
                "for the readout to separate. Make them differ in which attribute binds to "
                "which object, not in wording.")
    return None


def pooler(side: int, device, dtype=None) -> "torch.Tensor":
    """A (n_region, n_patch) mean-pooling matrix, so every region is one matmul.

    The obvious implementation loops the rectangles and calls .mean(0) on each. With 406
    regions that is 406 kernel launches, and it measured **65 ms** on an A10G against a
    6.4 ms vision pass - the readout cost seven times the model it was reading. As one
    matmul it is 406x196 @ 196x768, about 61 MFLOP, and disappears into the noise.
    """
    key = (side, str(device), str(dtype))
    m = _POOLERS.get(key)
    if m is None:
        reg = regions(side)
        m = torch.zeros(len(reg), side * side, dtype=dtype or torch.float32, device=device)
        for i, rect in enumerate(reg):
            m[i, torch.tensor(rect, device=device)] = 1.0 / len(rect)
        _POOLERS[key] = m
    return m


def regions(side: int) -> list[list[int]]:
    """Multi-scale rectangles over the patch grid, as flat index lists.

    Small rectangles behave like object readouts; large ones give a phrase spanning two
    objects the local pool it needs. Cached per grid size - the indices never change.
    """
    if side in _REGIONS:
        return _REGIONS[side]
    out = []
    for size in (2, 3, 4, 5, 7, 10, side):
        if size > side:
            continue
        step = max(1, size // 2)
        starts = sorted(set(list(range(0, side - size + 1, step)) + [side - size]))
        for r in starts:
            for c in starts:
                out.append([rr * side + cc for rr in range(r, r + size)
                            for cc in range(c, c + size)])
    _REGIONS[side] = out
    return out


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
    def answer_one(self, emb, kind, texts, labels, temp=None, patches=None) -> Answer:
        if kind.endswith("_grid"):
            return self.grid_read(patches, texts, temp, kind, labels)

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
            out = dict(type="noul", noul=round(p[0], 6), basis="contrast pair")
            # The same check the grid reading does, for the same reason. A negation pair is
            # not two alternatives, and it is not a rearrangement either, so it routes here
            # - which is exactly where the warning is needed, because nothing else about
            # this answer tells the caller their 0.65 is noise.
            warning = pair_warning(texts, self.encode_texts(texts).float())
            if warning:
                out["warning"] = warning
            return Answer(**out)

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

    def grid_read(self, patches, texts, temp=None, kind="noul_grid", labels=None) -> Answer:
        """Score captions by where things are, not just by what is present.

        The pooled vector answers "is a helmet in this picture". It cannot answer "is the
        helmet ON HIS HEAD", because pooling discards the arrangement. This reads the patch
        grid instead: every phrase in the criterion takes its best-matching region, and the
        captions are compared on that.

        Measured on BiVLC `swap` (359 real COCO instances whose negative caption contains
        the SAME WORDS in a different order, so a bag-of-words score is at chance):

            pooled vector (what `noul` does)   group score 10.03
            this readout                       group score 32.03

        and on controlled two-object images where the captions differ only in the relation:

            pooled vector    0.525 left/right, 0.470 above/below  (chance is 0.500)
            this readout     1.000 and 1.000

        Cost is one forward pass either way - the patches were already computed and thrown
        away. This is not a question type. `plan` routes `choice` and `noul` here on its own
        when the options are word-rearrangements of each other, because that is exactly and
        only when presence cannot separate them; on plain presence questions the pooled
        vector is better and is what runs. The answer keeps the shape of the type that was
        asked, so no caller has to know this ran.
        """
        if patches is None:
            raise ValueError(
                "these options are the same words in a different order, so telling them "
                "apart needs the patch grid rather than a stored embedding; call ask() "
                "with the image, not ask_embedding()")
        if patches.ndim == 3:
            patches = patches[0]
        side = int(round(patches.shape[0] ** 0.5))
        # every region in one matmul; see pooler() for why this is not a loop
        pooled = pooler(side, patches.device, torch.float32) @ patches.float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        scores = []
        for t in texts:
            tv = self.encode_texts(segments(t)).float()
            sim = pooled @ tv.T                                  # (n_region, n_seg)
            # softmax over regions rather than a hard max: measured better on BiVLC swap
            # (group 32.03 vs 30.64) because one noisy region cannot carry a segment
            w = torch.softmax(sim / 0.02, dim=0)
            scores.append(float((w * sim).sum(0).mean()))
        p = torch.softmax(torch.tensor(scores) / (temp or self.temperature), dim=-1).tolist()

        if kind == "choice_grid":
            # `choice` has no `basis` field, so it reports the readout under its own name
            return Answer(type="choice", readout="grid",
                          choice=labels[p.index(max(p))],
                          probabilities={k: round(v, 6) for k, v in zip(labels, p)},
                          confidence=round(confidence(p), 6))

        # No `pair_warning` here, and the reason is worth writing down. Both of its cases
        # belong to the POOLED reading. A rearrangement has the same words on both sides, so
        # it is always near-identical to its swap - measured 0.975 on a controlled two-object
        # image - which means the 0.90 similarity warning fires on essentially every question
        # this readout exists to answer, and suppressing the confidence there suppressed it
        # on exactly the right answers. High pooled similarity is the PRECONDITION for
        # reading the grid, not a reason to distrust it. The negation case cannot reach here
        # at all: adding "not" changes which words are present, so such a pair is not a
        # rearrangement and is planned as `noul_pair`, where the warning is emitted.
        #
        # `basis` is noul's existing field for how it was read, so the grid reading extends
        # that enum rather than adding a second field beside it.
        return Answer(type="noul", noul=round(p[0], 6), basis="grid readout",
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

    def answer_all(self, emb, questions: dict, temperature=None,
                   patches=None) -> dict[str, Answer]:
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
        return {qid: Answer(_finite(self.answer_one(emb, *p, temperature, patches), qid))
                for qid, p in plans.items()}

    @staticmethod
    def needs_grid(questions: dict) -> bool:
        """Whether any question in this request needs the patch grid.

        Checked before encoding so an ordinary request never pays for the extra readout,
        and so the grid path can be taken in one pass instead of re-encoding.

        The router decides this from the criteria, not from a type name, so the only
        honest way to answer is to plan. `plan` is pure and already runs again for real
        below; a malformed question raises here, and swallowing it leaves the real plan to
        produce the same error it always did.
        """
        for q in (questions.values() if isinstance(questions, dict) else questions):
            try:
                if plan(q)[0].endswith("_grid"):
                    return True
            except Exception:
                continue
        return False
