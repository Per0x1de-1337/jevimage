"""Train a readout on your own labelled images.

The encoder stays frozen. All that is fitted is a linear head over its embeddings -
multinomial logistic regression - which is why this takes under a second on a thousand
images and stores as a few tens of kilobytes. Inference afterwards is one matmul on an
embedding the request already computed, so a trained head costs nothing at query time.

The point of keeping the embeddings alongside the head is the COMPARISON. "Will training
help me?" has no general answer - measured on a 10-class set it was +5.7 points overall,
+33 on one class and -10 on another - so the honest thing is to answer it per class, on
held-out data, for the user's own images. That needs the embeddings, and they are small
(fp16: a thousand examples of a 768-dim encoder is 1.5MB).
"""
from __future__ import annotations

import random
import time

import torch

from .answers import check_template

__all__ = ["train", "compare", "prompts_for", "public",
           "MAX_EXAMPLES", "MAX_CLASSES", "FOLDS"]

MAX_EXAMPLES = 4000
MAX_CLASSES = 100
FOLDS = 5


def _fit(X, y, n_classes, idx, epochs: int = 500, wd: float = 2e-3, lr: float = 0.05):
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(X[idx] @ W + b, y[idx]).backward()
        opt.step()
    return W.detach(), b.detach()


def _folds(y, n_classes, k: int = FOLDS) -> list[list[int]]:
    """Stratified folds, deterministic per class so a retrain reports the same number."""
    buckets = [[] for _ in range(k)]
    for c in range(n_classes):
        ids = (y == c).nonzero().flatten().tolist()
        random.Random(c).shuffle(ids)
        for j, i in enumerate(ids):
            buckets[j % k].append(i)
    return buckets


def train(embs, labels: list[str], epochs: int = 500, wd: float = 2e-3) -> dict:
    """Fit a head. Returns the head plus held-out accuracy, overall and per class."""
    # the server enforces 1..5000; the floor belongs here, where local and remote both
    # route through. Zero epochs ran no optimiser step and still saved an all-zero head
    # that `heads` listed as trained and `ask` answered one class for, at confidence 0.
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1; got {epochs}")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("need at least 2 distinct labels")
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"at most {MAX_CLASSES} classes")
    # 'red' and 'red ' are two classes that print identically in every table this
    # produces, so the user sees one class scoring half of what it should and cannot see why
    by_stripped: dict[str, list[str]] = {}
    for c in classes:
        by_stripped.setdefault(c.strip(), []).append(c)
    clash = next((v for v in by_stripped.values() if len(v) > 1), None)
    if clash:
        raise ValueError(f"labels {clash!r} differ only by surrounding whitespace, so "
                         "they would print identically; make them distinct")
    K = len(classes)
    y = torch.tensor([classes.index(l) for l in labels])
    X = embs.float().cpu()

    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    thin = [c for c, n in counts.items() if n < 2]
    if thin:
        raise ValueError(f"every class needs at least 2 examples; too few for: "
                         f"{', '.join(sorted(thin))}")

    t0 = time.perf_counter()
    cv, per_class = None, {}
    # Cross-validation needs at least one example per class per fold; with fewer, report
    # no accuracy rather than a number computed on a fold that was missing a class.
    k = min(FOLDS, min(counts.values()))
    if k >= 2:
        buckets = _folds(y, K, k)
        ok = n = 0
        hit = dict.fromkeys(range(K), 0)
        tot = dict.fromkeys(range(K), 0)
        for f in range(k):
            te = buckets[f]
            tr = [i for g in range(k) if g != f for i in buckets[g]]
            W, b = _fit(X, y, K, tr, epochs, wd)
            pred = (X[te] @ W + b).argmax(1)
            ok += int((pred == y[te]).sum())
            n += len(te)
            for p_, t_ in zip(pred.tolist(), y[te].tolist()):
                tot[t_] += 1
                hit[t_] += p_ == t_
        cv = ok / n
        per_class = {classes[c]: round(hit[c] / tot[c], 4) for c in range(K) if tot[c]}

    W, b = _fit(X, y, K, list(range(len(y))), epochs, wd)   # final head on everything
    return {
        "W": W, "b": b, "classes": classes, "n": len(y),
        "counts": counts,
        "cv_accuracy": cv, "per_class": per_class, "folds": k if k >= 2 else None,
        "train_ms": round((time.perf_counter() - t0) * 1000, 1),
        "emb": X.half(),                       # kept so the comparison can be recomputed
        "labels": [classes.index(l) for l in labels],
    }


def compare(head: dict, text_emb) -> dict:
    """Trained vs zero-shot on the SAME images, per class.

    The trained side is the cross-validated number: a head scored on the examples it was
    fitted on would win by construction and tell the user nothing. The zero-shot side has
    no training to hold out, so every example counts.
    """
    classes = head["classes"]
    X = head["emb"].float()
    y = torch.tensor(head["labels"])
    K = len(classes)

    # predictions from the prompts alone - the baseline a trained head has to beat
    zs = (X @ text_emb.float().T).argmax(1)
    zs_overall = float((zs == y).float().mean())
    rows = []
    for c in range(K):
        m = y == c
        if not bool(m.any()):
            continue
        rows.append({
            "label": classes[c],
            "n": int(m.sum()),
            "zero_shot": round(float((zs[m] == y[m]).float().mean()), 4),
            "trained": head["per_class"].get(classes[c]),
        })
    trained_overall = head.get("cv_accuracy")
    return {
        "overall": {"zero_shot": round(zs_overall, 4), "trained": trained_overall,
                    "delta": round(trained_overall - zs_overall, 4)
                    if trained_overall is not None else None},
        "per_class": rows,
        "basis": "trained is k-fold held-out; zero-shot needs no holdout so it is scored "
                 "on every example",
    }


def prompts_for(classes: list[str], template: str = "a photo of {}") -> list[str]:
    """The prompts a user would plausibly write, so the baseline is a fair one."""
    # the same check a question's `prompt` gets: a template that cannot vary per class
    # captions every class identically, and the zero-shot side of the comparison becomes
    # an argmax tie-break reported as a measurement
    check_template(template, "template")
    return [template.format(c.replace("_", " ")) for c in classes]


def public(name: str, head: dict) -> dict:
    """The head without its weights or its stored embeddings - what an API returns and
    what `repr(head)` shows. Weights stay server-side; the metadata is the useful part."""
    return {
        "name": name,
        "encoder": head.get("encoder"),      # which encoder's space these weights live in
        "classes": head["classes"],
        "n": head["n"],
        "counts": head.get("counts", {}),
        "cv_accuracy": head.get("cv_accuracy"),
        "per_class": head.get("per_class", {}),
        "folds": head.get("folds"),
        "created": head.get("created"),
        "dim": int(head["W"].shape[0]) if "W" in head else None,
        "bytes": int(head["W"].numel() * 4 + head["b"].numel() * 4) if "W" in head else None,
    }


def demo() -> None:
    torch.manual_seed(0)
    d, per = 64, 30
    # three separable blobs plus noise, so a head should clearly beat a bad prompt
    centres = torch.randn(3, d) * 3
    X, labels = [], []
    for c in range(3):
        X.append(centres[c] + torch.randn(per, d))
        labels += [f"class_{c}"] * per
    X = torch.nn.functional.normalize(torch.cat(X), dim=-1)

    h = train(X, labels, epochs=200)
    assert h["classes"] == ["class_0", "class_1", "class_2"]
    assert h["n"] == 90 and h["counts"]["class_0"] == 30
    assert h["cv_accuracy"] > 0.9, h["cv_accuracy"]
    assert h["emb"].dtype == torch.float16 and h["emb"].shape == (90, d)

    # a deliberately useless prompt matrix: the head must beat it
    bad = torch.nn.functional.normalize(torch.randn(3, d), dim=-1)
    cmp = compare(h, bad)
    assert cmp["overall"]["trained"] > cmp["overall"]["zero_shot"]
    assert len(cmp["per_class"]) == 3
    assert all(r["trained"] is not None for r in cmp["per_class"])

    # a prompt matrix aimed AT the true centres should be strong - the comparison must
    # be able to report zero-shot winning, not just flatter training
    good = torch.nn.functional.normalize(centres, dim=-1)
    assert compare(h, good)["overall"]["zero_shot"] > 0.9

    # guards
    for bad_labels, msg in ((["only"] * 4, "2 distinct"), (["a", "a", "a", "b"], "at least 2")):
        try:
            train(X[:4], bad_labels, epochs=5)
            raise AssertionError(f"accepted bad labels: {bad_labels}")
        except ValueError as e:
            assert msg in str(e), (msg, str(e))

    p = public("t", {**h, "encoder": "e"})
    assert "W" not in p and "emb" not in p and p["dim"] == d
    print("training selftest ok - head beats a bad prompt, comparison is honest both "
          "ways, thin classes refused, weights never leak into the public shape")


if __name__ == "__main__":
    demo()
