"""Fitting a head, the honesty of its numbers, and what leaves the process."""
import pytest
import torch

from jevimage import training


def blobs(d=16, per=12, k=3, seed=0):
    """k separable clusters of unit vectors: a head must be able to learn these, so a
    low accuracy here means the fit is broken rather than the data being hard."""
    torch.manual_seed(seed)
    centres = torch.randn(k, d) * 3
    X = torch.cat([centres[c] + torch.randn(per, d) for c in range(k)])
    labels = [f"c{c}" for c in range(k) for _ in range(per)]
    return torch.nn.functional.normalize(X, dim=-1), labels, centres


def test_head_record_shape():
    X, labels, _ = blobs()
    h = training.train(X, labels, epochs=100)
    assert h["classes"] == ["c0", "c1", "c2"]
    assert h["n"] == 36 and h["counts"] == {"c0": 12, "c1": 12, "c2": 12}
    assert h["W"].shape == (16, 3) and h["b"].shape == (3,)
    assert h["folds"] == training.FOLDS
    assert h["emb"].dtype == torch.float16 and h["emb"].shape == (36, 16)
    assert h["labels"] == [0] * 12 + [1] * 12 + [2] * 12


def test_accuracy_is_held_out_not_fitted():
    X, labels, _ = blobs()
    h = training.train(X, labels, epochs=100)
    assert h["cv_accuracy"] > 0.9
    assert set(h["per_class"]) == {"c0", "c1", "c2"}

    # labels shuffled against the vectors: nothing to learn, so the held-out number has
    # to collapse to chance. A head scored on its own training set would not.
    scrambled = training.train(X, [labels[(i * 7) % len(labels)] for i in range(len(labels))],
                               epochs=100)
    assert scrambled["cv_accuracy"] < 0.6


def test_folds_shrink_to_the_thinnest_class():
    X, labels, _ = blobs(per=3, k=2)
    assert training.train(X, labels, epochs=20)["folds"] == 3


@pytest.mark.parametrize("labels, msg", [
    (["one"] * 6, "at least 2 distinct labels"),
    (["a"] * 5 + ["b"], "every class needs at least 2 examples"),
])
def test_train_refuses_data_it_cannot_score(labels, msg):
    X, _, _ = blobs(per=6, k=1)
    with pytest.raises(ValueError, match=msg):
        training.train(X[:len(labels)], labels, epochs=5)


def test_too_many_classes_refused():
    X = torch.nn.functional.normalize(torch.randn(202, 16), dim=-1)
    with pytest.raises(ValueError, match="at most 100 classes"):
        training.train(X, [f"c{i // 2}" for i in range(202)], epochs=5)


def test_comparison_is_honest_in_both_directions():
    X, labels, centres = blobs()
    h = training.train(X, labels, epochs=100)

    useless = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    cmp = training.compare(h, useless)
    assert set(cmp) == {"overall", "per_class", "basis"}
    assert set(cmp["overall"]) == {"zero_shot", "trained", "delta"}
    assert cmp["overall"]["trained"] > cmp["overall"]["zero_shot"]
    assert cmp["overall"]["delta"] == pytest.approx(
        cmp["overall"]["trained"] - cmp["overall"]["zero_shot"], abs=1e-4)
    assert [r["label"] for r in cmp["per_class"]] == ["c0", "c1", "c2"]
    assert all(r["n"] == 12 and r["trained"] is not None for r in cmp["per_class"])

    # prompts aimed at the true centres: the comparison must be able to report the
    # baseline winning, or it is just flattering the head
    good = training.compare(h, torch.nn.functional.normalize(centres, dim=-1))
    assert good["overall"]["zero_shot"] > 0.9


def test_prompts_for():
    assert training.prompts_for(["fire_hydrant", "bus"]) == ["a photo of fire hydrant",
                                                            "a photo of bus"]
    assert training.prompts_for(["bus"], "a blurry {}") == ["a blurry bus"]


def test_public_leaves_the_weights_behind():
    X, labels, _ = blobs()
    h = {**training.train(X, labels, epochs=20), "encoder": "toy", "created": 1.0}
    p = training.public("street", h)
    assert not {"W", "b", "emb", "labels"} & set(p)
    assert p["name"] == "street" and p["encoder"] == "toy"
    assert p["dim"] == 16 and p["bytes"] == (16 * 3 + 3) * 4
    assert p["classes"] == h["classes"] and p["n"] == 36


# -- through the public API ------------------------------------------------

def test_train_end_to_end(jev, trained, heads_dir, png):
    assert trained.name == "colours"
    assert trained.classes == ["blue", "red"]
    assert trained.n == 8 and trained.counts == {"blue": 4, "red": 4}
    assert trained.encoder == "toy"          # provenance, so the head cannot be misapplied
    assert 0.0 <= trained.accuracy <= 1.0
    assert set(trained.per_class) == {"blue", "red"}
    assert trained.comparison["overall"]["trained"] == trained.accuracy
    assert (heads_dir / "colours.pt").exists()

    listed = jev.list_heads()
    assert [h["name"] for h in listed] == ["colours"]
    assert listed[0]["encoder"] == "toy" and listed[0]["dim"] == 16
    assert "W" not in listed[0]

    assert jev.head("colours").classes == ["blue", "red"]
    assert jev.ask(png, {"q": {"type": "head", "head": "colours"}})["q"].head == "colours"

    assert jev.delete_head("colours") is True
    assert jev.delete_head("colours") is False
    with pytest.raises(ValueError, match="unknown head"):
        jev.ask(png, {"q": {"type": "head", "head": "colours"}})


def test_train_accepts_pairs_and_mappings(jev, ds):
    reds = sorted((ds / "red").iterdir())
    blues = sorted((ds / "blue").iterdir())

    mapped = jev.train("m", {"red": reds, "blue": blues}, epochs=20)
    paired = jev.train("p", [(p, "red") for p in reds] + [(p, "blue") for p in blues],
                       epochs=20)
    assert mapped.classes == paired.classes == ["blue", "red"]
    assert mapped.n == paired.n == 8

    capped = jev.train("c", {"red": reds, "blue": blues}, epochs=20, max_per_class=2)
    assert capped.counts == {"blue": 2, "red": 2}

    with pytest.raises(ValueError, match="image, label"):
        jev.train("bad", [(reds[0], "red", "extra")], epochs=5)
    with pytest.raises(TypeError, match="folder path"):
        jev.train("bad", 7, epochs=5)
