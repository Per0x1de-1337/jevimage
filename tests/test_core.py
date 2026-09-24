"""The readout engine, and the image/folder front door it is reached through."""
import base64

import pytest
import torch
from PIL import Image

import jevimage
from conftest import Toy
from jevimage import core

SCENE = {"type": "choice",
         "criteria": {"indoor": "an indoor scene", "outdoor": "the outdoors"}}
BUSY = {"type": "score", "criteria": ["empty", "busy", "packed"]}
PAIR = {"type": "noul", "criteria": {"true": "a cat", "false": "no cat"}}


def sums_to_one(p):
    return abs(sum(p.values()) - 1.0) < 1e-5


# -- the four primitives ---------------------------------------------------

def test_choice_keys_and_distribution(jev, png):
    a = jev.ask(png, {"scene": SCENE})["scene"]
    assert set(a) == {"type", "choice", "probabilities", "confidence"}
    assert a.type == "choice"
    assert set(a.probabilities) == {"indoor", "outdoor"}
    assert sums_to_one(a.probabilities)
    assert a.choice == max(a.probabilities, key=a.probabilities.get)
    assert 0.0 <= a.confidence <= 1.0


def test_score_keys_and_expected_level(jev, png):
    a = jev.ask(png, {"busy": BUSY})["busy"]
    assert set(a) == {"type", "score", "probabilities", "legend", "confidence"}
    assert a.legend == {"0": "empty", "1": "busy", "2": "packed"}
    assert set(a.probabilities) == {"0", "1", "2"}
    assert sums_to_one(a.probabilities)
    assert 0.0 <= a.score <= 2.0
    expected = sum(int(i) * v for i, v in a.probabilities.items())
    assert abs(a.score - expected) < 1e-3


def test_noul_pair_keys(jev, png):
    a = jev.ask(png, {"cat": PAIR})["cat"]
    assert set(a) == {"type", "noul", "basis"}
    assert a.type == "noul" and a.basis == "contrast pair"
    assert 0.0 <= a.noul <= 1.0


def test_noul_sigmoid_needs_a_calibrated_encoder(jev, toy_cal, heads_dir, png):
    bare = {"type": "noul", "instructions": "a cat"}
    with pytest.raises(ValueError) as e:
        jev.ask(png, {"cat": bare})
    assert "criteria pair" in str(e.value) and "toy" in str(e.value)

    a = jevimage.Jev(toy_cal, heads_dir=heads_dir).ask(png, {"cat": bare})["cat"]
    assert set(a) == {"type", "noul", "basis"}
    assert a.basis == "sigmoid readout" and 0.0 <= a.noul <= 1.0


def test_head_keys_and_distribution(jev, trained, png):
    a = jev.ask(png, {"q": {"type": "head", "head": "colours"}})["q"]
    assert set(a) == {"type", "head", "choice", "probabilities", "confidence",
                      "trained_on", "accuracy"}
    assert a.head == "colours"
    assert set(a.probabilities) == {"red", "blue"}
    assert sums_to_one(a.probabilities)
    assert a.choice in a.probabilities
    assert a.trained_on == 8


# -- confidence ------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 50])
def test_confidence_spans_uniform_to_peak(k):
    assert core.confidence([1 / k] * k) == pytest.approx(0.0, abs=1e-9)
    assert core.confidence([1.0] + [0.0] * (k - 1)) == pytest.approx(1.0)
    # the point of the rescaling: the same raw max means different things at K=2 and K=50
    half = [0.5] + [0.5 / (k - 1)] * (k - 1)
    assert core.confidence(half) == pytest.approx((0.5 - 1 / k) / (1 - 1 / k))


def test_confidence_reaches_both_ends_through_the_engine(jev, png):
    same = {"type": "choice", "criteria": {"a": "one caption", "b": "one caption"}}
    assert jev.ask(png, {"q": same})["q"].confidence == pytest.approx(0.0, abs=1e-5)
    # a near-zero temperature drives any real gap to a one-hot
    assert jev.ask(png, {"q": SCENE}, temperature=1e-6)["q"].confidence == pytest.approx(1.0)


# -- numbers that must never escape ---------------------------------------

@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_temperature_must_be_positive(jev, png, bad):
    with pytest.raises(ValueError, match="greater than 0"):
        jev.ask(png, {"scene": SCENE}, temperature=bad)


@pytest.mark.parametrize("bad", [0, -1, -0.01, "hot"])
def test_constructor_temperature_is_checked_too(toy, heads_dir, bad):
    """A negative one inverts every softmax and still reports confidence 0.999998; a
    zero used to fall through the `or` and be replaced by the encoder's default."""
    with pytest.raises(ValueError, match="greater than 0"):
        jevimage.Jev(toy, heads_dir=heads_dir, temperature=bad)


def test_a_rescaled_embedding_is_refused_not_saturated(jev, png):
    """The logits are cosines only if the embedding is unit-norm. A rescaled one gives
    p=1.0 and p=0.0 exactly - the one number a probability-first API must not invent."""
    with pytest.raises(ValueError, match="L2-normalised"):
        jev.ask_embedding(jev.embed(png) * 50, {"scene": SCENE})


def test_an_ensemble_accepts_a_member_without_a_calibration():
    """Same contract inside Ensemble: a member that omits logit_scale must ensemble,
    not raise AttributeError. An ensemble with one uncalibrated member has no scale."""
    from jevimage.encoders import Ensemble

    class Plain:                       # publishes neither, as the contract allows
        name, dim = "plain", 16

    class Cal(Plain):
        name, dim = "cal", 16
        logit_scale, logit_bias = 100.0, -10.0

    a, b = Cal(), Cal()
    b.name = "cal-b"
    assert Ensemble([a, b]).logit_scale == 100.0
    assert Ensemble([a, b]).logit_bias is None      # the bias never averages
    assert Ensemble([a, Plain()]).logit_scale is None


def test_an_encoder_without_a_calibration_still_loads(heads_dir, png):
    """The documented contract is four members; logit_scale/logit_bias are optional, so
    a plain object that omits them must not die inside Engine.__init__."""

    # deliberately NOT a jevimage.Encoder subclass: the base class declares
    # logit_scale/logit_bias, and the case under test is an encoder where they are
    # absent entirely. Only the four contract members are borrowed from Toy.
    class Plain:
        name, dim = "plain", 16
        __init__, _e = Toy.__init__, Toy._e
        img, txt = Toy.img, Toy.txt

    jev = jevimage.Jev(Plain(), heads_dir=heads_dir)
    assert sums_to_one(jev.ask(png, {"scene": SCENE})["scene"].probabilities)
    with pytest.raises(ValueError, match="no calibrated sigmoid"):
        jev.ask(png, {"q": {"type": "noul", "instructions": "a cat"}})


def test_nan_is_refused_and_named(jev):
    nan = torch.full((1, 16), float("nan"))
    with pytest.raises(ValueError) as e:
        jev.ask_embedding(nan, {"scene": SCENE})
    msg = str(e.value)
    # the field that went non-finite has to be named: that is the whole point of _finite
    assert "scene" in msg and "not finite" in msg
    # and it must NOT send the caller after the temperature. Temperature is validated
    # long before this point, so blaming it points at the one input already known good.
    assert "temperature" not in msg


# -- planning --------------------------------------------------------------

@pytest.mark.parametrize("q, msg", [
    ({"type": "choice", "criteria": {"only": "one option"}}, "2..255"),
    ({"type": "choice", "criteria": "indoor"}, "2..255"),
    ({"type": "score", "criteria": [str(i) for i in range(11)]}, "2..10"),
    ({"type": "score", "criteria": {"a": 1, "b": 2}}, "ordered criteria array"),
    ({"type": "vibes", "criteria": {"a": "x", "b": "y"}}, "unknown question type"),
    ({"type": "head", "head": "missing"}, "unknown head 'missing'"),
    ({"type": "head"}, "head needs a name"),
    ({"type": "noul"}, "criteria pair"),
    ("not an object", "must be an object"),
])
def test_plan_rejects(q, msg):
    with pytest.raises(ValueError) as e:
        core.plan(q, heads={"colours": {}})
    assert msg in str(e.value)


@pytest.mark.parametrize("crit", [{"true": "a cat"}, {"false": "no cat"}, {}])
def test_plan_rejects_a_half_specified_noul_pair(crit):
    # Half a pair is not a pair. Before _flat mapped None to "", the missing side became
    # the literal caption "None" and the question was answered against it.
    with pytest.raises(ValueError):
        core.plan({"type": "noul", "criteria": crit})


def test_plan_shapes():
    kind, texts, labels = core.plan(SCENE)
    assert (kind, labels) == ("choice", ["indoor", "outdoor"])
    assert texts == ["an indoor scene", "the outdoors"]

    # a bare list means the label is its own description
    kind, texts, labels = core.plan({"type": "choice", "criteria": ["cat", "dog"]})
    assert (kind, texts, labels) == ("choice", ["cat", "dog"], ["cat", "dog"])

    kind, texts, labels = core.plan({**BUSY, "prompt": "a photo of a {} room"})
    assert kind == "score" and labels == ["empty", "busy", "packed"]
    assert texts[0] == "a photo of a empty room"

    assert core.plan({"type": "head", "head": "colours"}, {"colours": {}}) == ("head", [], "colours")
    assert core.plan(PAIR)[0] == "noul_pair"
    assert core.plan({"type": "noul", "instructions": "a cat"}) == ("noul_sigmoid", ["a cat"], None)
    assert core.plan({"criteria": ["cat", "dog"]})[0] == "choice"   # choice is the default
    # both documented aliases (docs/questions.md): `instruction` for `instructions`, the
    # way `name` works for `head`. The singular used to be dropped in silence, so a noul
    # written the documented way failed as if it had no caption at all.
    assert core.plan({"type": "noul", "instruction": "a cat"}) == ("noul_sigmoid", ["a cat"], None)
    assert core.plan({"type": "head", "name": "colours"}, {"colours": {}})[0] == "head"
    # the canonical spelling wins when both are sent, rather than the alias shadowing it
    assert core.plan({"type": "noul", "instructions": "plural",
                      "instruction": "singular"})[1] == ["plural"]


def test_head_is_refused_under_another_encoder(jev, trained, toy_cal, heads_dir, png):
    other = jevimage.Jev(toy_cal, heads_dir=heads_dir)
    assert other.encoder.dim == jev.encoder.dim      # shapes agree; only the name saves us
    with pytest.raises(ValueError) as e:
        other.ask(png, {"q": {"type": "head", "head": "colours"}})
    assert "'toy'" in str(e.value) and "retrain" in str(e.value)


# -- Answer ----------------------------------------------------------------

def test_answer_attribute_access_and_its_error():
    a = core.Answer(type="noul", noul=0.25, basis="contrast pair")
    assert a.noul == 0.25 and a["noul"] == 0.25
    with pytest.raises(AttributeError) as e:
        a.choice
    assert "'noul' answers have no 'choice'" in str(e.value)
    assert "noul" in str(e.value) and "basis" in str(e.value)   # says what it does have


# -- to_image --------------------------------------------------------------

def test_to_image_accepts_path_bytes_pil_and_data_url(png):
    assert jevimage.to_image(png).size == (8, 8)
    assert jevimage.to_image(str(png)).size == (8, 8)
    raw = png.read_bytes()
    assert jevimage.to_image(raw).size == (8, 8)
    assert jevimage.to_image(bytearray(raw)).size == (8, 8)
    im = Image.new("RGB", (4, 4), (1, 2, 3))
    assert jevimage.to_image(im) is im
    url = "data:image/png;base64," + base64.b64encode(raw).decode()
    assert jevimage.to_image(url).size == (8, 8)


def test_to_image_refuses_a_bare_string_by_name(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        jevimage.to_image(str(tmp_path / "typo.png"))
    assert "typo.png" in str(e.value) and "data:image" in str(e.value)
    with pytest.raises(TypeError, match="int"):
        jevimage.to_image(42)


# -- read_folder -----------------------------------------------------------

def test_read_folder_labels_by_subfolder_and_ignores_the_rest(ds):
    (ds / "red" / "notes.txt").write_text("not an image")
    (ds / "loose.png").write_bytes((ds / "red" / "0.png").read_bytes())
    (ds / "empty_label").mkdir()

    out = jevimage.read_folder(ds)
    assert sorted(out) == ["blue", "red"]            # loose files and label-less dirs drop out
    assert len(out["red"]) == 4
    assert all(p.suffix == ".png" for p in out["red"])
    assert jevimage.read_folder(ds, max_per_class=2)["red"] == out["red"][:2]


def test_read_folder_refuses_nothing_to_train_on(tmp_path):
    (tmp_path / "bare").mkdir()
    with pytest.raises(ValueError, match="no labelled images"):
        jevimage.read_folder(tmp_path / "bare")
    with pytest.raises(NotADirectoryError):
        jevimage.read_folder(tmp_path / "nope")


# -- question ids ----------------------------------------------------------

def test_questions_may_be_a_list(jev, png):
    out = jev.ask(png, [SCENE, {**PAIR, "id": "cat"}])
    assert sorted(out) == ["cat", "q1"]
    with pytest.raises(TypeError, match="list of questions"):
        jev.ask(png, "scene")
