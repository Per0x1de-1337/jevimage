"""The readout engine, and the image/folder front door it is reached through."""
import base64
import io
import json
import os
import time
import types

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


# -- device fallback -----------------------------------------------------------
# A GPU is a preference, not a requirement: a card that cannot hold the model must not
# stop the job. But a device the caller named is theirs to have or to lose - swapping it
# silently turns a run into a 40x slower one with nothing in the output to say why.

class _Fussy:
    """An encoder that will not load on CUDA. Stands in for a full card."""

    name = "fussy"

    def __init__(self):
        self.tried = []

    def _place(self, device):
        self.tried.append(device)
        if device != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 376.00 MiB")
        self.device = device


def _fallback(monkeypatch, device=None, env=None, cuda=True):
    from jevimage import encoders
    monkeypatch.setattr(encoders, "resolve_device",
                        lambda d=None: d or env or ("cuda" if cuda else "cpu"))
    monkeypatch.setattr(encoders, "_free_cuda", lambda: None)
    monkeypatch.delenv("JEVIMAGE_DEVICE", raising=False)
    if env:
        monkeypatch.setenv("JEVIMAGE_DEVICE", env)
    enc = _Fussy()
    encoders._place_with_fallback(enc, device)
    return enc


def test_full_gpu_falls_back_to_cpu(monkeypatch, caplog):
    enc = _fallback(monkeypatch)
    assert enc.tried == ["cuda", "cpu"]
    assert enc.device == "cpu"
    assert "using the CPU instead" in caplog.text, "the swap has to be visible"


def test_named_device_is_not_swapped(monkeypatch):
    for kwargs in ({"device": "cuda"}, {"env": "cuda"}):
        with pytest.raises(RuntimeError, match="out of memory"):
            _fallback(monkeypatch, **kwargs)


def test_cpu_failure_is_not_retried_on_cpu(monkeypatch):
    """Nothing left to fall back to; retrying would raise the same error twice."""
    class Broken(_Fussy):
        def _place(self, device):
            self.tried.append(device)
            raise RuntimeError("no memory at all")

    from jevimage import encoders
    monkeypatch.setattr(encoders, "resolve_device", lambda d=None: "cpu")
    enc = Broken()
    with pytest.raises(RuntimeError):
        encoders._place_with_fallback(enc, None)
    assert enc.tried == ["cpu"]


# -- serve picks a free port ---------------------------------------------------

def test_bind_walks_past_a_taken_port():
    """`jevimage serve` on a laptop that already runs one loses the race for 8000. The
    walk is bind-and-keep, not check-then-bind, so nothing can take the port in the gap.
    """
    import socket

    from jevimage.cli import DEFAULT_PORT, PORT_TRIES, _bind

    held, first = _bind("127.0.0.1", 0)         # any free port, and keep it
    try:
        sock, port = _bind("127.0.0.1", first)
        try:
            assert port > first, "walked past the port we are holding"
            assert sock.getsockname()[1] == port, "the socket IS the port reported"
        finally:
            sock.close()
    finally:
        held.close()

    # all of them taken: one error naming the range, not errno 98 out of uvicorn
    keep = []
    try:
        base = _bind("127.0.0.1", 0)[1]
        blocked = [_bind("127.0.0.1", p, tries=1) for p in range(base, base + 3)]
        keep = [s for s, _ in blocked]
        with pytest.raises(OSError, match=f"{base}-{base + 2} .*in use"):
            _bind("127.0.0.1", base, tries=3)
    finally:
        for s in keep:
            s.close()

    assert PORT_TRIES > 1 and DEFAULT_PORT != 8000


def test_probe_sweeps_the_same_range_serve_walks(monkeypatch):
    """A server that walked to 8732 is still found; otherwise `ask` starts a second copy
    of the model beside the one already running."""
    import http.server
    import threading

    from jevimage import cli

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"primitives": ["choice"]}'
            self.send_response(200 if self.path == "/v1/health" else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Health)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        # as if serve had walked several ports up from the default
        monkeypatch.setattr(cli, "DEFAULT_PORT", port - 3)
        monkeypatch.setattr(cli, "PORT_TRIES", 8)
        assert cli._local_server() == f"http://127.0.0.1:{port}"

        # and a range with nothing in it is None, not a hang
        monkeypatch.setattr(cli, "DEFAULT_PORT", port + 1)
        monkeypatch.setattr(cli, "PORT_TRIES", 3)
        assert cli._local_server() is None
    finally:
        srv.shutdown()


# -- the image path on the way in ----------------------------------------------

def _exif_jpeg(size, orientation):
    """A JPEG carrying an Orientation tag, as a phone writes one."""
    import io

    from PIL import Image
    im = Image.new("RGB", size)
    for x in range(size[0] // 2):           # a left-half stripe, so a flip is visible
        for y in range(size[1]):
            im.putpixel((x, y), (255, 0, 0))
    ex = im.getexif()
    ex[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=ex, quality=95)
    return buf.getvalue()


def test_upright_rotates_only_when_it_has_to():
    """Both spellings of this copy the whole image to do nothing: exif_transpose returns
    a copy when the tag says upright, convert("RGB") returns one when the mode already
    is. On a 12MP photo that is two full-size copies for no change."""
    import io

    from PIL import Image
    from jevimage.images import upright

    plain = Image.open(io.BytesIO(_exif_jpeg((40, 30), 1)))
    plain.load()
    assert upright(plain) is plain, "already upright and already RGB: no copy"

    # orientation 6 is a phone held sideways: 90 degrees, so the sides swap
    turned = Image.open(io.BytesIO(_exif_jpeg((40, 30), 6)))
    out = upright(turned)
    assert out.size == (30, 40), "a rotated photo still has to come out upright"

    pal = Image.new("P", (8, 8))
    assert upright(pal).mode == "RGB", "a palette image is converted, not resampled"

    class NoExif:
        mode = "RGB"

        def getexif(self):
            raise ValueError("no exif here")
    n = NoExif()
    assert upright(n) is n, "an unreadable EXIF block is not a reason to fail"


def test_shrink_drafts_the_decode(monkeypatch):
    """A JPEG decoder can scale by 1/8 while it decodes. It only works on an image
    nothing has touched yet, so a `.convert()` before `thumbnail()` costs the whole win:
    the full-size decode has already happened by the time draft() is asked."""
    import io

    from PIL import Image
    from jevimage.client import _shrink

    big = Image.new("RGB", (4032, 3024), (200, 30, 30))
    buf = io.BytesIO()
    big.save(buf, "JPEG", quality=90)

    from PIL import JpegImagePlugin

    drafted = []
    real = JpegImagePlugin.JpegImageFile.draft

    def spy(self, mode, size):
        r = real(self, mode, size)
        drafted.append((size, self.size))
        return r

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy)
    out = _shrink(buf.getvalue(), 384)
    assert drafted, "draft() was never asked"
    assert drafted[0][1][0] < 4032, "draft() ran after the decode, so it did nothing"
    assert max(Image.open(io.BytesIO(out)).size) == 384

    # already small enough: passed through byte for byte, EXIF and all
    small = io.BytesIO()
    Image.new("RGB", (100, 80)).save(small, "JPEG")
    assert _shrink(small.getvalue(), 384) is None


# -- cpu threads ---------------------------------------------------------------
# torch defaults to the physical core count, which is wrong on a hybrid chip: every
# barrier in a forward pass waits on the slowest thread, so the E-cores set the pace for
# the P-cores too. Measured on an i7-12650H, one 224px forward went 158ms -> 86ms.

def test_fast_cores_is_sane_or_absent():
    import torch
    from jevimage.encoders import fast_cores

    n = fast_cores()
    assert n is None or 1 <= n <= os.cpu_count(), n
    if n is not None:
        assert os.path.isdir("/sys/devices/cpu_atom"), "only hybrid machines report one"


def _tune(monkeypatch, cores, **env):
    import torch
    from jevimage import encoders

    monkeypatch.setattr(encoders, "_TUNED", False)
    monkeypatch.setattr(encoders, "fast_cores", lambda: cores)
    for k in ("JEVIMAGE_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    before = torch.get_num_threads()
    try:
        return encoders.tune_cpu_threads()
    finally:
        torch.set_num_threads(before)


def test_thread_tuning_backs_off_only_when_it_helps(monkeypatch):
    import torch

    default = torch.get_num_threads()
    assert _tune(monkeypatch, None) is None, "uniform CPU: leave torch alone"
    assert _tune(monkeypatch, default + 4) is None, "more than the default is not a win"
    if default > 1:
        assert _tune(monkeypatch, default - 1) == default - 1


def test_thread_tuning_yields_to_the_caller(monkeypatch):
    import torch

    low = max(1, torch.get_num_threads() - 1)
    assert _tune(monkeypatch, low, OMP_NUM_THREADS="3") is None, \
        "a process that set OMP_NUM_THREADS chose for the whole application"
    assert _tune(monkeypatch, low, JEV_THREADS="0") is None, "0 opts out"
    assert _tune(monkeypatch, low, JEV_THREADS="2") == 2, "a number wins over the probe"
    with pytest.raises(ValueError, match="not a number of threads"):
        _tune(monkeypatch, low, JEV_THREADS="lots")


def test_thread_tuning_runs_once(monkeypatch):
    """It sets a process-wide global, so it does so once - not on every `load()`."""
    import torch

    from jevimage import encoders

    before = torch.get_num_threads()
    try:
        monkeypatch.setattr(encoders, "_TUNED", False)
        monkeypatch.setattr(encoders, "fast_cores", lambda: 1)
        for k in ("JEVIMAGE_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            monkeypatch.delenv(k, raising=False)
        encoders.tune_cpu_threads()
        assert encoders._TUNED
        assert encoders.tune_cpu_threads() is None, "second call is a no-op"
    finally:
        torch.set_num_threads(before)


# -- what is holding the GPU ---------------------------------------------------

def test_gpu_hogs_reads_nvidia_smi(monkeypatch, tmp_path):
    """"0.0 GB free of 3.7 GB" does not tell you what to close, and a *suspended*
    process holds its whole allocation while looking like nothing in top."""
    import subprocess

    from jevimage import cli

    class Done:
        stdout = "1779614, 900\n2122998, 896\nbad line\n72877, 860\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
    monkeypatch.setattr("builtins.open", _fake_proc(stopped={"1779614"}))
    rows = cli.gpu_hogs()
    assert len(rows) == 3, "the unparseable line is skipped, not counted"
    assert rows[0].startswith("pid 1779614"), "biggest first"
    assert "900 MiB" in rows[0] and "jevimage serve" in rows[0]
    assert "STOPPED" in rows[0] and "STOPPED" not in rows[1]
    assert cli.gpu_hogs(limit=1) == rows[:1]


def _fake_proc(stopped):
    import builtins
    real = builtins.open

    def fake(path, *a, **k):
        p = str(path)
        if "/cmdline" in p:
            return io.StringIO("/usr/bin/python3\x00/home/x/bin/jevimage\x00serve\x00")
        if "/stat" in p:
            pid = p.split("/")[2]
            return io.StringIO(f"{pid} (python3) {'T' if pid in stopped else 'S'} 1 2 3")
        return real(path, *a, **k)
    return fake


def test_gpu_hogs_is_silent_without_nvidia_smi(monkeypatch):
    import subprocess

    from jevimage import cli

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert cli.gpu_hogs() == [], "no NVIDIA tools is not an error, it is a machine"


# -- `ask --time` --------------------------------------------------------------

def test_time_breakdown_splits_image_from_answer(monkeypatch, capsys, tmp_path):
    """Two very different fixes hide behind one slow number - a smaller image, or a
    free GPU - so the two halves have to be told apart."""
    from jevimage import cli
    from jevimage.answers import Answer

    png = tmp_path / "shot.png"
    Image.new("RGBA", (2560, 1440)).save(png)

    class Stub:
        max_edge = 384
        url = "http://127.0.0.1:8731"

        def ask(self, image, questions):
            assert image.startswith("data:image/"), \
                "the encoded image is passed on, not re-encoded"
            return {"q": Answer({"type": "choice", "choice": "a", "confidence": 0.5,
                                 "probabilities": {"a": 0.75, "b": 0.25}})}

    monkeypatch.setattr(cli, "_backend", lambda a: Stub())
    code = cli.main(["ask", str(png), "--time", "--json",
                     "--choice", "a=an a", "b=a b"])
    out, err = capsys.readouterr()
    assert code == 0
    json.loads(out), "the timings go to stderr so --json stays parseable"
    assert "image" in err and "answer" in err and "total" in err
    assert "2560x1440 RGBA png" in err, err


def test_describe_reads_the_header_only(tmp_path):
    """Decoding the image to ask its size would double the number being reported."""
    from jevimage.cli import _describe

    p = tmp_path / "x.png"
    Image.new("RGBA", (2560, 1440)).save(p)
    t = time.perf_counter()
    d = _describe(p)
    assert d == "2560x1440 RGBA png", d
    assert (time.perf_counter() - t) < 0.05, "that was a full decode"
    assert _describe(tmp_path / "nope.png") == "image", "unreadable is not a crash"


# -- environment and URL mistakes ----------------------------------------------

def test_unread_jev_variable_is_called_out(monkeypatch, capsys):
    """`export JEV_BASE=...` is not an error anywhere: nothing reads it and the command
    carries on with the default, which is the quietest way a setting can fail.

    One line, though - listing all eleven turned a note about one variable into a wall.
    """
    from jevimage.cli import ENV_VARS, _check_env

    monkeypatch.setenv("JEVIMAGE_BASE", "https://localhost:8732")
    monkeypatch.setenv("JEVIMAGE_TIMEOUT", "30")
    _check_env()
    err = capsys.readouterr().err.strip()
    assert err.count("\n") == 0, f"one line, not a list: {err!r}"
    assert "$JEVIMAGE_BASE is set, but nothing reads it" in err
    assert "JEVIMAGE_TIMEOUT" not in err, "a variable that IS read must not be named"
    assert sum(v in err for v in ENV_VARS) == 0, "the catalogue belongs in --help"
    assert "doctor" in err, "it has to say where to look"


def test_a_near_miss_is_suggested_but_a_guess_is_not(monkeypatch, capsys):
    """difflib will answer $JEVIMAGE_API_KEY for $JEVIMAGE_BASE if you let it, and a confident
    wrong guess costs more than no guess."""
    from jevimage.cli import _check_env

    for bad, want in (("JEVIMAGE_TIMOUT", "Did you mean $JEVIMAGE_TIMEOUT?"),
                      ("JEVIMAGE_ENCODERS", "Did you mean $JEVIMAGE_ENCODER?")):
        monkeypatch.setenv(bad, "x")
        _check_env()
        assert want in capsys.readouterr().err
        monkeypatch.delenv(bad)

    for nonsense in ("JEVIMAGE_BASE", "JEVIMAGE_TOKEN", "JEVIMAGE_HOST"):
        monkeypatch.setenv(nonsense, "x")
        _check_env()
        err = capsys.readouterr().err
        assert "Did you mean" not in err, f"{nonsense} got a guess: {err!r}"
        monkeypatch.delenv(nonsense)


def test_https_at_a_plain_server_names_the_fix():
    """`jevimage serve` does not terminate TLS, and OpenSSL's WRONG_VERSION_NUMBER
    describes the bytes rather than the mistake."""
    import jevimage
    from jevimage.client import JevError

    r = jevimage.connect("https://127.0.0.1:1", timeout=2)
    with pytest.raises(JevError) as e:
        r.health()
    assert "cannot reach a jevimage server" in str(e.value)


# -- the two server envelopes --------------------------------------------------
# A self-hosted `jevimage serve` answers about one image and puts the answers at the top
# level. The hosted API answers about a list and puts each image's under results[i].
# A client that reads only one fails with a KeyError whose whole text is 'answers'.

def test_answers_read_from_either_envelope():
    from jevimage.client import JevError, _answers

    one = {"choice": {"type": "choice", "choice": "a", "confidence": 0.5,
                      "probabilities": {"a": 0.75, "b": 0.25}}}
    assert _answers({"answers": one, "timing": {}}) == one
    assert _answers({"results": [{"index": 0, "answers": one}], "usage": {}}) == one
    assert _answers({"results": [{"index": 0, "answers": one},
                                 {"index": 1, "answers": {}}]}) == one, "the image we sent"

    for junk in ({}, {"results": []}, {"results": [{"index": 0}]}, {"detail": "nope"}):
        with pytest.raises(JevError, match="without answers in it"):
            _answers(junk)


def test_health_falls_back_to_the_capability_document(monkeypatch):
    """The hosted API keeps /v1/health for liveness and puts the capability document
    behind /v1/models. Callers should not have to know which door they reached."""
    import jevimage
    from jevimage.client import JevError

    r = jevimage.connect("http://127.0.0.1:1")
    seen = []

    def fake(method, path, body=None, **kw):
        seen.append(path)
        if path == "/v1/health":
            return {"status": "ok", "version": "1", "request_id": "req_x"}
        return {"model": "jev-image-1", "primitives": ["choice", "head"],
                "heads": [{"name": "street_objects", "classes": ["a", "b"]}],
                "version": "1"}

    monkeypatch.setattr(r, "_call", fake)
    h = r.health()
    assert seen == ["/v1/health", "/v1/models"]
    assert h["encoder"] == "jev-image-1" and h["heads"] == 1
    assert h["primitives"] == ["choice", "head"] and h["auth"] == "bearer"

    # a server that answers /v1/health fully is not asked twice
    seen.clear()
    monkeypatch.setattr(r, "_call", lambda m, p, *a, **k: (
        seen.append(p) or {"encoder": "toy", "primitives": ["choice"]}))
    assert r.health()["encoder"] == "toy" and seen == ["/v1/health"]

    # and no key, so /v1/models 401s: say what we actually saw rather than raising
    seen.clear()

    def refused(method, path, body=None, **kw):
        seen.append(path)
        if path == "/v1/health":
            return {"status": "ok", "version": "1"}
        raise JevError("401", status=401)

    monkeypatch.setattr(r, "_call", refused)
    assert r.health() == {"status": "ok", "version": "1"}


def test_list_heads_falls_back_when_there_is_no_heads_route(monkeypatch):
    """The hosted API's heads are read-only and published in the capability document,
    so there is no route that could create one - but the list is the same list."""
    import jevimage
    from jevimage.client import JevError

    r = jevimage.connect("http://127.0.0.1:1")
    heads = [{"name": "street_objects", "classes": ["a", "b"], "n": 468}]

    def fake(method, path, body=None, **kw):
        if path == "/v1/heads":
            raise JevError("no such route", status=404)
        return {"model": "jev-image-1", "heads": heads}

    monkeypatch.setattr(r, "_call", fake)
    assert r.list_heads() == heads

    def other(method, path, body=None, **kw):
        raise JevError("boom", status=500)

    monkeypatch.setattr(r, "_call", other)
    with pytest.raises(JevError):
        r.list_heads(), "a 500 is not a missing route"


# -- several images, one connection --------------------------------------------
# Against a server a round trip away, TCP and TLS cost about half a second per process
# and the question itself costs a tenth of that. Measured over 8 files to a server
# ~200ms away: eight commands took 9.10s, one command took 2.86s.

def test_ask_takes_several_images_on_one_connection(monkeypatch, capsys, tmp_path):
    from jevimage import cli
    from jevimage.answers import Answer

    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (40, 30), (i * 60, 0, 0)).save(p)
        paths.append(str(p))

    class Stub:
        url = "http://h:8731"
        asks = 0

        def ask(self, image, questions):
            Stub.asks += 1
            return {"q": Answer({"type": "choice", "choice": f"c{Stub.asks}",
                                 "confidence": 0.5,
                                 "probabilities": {"a": 0.75, "b": 0.25}})}

    made = []
    monkeypatch.setattr(cli, "_backend", lambda a: made.append(1) or Stub())
    assert cli.main(["ask", *paths, "--json", "--choice", "a=an a", "b=a b"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert made == [1], "one backend, so one connection, for all three"
    assert Stub.asks == 3
    assert list(out) == paths, "keyed by path when there is more than one"
    assert out[paths[0]]["q"]["choice"] == "c1"

    # one image keeps the old shape exactly: a bare {question id: answer} object
    Stub.asks = 0
    assert cli.main(["ask", paths[0], "--json", "--choice", "a=an a", "b=a b"]) == 0
    one = json.loads(capsys.readouterr().out)
    assert set(one) == {"q"}, "a single image must not grow a path key"


def test_ask_still_refuses_no_images():
    from jevimage.cli import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["ask", "--choice", "a=an a", "b=a b"])


# -- embedding a whole training folder -----------------------------------------
# A folder is allowed 4000 images and `train()` passed the lot to the encoder in one
# call, so a real training set decoded 4000 pictures into memory and asked the GPU for
# all of them at once. On a 4GB card that is an OutOfMemoryError out of F.linear,
# reaching the caller as HTTP 500.

def test_embedding_is_chunked(monkeypatch, tmp_path):
    import jevimage
    from conftest import Toy

    seen = []
    toy = Toy()
    real = toy.img
    toy.img = lambda ims: seen.append(len(ims)) or real(ims)

    monkeypatch.setenv("JEVIMAGE_BATCH", "8")
    j = jevimage.load(toy, heads_dir=str(tmp_path))
    imgs = [Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(20)]
    out = j.embed(imgs)
    assert out.shape[0] == 20, "every image still comes back"
    assert seen == [8, 8, 4], f"should be three chunks of at most 8, got {seen}"
    assert max(seen) <= 8


def test_embedding_halves_on_out_of_memory(monkeypatch, tmp_path):
    """The right chunk depends on the card, the encoder and whatever else holds memory
    right now, none of which can be known beforehand. Back off instead of guessing."""
    import jevimage
    from conftest import Toy

    toy, tried, done = Toy(), [], []
    real = toy.img

    def greedy(ims):
        tried.append(len(ims))
        if len(ims) > 4:                       # this card only fits four
            raise torch.OutOfMemoryError("CUDA out of memory")
        done.append(len(ims))
        return real(ims)

    toy.img = greedy
    monkeypatch.setenv("JEVIMAGE_BATCH", "16")
    j = jevimage.load(toy, heads_dir=str(tmp_path))
    out = j.embed([Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(10)])
    assert out.shape[0] == 10, "it still finishes, and with every image"
    assert tried == sorted(tried, reverse=True), \
        f"the chunk only ever shrinks, so a failing size is never retried: {tried}"
    assert max(done) <= 4, f"everything that succeeded fitted the card: {done}"
    assert sum(done) == 10, f"each image embedded exactly once: {done}"


def test_jev_batch_is_validated(monkeypatch):
    from jevimage.local import _embed_batch

    monkeypatch.delenv("JEVIMAGE_BATCH", raising=False)
    assert _embed_batch() == 32
    monkeypatch.setenv("JEVIMAGE_BATCH", "4")
    assert _embed_batch() == 4
    for bad in ("nope", "0", "-3"):
        monkeypatch.setenv("JEVIMAGE_BATCH", bad)
        with pytest.raises(ValueError, match="JEVIMAGE_BATCH"):
            _embed_batch()


# -- what `train` prints -------------------------------------------------------

def test_train_output_leads_with_the_head(monkeypatch, capsys, tmp_path):
    """The zero-shot table is the evidence for whether training was worth doing, and on
    classes the encoder already knows the honest answer is often no. Printed by default
    it buried the one line people came for: the head exists, and this is how to call it.
    """
    from jevimage import cli
    from jevimage.images import Head

    cmp = {"basis": "k-fold", "overall": {"zero_shot": 0.919, "trained": 0.916,
                                          "delta": -0.003},
           "per_class": [{"label": "bus", "n": 120, "zero_shot": 0.908,
                          "trained": 0.875}]}
    head = Head("vehicles", {"name": "vehicles", "n": 431, "encoder": "toy",
                             "classes": ["bicycle", "bus"], "cv_accuracy": 0.916}, cmp)

    class Stub:
        def train(self, *a, **k):
            return head

    monkeypatch.setattr(cli, "_backend", lambda a: Stub())
    assert cli.main(["train", "vehicles", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "trained 'vehicles'" in out and "91.6% accurate" in out
    assert "--head vehicles" in out, "it has to say how to use what it just made"
    assert "zero-shot" not in out, "the table is opt-in now"
    assert "  91.6%" not in out, "_pct pads for a table; inline it must not"

    assert cli.main(["train", "vehicles", str(tmp_path), "--compare"]) == 0
    out = capsys.readouterr().out
    assert "zero-shot" in out and "delta" in out
    assert "did not beat the prompts" in out, "a negative result must still be stated"

    assert cli.main(["train", "vehicles", str(tmp_path), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["comparison"]["overall"]["delta"] == -0.003, "--json keeps everything"


def test_head_answer_prints_against_either_server(capsys):
    """jevimage's own server calls it `accuracy`; a hosted one may publish the same
    number as `cv_accuracy`. Reading only one turned a good answer into a KeyError
    whose entire text was the key name."""
    from jevimage.cli import _show

    base = {"type": "head", "head": "vehicles", "choice": "car", "confidence": 0.31,
            "probabilities": {"car": 0.6, "bus": 0.4}, "trained_on": 431}
    for extra in ({"accuracy": 0.9165}, {"cv_accuracy": 0.9165},
                  {"accuracy": 0.9165, "cv_accuracy": 0.9165}):
        _show("q", {**base, **extra})
        out = capsys.readouterr().out
        assert "held out 91.6%" in out, f"{extra} -> {out!r}"
        assert "431 examples" in out

    # a head too thin to hold anything out reports no accuracy at all, and that is not
    # a reason to fail
    _show("q", base)
    out = capsys.readouterr().out
    assert "vehicles" in out and "held out" not in out, out


# -- `jevimage` on its own -----------------------------------------------------

def test_bare_command_is_not_an_error(monkeypatch, capsys):
    """argparse answered `error: the following arguments are required: command` and
    exit 2. Nothing went wrong - someone installed it and typed its name."""
    from jevimage import cli

    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_config", dict)
    for k in ("JEVIMAGE_URL", "JEVIMAGE_API_KEY", "JEVIMAGE_BASE"):
        monkeypatch.delenv(k, raising=False)
    assert cli.main([]) == 0, "typing the name is a question, not a failure"
    out = capsys.readouterr().out
    assert "error" not in out.lower()
    for expected in ("jevimage ask", "jevimage train", "jevimage doctor",
                     "jevimage login", "jevimage serve", "docs.jevimage.org"):
        assert expected in out, expected


def test_bare_command_says_where_it_points(monkeypatch, capsys):
    """Three situations, three answers: a server named, one found, or neither."""
    from jevimage import cli

    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: None)
    monkeypatch.setenv("JEVIMAGE_URL", "https://api.example.org")
    monkeypatch.setenv("JEVIMAGE_API_KEY", "k")
    cli.main([])
    assert "Pointed at https://api.example.org (with $JEVIMAGE_API_KEY)" \
        in capsys.readouterr().out

    monkeypatch.delenv("JEVIMAGE_API_KEY")
    monkeypatch.setattr(cli, "_config", dict)      # nothing saved by `login` either
    cli.main([])
    out = capsys.readouterr().out
    assert "no key is set" in out, "a hosted server without a key is a 401"

    monkeypatch.delenv("JEVIMAGE_URL")
    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: "http://127.0.0.1:8731")
    cli.main([])
    assert "already running on http://127.0.0.1:8731" in capsys.readouterr().out


def test_a_real_mistake_is_still_an_error(capsys):
    """Softening the empty invocation must not soften a wrong one."""
    from jevimage.cli import _parser

    with pytest.raises(SystemExit) as e:
        _parser().parse_args(["frobnicate"])
    assert e.value.code == 2




def test_help_does_not_recite_the_environment(capsys):
    """`--help` answers "what can this do". A table of eleven variables most people
    will never set made it twice as long as the thing it documents."""
    from jevimage.cli import ENV_VARS, _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(["--help"])
    out = capsys.readouterr().out
    named = [n for n in ENV_VARS if n in out]
    assert not named, f"--help is not the place for these: {named}"
    assert "jevimage doctor" in out, "it has to say where configuration lives"


def test_doctor_shows_what_is_set_and_never_a_key(monkeypatch, capsys):
    """The pointer in --help has to lead somewhere that answers the question."""
    from jevimage import cli

    monkeypatch.setattr(cli, "_config", dict)
    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: None)
    for n in cli.ENV_VARS:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("JEVIMAGE_URL", "http://mine:8731")
    monkeypatch.setenv("JEVIMAGE_API_KEY", "jev_do_not_print_me")
    monkeypatch.setenv("JEVIMAGE_DEVICE", "cpu")

    cli.cmd_doctor(types.SimpleNamespace(url=None))
    out = capsys.readouterr().out
    assert "$JEVIMAGE_URL" in out and "http://mine:8731" in out
    assert "$JEVIMAGE_DEVICE" in out and "cpu" in out
    assert "jev_do_not_print_me" not in out, "a key is never echoed, only named"
    assert "$JEVIMAGE_API_KEY" in out and "set" in out
    assert "$JEVIMAGE_TIMEOUT" not in out, "only what is actually set"


# -- which backend a command reaches for ---------------------------------------
# A key names the service it belongs to, so holding one is the same as choosing it.
# Without one, a hardcoded default is a 401 - and this package is self-hostable, so
# pointing a working install at an endpoint it cannot authenticate to would break it.

def _args(**kw):
    import types
    return types.SimpleNamespace(**{"url": None, "local": False, "encoder": None, **kw})


def test_a_key_selects_the_hosted_api(monkeypatch):
    from jevimage import cli

    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: "http://127.0.0.1:8731")
    monkeypatch.setattr(cli, "_config", dict)
    for k in ("JEVIMAGE_URL", "JEVIMAGE_API_KEY", "JEVIMAGE_ENCODER"):
        monkeypatch.delenv(k, raising=False)

    assert cli._server_for(_args()) == "http://127.0.0.1:8731", \
        "no key: a server running here, not an endpoint we cannot authenticate to"

    monkeypatch.setenv("JEVIMAGE_API_KEY", "k")
    assert cli._server_for(_args()) == cli.DEFAULT_URL

    # anything named explicitly still wins over the default
    assert cli._server_for(_args(url="http://mine")) == "http://mine"
    monkeypatch.setenv("JEVIMAGE_URL", "http://env")
    assert cli._server_for(_args()) == "http://env"
    monkeypatch.delenv("JEVIMAGE_URL")


def test_local_means_this_machine_not_this_process(monkeypatch):
    """Someone with a server running here who asks for local means that server, not
    twelve seconds of reloading the same weights beside it."""
    from jevimage import cli

    monkeypatch.setenv("JEVIMAGE_API_KEY", "k")
    monkeypatch.delenv("JEVIMAGE_URL", raising=False)
    monkeypatch.delenv("JEVIMAGE_ENCODER", raising=False)
    monkeypatch.setattr(cli, "_config", dict)

    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: "http://127.0.0.1:8731")
    assert cli._server_for(_args(local=True)) == "http://127.0.0.1:8731"

    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: None)
    assert cli._server_for(_args(local=True)) is None, "nothing running: this process"

    # naming an encoder is how you insist on this process, and it reads as what it does
    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: "http://127.0.0.1:8731")
    assert cli._server_for(_args(encoder="toy")) is None


# -- staying signed in ---------------------------------------------------------
# The environment is the wrong place for this. A shell that has not exported
# $JEVIMAGE_API_KEY looks exactly like someone with no account, so the same command in two
# terminals reached two different servers and listed two different sets of heads.

def test_login_saves_a_key_every_shell_can_use(monkeypatch, tmp_path, capsys):
    import stat

    from jevimage import cli

    monkeypatch.setenv("JEVIMAGE_HOME", str(tmp_path))
    monkeypatch.delenv("JEVIMAGE_API_KEY", raising=False)
    monkeypatch.delenv("JEVIMAGE_URL", raising=False)
    monkeypatch.setattr(cli, "jevimage_connect",
                        lambda url, key: types.SimpleNamespace(
                            health=lambda: {"encoder": "jev-image-1", "heads": 2}))

    assert cli.main(["login", "--key", "jev_secret", "--url", "https://api.example"]) == 0
    out = capsys.readouterr().out
    assert "signed in to https://api.example" in out
    assert "jev_secret" not in out, "a secret must not be echoed back"

    cfg = tmp_path / "config.json"
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600, "a key file is nobody else's"
    assert json.loads(cfg.read_text()) == {"url": "https://api.example",
                                           "key": "jev_secret"}

    # and it is used with no environment at all
    monkeypatch.setattr(cli, "_local_server", lambda *a, **k: "http://127.0.0.1:8731")
    assert cli._server_for(_args()) == "https://api.example"
    assert cli._api_key() == "jev_secret"

    # the environment still wins, so one shell can point somewhere else
    monkeypatch.setenv("JEVIMAGE_API_KEY", "other")
    assert cli._api_key() == "other"
    monkeypatch.setenv("JEVIMAGE_URL", "http://staging")
    assert cli._server_for(_args()) == "http://staging"

    monkeypatch.delenv("JEVIMAGE_API_KEY")
    monkeypatch.delenv("JEVIMAGE_URL")
    assert cli.main(["logout"]) == 0
    assert not cfg.exists()
    assert cli._server_for(_args()) == "http://127.0.0.1:8731", "back to this machine"


def test_login_refuses_an_empty_key(monkeypatch, tmp_path):
    from jevimage import cli

    monkeypatch.setenv("JEVIMAGE_HOME", str(tmp_path))
    monkeypatch.delenv("JEVIMAGE_API_KEY", raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdin, "readline", lambda: "\n")
    with pytest.raises(ValueError, match="no key given"):
        cli.cmd_login(types.SimpleNamespace(key=None, url=None))
    assert not (tmp_path / "config.json").exists(), "nothing written on a refusal"


def test_a_broken_config_is_not_fatal(monkeypatch, tmp_path):
    """Half-written JSON should mean 'not signed in', not a crash on every command."""
    from jevimage import cli

    monkeypatch.setenv("JEVIMAGE_HOME", str(tmp_path))
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "config.json").write_text('{"url": "https://x", "key"')
    assert cli._config() == {}
    assert cli._api_key() is None


def test_every_advertised_variable_is_really_read():
    """ENV_VARS is a promise: a name in it is one the package looks at. A name that
    stops being read has to leave, or `$X is set but nothing reads it` starts lying in
    the other direction - staying silent about a setting that does nothing."""
    import pathlib

    from jevimage.cli import ENV_VARS

    src = "".join(p.read_text() for p in pathlib.Path("jevimage").glob("*.py")
                  if p.name != "cli.py")
    cli = pathlib.Path("jevimage/cli.py").read_text().replace(str(ENV_VARS), "")
    for name in ENV_VARS:
        assert f'"{name}"' in src + cli, f"{name} is advertised but never read"


def test_heads_names_the_account_it_is_showing(monkeypatch, capsys):
    """A key scopes what you can see, so two keys list two different sets of heads.
    Without saying whose, a head trained in the browser and missing from the CLI looks
    like a sync failure rather than a second account."""
    from jevimage import cli

    class Remote:
        url = "https://api.example"

        def __init__(self, acct):
            self._a = acct

        def health(self):
            return {"account": self._a}

        def list_heads(self):
            return [{"name": "h", "classes": ["a", "b"], "n": 4, "cv_accuracy": 0.5}]

    # `_server_for` decides remote vs local, and it reads a saved login and probes for a
    # local server. Unmocked, this test passed only on a machine that happened to have
    # one, and on any other it took the LOCAL branch, read the developer's real
    # ~/.jev/heads and asserted against whatever was in it.
    monkeypatch.setattr(cli, "_server_for", lambda a: "https://api.example")
    monkeypatch.setattr(cli, "_backend",
                        lambda a: Remote({"id": "gh:1", "login": "someone"}))
    cli.main(["heads"])
    assert "(account someone)" in capsys.readouterr().out

    # a server that does not say is not a reason to print an empty bracket
    monkeypatch.setattr(cli, "_backend", lambda a: Remote(None))
    cli.main(["heads"])
    out = capsys.readouterr().out
    assert "account" not in out, out

    # the uid alone when nobody has signed in to put a name to it
    monkeypatch.setattr(cli, "_backend", lambda a: Remote({"id": "gh:2"}))
    cli.main(["heads"])
    assert "(account gh:2)" in capsys.readouterr().out

# -- the grid readout: reading WHERE, not just what --------------------------------
# It is not a question type. `plan` routes choice and noul here when their options are the
# same words in a different order, so every test below asks an ordinary primitive.

@pytest.fixture
def grid_on(monkeypatch):
    """The grid routing is opt-in: measured worse than pooled on the one-image task this
    library performs, so it is off unless JEVIMAGE_GRID is set. These tests are about the
    readout itself, so they turn it on explicitly."""
    monkeypatch.setenv("JEVIMAGE_GRID", "1")


GRID = {"g": {"type": "noul",
              "criteria": {"true": "red circle above blue square",
                           "false": "blue square above red circle"}}}


def test_the_router_fires_only_on_a_rearrangement(grid_on):
    """The whole routing decision, and it must not fire on a presence question: the grid
    scores 75.24 against pooled 92.01 on SugarCrepe replace_obj, so a false positive here
    makes plain classification worse."""
    assert core.plan(GRID["g"], {})[0] == "noul_grid"
    assert core.plan({"type": "noul", "criteria": {"true": "a dog",
                                                   "false": "a cat"}}, {})[0] == "noul_pair"
    assert core.plan({"type": "choice", "criteria": {
        "x": "a red cube left of a blue ball",
        "y": "a blue ball left of a red cube"}}, {})[0] == "choice_grid"
    assert core.plan({"type": "choice", "criteria": {"x": "a cat",
                                                     "y": "a dog"}}, {})[0] == "choice"
    # identical captions are not a rearrangement, they are a mistake, and routing them to
    # the grid would dress a tie up as a structural answer
    assert core.plan({"type": "choice", "criteria": {"x": "a cat",
                                                     "y": "a cat"}}, {})[0] == "choice"
    # stopwords are ignored, so the same structure worded slightly differently still fires
    assert core.plan({"type": "noul", "criteria": {
        "true": "the cat is on a mat", "false": "a mat is on the cat"}}, {})[0] == "noul_grid"


def test_grid_noul_keeps_the_noul_shape(grid_on, toy_grid, png):
    """A caller who asked a noul gets a noul. `basis` is the only thing that changed, and
    it is the field noul already used to say how it was read."""
    a = jevimage.Jev(toy_grid).ask(png, GRID)["g"]
    assert set(a) == {"type", "noul", "basis", "confidence"}, set(a)
    assert a["type"] == "noul" and 0.0 <= a["noul"] <= 1.0
    assert a["basis"] == "grid readout"
    assert 0.0 <= a["confidence"] <= 1.0


def test_grid_choice_keeps_the_choice_shape(grid_on, toy_grid, png):
    a = jevimage.Jev(toy_grid).ask(png, {"c": {"type": "choice", "criteria": {
        "square_above": "blue square above green triangle",
        "triangle_above": "green triangle above blue square"}}})["c"]
    assert set(a) == {"type", "readout", "choice", "probabilities", "confidence"}, set(a)
    assert a["type"] == "choice" and a["readout"] == "grid"
    assert a["choice"] in ("square_above", "triangle_above")
    assert sums_to_one(a["probabilities"])


def test_grid_reading_needs_the_image_not_an_embedding(grid_on, toy_grid, png):
    j = jevimage.Jev(toy_grid)
    emb = j.embed(png)
    with pytest.raises(ValueError, match="no longer records"):
        j.ask_embedding(emb, GRID)


def test_grid_reading_refuses_an_encoder_without_a_grid(grid_on, toy, png):
    """An ensemble cannot expose patches, and silently answering from the pooled vector
    would hand back a number that is at chance on exactly this question."""
    with pytest.raises(ValueError, match="patch grid"):
        jevimage.Jev(toy).ask(png, GRID)


def test_the_warning_lands_on_the_pooled_pair_where_it_is_needed(grid_on, toy_grid, png):
    """A negation pair is one caption with "not" in it, not two alternatives. Dual encoders
    do not represent negation, so the answer is noise and has to say so. Measured on a real
    photo: cos(true, false) 0.927 against 0.83-0.85 for two genuine alternatives."""
    j = jevimage.Jev(toy_grid)
    a = j.ask(png, {"g": {"type": "noul", "criteria": {
        "true": "the girl is wearing a pink dress",
        "false": "the girl is not wearing a pink dress"}}})["g"]
    # inserting "not" changes WHICH words are present, so this is not a rearrangement and
    # it reads pooled. That is exactly where the warning matters: nothing else about a
    # contrast-pair answer tells the caller the number is noise.
    assert a["basis"] == "contrast pair", a
    assert a["warning"] and "negation" in a["warning"]
    b = j.ask(png, GRID)["g"]
    assert "warning" not in b and b["confidence"] is not None, b


def test_a_rearrangement_never_carries_a_similarity_warning(grid_on, toy_grid, png):
    """The regression this fixes. A rearrangement has the same words on both sides, so it is
    near-identical to its swap by construction - 0.975 measured on a controlled two-object
    image. Running the 0.90 similarity check on the grid path fired on essentially every
    question the readout exists for, and suppressed the confidence on the right answers."""
    a = jevimage.Jev(toy_grid).ask(png, GRID)["g"]
    assert "warning" not in a and "confidence" in a, a
    # The 0.975 similarity that would have tripped it is a real-encoder measurement; the toy
    # is a per-string hash, so two rearranged captions are unrelated under it and the check
    # would not fire here anyway. What this test pins is that the grid path never emits the
    # key, so the value cannot come back.


def test_segments_splits_on_spatial_words():
    """Splitting on prepositions is the whole mechanism: without it "a red circle above a
    blue square" stays one phrase and the readout degenerates to the pooled answer."""
    assert core.segments("a red circle above a blue square") == [
        "a red circle above a blue square", "a red circle", "a blue square"]
    assert core.segments("a plain phrase") == ["a plain phrase"]


def test_regions_are_multi_scale_and_cached(toy_grid):
    r = core.regions(14)
    assert r is core.regions(14)                       # cached, indices never change
    assert {len(x) for x in r} == {4, 9, 16, 25, 49, 100, 196}
    assert all(0 <= i < 196 for x in r for i in x)


def test_needs_grid_only_fires_on_a_rearrangement(grid_on):
    assert core.Engine.needs_grid(GRID)
    assert not core.Engine.needs_grid({"q": SCENE})
    assert not core.Engine.needs_grid({"q": {"type": "noul", "instructions": "x"}})
    # it plans to decide, so a question the planner rejects must not explode the check
    # that runs before encoding - the real plan raises for it a moment later
    assert not core.Engine.needs_grid({"q": {"type": "choice", "criteria": None}})
    assert not core.Engine.needs_grid({"q": "not even a dict"})

def test_the_old_JEV_names_still_work(monkeypatch):
    """The variables were renamed to JEVIMAGE_. Every machine that already exported the
    old names - including this project's own systemd units - must keep working, or the
    rename is an outage rather than a rename."""
    from jevimage.store import env

    for n in ("JEVIMAGE_URL", "JEV_URL"):
        monkeypatch.delenv(n, raising=False)
    assert env("JEVIMAGE_URL") is None
    assert env("JEVIMAGE_URL", "fallback") == "fallback"

    monkeypatch.setenv("JEV_URL", "http://old")          # legacy name alone
    assert env("JEVIMAGE_URL") == "http://old"

    monkeypatch.setenv("JEVIMAGE_URL", "http://new")     # both set: new wins
    assert env("JEVIMAGE_URL") == "http://new"

    # and the CLI's own resolution goes through it, not os.environ directly
    import types
    from jevimage import cli
    monkeypatch.delenv("JEVIMAGE_URL", raising=False)
    assert cli._opt(types.SimpleNamespace(url=None), "url", "JEVIMAGE_URL") == "http://old"


def test_a_legacy_JEV_name_is_not_reported_as_unread(monkeypatch, capsys):
    """`JEV_URL` is still read, so calling it unread would send someone to fix a setting
    that is working. A genuinely unknown suffix is still reported, under either prefix."""
    from jevimage import cli

    for n in list(cli.ENV_VARS) + ["JEV_URL", "JEV_NONSENSE", "JEVIMAGE_NONSENSE"]:
        monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("JEV_URL", "http://old")
    cli._check_env()
    assert capsys.readouterr().err == "", "a working legacy name must stay quiet"

    monkeypatch.setenv("JEV_NONSENSE", "x")
    cli._check_env()
    assert "$JEV_NONSENSE is set, but nothing reads it" in capsys.readouterr().err


def test_the_grid_routing_is_off_unless_asked_for(monkeypatch):
    """It loses on the task this library performs - one image, two captions - so a caller
    who upgrades must not silently get different answers. SugarCrepe swap, n=911, against
    a whole-caption baseline: pooled 70.25, this routing 68.50."""
    from jevimage.core import grid_readout_enabled

    monkeypatch.delenv("JEVIMAGE_GRID", raising=False)
    monkeypatch.delenv("JEV_GRID", raising=False)
    assert grid_readout_enabled() is False
    swap = {"type": "noul", "criteria": {"true": "a cube left of a ball",
                                         "false": "a ball left of a cube"}}
    assert core.plan(swap, {})[0] == "noul_pair", "a rearrangement must still read pooled"

    for off in ("", "0", "false", "no", "NO"):
        monkeypatch.setenv("JEVIMAGE_GRID", off)
        assert grid_readout_enabled() is False, off
    for on in ("1", "true", "yes"):
        monkeypatch.setenv("JEVIMAGE_GRID", on)
        assert grid_readout_enabled() is True, on
    assert core.plan(swap, {})[0] == "noul_grid"
