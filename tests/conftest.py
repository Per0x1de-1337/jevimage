"""Fixtures for the whole suite.

The encoder here is a hash function: same bytes in, same vector out, no weights, no
network, no download. Every test below therefore asserts on the framework's behaviour
and never on a model's opinion - the only numbers checked are ones arithmetic forces.
"""
import hashlib
import os

# Before torch is imported: the suite must never reach for an accelerator. torch's
# optimizers probe one while choosing a foreach kernel, and on a machine whose GPU is
# already busy that probe raises CUDA out-of-memory from inside a 16-dim fit.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from PIL import Image

torch.set_num_threads(1)      # the tensors here are tiny; threading them costs more

import jevimage


class Toy(jevimage.Encoder):
    """sha512 of the input projected to 16 dims. Deterministic across runs and machines."""

    name, dim = "toy", 16
    logit_scale = logit_bias = None

    def __init__(self):
        torch.manual_seed(0)
        self.W = torch.randn(64, self.dim)

    def _e(self, xs):
        v = torch.stack([torch.tensor([b / 255 for b in hashlib.sha512(str(x).encode()).digest()])
                         for x in xs])
        return torch.nn.functional.normalize(v @ self.W, dim=-1)

    def img(self, ims):
        return self._e([getattr(i, "tobytes", lambda: i)() for i in ims])

    def txt(self, ts):
        return self._e(list(ts))


class ToyCalibrated(Toy):
    """Same width and same space, different name, plus a sigmoid calibration.

    Deliberately shape-compatible with Toy: that is what makes running one encoder's
    head under another silently wrong, and what the encoder-name check has to catch.
    """

    name = "toy-calibrated"
    logit_scale, logit_bias = 100.0, -10.0


@pytest.fixture
def toy():
    return Toy()


@pytest.fixture
def toy_cal():
    return ToyCalibrated()


@pytest.fixture
def heads_dir(tmp_path):
    return tmp_path / "heads"


@pytest.fixture
def jev(toy, heads_dir):
    return jevimage.Jev(toy, heads_dir=heads_dir)


def write_png(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), rgb).save(path)
    return path


@pytest.fixture
def png(tmp_path):
    return write_png(tmp_path / "one.png", (200, 30, 30))


@pytest.fixture
def ds(tmp_path):
    """A labelled folder: ds/red/*.png, ds/blue/*.png. Every file differs from every
    other, or the hash encoder would hand identical embeddings to different examples."""
    root = tmp_path / "ds"
    for label, (r, b) in (("red", (200, 30)), ("blue", (30, 200))):
        for i in range(4):
            write_png(root / label / f"{i}.png", (r, 20 + i * 7, b))
    return root


@pytest.fixture
def trained(jev, ds):
    """A real head fitted through the public API. Few epochs: these tests check the
    plumbing around a head, not how well it separates hashes."""
    return jev.train("colours", ds, epochs=60)
