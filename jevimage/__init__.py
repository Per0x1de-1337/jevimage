"""jevimage - a probability-first image classifier you can self-host.

One image encode, many typed questions read against it. Nothing is generated: every
answer is a calibrated probability distribution, so you get a number you can threshold
instead of a sentence you have to parse.

There are two ways to use this, and both are first class.

RUN IT YOURSELF - the models execute in your process, nothing leaves the machine:

    pip install 'jevimage[local]'

    import jevimage
    jev = jevimage.load()                       # siglip2-base-224 by default
    a = jev.ask("photo.jpg", {
        "scene": {"type": "choice",
                  "criteria": {"indoor": "an indoor scene",
                               "outdoor": "an outdoor scene"}},
    })
    a["scene"].choice           # 'outdoor'
    a["scene"].probabilities    # {'indoor': 0.07, 'outdoor': 0.93}
    a["scene"].confidence       # 0.86

USE A SERVER - someone else runs the encoder; you need no torch, no weights, no GPU:

    pip install jevimage                        # a few hundred KB, stdlib + pillow

    jev = jevimage.connect("https://jev.example.com", api_key="...")
    a = jev.ask("photo.jpg", {...})             # the same call, the same Answer objects

`connect()` is a drop-in for `load()`: same method names, same return shapes. Write
against one and switch later by changing that line.

HOST IT FOR OTHERS:

    pip install 'jevimage[serve]'
    jev serve --host 0.0.0.0 --port 8000

Zero-shot not good enough? Train a head on your own labelled images. The encoder stays
frozen, so this takes about a second and costs nothing at query time - and it works the
same locally or through a server:

    head = jev.train("street", "dataset/")      # dataset/bus/*.jpg, dataset/bike/*.jpg
    head.accuracy                               # held-out, k-fold
    jev.ask("photo.jpg", {"q": {"type": "head", "head": "street"}})
"""
from __future__ import annotations

__version__ = "0.1.0"

# Only the light things are imported eagerly. torch and transformers live behind
# __getattr__ below, so `import jevimage` costs milliseconds and an API-only install
# never needs them on disk at all.
from .answers import Answer, confidence, normalise_questions
from .images import IMAGE_SUFFIXES, Head, as_examples, read_folder, to_image

__all__ = ["load", "connect", "register", "available", "Jev", "Head", "Answer",
           "Encoder", "JevError", "to_image", "read_folder", "as_examples",
           "confidence", "IMAGE_SUFFIXES", "DEFAULT", "__version__"]

_LAZY = {
    "load": ("local", "load"),
    "Jev": ("local", "Jev"),
    "Encoder": ("encoders", "Encoder"),
    "register": ("encoders", "register"),
    "available": ("encoders", "available"),
    "DEFAULT": ("encoders", "DEFAULT"),
    "JevError": ("client", "JevError"),
}


def __getattr__(name):
    """Resolve the rest of the API on first use.

    `load` and `Jev` need torch; the encoder-registry names and `JevError` do not - they
    are here to keep `import jevimage` down to milliseconds. Importing the heavy half
    lazily is what lets `pip install jevimage` be a small HTTP client, while
    `pip install 'jevimage[local]'` is the full framework under the identical name.
    """
    target = _LAZY.get(name)
    if target is None:
        # submodules too: error messages and the docs say jevimage.encoders.unit(),
        # and a plain `import jevimage` should be enough to reach one.
        import importlib.util
        if importlib.util.find_spec(f".{name}", __name__) is None:
            raise AttributeError(f"module 'jevimage' has no attribute {name!r}")
        target = (name, None)
    module, attr = target
    try:
        import importlib
        mod = importlib.import_module(f".{module}", __name__)
        return mod if attr is None else getattr(mod, attr)
    except ImportError as e:
        raise ImportError(
            f"jevimage.{name} runs the encoder in this process, which needs torch and "
            f"transformers: pip install 'jevimage[local]'\n"
            f"To use a server instead - no torch required - call "
            f"jevimage.connect(url).\n(underlying import error: {e})") from e


def __dir__():
    return sorted(__all__)


def connect(url: str, *, api_key: str | None = None, timeout: float = 60.0,
            max_edge: int | None = 384):
    """Talk to a `jev serve` instance. Same methods as `load()`, over HTTP.

    Needs no torch and downloads no weights, so this is the whole dependency footprint
    of an API-only install.

    `max_edge` downscales an image before uploading it, because no encoder reads more
    than 384px and on a fresh connection the upload is most of the latency. Pass None
    to send originals.
    The returned object keeps one connection open; close it with `.close()`, or use it
    as a context manager.
    """
    from .client import Remote
    return Remote(url, api_key=api_key, timeout=timeout, max_edge=max_edge)
