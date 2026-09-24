"""Reading images and labelled folders, and the Head record. PIL only, no torch.

Everything here is shared by the two ways to use jevimage - running the models locally
and calling a server over HTTP - so it must not drag a deep-learning stack into an
API-only install.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

__all__ = ["to_image", "read_folder", "as_examples", "Head", "IMAGE_SUFFIXES"]

# What the encoders decode. Deliberately short: an animated GIF, an SVG or a RAW file
# is not a thing to silently take the first frame of.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _short(s) -> str:
    """What an error message may quote back. A bare base64 blob handed over as a
    filename is 40KB long, and an error that pastes all of it is unreadable."""
    s = str(s)
    return s if len(s) <= 80 else s[:77] + "..."


def to_image(src):
    """Accept the four things people actually have, and refuse anything ambiguous.

        Path / str  an existing file on disk
        bytes       encoded image bytes (JPEG/PNG/WebP)
        PIL.Image   used as-is
        str         a 'data:image/...;base64,...' URL

    A plain string that is not a path and not a data URL raises, rather than being
    guessed at as base64 - a typo'd filename should say so, not fail three layers down.
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(src, Image.Image):
        return src
    if isinstance(src, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(src)))
    # str and Path share one branch on purpose: the docs present them as interchangeable,
    # and a separate `Image.open(src)` for Path answered the same mistakes with a bare
    # errno - and named a broken symlink's TARGET rather than the path the caller passed.
    if isinstance(src, (str, Path)):
        if isinstance(src, str) and src[:5].lower() == "data:":
            meta, comma, b64 = src.partition(",")
            try:
                if not comma:
                    raise ValueError("there is no ',' before the payload")
                return Image.open(io.BytesIO(base64.b64decode(b64)))
            except UnidentifiedImageError:
                raise ValueError(
                    f"could not read the data URL {_short(meta)!r}: what follows the "
                    "comma decodes, but not to a JPEG, PNG or WebP.") from None
            except ValueError as e:
                # b64decode raises binascii.Error (a ValueError); a data URL that is not
                # base64 at all, and base64 of something that is not an image, both reach
                # PIL. All three used to surface as "cannot identify image file
                # <_io.BytesIO object at 0x...>", which names nothing the caller wrote.
                raise ValueError(
                    f"could not read the data URL {_short(meta)!r}: {e}. What follows "
                    "the comma must be base64 of a JPEG, PNG or WebP.") from None
        p = Path(src)
        try:
            # Path("") is ".", so an empty or unset path variable reads as a directory.
            is_dir, exists = p.is_dir(), p.exists()
        except OSError:
            # A path component longer than the filesystem allows - which is what bare
            # base64 pasted as a filename is - does not exist. Errno 36 out of pathlib
            # is not the answer; the FileNotFoundError below is.
            is_dir = exists = False
        if is_dir:
            raise IsADirectoryError(f"{_short(src)!r} is a directory, not an image file")
        if exists:
            return Image.open(p)
        raise FileNotFoundError(
            f"no such image file: {_short(src)!r}. Pass a path that exists, raw image "
            "bytes, a PIL.Image, or a 'data:image/...;base64,...' string.")
    raise TypeError(f"cannot read an image from {type(src).__name__}; pass a path, "
                    "bytes, a PIL.Image, or a data URL")


def _cap(max_per_class):
    """One meaning for `max_per_class` in every input shape: a positive count, or None.

    Tested for truthiness it meant three different things - 0 was "no limit", -1 dropped
    the LAST example per label from a folder and EVERY example from a pair list, and a
    float died inside a slice - so the same argument produced three different datasets.
    """
    if max_per_class is None:
        return None
    if isinstance(max_per_class, bool) or not isinstance(max_per_class, int) \
            or max_per_class < 1:
        raise ValueError("max_per_class must be a positive int or None, got "
                         f"{max_per_class!r}")
    return max_per_class


def read_folder(folder, max_per_class: int | None = None):
    """A folder of labelled images -> {label: [Path, ...]}.

    One sub-folder per label, which is the layout every image dataset already uses:

        dataset/bus/0001.jpg        -> label 'bus'
        dataset/hydrant/0002.png    -> label 'hydrant'

    Loose files directly in `folder` are ignored, because they have no label.
    """
    n = _cap(max_per_class)
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{folder!r} is not a directory")
    out: dict[str, list[Path]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        # by NAME, not by path: rglob descends, and sorting by the full path ordered a
        # nested layout by sub-directory, so `max_per_class` kept 2024/01.png over
        # old/1999.png - the opposite of the documented "you get the oldest N".
        files = sorted((f for f in sub.rglob("*")
                        if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES),
                       key=lambda f: (f.name, str(f)))
        if files:
            out[sub.name] = files[:n] if n else files
    if not out:
        raise ValueError(
            f"no labelled images under {folder!r}. Expected one sub-folder per label, "
            f"each holding {'/'.join(sorted(IMAGE_SUFFIXES))} files.")
    return out


def as_examples(data, max_per_class=None):
    """Folder | {label: [images]} | [(image, label)] -> ([images], [labels])."""
    n = _cap(max_per_class)
    if isinstance(data, (str, Path)):
        # uncapped on purpose: the cap is applied once, in the mapping branch below, so
        # the folder shape is not truncated twice (which silently deleted a whole class)
        data = read_folder(data)
    if isinstance(data, dict):
        images, labels = [], []
        for label, items in data.items():
            if isinstance(items, (str, bytes, Path)):
                # a single path as the value was iterated character by character, so
                # each character became an "image" and the failure surfaced much later
                # as a FileNotFoundError about a one-character filename
                raise TypeError(f"value for label {label!r} must be a list of images, "
                                f"got {type(items).__name__}")
            for im in (items[:n] if n else items):
                images.append(im)
                labels.append(str(label))
        return images, labels
    if isinstance(data, (list, tuple)):
        images, labels, seen = [], [], {}
        for i, pair in enumerate(data):
            # checked per item, not `len(pairs[0]) != 2`: that inspected only the first
            # element, and calling len() on an element that has none (a Path, a
            # PIL.Image - exactly what a list of unlabelled images holds) raised a bare
            # TypeError instead of the sentence below
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                shape = (f"a {type(pair).__name__} of {len(pair)}"
                         if isinstance(pair, (tuple, list))
                         else f"a bare {type(pair).__name__}")
                raise ValueError("a list of examples must hold (image, label) pairs; "
                                 f"item {i} is {shape}")
            im, label = pair
            label = str(label)
            # max_per_class was honoured for the other two input shapes and silently
            # ignored here, so the same argument meant two different things
            seen[label] = seen.get(label, 0) + 1
            if n and seen[label] > n:
                continue
            images.append(im)
            labels.append(label)
        return images, labels
    raise TypeError("train() takes a folder path, a {label: [images]} mapping, or a "
                    "list of (image, label) pairs")


class Head:
    """A trained head: its metadata, and the comparison against zero-shot prompts.

    The same object comes back from a local `train()` and from a remote one, because
    a head is metadata plus numbers - the weights stay wherever the encoder is.
    """

    def __init__(self, name, record, comparison=None):
        self.name = name
        self._record = dict(record)
        self.comparison = comparison

    @property
    def classes(self): return self._record["classes"]
    @property
    def n(self): return self._record["n"]
    @property
    def counts(self): return self._record.get("counts", {})
    @property
    def accuracy(self):
        """Held-out accuracy, from stratified k-fold with k = min(5, thinnest class).
        Never the accuracy on the images it was fitted to. Read it next to `n` and
        `counts`: two examples in a class means two folds, which is a coin toss."""
        return self._record.get("cv_accuracy")
    @property
    def per_class(self): return self._record.get("per_class", {})
    @property
    def encoder(self): return self._record.get("encoder")

    def to_dict(self): return dict(self._record)

    def __repr__(self):
        acc = f"{self.accuracy:.1%} held out" if self.accuracy is not None else "no holdout"
        return f"<Head {self.name!r} {len(self.classes)} classes, {self.n} examples, {acc}>"
