"""Where trained heads live.

A head is tiny - a (dim x K) float matrix, tens of kilobytes - so each one is its own
file under a directory you choose. One file per head means a head can be copied,
committed, or handed to a colleague on its own, and two processes training different
heads never contend for one file.

    ~/.jev/heads/street_objects.pt

Default directory: $JEVIMAGE_HOME/heads, else ~/.jev/heads.
"""
from __future__ import annotations

import os
import re
import threading
import uuid
from pathlib import Path

def env(name: str, default=None):
    """Read a setting, preferring JEVIMAGE_ and falling back to the old JEV_ name.

    The variables were renamed to match the package. Dropping the old names would have
    broken every machine that already exported them, including this project's own
    systemd units, so both are read and the new one wins. `JEV_` is not deprecated
    loudly anywhere: it simply keeps working.
    """
    short = name[len("JEVIMAGE_"):] if name.startswith("JEVIMAGE_") else name
    v = os.environ.get("JEVIMAGE_" + short)
    if v is None:
        v = os.environ.get("JEV_" + short)
    return default if v is None else v


# torch is imported inside the methods that read and write files, not here, so that
# `valid_name` is usable from the HTTP client - an API-only install has no torch and
# still has to refuse a bad head name before it builds a URL out of it.

__all__ = ["HeadStore", "default_dir", "valid_name"]

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def default_dir() -> Path:
    home = Path(env("JEVIMAGE_HOME") or Path.home() / ".jev")
    return Path(env("JEVIMAGE_HEADS_DIR") or home / "heads")


def valid_name(name: str) -> str:
    """Head names become filenames, so they are checked rather than sanitised: silently
    rewriting a name means `ask` later cannot find what `train` said it saved."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"invalid head name {name!r}: use 1-64 characters from "
                         "[A-Za-z0-9._-], starting with a letter or digit")
    return name


class HeadStore(dict):
    """A dict of name -> head that is backed by a directory.

    It is a real dict, so `store["name"]` and `"name" in store` work as they look. It
    reloads any file that changed on disk, which is what lets several server workers
    share heads that any one of them trained.
    """

    def __init__(self, directory=None):
        super().__init__()
        self.dir = Path(directory) if directory else default_dir()
        self._mtimes: dict[str, tuple] = {}
        self.broken: dict[str, str] = {}      # name -> why, for files that would not load
        # sync() and save() both rewrite the dict, _mtimes and broken together. Two
        # threads running sync() at once each computed `set(self) - seen` and each
        # deleted the same name, so the second one raised KeyError out of a pure READ -
        # `name in store`, which is what gates every ask().
        self._lock = threading.Lock()
        self.sync()

    def path(self, name: str) -> Path:
        return self.dir / f"{valid_name(name)}.pt"

    def sync(self) -> "HeadStore":
        """Pick up heads added, changed or removed on disk since the last look."""
        # Under the lock: reads call this (a `head` question gates on `name in store`),
        # so two of them racing each other, or a read racing a train, used to raise
        # KeyError out of the removal loop below.
        with self._lock:
            return self._sync()

    def _sync(self) -> "HeadStore":
        if not self.dir.is_dir():
            return self
        # Once, not per file, and outside the per-file handler: catching this below
        # recorded every head in the directory as "unreadable", which reads as file
        # corruption when the only thing wrong is that torch is not installed.
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "reading trained heads needs torch: pip install 'jevimage[local]'\n"
                "To use heads on a server instead - no torch required - call "
                f"jevimage.connect(url).\n(underlying import error: {e})") from e
        try:
            # os.listdir, not Path.glob: glob swallows EACCES and yields nothing, so an
            # unreadable directory printed as "no heads" - the same lie as a vanished
            # head file, and `rm` in the same directory reported the real errno.
            files = [self.dir / f for f in os.listdir(self.dir) if f.endswith(".pt")]
        except OSError as e:
            raise OSError(f"cannot read heads directory {self.dir}: {e}") from None
        seen = set()
        for f in files:
            name = f.stem
            seen.add(name)
            try:
                valid_name(name)
            except ValueError as e:
                # `heads` and `ask` used to accept a stem that `rm` and path() refuse,
                # so the CLI listed and ran a head it could not delete. One rule: a file
                # that cannot be addressed by name is not a head, and says why.
                self.broken[name] = str(e)
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            # not mtime alone: a retrain landing inside one filesystem timestamp tick
            # is invisible to it. os.replace always installs a new inode, so the
            # identity of the file is the reliable signal.
            stamp = (st.st_mtime, st.st_ino, st.st_size)
            if self._mtimes.get(name) == stamp:
                continue
            try:
                # weights_only=True: a head is tensors plus plain metadata (strings,
                # numbers, lists, dicts), all of which this loader allows - so a head
                # file handed over by a colleague cannot execute code on the way in.
                obj = torch.load(f, map_location="cpu", weights_only=True)
                # a .pt that loads is not yet a head: someone else's checkpoint, or a
                # bare tensor, used to enter the store and then blow up in public() -
                # taking the whole listing (and GET /v1/heads) down with it.
                if not (isinstance(obj, dict) and {"W", "classes", "n"} <= set(obj)):
                    raise ValueError(f"not a jev head: a {type(obj).__name__} without "
                                     "the 'W'/'classes'/'n' a head has")
                super().__setitem__(name, obj)
                self._mtimes[name] = stamp
                self.broken.pop(name, None)
            except Exception as e:
                # a half-written or foreign .pt must not break startup, but it must not
                # vanish either: "unknown head" about a file sitting right there is a lie.
                # First line only - torch's unpickling error is several paragraphs.
                why = str(e).strip().split("\n")[0][:160] or "no detail"
                self.broken[name] = f"{type(e).__name__}: {why}"
        # pop, not __delitem__: two threads reaching this loop with the same `gone` both
        # deleted it, and the loser raised KeyError out of whatever read called sync()
        for gone in set(self) - seen:
            super().pop(gone, None)
            self._mtimes.pop(gone, None)
        for gone in set(self.broken) - seen:
            self.broken.pop(gone, None)
        return self

    def save(self, name: str, head: dict) -> dict:
        import torch

        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.path(name)
        payload = {**head, "W": head["W"].cpu(), "b": head["b"].cpu()}
        # unique per writer: a temp path derived from the head name alone meant two
        # processes overwriting the same head wrote the same file, the first os.replace
        # consumed it, and the second failed with a FileNotFoundError naming a .pt.tmp
        # the caller never chose. Still *.tmp, so the directory scan still ignores it.
        tmp = p.with_suffix(f".pt.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            torch.save(payload, tmp)
        except RuntimeError as e:
            # torch re-raises the underlying PermissionError as RuntimeError, which is
            # outside the CLI's caught set - a read-only volume printed a traceback.
            tmp.unlink(missing_ok=True)
            raise OSError(f"cannot write head {name!r} to {self.dir}: {e}") from None
        os.replace(tmp, p)      # atomic: a crash mid-write cannot leave a torn head
        with self._lock:
            super().__setitem__(name, payload)
            try:
                st = p.stat()
                self._mtimes[name] = (st.st_mtime, st.st_ino, st.st_size)
            except OSError:
                # another process deleted or replaced it between the rename and here.
                # No stamp means the next read re-syncs, which is the right answer.
                self._mtimes.pop(name, None)
            # retraining a head that was previously unreadable is the documented cure
            # for it, so it must stop being reported as broken
            self.broken.pop(name, None)
        return payload

    def delete(self, name: str) -> bool:
        p = self.path(name)
        existed = p.exists()
        p.unlink(missing_ok=True)
        with self._lock:
            super().pop(name, None)
            self._mtimes.pop(name, None)
            self.broken.pop(name, None)
        return existed

    def _stale(self, name) -> bool:
        """Has the file behind a cached head changed, or gone, since it was read?

        One stat, against the matmul that follows it. Without it a cache HIT short-
        circuited every freshness check, so a head retrained by another process kept
        answering with the old weights and the old class names until something else
        happened to call sync().
        """
        try:
            st = self.path(name).stat()
        except (OSError, ValueError):
            return True
        return self._mtimes.get(name) != (st.st_mtime, st.st_ino, st.st_size)

    def __getitem__(self, name):
        if super().__contains__(name) and self._stale(name):
            self.sync()
        return super().__getitem__(name)      # __missing__ still handles the absent case

    def __contains__(self, name):
        # `name in store` must see a head another process trained a moment ago, because
        # that check is what gates every ask() - a stale False reads as "no such head".
        if super().__contains__(name) and not self._stale(name):
            return True
        self.sync()
        # a file that exists but will not load counts as present. Saying "unknown head"
        # about a head the user can see on disk sends them looking for a typo that is
        # not there; __missing__ then explains what is actually wrong with it.
        return super().__contains__(name) or name in self.broken

    def __missing__(self, name):
        self.sync()             # trained by another process a moment ago
        # super(): `name in self` now also answers True for a broken head, which would
        # route straight back here
        if super().__contains__(name):
            return self[name]
        if name in self.broken:
            raise KeyError(f"{name}.pt is present in {self.dir} but unreadable: "
                           f"{self.broken[name]}. Delete it, or retrain the head.")
        known = ", ".join(sorted(self)) or "none"
        raise KeyError(f"no head named {name!r} in {self.dir} (have: {known})")
