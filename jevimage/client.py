"""The HTTP client: `jevimage.connect(url)` where you would have written `jevimage.load()`.

Remote has the method names, the arguments and the return types of `jevimage.Jev`, so
moving a script from an in-process encoder to a `jevimage serve` box is a one-line change and
nothing downstream notices - `ask` still hands back `Answer` objects, `train` still
hands back a `jevimage.Head`, and `embed` still hands back a (n, dim) fp32 tensor when
torch is installed (a list of lists when it is not, since an API-only install has none).

This module imports no torch and no transformers. That is the point: `pip install
jevimage` is a few hundred kilobytes and talks to someone else's GPU.

Stdlib only. A client is the file people vendor into an environment they do not control,
so it must not drag `requests` in behind it.

The wire is whatever `jevimage.server` serves - that module is the contract, this one
follows it. Images go up as `data:image/...;base64,...` strings, which is both what
`jevimage.to_image` accepts and what the server's `_decode` strips:

    GET    /v1/health                                            -> {encoder, dim, ...}
    POST   /v1/embed            {image | images}                 -> {embeddings: [[f]], dim}
    POST   /v1/ask              {image, questions, temperature}  -> {answers: {id: answer}, timing}
    POST   /v1/heads            {name, examples, epochs}         -> head + {comparison}
    GET    /v1/heads                                             -> [head]
    GET    /v1/heads/{name}                                      -> head
    GET    /v1/heads/{name}/compare                              -> {overall, per_class, basis}
    DELETE /v1/heads/{name}                                      -> {deleted: name}, 404 if absent

A `head` is `training.public()` - metadata only, never the weights. Errors are any
non-2xx with the human-readable reason in `detail`; that reason is what the caller
needs, so JevError carries it verbatim.
"""
from __future__ import annotations

import base64
import io
import http.client
import json
import threading
import urllib.parse
from pathlib import Path

from .answers import Answer, normalise_questions, positive_temperature
from .images import Head, as_examples, to_image, upright
from .store import valid_name

__all__ = ["Remote", "JevError", "MAX_IMAGES"]

MAX_IMAGES = 8      # jevimage.server.MAX_IMAGES; embed() chunks to it rather than 413


class JevError(ValueError):
    """A request the server refused. `status` is the HTTP code, `detail` the server's
    own message - a 422 saying which head is unknown is the useful part, not the 422.

    A ValueError, not a RuntimeError, because `connect()` is a drop-in for `load()`:
    a local `Jev` raises ValueError for these same refusals, so an existing
    `except ValueError` keeps working when the `load()` line becomes a `connect()` line.
    """

    def __init__(self, message, *, status=None, detail=None):
        super().__init__(message)
        self.status = status
        self.detail = detail


def _mime(raw: bytes) -> str | None:
    """The format of already-encoded bytes, by magic number."""
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


# No encoder in the registry sees more than 384px; the server resizes to that and
# discards the rest, so anything larger is upload spent on pixels nothing will read.
#
# 384 rather than some safer-looking larger number, because on a fresh connection the
# cost is not linear in bytes. TCP starts with a ~14KB congestion window, so a body
# needs a round trip per doubling before it is all on the wire. Measured against a
# server one RTT of 190ms away: 79KB took 940ms, 26KB took 660ms, and below that the
# RTT floor dominates. A custom encoder that reads more than 384px should raise this.
MAX_EDGE = 384


def _shrink(raw: bytes, max_edge: int) -> bytes | None:
    """Re-encode an oversized image smaller, or None to send the original untouched.

    Image.open reads the header only, so an image already within `max_edge` costs a
    few microseconds and is passed through byte for byte.
    """
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(raw))
        if max(im.size) <= max_edge:
            return None            # passed through with its EXIF; the server transposes

        # draft() BEFORE anything touches the pixels. A JPEG decoder can scale by 1/2,
        # 1/4 or 1/8 while it decodes, for free - a 4032px phone photo comes out of the
        # DCT at 504px instead of being decoded in full and then thrown away. It only
        # works on an image that has not been loaded yet, which is why `.convert("RGB")`
        # used to cost the whole win: thumbnail() calls draft() itself, and by then the
        # full-size decode had already happened. Not a JPEG, or already small enough:
        # draft returns None and nothing changes.
        im.draft("RGB", (max_edge, max_edge))
        # exif_transpose before re-encoding: save() drops the Orientation tag, so a
        # portrait phone photo would otherwise arrive sideways with nothing left on it
        # for the server to act on, and remote answers would diverge from local ones.
        im = upright(im)
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
    except Exception:
        return None            # unreadable here is the server's 415 to give, not ours
    out = buf.getvalue()
    return out if len(out) < len(raw) else None


def _data_url(src, max_edge: int | None = MAX_EDGE) -> str:
    """One image -> a data URL, accepting everything `to_image` accepts.

    Encoded bytes that are already JPEG/PNG/WebP go over the wire untouched unless they
    are larger than `max_edge`, in which case they are downscaled first: re-encoding a
    small JPEG to PNG would triple the upload and change the pixels the server decodes,
    so the remote answer would no longer match the local one for the same file. Anything
    else - a PIL image, an exotic format - becomes a PNG, which is lossless.
    """
    raw = None
    if isinstance(src, (bytes, bytearray, memoryview)):
        raw = bytes(src)
    elif isinstance(src, (Path, str)):
        if isinstance(src, str) and src.startswith("data:"):
            return src
        p = Path(src)
        if p.is_file():
            raw = p.read_bytes()

    if raw is not None:
        mime = _mime(raw)
        if mime:
            if max_edge:
                smaller = _shrink(raw, max_edge)
                if smaller is not None:
                    raw, mime = smaller, "image/jpeg"
            return f"data:{mime};base64,{base64.b64encode(raw).decode()}"

    im = upright(to_image(src))
    if max_edge and max(im.size) > max_edge:
        im.thumbnail((max_edge, max_edge), _resample())
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _resample():
    from PIL import Image
    return Image.LANCZOS


def _detail(body: bytes) -> str:
    """The server's reason, out of whatever shape it arrived in."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace").strip()
    if isinstance(parsed, dict) and isinstance(parsed.get("detail"), str):
        return parsed["detail"]
    # a validation error is a list of field reports; printing it beats hiding it
    return json.dumps(parsed)


def _answers(r: dict) -> dict:
    """The answers, out of either server's envelope.

    A self-hosted `jevimage serve` answers about one image and puts them at the top
    level. The hosted API answers about a list and puts each image's under `results[i]`.
    Same answers, two envelopes - and a client that reads only one of them fails with a
    KeyError whose whole text is 'answers', which is the least useful thing it could say.
    """
    if "answers" in r:
        return r["answers"]
    results = r.get("results")
    if isinstance(results, list) and results and "answers" in results[0]:
        return results[0]["answers"]
    raise JevError("the server replied without answers in it; its keys were "
                   f"{', '.join(sorted(r)) or '(none)'}. Is that URL a jevimage server?")


class Remote:
    """A `jevimage serve` instance, with the same methods as `jevimage.Jev`.

        jev = jevimage.connect("http://localhost:8731")
        jev.ask("photo.jpg", {"scene": {"type": "choice", "criteria": {...}}})

    One instance is safe to share between threads: the socket it keeps open is held
    under a lock, so calls queue rather than interleave on the wire.
    """

    def __init__(self, url: str, *, api_key: str | None = None, timeout: float = 60.0,
                 max_edge: int | None = MAX_EDGE):
        if "://" not in url:
            # the common typo for a URL handed over in chat; urllib's own message is
            # "unknown url type: 127.0.0.1", which points at the port, not the scheme
            raise ValueError(f"url {url!r} has no scheme; use 'http://{url}'")
        self.url = url.rstrip("/")
        if api_key is not None:
            # a key read from a file or $(cat) carries a newline, and http.client then
            # fails deep inside putheader with the key itself in the traceback. The
            # server splits $JEVIMAGE_API_KEYS on whitespace, so such a key is never valid.
            if api_key != api_key.strip() or not api_key:
                raise ValueError("api_key has surrounding whitespace or is empty; "
                                 "strip it (the value is not shown here on purpose)")
            try:
                api_key.encode("latin-1")
            except UnicodeEncodeError:
                raise ValueError("api_key has characters that cannot go in an HTTP "
                                 "header; keys are ASCII") from None
        self.api_key = api_key
        self.timeout = timeout
        # images larger than this are downscaled before upload; None sends originals
        self.max_edge = max_edge
        self._parts = urllib.parse.urlsplit(self.url)
        if self._parts.scheme not in ("http", "https"):
            raise ValueError(f"url {url!r} must be http or https, not "
                             f"{self._parts.scheme!r}")
        self._lock = threading.Lock()
        self._conn = None

    # A fresh TCP handshake costs a full round trip, which against a server on another
    # continent is ~200ms - an order of magnitude more than the model spends answering.
    # One connection is kept and reused; a server that closed it gets one reconnect.
    def _open(self):
        cls = (http.client.HTTPSConnection if self._parts.scheme == "https"
               else http.client.HTTPConnection)
        return cls(self._parts.netloc, timeout=self.timeout)

    def close(self) -> None:
        """Drop the kept connection. Safe to call at any time; the next call reopens."""
        with self._lock:
            self._shut()

    def _shut(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- transport -------------------------------------------------------
    def _call(self, method: str, path: str, body: dict | None = None):
        url = self.url + path
        target = (self._parts.path or "") + path
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(data))

        with self._lock:
            for attempt in (0, 1):
                reused = self._conn is not None
                try:
                    if self._conn is None:
                        self._conn = self._open()
                    self._conn.request(method, target, body=data, headers=headers)
                    r = self._conn.getresponse()
                    payload = r.read()
                    status = r.status
                    location = r.getheader("Location")
                    break
                except TimeoutError:
                    self._shut()
                    raise JevError(
                        f"{method} {url} did not answer within {self.timeout}s; raise "
                        "timeout= if the server is loading a model") from None
                except (http.client.HTTPException, OSError) as e:
                    self._shut()
                    # A kept connection the server has since closed is worth exactly one
                    # silent retry - but only for a method that is safe to run twice.
                    # The failure surfaces on getresponse(), i.e. possibly after the
                    # server has already trained the head, so replaying a POST would
                    # train it again (and bill it again) with the caller none the wiser.
                    if reused and attempt == 0 and method in ("GET", "DELETE"):
                        continue
                    # `jevimage serve` does not terminate TLS, so an https:// URL
                    # pointed at one gets WRONG_VERSION_NUMBER out of OpenSSL - which
                    # describes the bytes rather than the mistake.
                    plain = (self.url.startswith("https://")
                             and "WRONG_VERSION_NUMBER" in str(e))
                    raise JevError(
                        f"cannot reach a jevimage server at {self.url} ({e}). "
                        + ("That server is speaking plain HTTP: use http://, or put a "
                           "proxy in front of it to terminate TLS."
                           if plain else
                           "Start one with `jevimage serve`, or pass the URL it is "
                           "actually listening on.")
                    ) from None

        if 300 <= status < 400:
            # the explanation first, the body after: every real proxy answers a
            # redirect with an HTML page, and dumping that page hides the one fact the
            # caller cannot get anywhere else - where it was being sent, and why we
            # did not go there.
            detail = (
                f"{url} redirected to {location!r}; jevimage does not follow redirects, "
                "because that would carry your Authorization header to wherever they "
                "point. Connect to the final URL directly.")
            body = _detail(payload)
            if body:
                detail = f"{detail} (server said: {body[:200]})"
            raise JevError(f"{detail} [HTTP {status} from {method} {url}]",
                           status=status, detail=detail)
        if status >= 400:
            detail = _detail(payload)
            raise JevError(f"{detail} [HTTP {status} from {method} {url}]",
                           status=status, detail=detail)
        try:
            return json.loads(payload or b"{}")
        except ValueError:
            snippet = payload[:80].decode("utf-8", "replace").strip()
            detail = (f"{url} answered {status} but not JSON ({snippet!r}); "
                      "is that a jevimage server?")
            raise JevError(detail, status=status, detail=detail) from None

    def health(self) -> dict:
        """What the server is running: encoder, dim, device, auth mode, version.

        `Jev.encoder` is local-only - there is no encoder object on this side of the
        wire - so this is how you ask which encoder your heads will be trained under,
        and whether single-caption `noul` will work.
        """
        h = self._call("GET", "/v1/health")
        if "primitives" in h:
            return h
        # The hosted API keeps /v1/health for liveness alone and puts the capability
        # document behind /v1/models. Same question, different door; answered in the
        # shape this method promises so a caller does not have to know which it reached.
        try:
            m = self._call("GET", "/v1/models")
        except JevError:
            return h                # no key, or not a jevimage server: say what we saw
        return {"encoder": m.get("model"), "dim": m.get("dim"),
                "device": m.get("device", "hosted"),
                "primitives": m.get("primitives", []),
                "heads": len(m.get("heads") or []),
                # who the key belongs to, when the server says. Two accounts see two
                # different lists of heads, correctly, and the difference is otherwise
                # invisible from a shell.
                "account": m.get("account"),
                "auth": "bearer", "version": str(m.get("version", ""))}

    # -- embedding -------------------------------------------------------
    def embed(self, images):
        """Unit embeddings for one image or a list. Returns (n, dim) fp32.

        The round trip costs what the upload costs, so pass a list rather than looping -
        and cache what comes back if you will ask about it again.
        """
        single = not isinstance(images, (list, tuple))
        batch = [images] if single else list(images)
        if not batch:
            raise ValueError("embed() needs at least one image")
        # the server caps a request at MAX_IMAGES; Jev.embed() has no cap, and it is
        # stateless either way, so chunking here is what makes the two interchangeable
        rows = []
        for i in range(0, len(batch), MAX_IMAGES):
            chunk = batch[i:i + MAX_IMAGES]
            rows += self._call("POST", "/v1/embed",
                               {"images": [_data_url(x, self.max_edge)
                                           for x in chunk]})["embeddings"]
        # A torch.Tensor when torch is installed, so this is interchangeable with
        # Jev.embed(); a plain list of lists otherwise, because an API-only install
        # deliberately has no torch and should still be able to read embeddings.
        try:
            import torch
        except ImportError:
            return rows
        return torch.tensor(rows, dtype=torch.float32)

    # -- asking ----------------------------------------------------------
    def ask(self, image, questions, *, temperature=None) -> dict[str, Answer]:
        """Answer every question about one image. Returns {question id: Answer}."""
        # the same refusal Jev.ask gives - same type, same first sentence - so an
        # `except TypeError` written against the local half keeps working here
        if isinstance(image, (list, tuple)):
            raise TypeError(
                f"ask() takes one image, got a {type(image).__name__} of {len(image)}. "
                # deliberately not the local wording: ask_embedding is local-only, so
                # sending a remote caller to it lands them on an AttributeError
                "Loop over them, one ask() each.")
        # validated here, with the torch-free half of the core, so a malformed request
        # gets the same sentence a local Jev gives instead of pydantic's field dump
        questions = normalise_questions(questions)
        if temperature is not None:
            positive_temperature(temperature)
        r = self._call("POST", "/v1/ask", {"image": _data_url(image, self.max_edge),
                                           "questions": questions,
                                           "temperature": temperature})
        return {qid: Answer(a) for qid, a in _answers(r).items()}

    # -- training --------------------------------------------------------
    def train(self, name, data, *, epochs=500, max_per_class=None) -> Head:
        """Fit a head on the server from your own labelled images.

        `data` is a folder, a {label: [images]} mapping, or a list of (image, label)
        pairs - the same three shapes `Jev.train` takes. Every image is uploaded once;
        the head itself stays on the server, and what comes back is its metadata.
        """
        # the server bounds this (server.py TrainIn), and a pydantic field dump is
        # not the sentence a local Jev would have given
        if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 5000:
            raise ValueError(f"epochs must be an int from 1 to 5000, got {epochs!r}")
        images, labels = as_examples(data, max_per_class)
        r = self._call("POST", "/v1/heads",
                       {"name": valid_name(name),
                        "examples": [{"image": _data_url(i, self.max_edge), "label": l}
                                     for i, l in zip(images, labels)],
                        "epochs": epochs})
        # the server returns the public head flat, with the comparison alongside it;
        # Head keeps them apart, so the comparison does not end up in to_dict()
        return Head(name, {k: v for k, v in r.items() if k != "comparison"},
                    r.get("comparison"))

    def compare(self, name):
        """Trained vs zero-shot prompts on the same images, per class. Same shape as
        `jevimage.load().compare()`, including None for a head saved without its
        embeddings."""
        try:
            return self._call("GET", f"/v1/heads/{valid_name(name)}/compare")
        except JevError as e:
            # the server says 409 for a head with no stored embeddings; Jev.compare says
            # None, and this is the drop-in, so it says None too
            if e.status == 409:
                return None
            raise

    def head(self, name) -> Head:
        return Head(name, self._call("GET", f"/v1/heads/{valid_name(name)}"))

    def list_heads(self) -> list[dict]:
        try:
            return self._call("GET", "/v1/heads")
        except JevError as e:
            if e.status != 404:
                raise
            # An older hosted API published its heads only in the capability document
            # and had no /v1/heads at all. The list is the same list, so answer it
            # rather than handing back a 404 for something that is right there.
            return (self._call("GET", "/v1/models").get("heads") or [])

    def delete_head(self, name) -> bool:
        try:
            self._call("DELETE", f"/v1/heads/{valid_name(name)}")
        except JevError as e:
            # the server says 404 for a head that was not there; Jev.delete_head says
            # False, and this is the drop-in, so it says False too
            if e.status == 404:
                return False
            raise
        return True

    def __repr__(self):
        return f"<Remote {self.url!r}{' keyed' if self.api_key else ''}>"


def demo():
    """python3 -m jevimage.client - the package's own server in a thread, driven over
    real HTTP. It runs `jevimage.server.build_app`, not a stand-in, because the only
    bug worth catching here is the client and the server disagreeing about the wire."""
    import socket
    import tempfile
    import threading
    import time

    import torch
    import uvicorn
    from PIL import Image

    import jevimage
    from .server import _Toy, build_app

    tmp = Path(tempfile.mkdtemp(prefix="jev-client-demo-"))
    heads_dir = tmp / "heads"
    ds = tmp / "ds"
    for label, rgb in (("red", (200, 40, 40)), ("blue", (40, 40, 200))):
        (ds / label).mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (32, 32), (rgb[0] - i * 5, rgb[1], rgb[2])).save(
                ds / label / f"{i}.png")
    served = jevimage.load(_Toy(), heads_dir=str(heads_dir))  # what the server runs
    local = jevimage.load(_Toy(), heads_dir=str(heads_dir))   # the drop-in comparison

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    # keyed, because the Authorization header is the one thing a TestClient in
    # server.py's own selftest cannot prove travels correctly over a real socket
    app = build_app(served, api_keys=["s3cret"])
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                        log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.01)

    jev = jevimage.connect(f"http://127.0.0.1:{port}", api_key="s3cret")
    img = ds / "red" / "0.png"
    qs = {"colour": {"type": "choice", "criteria": {"red": "a red square",
                                                    "blue": "a blue square"}},
          "busy": {"type": "score", "criteria": ["empty", "half full", "packed"]},
          "outside": {"type": "noul", "criteria": {"true": "outdoors",
                                                   "false": "indoors"}}}

    # the whole promise: same call, same objects. Compared with a tolerance, not ==:
    # captions are encoded in whatever batch the request needs, so a warm server and a
    # fresh script can differ in the last decimal of a float.
    def close(a, b, where=""):
        assert type(a) is type(b), (where, a, b)
        if isinstance(a, float):
            assert abs(a - b) < 1e-4, (where, a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), (where, a, b)
            for k in a:
                close(a[k], b[k], f"{where}.{k}")
        else:
            assert a == b, (where, a, b)

    remote_a, local_a = jev.ask(img, qs), local.ask(img, qs)
    close(dict(remote_a), dict(local_a), "answers")
    assert all(isinstance(a, Answer) for a in remote_a.values())
    assert remote_a["colour"].choice == local_a["colour"].choice
    assert isinstance(remote_a["colour"].probabilities, dict)
    assert remote_a["busy"].score == local_a["busy"].score

    e = jev.embed([img, ds / "blue" / "0.png"])
    assert e.shape == (2, 16) and e.dtype == torch.float32, (e.shape, e.dtype)
    assert torch.allclose(e, local.embed([img, ds / "blue" / "0.png"]), atol=1e-6)
    assert jev.embed(img).shape == (1, 16)
    # past the server's 8-image cap: embed() chunks, so it matches Jev.embed()
    many = [img, ds / "blue" / "0.png"] * 6
    assert torch.allclose(jev.embed(many), local.embed(many), atol=1e-6)

    assert jev.health()["encoder"] == "toy", jev.health()
    # a bad head name is the local ValueError, not a 404 or a silent False
    for bad in ("../colours", "bad name!", ""):
        for call in (jev.head, jev.delete_head, jev.compare):
            try:
                call(bad)
                raise AssertionError(f"{call.__name__} accepted {bad!r}")
            except ValueError as err:
                assert "invalid head name" in str(err), err

    # a PIL image and raw bytes must reach the server as the same pixels as the path
    assert torch.allclose(jev.embed(Image.open(img)), e[0:1], atol=1e-6)
    assert torch.allclose(jev.embed(img.read_bytes()), e[0:1], atol=1e-6)

    head = jev.train("colours", str(ds))
    assert isinstance(head, jevimage.Head) and head.classes == ["blue", "red"]
    assert head.n == 6 and head.encoder == "toy", head.to_dict()
    assert head.accuracy is not None and head.comparison["overall"]["zero_shot"] >= 0
    assert "W" not in head.to_dict(), "weights must not cross the wire"

    hq = {"c": {"type": "head", "head": "colours"}}
    assert jev.ask(img, hq) == local.ask(img, hq)
    assert [h["name"] for h in jev.list_heads()] == ["colours"]
    assert jev.head("colours").classes == ["blue", "red"]
    assert head.comparison["per_class"][0]["label"] == "blue"
    # compare() must agree with what the local engine computes for the same head
    cmp = jev.compare("colours")
    assert set(cmp) == {"overall", "per_class", "basis"}, cmp
    assert cmp["overall"]["trained"] is not None
    assert {r["label"] for r in cmp["per_class"]} == {"red", "blue"}

    # the server's own words, not "HTTP 422"
    try:
        jev.ask(img, {"c": {"type": "head", "head": "nope"}})
        raise AssertionError("unknown head was accepted")
    except JevError as err:
        assert err.status == 422 and "unknown head 'nope'" in str(err), err
    try:
        jev.head("nope")
        raise AssertionError("unknown head was accepted")
    except JevError as err:
        assert err.status == 404 and "no head named 'nope'" in str(err), err

    assert jev.delete_head("colours") is True
    assert jev.delete_head("colours") is False      # 404, but Jev.delete_head says False
    assert jev.list_heads() == []

    # the key is really being checked, so the keyed run above proved the header travels
    try:
        jevimage.connect(f"http://127.0.0.1:{port}", api_key="wrong").list_heads()
        raise AssertionError("a wrong api key was accepted")
    except JevError as err:
        assert err.status == 401, err

    # nothing listening: say where, and what to run
    try:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()
        jevimage.connect(f"http://127.0.0.1:{dead}", timeout=2).list_heads()
        raise AssertionError("a closed port did not raise")
    except JevError as err:
        assert f"127.0.0.1:{dead}" in str(err) and "jevimage serve" in str(err), err

    # a redirect must not carry the key onward: an HTTP client that followed one would
    # copy Authorization to the
    # target host, so the 3xx has to surface as an error instead of being followed
    import http.server

    seen = []

    class _Hop(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            # with a body, because every real proxy sends its own error page with the
            # 3xx, and the explanation must survive it rather than be replaced by it
            page = b"<html><body>301 Moved Permanently (nginx)</body></html>"
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/v1/health")
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *a):
            pass

    hop = http.server.HTTPServer(("127.0.0.1", 0), _Hop)
    threading.Thread(target=hop.serve_forever, daemon=True).start()
    try:
        jevimage.connect(f"http://127.0.0.1:{hop.server_address[1]}",
                         api_key="s3cret").health()
        raise AssertionError("a redirect was followed")
    except JevError as err:
        assert err.status == 302 and "does not follow redirects" in str(err), err
        assert "example.invalid" in str(err), err        # where, not just that
        assert "nginx" in str(err), err                  # the proxy's page, after it
    assert seen == ["Bearer s3cret"], seen      # sent once, to the host we named
    hop.shutdown()

    print(f"client selftest ok - ask/embed/train/head/delete against jevimage.server "
          f"over real HTTP on :{port}, answers identical to a local Jev, bearer key "
          f"enforced, redirects refused rather than followed, server errors surfaced "
          f"verbatim")


if __name__ == "__main__":
    # `python3 -m jevimage.client` loads this file twice - once as __main__, once as
    # jevimage.client when connect() imports it - and the two JevError classes are then
    # not the same class. Run the package's copy, so `except JevError` means one thing.
    from jevimage.client import demo as _demo

    _demo()
