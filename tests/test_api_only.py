"""`pip install jevimage` - the API-only install - must never need torch.

That is the product's headline claim and nothing else in the suite can see it break:
conftest imports torch, so a stray `import torch` at the top of client.py or images.py
would leave the rest of the suite green and every API-only install broken.

The install is simulated with a meta_path hook that refuses torch, transformers and
open_clip, in a subprocess so the hook cannot leak into the rest of the session - by the
time pytest runs, torch is already imported and cannot be un-imported.
"""
import subprocess
import sys
import textwrap

import pytest

BLOCK = """
import sys


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "transformers", "open_clip"):
            raise ImportError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, Block())
"""


def run(body):
    """Run `body` in a fresh interpreter that cannot import torch. Returns its stdout."""
    r = subprocess.run([sys.executable, "-c", BLOCK + textwrap.dedent(body)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_torch_is_really_blocked():
    """The guard itself, or every assertion below proves nothing."""
    out = run("""
        try:
            import torch
            print("LEAKED")
        except ImportError:
            print("blocked")
    """)
    assert out.strip() == "blocked"


def test_import_and_the_light_half_of_the_api():
    out = run("""
        import jevimage
        print(jevimage.__version__)
        print(jevimage.DEFAULT)
        print(len(jevimage.available()) >= 7)
        print(jevimage.Encoder.__name__, jevimage.JevError.__name__)
        print(callable(jevimage.to_image), callable(jevimage.read_folder))
        print(jevimage.confidence([0.5, 0.5]))
    """).split("\n")
    assert out[1] == "siglip2-base-224"
    assert out[2] == "True"
    assert out[3] == "Encoder JevError"
    assert out[4] == "True True"
    assert out[5] == "0.0"


def test_load_names_the_fix_instead_of_failing_obscurely():
    out = run("""
        import jevimage
        try:
            jevimage.load()
            print("LOADED")
        except ImportError as e:
            print(e)
    """)
    assert "pip install 'jevimage[local]'" in out
    assert "jevimage.connect(url)" in out
    assert "No module named" in out          # the underlying cause is still shown


def test_connect_builds_a_client_and_validates_without_torch():
    """No server needed: these are the checks that must happen before a request."""
    out = run("""
        import jevimage
        jev = jevimage.connect("http://127.0.0.1:1")
        print(type(jev).__name__)
        for call, arg in ((jev.head, "../etc/passwd"), (jev.delete_head, "bad name!")):
            try:
                call(arg)
                print("ACCEPTED")
            except ValueError as e:
                print("invalid head name" in str(e))
        try:
            jevimage.connect("127.0.0.1:8123")
            print("ACCEPTED")
        except ValueError as e:
            print("has no scheme" in str(e))
        try:
            jev.ask("x.png", {"a": {}, "b": {}}, temperature=0)
            print("ACCEPTED")
        except ValueError as e:
            print("greater than 0" in str(e))
    """).split()
    assert out == ["Remote", "True", "True", "True", "True"]


def test_jev_encoders_runs():
    out = run("""
        from jevimage.cli import main
        raise SystemExit(main(["encoders"]))
    """)
    assert "siglip2-base-224" in out and "*" in out


def test_jev_rm_deletes_a_head_without_torch(tmp_path):
    """The one LOCAL command docs promise on the base install (docs/install.md:88).

    It has to be exercised with a head file actually PRESENT. `cmd_rm` used to build a
    HeadStore first, whose __init__ syncs the directory and imports torch as soon as it
    exists - so the command failed in exactly the case where there was something to
    delete, and "worked" only when the directory was absent and there was nothing to
    remove. A test that deletes nothing cannot see that.
    """
    heads = tmp_path / "heads"
    heads.mkdir()
    (heads / "quick.pt").write_bytes(b"not a real head, and nothing should read it")
    out = run(f"""
        import os
        os.environ["JEVIMAGE_HEADS_DIR"] = {str(heads)!r}
        from jevimage.cli import main
        raise SystemExit(main(["rm", "quick"]))
    """)
    assert "deleted 'quick'" in out
    assert not (heads / "quick.pt").exists(), "reported success without unlinking"


def test_jev_rm_refuses_a_bad_name_without_torch(tmp_path):
    """Name validation must be reached, not hidden behind the torch import."""
    heads = tmp_path / "heads"
    heads.mkdir()
    r = subprocess.run(
        [sys.executable, "-c", BLOCK + textwrap.dedent(f"""
            import os
            os.environ["JEVIMAGE_HEADS_DIR"] = {str(heads)!r}
            from jevimage.cli import main
            raise SystemExit(main(["rm", "../etc"]))
        """)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "invalid head name" in (r.stdout + r.stderr)
    assert "needs torch" not in (r.stdout + r.stderr), \
        "an unlink must not claim it runs the encoder locally"


def test_answers_and_images_are_torch_free():
    """The two modules both halves share: a torch import in either breaks the client."""
    run("""
        import jevimage.answers, jevimage.images, jevimage.client, jevimage.store
        import sys
        assert "torch" not in sys.modules
    """)


@pytest.mark.parametrize("module", ["jevimage.local", "jevimage.core", "jevimage.training"])
def test_the_heavy_half_still_asks_for_torch(module):
    """The mirror image: these are the modules that legitimately need it."""
    r = subprocess.run([sys.executable, "-c", BLOCK + f"import {module}"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "No module named" in r.stderr


# --- upload shrinking ------------------------------------------------------
# No encoder reads more than 384px, so an oversized upload is pure latency. These
# guard the two halves: that a big image really is cut, and that a small one is
# passed through byte for byte rather than silently re-encoded.

def _png(w, h, noise=False):
    import io

    from PIL import Image

    if noise:
        # flat colour compresses to almost nothing, which is the one case where
        # shrinking looks pointless; real photographs behave like noise
        import random
        r = random.Random(0)
        im = Image.new("RGB", (w, h))
        im.putdata([(r.randrange(256), r.randrange(256), r.randrange(256))
                    for _ in range(w * h)])
    else:
        im = Image.new("RGB", (w, h), "red")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_a_large_image_is_downscaled_before_upload():
    """The contract is the pixel count, not the byte count: how well the source
    compressed is not something the client controls."""
    import base64
    import io

    from PIL import Image

    from jevimage.client import _data_url

    raw = _png(3000, 2250)
    small = _data_url(raw, 768)
    assert small.startswith("data:image/jpeg;base64,")
    sent = Image.open(io.BytesIO(base64.b64decode(small.split(",", 1)[1])))
    assert max(sent.size) == 768, sent.size


def test_a_photographic_image_shrinks_by_an_order_of_magnitude():
    from jevimage.client import _data_url

    raw = _png(1600, 1200, noise=True)
    full, small = _data_url(raw, None), _data_url(raw, 768)
    assert len(small) < len(full) / 10, (len(small), len(full))


def test_an_image_already_small_enough_goes_over_untouched():
    import base64

    from jevimage.client import _data_url

    raw = _png(200, 150)
    url = _data_url(raw, 768)
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw      # byte for byte


def test_max_edge_none_disables_shrinking():
    import base64

    from jevimage.client import _data_url

    raw = _png(3000, 2250)
    url = _data_url(raw, None)
    assert base64.b64decode(url.split(",", 1)[1]) == raw


def test_an_unreadable_blob_is_left_for_the_server_to_reject():
    from jevimage.client import _shrink

    assert _shrink(b"not an image at all", 768) is None


def test_remote_passes_max_edge_through_to_the_upload():
    """The shrinking tests above call `_data_url` directly, so they cannot see the
    constructor argument being dropped between `connect()` and the wire."""
    out = run("""
        import base64, io, json, threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from PIL import Image
        import jevimage

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                one = body["images"][0] if "images" in body else body["image"]
                raw = base64.b64decode(one.split(",", 1)[1])
                print(max(Image.open(io.BytesIO(raw)).size))
                out = json.dumps({"embeddings": [[0.0]], "dim": 1}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        buf = io.BytesIO(); Image.new("RGB", (800, 600), "red").save(buf, "PNG")
        raw = buf.getvalue()
        for max_edge in (None, 64, 384):
            jevimage.connect("http://127.0.0.1:%d" % srv.server_address[1],
                             max_edge=max_edge).embed(raw)
        srv.shutdown()
    """).split()
    assert out == ["800", "64", "384"], out


def test_the_downscaled_upload_carries_the_exif_rotation():
    """A phone photo is both over the limit and EXIF-rotated. `Image.save` drops the
    Orientation tag, so the rotation has to be baked in before it goes, or the server
    encodes pixels a local Jev would never have encoded."""
    out = run("""
        import base64, io
        from PIL import Image, ImageOps
        from jevimage.client import _data_url

        exif = Image.Exif(); exif[274] = 6                     # portrait phone photo
        buf = io.BytesIO()
        Image.new("RGB", (800, 600), "red").save(buf, "JPEG", exif=exif.tobytes())
        raw = buf.getvalue()
        upright = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).size   # (600, 800)
        for src in (raw, Image.open(io.BytesIO(raw))):
            sent = Image.open(io.BytesIO(base64.b64decode(
                _data_url(src, 384).split(",", 1)[1])))
            # taller than wide, like the transposed original, and nothing left to rotate
            print(sent.height > sent.width, sent.getexif().get(274))
        # under the limit the original bytes go untouched, tag and all: the server
        # transposes them itself, so it must still be there
        small = io.BytesIO()
        Image.new("RGB", (300, 200), "red").save(small, "JPEG", exif=exif.tobytes())
        kept = Image.open(io.BytesIO(base64.b64decode(
            _data_url(small.getvalue(), 384).split(",", 1)[1])))
        print(kept.size == (300, 200), kept.getexif().get(274))
        print(upright)
    """).split("\n")
    assert out[0] == "True None", out
    assert out[1] == "True None", out
    assert out[2] == "True 6", out
