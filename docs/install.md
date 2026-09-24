---
title: "Install"
description: "The four installs, what each pulls in, and how to verify the one you chose."
---

jevimage ships as one package with three runtime extras, plus `[dev]` and `[docs]` for
working on the package itself. The split exists because the encoder is the expensive
part and not everybody runs it. Someone who only calls a server should not download a
multi-gigabyte deep-learning stack to send an HTTP request, so the base install is the
HTTP client and nothing else.

Both paths are first class. `jevimage.connect(url)` and `jevimage.load()` answer the same
calls and return the same objects. Two methods differ: `ask_embedding` is local only
because its point is skipping the network, and `health` is remote only because there is
no server to report on locally. One argument differs too: `template=` on `train()` and
`compare()` has no remote equivalent, because the server picks the template. The full
list is the [difference table](/docs/reference-python#what-differs-from-jev).

Which one you install is a deployment decision, not an API decision. The server can be
yours: the axis is how many processes need the model, not who owns the machine. One
process (a script, a batch job, a single worker): use `load()`. Several worker
processes on one box: run `jev serve` there and `connect()` to it from each, because
`uvicorn --workers N` or N app processes each load their own copy of the weights, and
you pay N times 1.5 GB and N cold starts for a model that is idle most of the time.

## Which one do you want

| Command | Adds | Use it when |
|---|---|---|
| `pip install jevimage` | pillow | Something else runs the encoder. You call it over HTTP. |
| `pip install 'jevimage[local]'` | torch, transformers | You want the model in your process. Images never leave the machine. |
| `pip install 'jevimage[serve]'` | `[local]` + fastapi, uvicorn | You are the server other people connect to. |
| `pip install 'jevimage[openclip]'` | `[local]` + open_clip_torch | You want the `dfn5b-h-14-384` registry entry. |

`[serve]` and `[openclip]` both depend on `[local]`, so you never need to list it twice.
`pip install 'jevimage[serve,openclip]'` is valid and gives you all of it.

Sizes measured on the machine this page was written on (Linux, CPython 3.12, x86-64;
yours will differ, especially for torch):

| Install | On disk |
|---|---|
| `jevimage` | 336 KB package + 21 MB pillow. The whole venv, pip included, was 37 MB. |
| `[local]` | + 1.2 GB torch, 101 MB transformers — and 3.2 GB of `nvidia-*` CUDA wheels, which the default torch build pulls in whether or not you have a GPU. |
| `[serve]` | + about 12 MB (fastapi, starlette, uvicorn, pydantic). |
| `[openclip]` | Not measured here — open_clip_torch was not installed on this machine, and quoting a number I did not weigh would be a guess. |

Model weights are a separate download on top of `[local]`, and they are the big number:
see [Model weights and offline use](#model-weights-and-offline-use).

## The API-only install

This is the one most people should start with. Asking, embedding, listing heads,
deleting heads **and training** all work through it, because the encoder that fits the
head lives on the server:

```console
$ pip install jevimage
$ python -c "import jevimage; print(jevimage.__version__)"
0.1.0
```

Training from a machine with no torch on it at all:

```console
$ python examples/make_fixtures.py          # writes ds/red/0..5.png, ds/blue/0..5.png
$ jev train colour ds/ --url http://127.0.0.1:8123
'colour': 12 examples, 2 classes, encoder toy

class   n  zero-shot  trained   delta
blue    6      50.0%    50.0%    0.0%
red     6     100.0%    50.0%  -50.0%
ALL    12      75.0%    50.0%  -25.0%

trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example.
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
```

That server was running a deterministic stand-in encoder (`encoder toy` in the output);
against a real encoder the accuracies are real accuracies and everything else about the
exchange is identical.

Two things behave differently without torch, both deliberately:

- `Remote.embed()` returns a plain list of lists instead of a `(n, dim)` tensor. With
  torch installed it returns the tensor, so the local and remote calls stay
  interchangeable.
- `jev heads` **without** `--url` still needs torch. Not because of the files: the local
  branch imports `jevimage.training` for its one formatting helper, and that module
  imports torch at module scope, so the command fails even when the head directory is
  empty or absent. The CLI's error says "this command runs the encoder locally", which
  overstates it slightly (no encoder is loaded), but the fix it names is the right one.
  `jev rm` is the exception: it is an `unlink` and never takes that branch, so it works
  on the base install.

```console
$ jev heads
jev: this command runs the encoder locally, which needs torch:
       pip install 'jevimage[local]'
     or point it at a server that already has one:
       jev heads --url http://host:8000   (also $JEV_URL)
```

`jev encoders` works in the API-only install — listing which encoders exist should not
require the stack that runs them:

```console
$ jev encoders
   clip-vit-l-14
   dfn5b-h-14-384
   metaclip2-huge
 * siglip2-base-224
   siglip2-base-384
   siglip2-giant-384
   siglip2-so400m-384

* default (siglip2-base-224). Join names with '+' for an ensemble, e.g. siglip2-base-224+dfn5b-h-14-384.
```

## What `load()` does without `[local]`

It raises `ImportError` naming both fixes, rather than letting a `ModuleNotFoundError`
from three frames down be your first hint:

```console
$ python -c "import jevimage; jevimage.load()"
Traceback (most recent call last):
  ...
    import torch
ModuleNotFoundError: No module named 'torch'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File ".../jevimage/__init__.py", line 88, in __getattr__
    raise ImportError(
ImportError: jevimage.load runs the encoder in this process, which needs torch and transformers: pip install 'jevimage[local]'
To use a server instead - no torch required - call jevimage.connect(url).
(underlying import error: No module named 'torch')
```

The same applies to `jevimage.Jev`. `jevimage.Encoder`, `register`, `available`,
`DEFAULT` and `JevError` resolve without torch, so you can inspect the registry and
catch errors in an API-only process.

## Verifying an install landed

Run the check for the level you installed. Each of these was run as printed.

**Base.** Imports, version, and the encoder registry:

```console
$ python -c "import jevimage; print(jevimage.__version__)"
0.1.0
$ python -c "import jevimage; print(jevimage.available())"
['clip-vit-l-14', 'dfn5b-h-14-384', 'metaclip2-huge', 'siglip2-base-224', 'siglip2-base-384', 'siglip2-giant-384', 'siglip2-so400m-384']
```

Against a server you can reach, `health()` is the end-to-end check — it tells you which
encoder your answers and heads will come from:

```console
$ python -c "import jevimage; print(jevimage.connect('http://127.0.0.1:8199').health())"
{'encoder': 'siglip2-base-224', 'dim': 768, 'device': 'cpu', 'primitives': ['choice', 'score', 'noul', 'head'], 'heads': 0, 'auth': 'open', 'version': '0.1.0'}
```

**`[local]`.** The real check is loading an encoder, because that is what exercises
torch, transformers and the weight cache at once:

```console
$ JEV_DEVICE=cpu python -c "import jevimage, time; t=time.time(); j=jevimage.load(); print(j, round(time.time()-t,1), 's')"
<Jev encoder='siglip2-base-224' dim=768 heads=0> 9.7 s
```

That was a warm cache on a 16-thread laptop CPU (12th Gen Core i7-12650H): repeated runs
on a quiet box land between 8 and 10 s, and one while the machine was otherwise busy took
32.8 s, because this is dominated by pulling 1.5 GB off disk. The first ever call also
downloads that 1.5 GB.

**`[serve]`.** Start it and ask it how it is:

```console
$ JEV_DEVICE=cpu jev serve --port 8199 &
$ curl -s http://127.0.0.1:8199/v1/health
{"encoder":"siglip2-base-224","dim":768,"device":"cpu","primitives":["choice","score","noul","head"],"heads":0,"auth":"open","version":"0.1.0"}
```

`auth: open` means no API key is set and anyone who can reach the port can use it. See
[Serving](/docs/serving) before you bind it to anything but localhost.

**`[openclip]`.** `python -c "import open_clip"` and then `jevimage.load("dfn5b-h-14-384")`.
Without the extra, the failure is the plain import error from the encoder's constructor.
It is not wrapped, so read the last line:

```console
$ python -c "import jevimage; jevimage.load('dfn5b-h-14-384')"
  ...
  File ".../jevimage/encoders.py", line 164, in __init__
    import open_clip
ModuleNotFoundError: No module named 'open_clip'
```

Those DFN5B weights are Apple's and carry Apple's own licence. That is why open_clip is
an extra and the weights are never a dependency: whether to accept that licence is your
call, not the package's.

## Python versions

`requires-python = ">=3.10"`. The source is written in PEP 604 union syntax
(`str | None`) throughout, and while `from __future__ import annotations` keeps most of
that from being evaluated, pydantic resolves the server's request-model annotations at
runtime, so 3.10 is a real floor. Everything on this page was run on CPython 3.12.3;
the declared floor is what pip enforces, and 3.10 and 3.11 are not exercised here.

There is no upper bound in the metadata, so on a very new Python the thing likely to
stop you is torch, not jevimage.

## From source

**The package is not on PyPI yet.** Until it is, every `pip install jevimage...`
line in these docs means the equivalent against a clone: `pip install .`,
`pip install '.[local]'`, and so on.

```console
$ git clone <repo> && cd jevimage-oss
$ pip install .                  # or: pip install -e '.[local]'
```

The build needs `setuptools>=77`. That is the release where `license = "MIT"` became
the correct spelling in `pyproject.toml`, and an older setuptools will reject the
metadata. Normally pip fetches it into an isolated build environment for you. On a
machine with no index reachable, supply it yourself:

```console
$ pip install --no-index --no-build-isolation --no-deps /path/to/jevimage-oss
Successfully built jevimage
Installing collected packages: jevimage
Successfully installed jevimage-0.1.0
$ python -c "import jevimage; print(jevimage.__file__, jevimage.__version__)"
.../site-packages/jevimage/__init__.py 0.1.0
```

The built wheel is 51 KB. `--no-deps` there means you still have to have pillow (and,
for `[local]`, torch and transformers) present by some other route.

## CPU, GPU, and how the device is chosen

`jevimage.encoders.resolve_device()` decides, in this order:

1. an explicit `device=` argument to the encoder constructor,
2. `$JEV_DEVICE`,
3. `"cuda"` if `torch.cuda.is_available()`,
4. `"cpu"`.

The dtype follows from the device and is not separately configurable: bfloat16 on CUDA,
float32 on CPU. bf16 is faster on a GPU at no measured accuracy cost, and slower than
fp32 on a CPU, so CPU stays fp32. Embeddings are normalised back to fp32 either way,
because bf16 has too few mantissa bits for a stable softmax, and a probability that
changes with the dtype is not a probability.

```console
$ python -c "from jevimage.encoders import resolve_device, resolve_dtype; d=resolve_device(); print(d, resolve_dtype(d))"
cuda torch.bfloat16
$ JEV_DEVICE=cpu python -c "from jevimage.encoders import resolve_device, resolve_dtype; d=resolve_device(); print(d, resolve_dtype(d))"
cpu torch.float32
```

`$JEV_DEVICE` takes any string torch accepts, so `cuda:1` and `mps` work as far as torch
does. A value torch does not recognise, or a CUDA ordinal this machine does not have, is
refused before any weights load - one line, exit 1, no model download for a typo.

Two caveats:

- Only the built-in `HFCLIP` and `OpenCLIP` encoders consult `resolve_device`. A custom
  encoder you pass to `load()` does whatever you wrote.
- `/v1/health` reports `getattr(jev.encoder, "device", "cpu")`. An encoder without a
  `.device` attribute is reported as `cpu` whether or not that is true. The built-ins
  always set it.

CPU is a supported way to run this. A base-224 encode on CPU is slow by GPU standards
and fine for a batch job. If you have no GPU, install torch from the CPU-only index
first and then jevimage; that is what avoids the several gigabytes of `nvidia-*` wheels
listed above.

## Model weights and offline use

jevimage downloads nothing itself. `HFCLIP` hands the model id to transformers and
`OpenCLIP` hands it to open_clip; both cache through huggingface_hub, and all the
offline knobs are huggingface's, not jevimage's.

Default cache, as reported by the installed huggingface_hub:

```console
$ python -c "from huggingface_hub import constants as c; print(c.HF_HOME); print(c.HF_HUB_CACHE)"
/home/you/.cache/huggingface
/home/you/.cache/huggingface/hub
$ HF_HOME=/srv/models python -c "from huggingface_hub import constants as c; print(c.HF_HOME, c.HF_HUB_CACHE)"
/srv/models /srv/models/hub
```

Setting `HF_HOME` moves everything; `HF_HUB_CACHE` moves just the model blobs. Point
them at a shared volume if several processes or containers should share one copy. The
cache is content-addressed and safe to share read-only once it is populated.

What the default encoder actually occupies, from the cache on this machine:

```console
$ ls -lL ~/.cache/huggingface/hub/models--google--siglip2-base-patch16-224/snapshots/*/ | awk '{print $5, $9}'
      253 config.json
1500800904 model.safetensors
      394 preprocessor_config.json
      636 special_tokens_map.json
    47164 tokenizer_config.json
 34363039 tokenizer.json
```

1.5 GB for `siglip2-base-224`. `siglip2-so400m-384` came to 4.3 GB in the same cache.
Budget accordingly before picking a bigger encoder in a container image.

**Air-gapped.** Pre-download on a connected machine with the same encoder name, copy the
cache directory over, and set `HF_HUB_OFFLINE=1` on the air-gapped one. Loading then
never touches the network:

```console
$ HF_HUB_OFFLINE=1 python -c "import jevimage; print(jevimage.load())"
<Jev encoder='siglip2-base-224' dim=768 heads=0>
```

The pre-download itself is a plain `hf download google/siglip2-base-patch16-224`. That is
huggingface_hub's own CLI, and on hub 1.x it is the only spelling that works; the older
`huggingface-cli download` now exits with "`huggingface-cli` is deprecated and no longer
works". Calling `jevimage.load("siglip2-base-224")` once while the network is up does the
same job. The registry name to model id mapping is in `jevimage/encoders.py`; see
[Choosing an encoder](/docs/encoders).

A cache miss under `HF_HUB_OFFLINE=1` fails from transformers, not from jevimage, and
the message is about the hub rather than about the encoder name. Do not read it as
"no such encoder":

```console
$ HF_HUB_OFFLINE=1 python -c "import jevimage; jevimage.load('siglip2-base-384')"
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

A wrong *name* looks completely different, and lists what it has:

```console
$ python -c "import jevimage; jevimage.load('siglip3')"
KeyError: "unknown encoder 'siglip3'. Available: clip-vit-l-14, dfn5b-h-14-384, metaclip2-huge, siglip2-base-224, siglip2-base-384, siglip2-giant-384, siglip2-so400m-384. Register your own with jevimage.register(name, factory) - from Python, in the process that will serve it; the `jev` command only sees these built-in names, because it never imports your code."
```

The other fully offline option is to skip the hub entirely: `load()` accepts any object
with `.name`, `.dim`, `.img()` and `.txt()`, so an encoder you load from a local
directory needs no registry entry and no network. That is [Choosing an
encoder](/docs/encoders).

Nothing about the API-only install touches the hub at all — no weights, no cache, no
`HF_*` variables. That is often the simplest answer to an air-gap: one machine holds the
weights, everything else connects to it.

## Environment variables

Every variable the package itself reads. These are the complete set; anything else is
read by torch, transformers or huggingface_hub, not here.

| Variable | Read by | Effect | Default |
|---|---|---|---|
| `JEV_DEVICE` | `encoders.resolve_device()` | Torch device for the built-in encoders. Loses to an explicit `device=`, beats CUDA autodetection. | CUDA if available, else CPU |
| `JEV_ENCODER` | `encoders.load()` when no name is passed; shown by `jev encoders` | Which registry name `jevimage.load()` and `jev serve` use. | `siglip2-base-224` |
| `JEV_HOME` | `store.default_dir()` | Base directory for trained heads; they land in `$JEV_HOME/heads`. | `~/.jev` |
| `JEV_HEADS_DIR` | `store.default_dir()` | The heads directory itself. Overrides `JEV_HOME` outright. | `$JEV_HOME/heads` |
| `JEV_API_KEYS` | `server.build_app()` | Keys the server accepts as `Authorization: Bearer`. Comma- or whitespace-separated. Empty means the API is open. | unset (open) |
| `JEV_URL` | the CLI, on `ask`, `train`, `heads`, `rm` | Run that command against a server instead of local weights. Same as `--url`. | unset (local) |
| `JEV_API_KEY` | the CLI, with `JEV_URL`/`--url` | The single key the CLI sends. Note the singular: the server reads `JEV_API_KEYS`. | unset |
| `JEV_TIMEOUT` | the CLI, with `JEV_URL`/`--url` | Seconds to wait on a server call. `--timeout` beats it; that flag is on `ask` and `train`, the variable applies to all four remote commands. Raise it for a big `train` folder. | 60 |

```console
$ python -c "from jevimage.store import default_dir; print(default_dir())"
/home/you/.jev/heads
$ JEV_HOME=/tmp/x python -c "from jevimage.store import default_dir; print(default_dir())"
/tmp/x/heads
$ JEV_HEADS_DIR=/tmp/y python -c "from jevimage.store import default_dir; print(default_dir())"
/tmp/y
```

`jev serve --encoder`, `--heads-dir` and `--api-key` work by setting `JEV_ENCODER`,
`JEV_HEADS_DIR` and `JEV_API_KEYS` in the process before the app is built, so the flags
and the variables are the same knob, and a unit file or container that sets the
variables directly behaves identically.

`heads_dir=` passed to `jevimage.load()` wins over both directory variables. `JEV_URL`,
`JEV_API_KEY` and `JEV_TIMEOUT` are read by the CLI only: from Python you pass the URL,
key and `timeout=` to `jevimage.connect()` yourself.

## Failure modes

| What you see | What it means |
|---|---|
| `ImportError: jevimage.load runs the encoder in this process...` | API-only install. Either add `[local]` or switch the line to `connect(url)`. |
| `jev: this command runs the encoder locally, which needs torch` | Same, from the CLI. On `heads` it is the local branch's `import jevimage.training` that pulls torch in, not the encoder, so it raises even with no head files at all; `rm` never takes that branch and does not raise this. |
| `jev: jev serve needs uvicorn: pip install 'jevimage[serve]'` | `[serve]` is missing. |
| `ModuleNotFoundError: No module named 'open_clip'` | You asked for `dfn5b-h-14-384` without `[openclip]`. |
| `KeyError: unknown encoder '...'` | A name that is not in the registry. The message lists the registry. |
| `OSError: We couldn't connect to 'https://huggingface.co'` | Weights are not cached and the hub is unreachable. Pre-download, or point `HF_HOME` at a populated cache. |
| `JevError: cannot reach jev at http://...` | Nothing is listening. The client says which URL it tried. |
| `ValueError: url '127.0.0.1:8000' has no scheme` | `connect()` wants `http://` on the front. |
| `No module named 'setuptools'` while installing from source | Build isolation is off and setuptools is not in the environment. Install `setuptools>=77`, or drop `--no-build-isolation`. |

## Next

[Quickstart](/docs/quickstart) runs the same walkthrough twice, once against a server and
once in-process. [Serving](/docs/serving) covers `jev serve` for real traffic. [Choosing an
encoder](/docs/encoders) covers the registry, ensembles and bringing your own model.

---

[← Overview](/docs/index) · [Quickstart →](/docs/quickstart)
