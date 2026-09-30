# jevimage
> ⚠️jevimage sdk is still in beta; if you encounter any error; ask claude to fix it; and would appreciate if you open a pull request ⚠️


jevimage answers typed questions about an image with probabilities instead of prose. The
image goes through a frozen dual encoder once and becomes a single unit-norm vector.
Every question after that (pick one of these options, place it on this rubric, yes or
no, run the head I trained on my own photos) is a matmul against that same vector.

Nothing is generated. An answer is a distribution over names *you* chose, plus a
confidence rescaled so one threshold works whether the question had two options or fifty.
So there is no sentence to parse and no retry loop. The same image and question give the
same numbers, to the six decimals they are printed to, every time, and the tenth question
about an image costs a matmul rather than another forward pass once its captions are
cached. A repeated question set is what gets cheap; a caption the process has never
seen still costs one text-tower pass. When prompts are not good enough you fit a linear
head on a folder of your own labelled images in seconds - and jevimage tells you, per
class, whether that head beat the prompts. Sometimes it did not, and it says so.

Full documentation is in [docs/](https://docs.jevimage.org/docs/index/).

## Two ways to use it

Both are first class, and they are the same API. Pick the row you are in.

|  | Use someone's server | Run it yourself |
|---|---|---|
| install | `pip install jevimage` | `pip install 'jevimage[local]'` |
| what that pulls in | pillow and the standard library. **No torch, no transformers, no weights.** | torch, transformers, and encoder weights on first use |
| entry point | `jevimage.connect(url)` | `jevimage.load()` |
| where the model runs | on that server | in your process |
| where your images go | up to that server | nowhere |

The base install is an HTTP client. Asking, embedding, managing heads **and training**
all work through it, because the encoder that fits the head lives on the server.
`jevimage encoders` works with no torch on the machine, and `jevimage ask/train/heads/rm --url ...`
(or `$JEVIMAGE_URL` with `$JEVIMAGE_API_KEY`) drive a server from the shell.

In that install `jevimage.load()` does not fail obscurely; it names both fixes:

```console
$ python -c "import jevimage; jevimage.load()"
ImportError: jevimage.load runs the encoder in this process, which needs torch and
transformers: pip install 'jevimage[local]'
To use a server instead - no torch required - call jevimage.connect(url).
(underlying import error: No module named 'torch')
```

## Install

Python 3.10+. Four installs, each a superset of the base:

```bash
pip install jevimage                # API only: pillow + stdlib. connect(), and the CLI's --url mode.
pip install 'jevimage[local]'       # + torch, transformers. load() runs the encoder here.
pip install 'jevimage[serve]'       # + fastapi, uvicorn. `jevimage serve` hosts it for others.
pip install 'jevimage[openclip]'    # + open_clip, for the dfn5b-h-14-384 entry (Apple's licence - see below).
```

`[serve]` and `[openclip]` both include `[local]`, because a server runs the encoder and
open_clip is an encoder. `jevimage doctor` reports which of these this machine has and
prints the command for anything missing.

Then point it at a model. Either someone else's server, signed in once so every shell
agrees:

```bash
jevimage login                    # key from console.jevimage.org
```

or your own, which needs no key and is found automatically:

```bash
jevimage serve                    # 127.0.0.1:8731; ask/train/heads probe for it
```

Having a key is what chooses the hosted API, whether `login` saved it or the shell
exported `$JEVIMAGE_API_KEY` - a key is only good against the service that issued it.
`--local` overrides it and means this machine: the server running here, or the weights
in this process. Typing `jevimage` with no arguments says which of those you are on, and
what is missing if it is neither.

Sizes, offline and air-gapped use, and what the first `load()` downloads:
[docs/install.md](https://docs.jevimage.org/docs/install/).

## Quickstart

Three questions, one encode. This runs on the API-only install:

```python
import jevimage

jev = jevimage.connect("http://127.0.0.1:8123")     # or jevimage.load(), same methods

a = jev.ask("ds/red/0.png", {           # or any image of your own
    "colour": {"type": "choice",
               "criteria": {"red": "a red square", "blue": "a blue square"}},
    "bright": {"type": "score",
               "criteria": ["a dark square", "a mid-tone square", "a bright square"]},
    "plain":  {"type": "noul",
               "criteria": {"true": "one flat colour", "false": "a detailed photo"}},
})

print(a["colour"])
print(a["colour"].choice, a["colour"].probabilities, a["colour"].confidence)
print(a["bright"], a["plain"])
```

```text
<choice 'red' p=1.000>
red {'red': 1.0, 'blue': 0.0} 0.999999
<score 0.02/2> <noul 0.038>
```

> That transcript is real, against a red square: `ds/red/0.png`, one of the twelve
> `python examples/make_fixtures.py` writes. The server was hosting a deterministic
> stand-in encoder, so the numbers are reproducible and the example costs nothing to
> re-run; the code, the shapes and the reprs are exactly what you get. On a real encoder
> the numbers move but the saturation does not necessarily: the same `colour` question on
> `siglip2-base-224` measured `{'red': 0.997553, 'blue': 0.002447}`, and `bright` came
> out `<score 1.52/2>` where the stand-in said `0.02`.

An `Answer` is a `dict` subclass with attribute access: `a["colour"].choice` and
`a["colour"]["choice"]` are the same thing, printing one gives the short repr above, and
`json.dumps(a)` gives the full shape with no conversion step. Swapping `connect(url)` for
`load()` changes nothing else in that file.

The four question types are `choice` (one of 2–255 named options), `score` (a position
on an ordered rubric), `noul` ("no-or-yes level": one yes/no probability) and `head` (a
classifier you trained). Up to 64 of them ride one encode. See
[docs/questions.md](https://docs.jevimage.org/docs/questions/).

Training, in one line and with the honest number attached:

```console
$ jevimage train colour ds/ --url http://127.0.0.1:8123
'colour': 12 examples, 2 classes, encoder toy

class   n  zero-shot  trained   delta
blue    6      50.0%    50.0%    0.0%
red     6     100.0%    50.0%  -50.0%
ALL    12      75.0%    50.0%  -25.0%

trained is k-fold held-out; zero-shot needs no holdout so it is scored on every example.
Training did not beat the prompts here - try more examples per class, or a bigger encoder.
```

The zero-shot column is what the prompts alone scored on the same images, because
training does not always help, and the only useful answer is the one for your images,
per class.

## What it will not do

**It answers what is *present*, not how things are arranged.** Measured: a man wearing a
hat scores 0.993 for "the man is wearing a hat", a bare-headed man with no hat in frame
scores 0.092, and both are right. A man *holding* a hat above his head also scores 1.000,
because the hat is in the picture and the pooled vector is not aligned to where it is.
Ask whether something is there and it is reliable. Ask whether it is on, in, left of or
held by something else and it will confidently answer the first question instead.

That information is not lost, only unreachable from words. On 400 controlled images of two
objects with the sides swapped, asking in words scores 50.00% (chance) while a linear head
fitted on the same embeddings scores 80.83% on images it had not seen. So **arrangement is
a `head` question, not a wording question** - and the head reports the accuracy it reached
on held-out examples, so the number is yours rather than ours.

**Do not phrase a `noul` pair as a negation.** `"wearing a pink dress"` against `"not
wearing a pink dress"` is one caption with a word inserted, not two alternatives, and a
dual encoder has no representation of "not": measured on a real photo those two are 0.927
similar, where two genuine alternatives sit at 0.83-0.85. The answer carries a `warning`
when it detects this, rather than handing back a confident-looking number. Name the
alternative - pink against white - and the same model answers correctly.

It also will not read a document, count reliably, or reason about what happened outside
the frame.

## Documentation

| Page | Read it when |
|---|---|
| [Overview](https://docs.jevimage.org/docs/index/) | You want the shape of the whole thing, including what it deliberately does not do. |
| [Install](https://docs.jevimage.org/docs/install/) | Choosing between the four installs, or working offline. |
| [Quickstart](https://docs.jevimage.org/docs/quickstart/) | The same walkthrough twice, once against a server and once in-process. |
| [Question types](https://docs.jevimage.org/docs/questions/) | Writing questions: `choice`, `score`, `noul`, `head`, and the two forms of `noul`. |
| [Training a head](https://docs.jevimage.org/docs/training/) | Prompts are not good enough and you have labelled images. Also: how to tell whether training helped. |
| [Choosing an encoder](https://docs.jevimage.org/docs/encoders/) | Picking from the registry, joining two with `+`, or plugging in your own model. |
| [Serving](https://docs.jevimage.org/docs/serving/) | Running `jevimage serve` for other people: routes, API keys, limits, deployment. |
| [CLI reference](https://docs.jevimage.org/docs/reference-cli/) | `jevimage ask`, `train`, `heads`, `rm`, `encoders`, `serve` - local or `--url`. |
| [Python reference](https://docs.jevimage.org/docs/reference-python/) | The exact public surface: every function, method and return shape. |
| [HTTP reference](https://docs.jevimage.org/docs/reference-http/) | The routes, if you are writing your own client. |
| [Cookbook](https://docs.jevimage.org/docs/cookbook/) | Complete scripts: screening an upload, caching an embedding, routing a directory. |
| [Troubleshooting](https://docs.jevimage.org/docs/troubleshooting/) | The error's sentence is clear but the reason it fired is not. |

The pages are plain Markdown with relative links, so they read on GitHub as they are.
[`mkdocs.yml`](https://github.com/Per0x1de-1337/jevimage/blob/main/mkdocs.yml) is there if you want a searchable site instead:
`pip install 'jevimage[docs]' && mkdocs serve`.

Runnable examples live in [`examples/`](https://github.com/Per0x1de-1337/jevimage/blob/main/examples/). The test suite is
`pip install 'jevimage[dev]' && python -m pytest tests/ -q`.

## Licence

MIT. See [LICENSE](https://github.com/Per0x1de-1337/jevimage/blob/main/LICENSE).

The model weights are not MIT and are not redistributed here - each one downloads from
its own source under its own terms. In particular, `dfn5b-h-14-384` pulls Apple's DFN5B
checkpoint, which carries **Apple's own licence**. Read it before you ship or redistribute
anything built on that encoder. It sits behind the optional `[openclip]` extra for exactly
that reason: the choice is yours to make, so it is never installed on your behalf.
