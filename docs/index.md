---
title: "Overview"
description: "One image encode, many typed questions, calibrated probabilities instead of generated text."
---

jevimage answers typed questions about an image with probabilities instead of prose.
The image goes through a frozen dual encoder once and becomes a single unit-norm
vector. Every question after that (pick one of these options, place it on this rubric,
yes or no, run the head I trained) is a matmul against that same vector.

Nothing is generated. An answer is a distribution over names *you* chose, plus a
confidence rescaled so that one threshold works whether the question had two options or
fifty. So there is no sentence to parse and no retry loop when the model phrases it
differently. The same image and the same question give the same numbers, to the six
decimals they are printed to, every time. The second, third and tenth question about one
image cost a matmul each, not another forward pass, once their captions are in the
per-process cache. A caption the process has not seen costs a text-tower pass the first
time; see [the timings](encoders.md#timings) for what that is worth. When prompts are
not good enough you fit a linear head on your own labelled folder in seconds, and
jevimage reports per class whether that head beat the prompts. Sometimes it does not,
and it says so.

## Two ways to use it

Both are supported paths, not a main one and a fallback. They are the same API.

|  | Call a server | Run it in-process |
|---|---|---|
| install | `pip install jevimage` | `pip install 'jevimage[local]'` |
| what that pulls in | pillow and the standard library | torch, transformers, and encoder weights on first use |
| entry point | `jevimage.connect(url)` | `jevimage.load()` |
| where the model runs | on that server - yours or someone else's | in your process |
| where your images go | up to that server | nowhere |

The base install is an HTTP client. It has no torch, no transformers and no weights,
and everything works through it: asking, embedding, **and training**, since the
encoder that fits the head lives on the server. `jevimage encoders` works offline, and
`jevimage ask/train/heads/rm --url ...` (or `$JEVIMAGE_URL`, with `$JEVIMAGE_API_KEY`) drive a server
from the command line.

`jevimage.load()` in that install does not fail obscurely; it names both fixes:

```console
$ python -c "import jevimage; jevimage.load()"
ImportError: jevimage.load runs the encoder in this process, which needs torch and
transformers: pip install 'jevimage[local]'
To use a server instead - no torch required - call jevimage.connect(url).
(underlying import error: No module named 'torch')
```

Going the other way, `pip install 'jevimage[serve]'` adds fastapi and uvicorn so you can
be the server other people connect to: `jevimage serve`. See [Serving](serving.md).

## 60 seconds

Against a server, needing nothing but `pip install jevimage`:

```python
import jevimage

jev = jevimage.connect("http://127.0.0.1:8123")

answers = jev.ask("ds/red/0.png", {
    "colour": {"type": "choice",
               "criteria": {"red": "a red square", "blue": "a blue square"}},
    "busy":   {"type": "score", "criteria": ["empty", "half full", "packed"]},
    "indoor": {"type": "noul",
               "criteria": {"true": "an indoor scene", "false": "the outdoors"}},
})

print(answers["colour"])
print(answers["colour"].choice, answers["colour"].probabilities)
print(answers["colour"].confidence)
print(answers["busy"], answers["indoor"])
```

```text
<choice 'red' p=1.000>
red {'red': 1.0, 'blue': 0.0}
0.999999
<score 1.28/2> <noul 0.000>
```

Three questions, one encode. An `Answer` is a `dict` subclass with attribute access, so
`answers["colour"]` prints as that short repr, `a.choice` and `a["choice"]` are the same
thing, and `json.dumps(answers)` gives you the full shape with no conversion step.

To run the encoder in your own process instead, change the `connect` line to
`jevimage.load()` and the rest of the file stands. The two produce the same objects; the
numbers match when both sides run the same encoder, because what the probabilities depend
on is the encoder, not which side of the wire it runs on.

> The numbers above came from a deterministic stand-in encoder, so the transcript is
> reproducible and costs nothing to re-run: `python examples/make_fixtures.py` writes
> the `ds/` squares it asks about, and the server was the toy one from
> [Serving](serving.md#build_app-your-own-encoder) (a custom encoder has to be served
> from Python - `jevimage serve` only knows the built-in registry). The shapes, the repr and
> the code are real.
> How peaked a real distribution is depends on the question and the encoder's
> temperature: the same two-option question on `siglip2-base-224` measured
> `{'red': 0.997553, 'blue': 0.002447}` (also saturated), while a three-level `score`
> on the same image came out `{0: 0.195, 1: 0.094, 2: 0.711}`. Saturation is the
> encoder's doing, not the library's; see [Choosing an encoder](encoders.md).

## What it does not do

- **No generation.** There is no text output, no captioning, no VQA in free text. If you
  need a sentence, this is the wrong tool; if you need a number you can threshold, it is
  the right one.
- **No detection, no boxes, no segmentation.** Answers are about the whole image. "Is
  there a bicycle" is a question it can answer; "where is the bicycle" is not.
- **No OCR.** Text in the image is whatever the encoder made of it, which is not reading.
- **No comparison across images.** `ask()` answers about one image. Handed a list, a
  local `Jev` and a `connect()` client refuse with the same sentence: `TypeError: ask()
  takes one image, got a list of 2. Loop over them, or call embed() once and
  ask_embedding() per row.` Pass one image, and use `embed()` if you
  want the vectors for several. (`compare()` is unrelated: it compares a trained
  head against zero-shot prompts, not one image against another.)
- **No open-ended questions.** You supply the options. A `choice` with an option missing
  returns a confident answer among the options you gave, because the probabilities are
  over your list and nothing else.

## Where to go next

| Page | Read it when |
|---|---|
| [Install](install.md) | Choosing between `jevimage`, `[local]`, `[serve]` and `[openclip]`, and what each pulls in. |
| [Quickstart](quickstart.md) | You want the same walkthrough twice, once against a server and once in-process. |
| [Question types](questions.md) | Writing questions: `choice`, `score`, `noul`, `head`, what each answer carries, and the two forms of `noul`. |
| [Training a head](training.md) | Prompts are not good enough and you have labelled images. Also: how to tell whether training helped. |
| [Choosing an encoder](encoders.md) | Picking from the registry, joining two with `+`, or plugging in your own model. |
| [Serving](serving.md) | Running `jevimage serve` for other people: routes, API keys, limits. |
| [CLI reference](reference-cli.md) | Driving it from a shell - `jevimage ask`, `jevimage train`, `jevimage heads`, local or `--url`. |
| [Python reference](reference-python.md) | The exact public surface: every function, method and return shape. |
| [HTTP reference](reference-http.md) | The routes, if you are writing your own client against a server. |
| [Cookbook](cookbook.md) | Complete scripts: screening an upload, caching an embedding, routing a directory. |
| [Troubleshooting](troubleshooting.md) | The error's sentence is clear but the reason it fired is not. |

---

[Install →](install.md)
