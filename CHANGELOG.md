# Changelog

## 0.1.20

**`JEV_*` environment variables are now `JEVIMAGE_*`.** `JEVIMAGE_URL`,
`JEVIMAGE_API_KEY`, `JEVIMAGE_TIMEOUT`, `JEVIMAGE_ENCODER`, `JEVIMAGE_DEVICE`,
`JEVIMAGE_THREADS`, `JEVIMAGE_BATCH`, `JEVIMAGE_HOME`, `JEVIMAGE_HEADS_DIR`,
`JEVIMAGE_API_KEYS`.

**The old names still work.** They are read whenever the new name is unset, so nothing
that already exports `JEV_*` breaks, and `jevimage doctor` does not report a working
`JEV_` name as unread. When both are set, the new name wins.

**The CLI now says when it is about to run the weights locally.** Setting
`JEVIMAGE_URL=...` on its own line creates a shell variable, which a child process never
sees, so a caller who believed they had pointed at the hosted API silently got a 400MB
download and an answer from their own machine instead. Before loading anything, with no
`--encoder` named:

    no server selected, so the weights will run in this process (first use downloads them).
       for the hosted API:  export JEVIMAGE_URL=https://api.jevimage.org
                            export JEVIMAGE_API_KEY=jevimage_...
       `export` matters: a bare VAR=value is not passed to this command.

**`jevimage heads` no longer crashes without a server.** It bound the client only on the
remote path but used it unconditionally in the footer, so listing local heads died with
`UnboundLocalError` after building the table it was about to print.

**A `noul` contrast pair now carries `warning` when the pair cannot be answered.** A
negation pair - `"wearing a pink dress"` against `"not wearing a pink dress"` - is one
caption with a word inserted, not two alternatives, and dual encoders do not represent
negation: measured on a real photo those two are 0.927 similar, where two genuine
alternatives sit at 0.83-0.85. The warning previously existed only on a readout that
such a pair never reaches, which is precisely where it was not needed.

### The patch-grid readout is present but off

`choice` and `noul` can read the encoder's patch grid instead of the pooled vector when
the options are the same words in a different order. It is **disabled by default** and
enabled with `JEVIMAGE_GRID=1`.

It wins on the benchmark protocol that scores two images against two captions. It does
not win on the task this library performs - one image, two captions - measured afterwards
on SugarCrepe swap, n=911, against a whole-caption baseline:

| readout | accuracy |
|---|---|
| pooled | **70.25** |
| grid routing | 68.50 |
| the benchmark's own configuration | 64.22 |

Earlier figures for this readout compared against a baseline that scored a caption as the
mean of its segment embeddings, roughly 18 points weaker than encoding the caption, which
is where the reported gain came from. The mechanism is real and the code is here; the
default is pooled until it beats pooled on one image.

## 0.1.19

**`jevimage heads` says whose heads it is showing.** A key scopes what you can see, so
two keys list two different sets - correctly. Without saying which account, a head
trained in the browser and missing from the CLI looks like a sync failure rather than a
second account:

```console
$ jevimage heads
head                encoder  classes    n  held out  created
street_objects  jev-image-1        4  468     95.9%        -

https://api.jevimage.org   (account Hemanshu-codeant)
```

`jevimage doctor` names it on the server line too. A server that does not report an
account prints no bracket rather than an empty one, and an account nobody has signed in
with shows its id, because an unreadable answer beats a missing one.

This needs the server to publish it: `/v1/models` gained `account`, `{id, login}`, where
`login` is filled in when that account has signed in to the console at least once.

## 0.1.18

**`jevimage --help` no longer recites the environment.** The table of eleven variables
made the help twice as long as the commands it documents, and most people will never set
any of them. `--help` answers "what can this do"; configuration moved to where "how is
this set up" is already answered:

```console
$ jevimage doctor
...
  yes $JEV_URL               http://127.0.0.1:8731
  yes $JEV_API_KEY           set
  yes $JEV_DEVICE            cpu
```

Only what is actually set, and a key reads `set` rather than its value - configuration
output that echoes a secret is one screenshot away from leaking it. With nothing set it
says so and names the two ways to fix that.

**`$JEV_DEFAULT_URL` is gone.** It existed for a day so that someone hosting the service
themselves could move the endpoint a key points at, which `$JEV_URL` already does and
does more directly. A variable whose whole job is covered by another one is a variable
to delete.

## 0.1.17

**The "nothing reads it" note is one line again.** Telling you `$JEV_BASE` does nothing
by listing all nine variables that do something turned a note about one setting into a
wall, and left you to find your own name in it.

```console
$ export JEV_BASE=https://localhost:8732
$ jevimage heads
jevimage: $JEV_BASE is set, but nothing reads it. `jevimage --help` lists the ones that are.
```

A genuine typo of a real name is suggested; a name that merely looks similar is not:

```console
$ JEV_TIMOUT=30 jevimage heads
jevimage: $JEV_TIMOUT is set, but nothing reads it. Did you mean $JEV_TIMEOUT?
```

difflib answers `$JEV_API_KEY` for `$JEV_BASE` at its default threshold, and
`$JEV_HOME` for `$JEV_TOKEN`. A confident wrong guess costs more than no guess, so the
cutoff is set where only a real misspelling of an existing name gets through.

`jevimage --help` now ends with the environment table, so the message points somewhere
true rather than at a list it used to recite itself. `$JEV_BATCH` and `$JEV_DEFAULT_URL`
were missing from the set the CLI recognised, so both were being reported as unread
while being read.

## 0.1.16

**`jevimage login` - sign in once instead of exporting a key in every shell.**

The environment was the wrong place to keep this. A terminal that had not exported
`$JEV_API_KEY` was indistinguishable from someone with no account, so the same command
in two windows reached two different servers and listed two different sets of heads:
the console showing what you had trained on the service, `jevimage heads` showing a
local server's. A browser does not ask you to re-authenticate per tab.

```console
$ jevimage login
API key (not echoed):
signed in to https://api.jevimage.org
  model jev-image-1, 2 head(s) you can use
  saved to /home/you/.jev/config.json (0600). `jevimage logout` removes it.
```

The key is checked against the server before anything is written, so a typo fails there
rather than on the next command. The file is created `0600` **before** the key goes into
it, because a secret must not exist even for an instant at whatever the umask happens to
be. `--key` takes it from the command line and stdin is read when piped, for CI.

`$JEV_API_KEY` and `$JEV_URL` still win when set, so one shell can point elsewhere
without signing out. A half-written config reads as "not signed in" rather than
crashing every command. `jevimage logout` forgets it, and `jevimage` on its own says
which server you are on and why.

## 0.1.15

**A key now selects the hosted API.** `$JEV_API_KEY` with no `--url` sends `ask`,
`train` and `heads` to `https://api.jevimage.org`, which is what someone holding a key
meant: a key is only good against the service that issued it. Before this, a
`jevimage serve` running on the same machine won, so `jevimage heads` listed local heads
while the key's own heads sat unseen on the service.

It is a key that selects it, not a hardcoded default. Without one that default is a 401,
and this package is self-hostable - redirecting a working local install to an endpoint
it cannot authenticate to would break it to make a sale. `$JEV_DEFAULT_URL` moves the
endpoint if you host the service yourself.

The order, first line wins:

| | |
|---|---|
| `--local` | this machine: a `jevimage serve` running here, or this process |
| `--url` / `$JEV_URL` | that server |
| `--encoder` / `$JEV_ENCODER` | this process - you named an encoder, so you meant one |
| `$JEV_API_KEY` set | the hosted API |
| a `jevimage serve` on `127.0.0.1:8731-8740` | that server |
| nothing above | this process |

**`--local` now means the machine, not the process.** Someone with a server running here
who asks for local means that server, not twelve seconds of reloading the same weights
beside it. `--encoder NAME` is how you insist on this process, and it reads as what it
does.

## 0.1.14

**`jevimage` with no arguments is no longer an error.** argparse answered
`error: the following arguments are required: command` and exited 2, which reads as a
failure to someone who has just installed it and typed its name to see what it does.
Nothing went wrong. It now prints what the thing is, the four commands worth knowing,
and - the part that was missing everywhere - where it is currently pointed:

```console
$ jevimage
jevimage 0.1.14 - typed questions about images.

  jevimage ask photo.jpg --choice indoor="an indoor scene" outdoor="the outdoors"
  jevimage train mine dataset/      one sub-folder per label
  jevimage heads                    the trained heads you can ask for
  jevimage doctor                   what this machine can do, and what is missing

Not pointed anywhere yet. Either:
  export JEV_URL=https://api.jevimage.org   # the hosted API
  export JEV_API_KEY=...                    # a key from console.jevimage.org
or run your own, which needs no key:
  pip install 'jevimage[serve]' && jevimage serve
```

That middle section changes: it names the server `$JEV_URL` points at, warns when a
hosted one has no `$JEV_API_KEY` to go with it, or names the local `jevimage serve` it
found. Exit status is 0. A subcommand that does not exist is still an error and still
exits 2.

Docs: the quickstart and the README now have a "point it at something" step. The two
ways to make the CLI work were documented, but scattered across pages nobody reads
before their first command fails.

## 0.1.13

**`jevimage ask --head NAME` no longer fails with `jevimage: accuracy`.** A head answer
carries the head's held-out accuracy, and jevimage's own server calls that field
`accuracy` while a hosted server may publish the same number as `cv_accuracy`. The CLI
read one of them straight out of the dict, so against the other it raised a `KeyError`
whose entire text was the key name - a working answer, discarded over a spelling.

It reads either now, and treats both as optional: a head too thin for k-fold has no
accuracy to report, which is a fact about the head rather than a failure.

## 0.1.12

**`jevimage train` leads with the head, not with the statistics.** It printed a
five-column table of trained-versus-zero-shot accuracy before anything else, which
buried the one line most people ran the command for:

```console
$ jevimage train vehicles ds/
trained 'vehicles': 431 examples, 4 classes, 91.6% accurate
  classes: bicycle, bus, car, motorcycle
  encoder: siglip2-base-224

Use it:
  jevimage ask IMAGE --head vehicles
```

The comparison is behind `--compare` now, and `--json` still carries all of it. It is
not gone, because it is the evidence for whether training was worth doing at all - on
classes the encoder already knows, the honest answer is often no, and that table is
where it shows. But it is an answer to a question you have to ask.

`jevimage train --json` is new: the head record and the comparison, for a script.

## 0.1.11

**Training a real folder no longer runs out of GPU memory.** `embed()` put the whole
list through the encoder in one forward pass, and a training folder is allowed 4000
images - so `train()` decoded 4000 pictures into memory and asked the card for all of
them at once. On a 4 GB GPU that is an `OutOfMemoryError` out of `F.linear`, and through
`jevimage serve` it reached the caller as a bare HTTP 500. Training 1071 images failed
this way every time.

Images now go through in chunks of 32, and the chunk **halves on an out-of-memory**
until it fits - 16, 8, 4 on a starved card - then carries on. The size is not tuned
because the right one depends on the card, the encoder and whatever else holds memory
at that moment, none of which can be known beforehand; starting high and backing off
costs one failed pass and then works on a 3050 or an A100. `$JEV_BATCH` sets it by hand.

This also bounds the decode: only a chunk's worth of images is held as PIL bitmaps at a
time, rather than the entire folder.

## 0.1.10

**`jevimage ask` takes several images, and they share one connection.**

Against a server a round trip away, the handshake is the cost, not the question. To
api.jevimage.org from ~200ms away, one `ask` breaks down as:

| | |
|---|---|
| DNS | 46 ms |
| TCP connect | 200 ms (1 RTT) |
| TLS handshake | 265 ms (1 RTT) |
| upload + response | ~590 ms (the body needs 2 slow-start rounds) |
| **the server itself** | **9.8 ms** |

So a command per file pays half a second of handshake per file. Eight files measured:

```
  8 commands, one per file    9.10 s      (1138 ms each)
  1 command, 8 arguments      2.86 s      ( 358 ms each)
```

`--json` keys the object by path when there is more than one image, and keeps the bare
`{question id: answer}` shape for one, so existing scripts are unaffected. From Python,
a `connect()` object kept alive has always done this: 225 ms median per `ask()` on a
warm connection.

Two things were measured and **not** shipped. Cutting the payload below one congestion
window would save a round trip, but needs the JPEG down to about 10 KB, and the
probabilities move by up to 0.07 - a silent accuracy change is not worth 200 ms. And
there is nothing left in the server: 9.8 ms of a 1110 ms request.

## 0.1.9

**The SDK works against the hosted API at api.jevimage.org, not only against your own
`jevimage serve`.** Three places assumed the self-hosted shape:

- **`ask()` read `answers` from the top level only.** A self-hosted server answers about
  one image and puts them there; the hosted API answers about a list and puts each
  image's under `results[i]`. Same answers, two envelopes - and reading only one failed
  with a `KeyError` whose entire text was `answers`. Either is read now, and a reply
  with neither says so, with the keys it did get.
- **`health()` falls back to `/v1/models`.** The hosted API keeps `/v1/health` for
  liveness alone and publishes the capability document behind `/v1/models`. Same
  question, different door; it is answered in the shape this method promises, so
  `jevimage doctor --url https://api.jevimage.org` now reports the model, its
  primitives and its heads instead of nothing.
- **`heads` no longer 404s against a hosted server.** Its heads are read-only, so there
  is no route that could create one - but they are published in the capability document,
  and the list is the same list. The table also reads its columns with `.get`: a hosted
  head carries no encoder name or created timestamp, and a listing should not raise
  `KeyError` on a column when the heads are plainly there.

```console
$ export JEV_URL=https://api.jevimage.org
$ export JEV_API_KEY=<key>
$ jevimage ask photo.jpg --choice indoor="an indoor scene" outdoor="an outdoor scene"
photo.jpg   server https://api.jevimage.org

choice  [choice]  outdoor   confidence 0.480
    outdoor   0.740  ################
    indoor    0.260  ######
```

## 0.1.8

Three things that were quietly misleading.

- **`--time` blamed the model load on "startup".** Loading the weights is its own line
  now, so `--local` reads honestly - 10638 ms of `load` and 41 ms of `startup`, not
  13402 ms of "python, and importing pillow". The breakdown also prints *after* the
  answer rather than before it, with stdout flushed first so a pipe does not reorder
  them.

  ```
  load        10638 ms   encoder siglip2-base-224, loaded into this process
  answer        161 ms   encoder siglip2-base-224
  startup        41 ms   python, argparse, and importing pillow and http.client
                         ----
  total       10840 ms
  ```

- **A `JEV_` variable nothing reads now says so.** `export JEV_BASE=...` was not an
  error anywhere: it was simply never looked at, and the command carried on with the
  default as though nothing had been asked for. One line on stderr names the variables
  that do exist.

- **`https://` at a `jevimage serve` names the fix.** It does not terminate TLS, so
  OpenSSL answered `WRONG_VERSION_NUMBER`, which describes the bytes rather than the
  mistake. It now adds: that server is speaking plain HTTP, use `http://`, or put a
  proxy in front of it.

## 0.1.7

**`jevimage ask --time` says where the time went.** "Why is this 250ms" has two common
answers and they call for opposite fixes, so the command now separates them:

```console
$ jevimage ask a.png --time --choice indoor="an indoor scene" outdoor="an outdoor scene"
  image        121 ms   2560x1440 RGBA png -> 19 KB
  answer       124 ms   server http://127.0.0.1:8731
  startup       45 ms   python, and importing pillow and http.client
                        ----
  total        290 ms
```

A large `image` is a large image: a 2560x1440 RGBA PNG is 14 MB once inflated, and zlib
is serial, so fifteen cores watch one work. The same picture as a JPEG measured 0.17s
against the PNG's 0.26s. A large `answer` is usually a server that fell back to the CPU,
where the forward pass is about ten times the GPU's - `jevimage doctor` names what is
holding the card.

The breakdown goes to stderr, so `--time --json` still pipes. The image is encoded once
and handed to `ask()` as a data URL rather than being encoded twice to be measured, and
the dimensions come from the file header rather than a decode, which would have doubled
the number being reported.

## 0.1.6

- **`jevimage doctor` names what is holding the GPU.** "0.0 GB free of 3.7 GB" tells you
  the server will fall back to the CPU, but not what to close. It now lists the
  processes, largest first, and flags a *suspended* one: a Ctrl-Z'd server keeps its
  whole allocation for as long as it exists, and shows up as nothing at all in `top`.

  ```
  yes gpu       NVIDIA GeForce RTX 3050 Laptop GPU, 0.0 GB free of 3.7 GB
                -> too tight for a model; it will load on the CPU instead, which is
                   about ten times slower
                   pid 1779614    900 MiB  python3 jevimage serve  <- STOPPED, still holding it
                   pid 2122998    896 MiB  python3 jevimage serve
                   pid 1786123    896 MiB  python3 jevimage serve  <- STOPPED, still holding it
  ```

  It asks nvidia-smi, which is installed wherever CUDA is; where it is not, the line is
  simply absent.

Nothing else changed. Two things were measured and rejected: fused attention (`sdpa`) is
no faster than eager on a CPU, because the forward is matmul-bound at about half this
CPU's peak FLOPs; and int8 dynamic quantization moves the probabilities, which are the
product here, through a torch API that is already deprecated.

## 0.1.5

Latency. Every number below is a median of seven, one `jevimage ask` per process against
a server on the same machine, on an i7-12650H with an RTX 3050.

|  | 640x480 | 12MP jpg | 12MP png |
|---|---|---|---|
| 0.1.4, CPU server | 0.20s | 0.22s | 0.38s |
| **0.1.5, CPU server** | **0.16s** | **0.17s** | **0.31s** |
| **0.1.5, GPU server** | **0.08s** | **0.09s** | **0.25s** |

- **The CPU path picks the performance cores.** torch defaults to the physical core
  count, which is wrong on a hybrid chip: a forward pass waits on its slowest thread, so
  four E-cores set the pace for six P-cores as well. On an i7-12650H one 224px forward
  took 158ms at torch's default of 10 threads, 95ms at 6, and 905ms at all 16. Linux
  publishes the split under `/sys/devices/cpu_core`, so this is the kernel's own
  topology, not a guess - on a uniform CPU, or off Linux, nothing changes.
  `$JEV_THREADS` sets it by hand, `JEV_THREADS=0` restores torch's default, and an
  `$OMP_NUM_THREADS` already in the environment is left alone.
- **Big JPEGs decode at 1/8 scale instead of in full.** `thumbnail()` asks `draft()` to
  scale during the DCT, but only works on an image nothing has touched - and the client
  called `.convert("RGB")` first, so the full 4032px decode had already happened. A 12MP
  photo now costs 60ms less on the way out.
- **Two full-size pixel copies per image are gone.** `ImageOps.exif_transpose` returns
  `image.copy()` when the photo is already upright, and `Image.convert("RGB")` returns
  one when the mode already is - together about 30ms and 70MB on a 12MP image, to
  produce the image we started with. One `jevimage.images.upright()` now does both jobs
  and copies only when that changes something. Every image on the way in, local and
  remote, goes through it.
- `jevimage doctor` reports the thread choice.

If a question still takes longer than this, `jevimage doctor` will say why: the usual
answer is that something else has the GPU, so the server fell back to the CPU.

## 0.1.4

- **`--encoder` typos get the useful message again.** The CLI checks the name against
  the registry before importing torch - otherwise a typo on an API-only install answered
  "this needs torch", so you installed torch, ran the same command, and only then learned
  the name was wrong. That early check had its own shorter copy of the message, without
  the "register your own with `jevimage.register(name, factory)`" half, so `jevimage ask
  --encoder nope` and `jevimage serve --encoder nope` said different things about the
  same mistake. There is one sentence now, `jevimage.encoders.check()`, raised from both.
- Documentation: the CLI reference, install and troubleshooting pages still called the
  command `jev` in three places, one of them a Dockerfile `CMD` that would not run.
  Every documented version string, port and invocation has been re-checked against the
  real output - 85 documented commands, every flag and subcommand verified to exist, and
  every internal link and anchor resolved.

## 0.1.3

**`jevimage serve` no longer defaults to port 8000, and no longer dies when its port is
taken.** 8000 is the default of every other Python web server on the machine, so losing
that race is normal - and the way you found out was `[Errno 98] address already in use`
*after* twelve seconds of loading the model, with the process exiting. The default is
now 8731, which is unassigned, and if that is taken too `serve` walks up to 8740 and
says which port it settled on:

```
$ jevimage serve
port 8731 is in use; serving on 8732 instead
INFO:     Application startup complete.
```

The port is bound before uvicorn is handed the socket, rather than tested and then
bound, so nothing can take it in the gap. `--port` moves where the walk starts,
`--port 0` lets the kernel choose, and all ten busy gives one error naming the range.
`ask`, `train`, `heads` and `doctor` sweep the same range, so a server that walked is
still the one they use; a closed loopback port refuses immediately, so the sweep costs
about a millisecond when nothing is running.

If you run `jevimage serve` behind a reverse proxy, the walk is not what you want: run
`uvicorn jevimage.server:app --port N` instead, which binds that port or fails. See
[Serving](https://docs.jevimage.org/docs/serving/).

## 0.1.2

Runs on whatever machine you have, and tells you when it cannot.

- **A GPU is a preference, not a requirement.** A card that will not hold the model,
  because something else already has it or because it is a 4 GB laptop GPU, no longer
  stops the job: the encoder loads on the CPU and logs one warning, instead of raising
  `CUBLAS_STATUS_ALLOC_FAILED` from inside torch. A device you named yourself, with
  `device=` or `$JEV_DEVICE`, is still never swapped - a job that quietly runs 40x
  slower is worse than one that stops. `OpenCLIP` and `HFCLIP` share the one path.
- **New `jevimage doctor`.** Reports the Python version, pillow, whether a server
  answers, torch and transformers, the GPU and how much of it is free, whether the
  weights are cached, and free disk - naming the one command that fixes anything
  missing. An API-only install with no torch is a pass, not a failure; the exit status
  is 1 only when nothing on the machine could answer a question.
- **`jevimage ask` and `jevimage train` use a server you already have running.** Loading
  the encoder costs about 12 seconds and the question itself costs 20 milliseconds, so a
  CLI that loaded its own copy spent 99.8% of its life starting up - and on a small GPU
  it collided with the server it was ignoring. If `127.0.0.1:8000` answers `/v1/health`,
  the question goes there: 12.57 s to 0.09 s. `--local` forces the old behaviour.
- A CUDA failure during `jevimage ask` prints what to do about it rather than a 60-line
  traceback naming `cublasCreate`.
- The `--url` hint in the torch-missing message said `jev`, not `jevimage`, and the
  connection error said "cannot reach jev at". Both name the real command now.

## 0.1.1

Local SDK fixes - input that was accepted and then failed somewhere else, and three
things that were wrong rather than unclear:

- **`batch=True` no longer fails a whole batch for one bad image.** One caller's
  truncated upload answered every other caller who rode the same forward pass with an
  error about a file they never sent - on a server, up to 15 healthy requests at a time.
  A failing batch is now re-run one item at a time, so each caller gets its own result
  or its own error.
- **A head retrained by another process is picked up by `head()` and `ask()`,** not only
  by `list_heads()`. The freshness check ran on a cache MISS, so once a name was cached
  every later read served the pre-retrain weights and class names, with no error.
  Deleting a head propagates the same way.
- **`HeadStore` is thread-safe for reads again.** Two threads syncing at once both
  removed the same vanished head, and the loser raised `KeyError` out of `name in store`
  - the check that gates every `head` question.
- Two processes saving the same head name no longer collide: the temp file is per
  writer, so the loser is overwritten instead of failing with a `FileNotFoundError`
  about a `.pt.tmp` it never named. Retraining a broken head clears it from
  `heads.broken`.
- `load()` refuses to rename an encoder instance that is already cached under another
  registry name, instead of silently renaming it and invalidating every head trained
  under the first name. `register()` also drops cached ensembles built from that name.
- `Ensemble` normalises each member before concatenating, as `docs/encoders.md` always
  said it did. Without it a member whose output is not unit-norm outvoted the others,
  and the engine's L2 check could not see it.
- SigLIP entries truncate a caption to the tokeniser's 64 tokens instead of failing with
  a transformers error about `max_position_embeddings`.
- A `prompt` or `template` is validated by trial substitution, so `'{} and {}'`,
  `'a photo of {x}'` and `'{.__class__}'` are refused by name instead of escaping as a
  raw `IndexError`/`KeyError`, or captioning every option `"<class 'str'>"`. `train()`
  and `compare()` check `template=` before any work, so a bad one no longer raises
  *after* the head is on disk.
- `ask_embedding()` checks what it was handed: a numpy array, a list, a stray batch
  dimension or the wrong width are named, not left to fail inside the engine.
- `max_per_class` means one thing in all three `data` shapes: a positive int, or `None`.
  `0` used to mean "no limit", `-1` dropped different examples in each shape, and a
  folder was truncated twice - which could delete a whole class.
- `to_image()` treats `Path` exactly like `str`, truncates what it quotes back, reports a
  malformed `data:` URL as a `ValueError` naming the URL, and accepts `DATA:`.
  `read_folder` sorts by filename, not by full path, which is what the docs promised
  once sub-directories are involved.
- `temperature` accepts any real number (numpy scalars included) in `1e-6 .. 1e6`, and
  says so. Outside that range fp32 underflowed and the reported winner could flip. An
  encoder's `logit_scale` goes through the same guard, so a negative one is an error
  rather than a confident answer on the wrong option.
- The singular `instruction` is read again as a spelling of `instructions`, which is
  what `docs/questions.md` documents. (0.1.0 removed it as "undocumented"; the docs row
  says otherwise, so the docs win.)
- `jevimage[local]` requires `transformers>=4.56`: `metaclip2-huge` is `model_type
  metaclip_2`, which older versions do not recognise, so the old floor allowed an
  install that could not load a listed encoder.
- `dfn5b-h-14-384` names the extra it needs when `open_clip` is missing.

Client fixes, all in `jevimage.connect()`:

- `max_edge=` is now actually used. It was stored and never passed to the upload, so
  every image was downscaled to 384px whatever you asked for, and `None` did nothing.
- An oversized image is exif-transposed before it is re-encoded for upload. A portrait
  phone photo used to reach the server sideways, with its Orientation tag dropped by the
  re-encode, so remote answers disagreed with local ones.
- A `POST` (`ask`, `embed`, `train`) is no longer silently replayed when a kept
  connection dies. The failure can surface after the server has done the work, so a
  `train()` could run twice. `GET` and `DELETE` still reconnect once.
- `api_key` is validated in `connect()`: a trailing newline or a non-ASCII character
  used to blow up inside `http.client`, with the key itself in the traceback.
- `Remote.ask` gives `Jev.ask`'s wording for a bad temperature, `Remote.train` refuses
  an out-of-range `epochs` (the server's 1..5000) with a sentence instead of a pydantic
  field dump, a 2xx that is not JSON is a `JevError` naming the URL, and a redirect
  leads with where it pointed instead of the proxy's HTML page.

Docs:

- **`examples/make_fixtures.py` ships the images the docs read.** Every page's
  transcripts were run over `ds/`, `shots/` and `colours/` folders that were never in
  the repo, so no block on any page ran as written, and the stand-in encoders key off
  raw pixels, so no number could be reproduced either. The script writes all three, and
  every printed number on quickstart, questions, training, cookbook, encoders, the three
  references, index and README was re-run against them and re-pasted.
- Blocks that could not run as printed now run: `import json` in the quickstart's first
  block, `ToyCalibrated` in the questions prelude, `import base64` and `import tempfile`
  where the reference and encoders pages use them, a `uvicorn` line for the mounted-app
  transcript in serving, and `jevimage.load(Toy())` where troubleshooting said
  `jevimage.load("toy")`, which is not a registry name.
- `jevimage.encoders` (and any other submodule) resolves after a plain `import
  jevimage`, so `troubleshooting.md`'s ensemble-name example and the advice in the
  engine's own "wrap your encoder's output in `jevimage.encoders.unit()`" both work.
- Claims corrected against measurement: warm `load()` of `siglip2-base-224` is 8-10 s on
  the 16-thread laptop CPU these docs are measured on, not 4.8/5.0 s; 5000 epochs on
  4000 examples is about 70 s, not "seconds"; 16 concurrent single-image embeds coalesce
  into 2 forward passes on the toy encoder rather than staying 16; several keys may be
  configured so callers can migrate, but withdrawing one is a restart, not zero
  downtime; and N questions are only near-free once their captions are cached.

## 0.1.0

First release.

- `choice`, `score`, `noul` and `head` questions against one image encode.
- `jevimage.load()` for in-process use, `jevimage.connect(url)` for a server. Same
  methods, same return shapes.
- `jevimage` CLI: `ask`, `train`, `heads`, `rm`, `encoders`, `serve`. The first four take
  `--url`.
- Linear heads trained on a labelled folder, with a per-class comparison against the
  zero-shot prompts.
- Seven built-in encoders plus `'+'` ensembles, and `register()` for your own.

Two notes for anyone who tracked the pre-release source:

- `plan()` no longer accepts the singular `instruction` key as an alias for
  `instructions`. It was undocumented; the singular spelling now raises.
- `Encoder` has no `preprocess()` / `img_from_px()` pair. Nothing called it, and the
  cross-request batching it was meant to enable was never wired up. `Batcher` batches
  whole images through `img()`.
