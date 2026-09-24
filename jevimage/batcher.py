"""Dynamic batching for the image encoder.

One image through the encoder costs 9.7ms; eight together cost 20.6ms - 2.6ms each.
The GPU is idle between the small matmuls of a single image, so the marginal image in a
batch is nearly free. Measured on the A10G:

    batch  1   9.68 ms/img   103 img/s
    batch  8   2.57 ms/img   388 img/s   3.8x
    batch 16   2.05 ms/img   487 img/s   4.7x

Requests arriving within a few milliseconds of each other are therefore worth holding
briefly so they can ride one forward pass. The wait is bounded and small: under load the
batch fills before the timer, and when traffic is thin a request waits at most
`max_wait` and still beats the un-batched path on throughput while losing only that.

Thread-based on purpose - the server's handlers are sync and uvicorn runs them in its
threadpool, so this batches whatever that pool is holding without making anything async.
"""
from __future__ import annotations

import threading
import time

__all__ = ["Batcher"]


class Batcher:
    """Collect concurrent calls into one call of `fn(list) -> list`."""

    def __init__(self, fn, max_batch: int = 16, max_wait: float = 0.006,
                 name: str = "batch"):
        self.fn = fn
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.name = name
        self._q = []                       # [(item, slot)]
        self._ema = 1.0                    # recent batch size; drives the adaptive wait
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._worker = None
        self.stats = {"calls": 0, "batches": 0, "items": 0, "max_seen": 0}

    def _ensure_worker(self):
        # started lazily so importing this module never spawns a thread, and so a
        # forked uvicorn worker gets its own rather than inheriting a dead one
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name=f"batcher-{self.name}")
            self._worker.start()

    def submit(self, item):
        """Queue one item and block until its result is ready. Raises what fn raised."""
        slot = {"event": threading.Event(), "out": None, "err": None}
        self.stats["calls"] += 1
        with self._wake:
            self._q.append((item, slot))
            self._ensure_worker()
            self._wake.notify()
        slot["event"].wait()
        if slot["err"] is not None:
            raise slot["err"]
        return slot["out"]

    def _drain(self):
        """Wait for work, then let a batch accumulate for at most max_wait.

        The wait is ADAPTIVE. Holding a request for 6ms only pays if someone else
        arrives inside that window; on an idle service nobody does, and every request
        pays the delay for nothing - measured as p50 13ms -> 20ms at concurrency 1.
        So the recent batch size decides: while traffic is thin the batch goes straight
        through, and the wait switches on only once batches are actually forming.
        """
        with self._wake:
            while not self._q:
                self._wake.wait(0.5)
                if not self._q:
                    return []
            wait = self.max_wait if self._ema >= 1.5 else 0.0
            if wait:
                deadline = time.monotonic() + wait
                while len(self._q) < self.max_batch:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self._wake.wait(left)
            batch, self._q = self._q[:self.max_batch], self._q[self.max_batch:]
            # a queue that refilled while we worked is itself evidence of load, so count
            # the backlog too - otherwise the EMA can never climb out of "thin"
            self._ema = 0.7 * self._ema + 0.3 * (len(batch) + min(len(self._q), 4))
            return batch

    def _run(self):
        while True:
            batch = self._drain()
            if not batch:
                continue
            items = [b[0] for b in batch]
            try:
                out = self.fn(items)
                if len(out) != len(items):
                    raise RuntimeError(f"batched fn returned {len(out)} for {len(items)}")
                for (_, slot), o in zip(batch, out):
                    slot["out"] = o
            except BaseException as e:      # one bad batch must not kill the worker
                # Nor may it fail its neighbours. Batching is an internal throughput
                # trick, so one caller's undecodable image must not answer seven other
                # callers with an error about a file they never sent. Re-run the batch
                # one item at a time and route each result - or each failure - to the
                # caller it belongs to. Only on the error path, so it costs nothing
                # when nothing is wrong.
                for item, slot in batch:
                    if len(batch) == 1:
                        slot["err"] = e
                        continue
                    try:
                        out = self.fn([item])
                        if len(out) != 1:
                            raise RuntimeError(f"batched fn returned {len(out)} for 1")
                        slot["out"] = out[0]
                    except BaseException as e2:
                        slot["err"] = e2
            finally:
                self.stats["batches"] += 1
                self.stats["items"] += len(batch)
                self.stats["max_seen"] = max(self.stats["max_seen"], len(batch))
                for _, slot in batch:
                    slot["event"].set()


def demo() -> None:
    """python3 batcher.py - proves batching, ordering, errors and the wait bound."""
    seen = []

    def fn(xs):
        seen.append(len(xs))
        time.sleep(0.005)
        return [x * 2 for x in xs]

    b = Batcher(fn, max_batch=8, max_wait=0.02)

    # concurrent callers share a pass, and each gets ITS OWN answer back
    out = {}
    def call(i):
        out[i] = b.submit(i)
    ts = [threading.Thread(target=call, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out == {i: i * 2 for i in range(8)}, out
    assert max(seen) > 1, f"nothing batched: {seen}"

    # a lone caller on a quiet batcher pays no waiting penalty at all
    quiet = Batcher(lambda xs: xs, max_batch=8, max_wait=0.050)
    quiet.submit(1)                                   # prime the worker
    t0 = time.perf_counter()
    assert quiet.submit(21) == 21
    solo = time.perf_counter() - t0
    assert solo < 0.010, f"idle solo call waited {solo*1000:.0f}ms, should not wait"

    # an exception reaches every caller in the batch, and the worker survives
    def boom(xs):
        raise ValueError("upstream failed")
    b2 = Batcher(boom, max_batch=4, max_wait=0.01)
    errs = []
    def call2(i):
        try:
            b2.submit(i)
        except ValueError as e:
            errs.append(str(e))
    ts = [threading.Thread(target=call2, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(errs) == 4 and all("upstream failed" in e for e in errs), errs

    # ONE bad item must not fail the callers who rode the same pass. This is the whole
    # point: on a server, one truncated upload used to error every concurrent request.
    def picky(xs):
        if any(x == "bad" for x in xs):
            raise ValueError("cannot decode 'bad'")
        return [x * 2 for x in xs]
    b4 = Batcher(picky, max_batch=8, max_wait=0.02)
    got = {}
    def call4(i, x):
        try:
            got[i] = b4.submit(x)
        except ValueError as e:
            got[i] = f"ERR {e}"
    ts = [threading.Thread(target=call4, args=(i, i)) for i in range(7)]
    ts.append(threading.Thread(target=call4, args=(7, "bad")))
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert got[7] == "ERR cannot decode 'bad'", got[7]
    assert all(got[i] == i * 2 for i in range(7)), got

    # a wrong-length return is an error, not silent misrouting of answers
    b3 = Batcher(lambda xs: xs[:-1], max_batch=4, max_wait=0.01)   # always short by one
    try:
        b3.submit(1)
        raise AssertionError("short return was not caught")
    except RuntimeError:
        pass

    print(f"batcher selftest ok - batches {seen}, solo wait {solo*1000:.0f}ms")


if __name__ == "__main__":
    demo()
