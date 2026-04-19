"""Subscription: unified owner of a consumer's queue, worker, and policy.

A ``Subscription`` pairs a ``Consumer`` (the "what to do" — only
``on_message``) with a ``filter``, a ``transform``, and a delivery
``policy``. The engine treats subscriptions as data and never inspects
consumer internals; adding a new consumer type no longer means touching
the engine.

Three policies:
    sync         — deliver inline; ``deliver`` returns only after
                   ``on_message`` returns. No queue, no worker.
    async        — enqueue and return immediately; a background worker
                   drains the queue. Unbounded queue (same capacity as
                   the old ``QueuedConsumer``).
    drop_oldest  — bounded queue; on full, evict the oldest item and
                   increment ``stats.dropped``.

Lifecycle is lazy: ``deliver`` auto-starts the worker on first use, so
consumers subscribed after ``boot()`` (e.g. in tests) still work. Explicit
``start()`` in ``QueryEngine.boot`` is a belt-and-suspenders guarantee
that the worker is ready before the first dispatch — not a correctness
requirement.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from mini_cc.engine.messages import Message
from mini_cc.engine.predicates import Predicate, accept_all
from mini_cc.engine.transforms import Transform, identity

Policy = Literal["sync", "async", "drop_oldest"]


@runtime_checkable
class Consumer(Protocol):
    async def on_message(self, msg: Message) -> None: ...


class SubStats(BaseModel):
    handled: int = 0
    errors: int = 0
    dropped: int = 0
    last_error: str | None = None


@dataclass
class Subscription:
    consumer: Consumer
    name: str
    filter: Predicate = accept_all
    transform: Transform = identity
    policy: Policy = "sync"
    drop_oldest_maxsize: int = 256
    stats: SubStats = field(default_factory=SubStats)

    _queue: asyncio.Queue[Message] | None = field(default=None, init=False, repr=False)
    _worker: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _stopped: bool = field(default=False, init=False, repr=False)

    async def start(self) -> None:
        if self._started or self.policy == "sync":
            self._started = True
            return
        maxsize = self.drop_oldest_maxsize if self.policy == "drop_oldest" else 0
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._worker = asyncio.create_task(self._drain())
        self._started = True

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.policy == "sync" or self._worker is None or self._queue is None:
            return

        if self._worker.done():
            exc = self._worker.exception()
            if exc is not None:
                self.record_error(f"worker died: {exc!r}")
            return

        await self._queue.join()
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass

    async def deliver(self, msg: Message) -> None:
        """Dispatch a filtered message. Precondition: ``filter(msg) is True``."""
        try:
            payload = self.transform(msg)
        except Exception as e:  # noqa: BLE001
            self.record_error(f"transform: {e!r}")
            return

        if self.policy == "sync":
            try:
                await self.consumer.on_message(payload)
                self.stats.handled += 1
            except Exception as e:  # noqa: BLE001
                self.record_error(f"on_message: {e!r}")
            return

        if not self._started and not self._stopped:
            await self.start()
        if self._stopped or self._queue is None:
            return

        if self.policy == "drop_oldest" and self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.stats.dropped += 1
            except asyncio.QueueEmpty:
                pass

        self._queue.put_nowait(payload)
        self.stats.handled += 1

    def record_error(self, detail: str) -> None:
        self.stats.errors += 1
        self.stats.last_error = detail[:200]
        print(f"[sub {self.name}: {detail}]", file=sys.stderr, flush=True)

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            msg = await self._queue.get()
            try:
                await self.consumer.on_message(msg)
            except Exception as e:  # noqa: BLE001
                self.record_error(f"on_message: {e!r}")
            finally:
                self._queue.task_done()
