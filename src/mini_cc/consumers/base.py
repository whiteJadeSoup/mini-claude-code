"""QueuedConsumer — fire-and-forget base for engine consumers.

on_message() puts the message in an asyncio.Queue and returns immediately,
so _dispatch() is never blocked by slow consumer logic. Each subclass runs
its own _drain() task that processes messages independently.

Lifecycle (managed by QueryEngine):
    boot()     → consumer.start()   — creates the drain task
    shutdown() → consumer.stop()    — drains the queue, then cancels the task
"""
from __future__ import annotations

import asyncio
import sys

from mini_cc.engine.messages import Message


class QueuedConsumer:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Message] = asyncio.Queue()
        self._drain_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        await self._queue.join()
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

    async def on_message(self, msg: Message) -> None:
        if self._drain_task is None:
            await self.start()
        self._queue.put_nowait(msg)

    async def _drain(self) -> None:
        while True:
            msg = await self._queue.get()
            try:
                await self._handle(msg)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[{type(self).__name__}: drain error: {e}]",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                self._queue.task_done()

    async def _handle(self, msg: Message) -> None:
        raise NotImplementedError
