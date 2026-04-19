"""Unit tests for engine.subscription — Subscription + SubStats + policies."""
import asyncio

import pytest

from mini_cc.engine.messages import UserMessage
from mini_cc.engine.predicates import accept_all
from mini_cc.engine.subscription import Subscription
from mini_cc.engine.transforms import identity


class _Collecting:
    def __init__(self):
        self.messages = []

    async def on_message(self, msg):
        self.messages.append(msg)


class _Raising:
    def __init__(self):
        self.count = 0

    async def on_message(self, msg):
        self.count += 1
        raise RuntimeError("consumer boom")


def _sub(policy="sync", **kw):
    return Subscription(
        consumer=kw.pop("consumer", _Collecting()),
        name=kw.pop("name", "test"),
        filter=kw.pop("filter", accept_all),
        transform=kw.pop("transform", identity),
        policy=policy,
        **kw,
    )


def _msg(text="hi"):
    return UserMessage(content=text)


# ---------------------------------------------------------------------------
# sync policy
# ---------------------------------------------------------------------------


class TestSyncPolicy:
    @pytest.mark.asyncio
    async def test_delivers_inline(self):
        c = _Collecting()
        sub = _sub(policy="sync", consumer=c)
        await sub.deliver(_msg("one"))
        assert len(c.messages) == 1
        assert sub.stats.handled == 1

    @pytest.mark.asyncio
    async def test_consumer_exception_recorded(self, capsys):
        sub = _sub(policy="sync", consumer=_Raising())
        await sub.deliver(_msg())
        assert sub.stats.errors == 1
        assert "consumer boom" in (sub.stats.last_error or "")
        err = capsys.readouterr().err
        assert "consumer boom" in err

    @pytest.mark.asyncio
    async def test_start_stop_noop(self):
        sub = _sub(policy="sync")
        await sub.start()
        await sub.stop()
        assert sub._queue is None
        assert sub._worker is None


# ---------------------------------------------------------------------------
# async policy
# ---------------------------------------------------------------------------


class TestAsyncPolicy:
    @pytest.mark.asyncio
    async def test_lazy_start_on_first_deliver(self):
        c = _Collecting()
        sub = _sub(policy="async", consumer=c)
        await sub.deliver(_msg("one"))
        assert sub._started is True
        await sub.stop()
        assert c.messages and c.messages[0].content == "one"

    @pytest.mark.asyncio
    async def test_drain_preserves_order(self):
        c = _Collecting()
        sub = _sub(policy="async", consumer=c)
        for i in range(5):
            await sub.deliver(_msg(str(i)))
        await sub.stop()
        assert [m.content for m in c.messages] == ["0", "1", "2", "3", "4"]
        assert sub.stats.handled == 5

    @pytest.mark.asyncio
    async def test_worker_survives_consumer_exception(self, capsys):
        r = _Raising()
        sub = _sub(policy="async", consumer=r)
        await sub.deliver(_msg())
        await sub.deliver(_msg())
        await sub.stop()
        assert r.count == 2
        assert sub.stats.errors == 2

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        sub = _sub(policy="async")
        await sub.deliver(_msg())
        await sub.stop()
        await sub.stop()

    @pytest.mark.asyncio
    async def test_stop_when_worker_dead_does_not_hang(self):
        sub = _sub(policy="async")
        await sub.start()

        async def die():
            raise RuntimeError("worker died")

        sub._worker.cancel()
        try:
            await sub._worker
        except BaseException:
            pass
        sub._worker = asyncio.create_task(die())
        await asyncio.sleep(0)  # let it run
        await asyncio.wait_for(sub.stop(), timeout=1.0)


# ---------------------------------------------------------------------------
# drop_oldest policy
# ---------------------------------------------------------------------------


class TestDropOldestPolicy:
    @pytest.mark.asyncio
    async def test_drops_when_full(self):
        gate = asyncio.Event()
        seen = []

        class Slow:
            async def on_message(self, msg):
                if not gate.is_set():
                    await gate.wait()
                seen.append(msg.content)

        sub = _sub(policy="drop_oldest", consumer=Slow(), drop_oldest_maxsize=2)
        await sub.deliver(_msg("a"))  # gets picked up by worker, blocks
        await asyncio.sleep(0)
        await sub.deliver(_msg("b"))  # enqueued
        await sub.deliver(_msg("c"))  # enqueued, queue is now full
        await sub.deliver(_msg("d"))  # should evict 'b'
        gate.set()
        await sub.stop()
        assert sub.stats.dropped == 1
        assert "d" in seen
        assert "b" not in seen


# ---------------------------------------------------------------------------
# transform + filter error handling
# ---------------------------------------------------------------------------


class TestTransformErrors:
    @pytest.mark.asyncio
    async def test_transform_raises_blocks_delivery(self, capsys):
        c = _Collecting()

        def bad(m):
            raise ValueError("bad transform")

        sub = _sub(policy="sync", consumer=c, transform=bad)
        await sub.deliver(_msg())
        assert c.messages == []
        assert sub.stats.errors == 1
        assert "bad transform" in (sub.stats.last_error or "")

    @pytest.mark.asyncio
    async def test_transform_applied_before_consumer(self):
        c = _Collecting()

        def shout(m):
            return UserMessage(content=m.content.upper())

        sub = _sub(policy="sync", consumer=c, transform=shout)
        await sub.deliver(_msg("quiet"))
        assert c.messages[0].content == "QUIET"


# ---------------------------------------------------------------------------
# stats bookkeeping
# ---------------------------------------------------------------------------


class TestStats:
    @pytest.mark.asyncio
    async def test_handled_counts_up_sync(self):
        c = _Collecting()
        sub = _sub(policy="sync", consumer=c)
        for _ in range(3):
            await sub.deliver(_msg())
        assert sub.stats.handled == 3
        assert sub.stats.errors == 0

    @pytest.mark.asyncio
    async def test_handled_counts_up_async(self):
        c = _Collecting()
        sub = _sub(policy="async", consumer=c)
        for _ in range(3):
            await sub.deliver(_msg())
        await sub.stop()
        assert sub.stats.handled == 3

    @pytest.mark.asyncio
    async def test_last_error_truncated(self):
        class Boom:
            async def on_message(self, m):
                raise RuntimeError("x" * 500)

        sub = _sub(policy="sync", consumer=Boom())
        await sub.deliver(_msg())
        assert len(sub.stats.last_error) <= 200
