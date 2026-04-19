"""Message transforms and the ``compose`` combinator.

A transform is any ``Callable[[Message], Message]``. Subscribers use these
to declare *how* a message should be adapted before delivery (e.g.
redact sensitive fields, summarize long tool output). This module ships
the plumbing — ``identity`` and ``compose`` — so real transforms can land
as data in a future commit without changing the engine.
"""
from __future__ import annotations

from functools import reduce
from typing import Callable

from mini_cc.engine.messages import Message

Transform = Callable[[Message], Message]


def identity(m: Message) -> Message:
    return m


def compose(*fs: Transform) -> Transform:
    """Left-to-right composition: ``compose(f, g, h)(m) == h(g(f(m)))``."""
    if not fs:
        return identity
    if len(fs) == 1:
        return fs[0]
    return lambda m: reduce(lambda x, f: f(x), fs, m)
