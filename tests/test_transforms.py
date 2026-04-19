"""Unit tests for engine.transforms — identity + compose."""
from mini_cc.engine.messages import UserMessage
from mini_cc.engine.transforms import compose, identity


def _user(text="hi"):
    return UserMessage(content=text)


class TestIdentity:
    def test_returns_same_object(self):
        m = _user()
        assert identity(m) is m


class TestCompose:
    def test_empty_is_identity(self):
        m = _user()
        assert compose()(m) is m

    def test_single_returns_same_function(self):
        def shout(m):
            return UserMessage(content=m.content.upper())

        assert compose(shout) is shout

    def test_two_applies_left_to_right(self):
        def f(m):
            return UserMessage(content=m.content + "-f")

        def g(m):
            return UserMessage(content=m.content + "-g")

        out = compose(f, g)(_user("start"))
        assert out.content == "start-f-g"

    def test_three_applies_left_to_right(self):
        def f(m):
            return UserMessage(content=m.content + "a")

        def g(m):
            return UserMessage(content=m.content + "b")

        def h(m):
            return UserMessage(content=m.content + "c")

        assert compose(f, g, h)(_user("")).content == "abc"

    def test_compose_with_identity(self):
        def shout(m):
            return UserMessage(content=m.content.upper())

        assert compose(shout, identity)(_user("x")).content == "X"
        assert compose(identity, shout)(_user("x")).content == "X"
