import re

# LLMs insert spurious spaces between CJK characters during streaming
# (token boundaries often split mid-word). These regexes detect and strip them.
_CJK_CHARS = r'\u4e00-\u9fff\uff00-\uffef，。！？、：；\u201c\u201d\u2018\u2019（）【】'
_CJK_RE = re.compile(f'[{_CJK_CHARS}]')
_CN_SPACE = re.compile(f'(?<=[{_CJK_CHARS}])\\s+(?=[{_CJK_CHARS}])')


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


class StreamPrinter:
    """Centralizes all agent output. Strips spaces between Chinese characters in streamed text."""

    def __init__(self, prefix: str = ""):
        self._prefix = prefix
        # Spaces after CJK chars are held back — if the next chunk starts
        # with a CJK char, they're spurious and get stripped. Otherwise flushed.
        self._held = ""
        self._last_printed = ""  # needed for cross-chunk CJK space detection
        self._at_line_start = True  # prefix is only printed at the start of a line

    def write(self, text: str):
        """Stream a content chunk, stripping inter-Chinese spaces across boundaries."""
        combined = self._held + text
        self._held = ""

        # Strip spaces between CJK characters within the combined text
        combined = _CN_SPACE.sub("", combined)

        # Cross-boundary: last printed was CJK, combined starts with spaces + CJK → strip
        if self._last_printed and _is_cjk(self._last_printed):
            stripped = combined.lstrip(" \t")
            if stripped and _is_cjk(stripped[0]):
                combined = stripped

        if not combined:
            return

        # Hold back trailing spaces that follow a CJK character
        m = re.search(r'[ \t]+$', combined)
        if m:
            before = combined[:m.start()]
            last_ch = before[-1] if before else self._last_printed
            if last_ch and _is_cjk(last_ch):
                self._held = combined[m.start():]
                combined = before

        if combined:
            if self._prefix and '\n' in combined:
                # Indent every line — replace internal newlines with newline+prefix,
                # prepend prefix if we're at the start, trim trailing prefix after final \n.
                output = combined.replace('\n', '\n' + self._prefix)
                if self._at_line_start:
                    output = self._prefix + output
                if combined.endswith('\n') and output.endswith(self._prefix):
                    output = output[:-len(self._prefix)]
            else:
                output = (self._prefix if self._at_line_start else "") + combined
            print(output, end="", flush=True)
            self._last_printed = combined[-1]
            self._at_line_start = combined[-1] == '\n'

    def flush(self):
        """Flush any held spaces at end of response."""
        if self._held:
            print((self._prefix if self._at_line_start else "") + self._held, end="", flush=True)
            self._at_line_start = self._held[-1] == '\n'
            self._held = ""

    def tool_start(self, name: str):
        self.flush()
        print(f"\n{self._prefix}> {name}(", end="", flush=True)
        self._at_line_start = False

    def tool_args(self, args_chunk: str):
        print(args_chunk, end="", flush=True)

    def tool_end(self):
        print(")")
        self._at_line_start = True

    def tool_result(self, result: str):
        print(f"{self._prefix}  = {result}")
        self._at_line_start = True

    def newline(self):
        self.flush()
        print()
        self._at_line_start = True
