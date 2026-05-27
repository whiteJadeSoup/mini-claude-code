"""B-0: the main system prompt must declare the <system-reminder> protocol,
so the model treats injected context messages (channel B-1) as system
metadata rather than user speech. CC parity: prompts.ts:190."""
from mini_cc.prompts import build_system_prompt


def test_system_prompt_declares_system_reminder_protocol():
    prompt = build_system_prompt(available_tools=set())
    assert "<system-reminder>" in prompt
    assert "information from the system" in prompt
    assert "bear no direct relation" in prompt
