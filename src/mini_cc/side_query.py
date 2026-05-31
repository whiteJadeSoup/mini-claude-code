"""单发 side query 原语：在主对话循环之外调一次便宜、快、确定的副模型，返回原始文本。

dumb 通用原语：不解析、不校验、不吞异常——prompt / 解析 / best-effort 策略全归调用方
(让"失败→[]"只活在一处)。唯一调用方见 memdir/selector.py；它也是 selector 测试的
monkeypatch 缝。
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()  # 构造 _llm_flash 前确保 env 已读(幂等，同 llm.py)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)

_FLASH_MODEL = "deepseek-v4-flash"
"""副模型名。select 质量不够时改这一行升 deepseek-v4-pro(仍非思考)，不碰思考轴。"""

# 策略焊死在实例上：每个 side query 都"便宜/快/确定/非思考"。
# 朴素 ChatDeepSeek(非 llm._ChatDeepSeekRoundTrip)：单发 [system,user] 无历史 assistant，
# reasoning_content round-trip 补丁是 no-op。
_llm_flash = ChatDeepSeek(
    model=_FLASH_MODEL,
    temperature=0,
    timeout=10,
    max_retries=1,
    extra_body={"thinking": {"type": "disabled"}},
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


async def side_query(
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    max_tokens: int = 256,
) -> str:
    """跑一轮 [system, user]，返回模型原始文本。

    - 不解析/校验/吞异常(API 错、timeout、CancelledError 全向上抛)。
    - json_mode=True 置 response_format json_object；调用方 prompt 须含字面 "json"+示例+空目标。
    - system=稳定(缓存前缀) / user=变化(miss 尾)，拆分即 prompt-cache 边界。
    """
    binds: dict = {"max_tokens": max_tokens}
    if json_mode:
        binds["response_format"] = {"type": "json_object"}
    resp = await _llm_flash.bind(**binds).ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    logger.debug("side_query usage: %s", resp.response_metadata)
    content = resp.content
    return content if isinstance(content, str) else ""
