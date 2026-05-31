# Memory Selector 设计：`side_query` + `find_relevant_memories`

> **状态**：`side_query` 原语已锁定（§2）；`find_relevant_memories` 待讨论后补（§3 占位）。
> **配套**：本文是 `docs/L2-spec.md` 的 **M5（selector）/ M3（scan）** 的深化实现蓝图——L2-spec 给"做什么"，本文给"怎么做 + 为什么"。
> **上下文**：L2「可取」B-2——把 CC 的 LLM selector（`src/memdir/findRelevantMemories.ts`）移植到 mini-cc / DeepSeek。

---

## 0. 背景与边界

**为什么**：mini-cc 当前「可取」只做了 B-1（MEMORY.md 索引注入 messages[0]）。B-2 缺 LLM selector——即"按当前 query 语义选出 top-N 相关 memory"的能力。CC 用一次 Sonnet sideQuery 做这件事；我们在 DeepSeek 上复刻。

**Scope（Layer 1 纯召回）**：
- ✅ `side_query` 通用原语（调便宜副模型，返回原始文本）
- ✅ `find_relevant_memories` 召回函数（scan → manifest → side_query → 解析校验过滤 → top-N）
- ❌ **不做**（Layer 2，第二轮 brainstorm）：异步 prefetch、attachment 注入、三重去重、session 字节预算、主流程接线

**模型**：`deepseek-v4-flash` 非思考模式——CC「Sonnet 而非 Opus 做 selector」的 DeepSeek 同构。

---

## 1. 跨切面决策（side_query 与 selector 共享）

| ID | 决策 | 依据 |
|---|---|---|
| **D-JSON** | `response_format: json_object` + **手工校验形状** + **手工过滤真实文件名**；**不**用 strict-schema | json_object 只保证语法不保证 schema → 须自校验（[JSON 指南](https://api-docs.deepseek.com/guides/json_mode)）。strict 在 DeepSeek = beta 工具调用 + 已知畸形 JSON bug（[#1069](https://github.com/deepseek-ai/DeepSeek-V3/issues/1069)）+ **丢缓存**；而真正的安全网"文件名真实"只能手工过滤（CC `findRelevantMemories.ts:130`），strict 给不了 |
| **D-CACHE** | DeepSeek 自动前缀缓存（默认开、标准 `/v1`、命中 ~98% 便宜）→ 布局 **system=稳定内容(缓存前缀)** / **user=变化内容(miss 尾)** | [Context Caching 指南](https://api-docs.deepseek.com/guides/kv_cache)：缓存最长公共前缀，稳定内容前置可最大化命中。CC selector 本身不缓存（`sideQuery.ts` 未设 cache_control），这是我们因底层不同新增的优化 |
| **D-THINK** | 非思考（`extra_body={"thinking":{"type":"disabled"}}`） | flash 默认开思考；非思考=省钱/快 + 不让推理吃掉 `max_tokens`→空 content。select 是单遍分类不是多步推理，CC selector 亦非思考（Sonnet 默认关 extended thinking）。质量不够时爬**能力轴**（flash→pro）不碰**思考轴**。机制见 [Thinking Mode 指南](https://api-docs.deepseek.com/guides/thinking_mode) |
| **D-BESTEFFORT** | 失败→`[]` 的 best-effort 策略**只在 selector 一处**；side_query 不解析/不吞异常 | 让兜底集中在知道"`[]` 是安全默认"的领域层；side_query 保持可复用的 dumb 原语 |

---

## 2. `side_query` 原语（已锁定）

### 2.1 角色与边界

单发 side query 原语：在主对话循环之外调一次便宜、快、确定的副模型，返回**原始 str**。

- **dumb**：不解析、不校验、不过滤、不吞异常——这些全归调用方。
- **通用**：对 system/user 内容无知，原样透传。可被任意 JSON/非 JSON 调用方复用。
- **单调用方**：现在只有 selector 一个调用方，就按一个设计（YAGNI）。CC 的 `sideQuery` 13 参数是因为服务 N 个调用方（权限解释器、会话搜索、模型校验…），我们不抄大杂烩。
- **测试缝**：它是 selector 测试的 monkeypatch 点（同 `test_boot_memory_injection.py:24` 打 `build_memory_context`）。

### 2.2 实例策略（焊死在 `_llm_flash` 上 = 全局不变量）

| 项 | 值 | 焊死理由 / 依据 |
|---|---|---|
| model | `deepseek-v4-flash`（常量 `_FLASH_MODEL`） | 便宜副模型；升级=改一行（爬能力轴） |
| thinking | `extra_body={"thinking":{"type":"disabled"}}` | 非思考。`extra_body` 是 `BaseChatOpenAI` 直接字段（`base.py:795`），且文档指明 provider 专属参数就用它（`base.py:811-813`） |
| temperature | `0` | 分类 → 确定性 |
| timeout | `10`（→ `request_timeout`） | 防挂死兜底；非阻塞 prefetch 下非延迟控制 |
| max_retries | `1` | best-effort 快路径 |
| 类 | 朴素 `ChatDeepSeek`（**非** `_ChatDeepSeekRoundTrip`） | 单发无历史 assistant → `reasoning_content` 补丁是 no-op（`chat_models.py:235-257`） |
| 端点 | 默认 `/v1`（无 beta） | json_object + 缓存都在标准端点 |

### 2.3 签名与契约

```python
async def side_query(system: str, user: str, *, json_mode: bool = False, max_tokens: int = 256) -> str
```

| 参数 | 设计理由 |
|---|---|
| `system` / `user` 分两个 | **此拆分即 prompt-cache 边界**：稳定→system(缓存前缀)、变化→user(miss 尾) |
| `json_mode: bool`（非 `response_format: dict`） | 命名意图、藏 wire 格式；刻意不做 strict-schema（见 D-JSON） |
| `max_tokens: int = 256` | per-call，把截断坑摆在调用点；非思考后 256 全给 JSON，≤5 文件名绰绰有余 |

**契约**：
- 不解析/校验/吞异常——API 错、timeout、`CancelledError` 全向上抛（`CancelledError` 是 `BaseException`，调用方 `except Exception` 不会误吞 → Layer-1 靠 asyncio 取消成立）。
- `json_mode=True` 时**调用方 prompt 必须**含字面 `"json"` + 示例 + 明确空目标，否则 API 400 / 空 content。

### 2.4 调用机制（已核实，file:line）

| 机制 | 依据 |
|---|---|
| `json_mode` → `.bind(response_format={"type":"json_object"})` | `base.py:2848` 原样示例 |
| `max_tokens` → 同次 `.bind(max_tokens=…)` | `base.py:1127` 进 payload |
| `await _llm_flash.bind(**binds).ainvoke([SystemMessage, HumanMessage])` | 单发非流式 |
| 返回 `content if isinstance(str) else ""` | 非 str(多模态块)→ "" → 让 selector 的 `json.loads` 干净走 `[]` |

### 2.5 空 content 三层防御

DeepSeek 文档警告 json_object「偶尔返回空 content」。三层互补：

```
① 代码：selector  except Exception → []        (兜底，不崩)
② prompt：SELECT_PROMPT 的 output style 章节     (治本，给明确 JSON 目标 + 空目标)
③ 模型：非思考                                  (去掉"思考吃光预算"这个诱因)
```

**缓存命中率诊断（方案 A，已定）**：为支撑 §3 的 manifest 排序决策，side_query 内加一行 **debug 级**日志读 `resp` 的 cache 字段（`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`），**返回类型不变**（仍 `-> str`），非遥测管道，仅 dev 诊断。确切 metadata 路径落地读包确认。

### 2.6 锁定代码

```python
# src/mini_cc/side_query.py
"""单发 side query 原语：在主对话循环之外调一次便宜、快、确定的副模型，返回原始文本。

dumb 通用原语：不解析、不校验、不吞异常——prompt / 解析 / best-effort 策略全归调用方
(让"失败→[]"只活在一处)。唯一调用方见 memdir/selector.py；它也是 selector 测试的
monkeypatch 缝。
"""
from __future__ import annotations

import logging
import os
from dotenv import load_dotenv
load_dotenv()  # 构造 _llm_flash 前确保 env 已读(幂等，同 llm.py:2-3)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)

_FLASH_MODEL = "deepseek-v4-flash"
"""副模型名。select 质量不够时改这一行升 deepseek-v4-pro(仍非思考)，不碰思考轴。"""

# 策略焊死在实例上：每个 side query 都"便宜/快/确定/非思考"。
# 朴素 ChatDeepSeek(非 _ChatDeepSeekRoundTrip)：单发 [system,user] 无历史 assistant，
# reasoning_content round-trip 补丁是 no-op(见 chat_models.py:235-257)。
_llm_flash = ChatDeepSeek(
    model=_FLASH_MODEL,
    temperature=0,                                  # 分类任务 → 确定性
    timeout=10,                                     # 防挂死兜底(→ request_timeout)
    max_retries=1,                                  # best-effort 快路径
    extra_body={"thinking": {"type": "disabled"}},  # 非思考：省钱/快 + 不让思考吃 max_tokens→空content
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
        [SystemMessage(system), HumanMessage(user)]
    )
    # 缓存命中率诊断(方案 A)：debug 级，返回类型不变(仍 -> str)，非遥测管道。
    # DeepSeek 的 prompt_cache_hit_tokens / prompt_cache_miss_tokens 在 resp 的 usage 元数据里；
    # 确切路径(response_metadata / usage_metadata)落地时读已装包确认。
    logger.debug("side_query usage: %s", resp.response_metadata)
    content = resp.content
    return content if isinstance(content, str) else ""
```

### 2.7 测试（`tests/test_side_query.py`）

side_query 是 API 边界，自有单测；底层模型换假对象（记录 bind kwargs + 返回 canned `AIMessage`）：

| 用例 | 验证 |
|---|---|
| `json_mode=True` | 绑了 `response_format={"type":"json_object"}` + `max_tokens`；返回 `.content` |
| `json_mode=False` | **不**绑 `response_format` |
| `.content` 是 list（多模态块） | 返回 `""` |
| （可选）live smoke | `skipif(not DEEPSEEK_API_KEY)` 真打一次，端到端 + 看 `prompt_cache_*` |

selector 测试则 monkeypatch `selector.side_query`，全程不打 API。

---

## 3. `find_relevant_memories` 召回函数（待讨论后补）

> 下一轮 brainstorm 议题，先占位。预期覆盖：
> - 签名：`async def find_relevant_memories(query, memdir, *, recent_tools=(), already_surfaced=frozenset()) -> list[MemoryHeader]`（返回选中的 `MemoryHeader` 子集，≤5；不新建 `RelevantMemory` 数据类）
> - 逻辑链：scan → 滤 `already_surfaced` → 空则短路返回 `[]` → manifest → side_query → `json.loads` → 校验 `selected_memories` 形状 → 过滤真实文件名 → 截断 5 → `except Exception: return []`
> - **缓存布局**：`system = SELECT_PROMPT + manifest`（稳定）/ `user = query (+ recent_tools)`（变化）；manifest 须在 query **之前**（与 CC 的 query-first 相反，因 DeepSeek 自动前缀缓存）
> - **manifest 排序与缓存稳定性**：mtime 倒序的写入重排会从第 1 行击穿缓存；候选可选 L2「top-N by mtime 取淘汰、再 filename 排序用于显示」
> - **SELECT_PROMPT**（干净 markdown）：选择规则（移植 CC `findRelevantMemories.ts:18-24`，含 recent_tools 反噪声规则）+ output style 章节（含字面 "json" + 示例 + 空目标）
> - 测试：happy ≤5 / `already_surfaced` 在调用前过滤 / recent_tools 进 prompt / 幻觉文件名丢弃 / 坏 JSON → `[]` / side_query 抛 → `[]`
