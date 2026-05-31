# Memory Prefetch (Layer 2) 设计

> **状态**：**① prefetch+consume+inject 已锁定**（§1–§7，本文）；②③④⑤⑥ 待续（§8 占位）。
> **配套**：Layer 1 已 ship 到 master（`side_query` + `find_relevant_memories`，见 `2026-05-31-memory-selector-design.md`）。
> **上下文**：把 CC 的 `startRelevantMemoryPrefetch` / `getRelevantMemoryAttachments` / `collectSurfacedMemories` / `collectRecentSuccessfulTools` 管道移植到 mini-cc。
> **范围**：全 CC-faithful 六件 —— **① prefetch+consume+inject** · ② already_surfaced 去重 · ③ recent_tools 反噪声 · ④ session 字节预算 · ⑤ 单文件截断 · ⑥ file_read 去重。本文锁 ①（含 ⑥ 的 mark-after）。

---

## 1. 核心思想：speculative prefetch（把慢 selector 藏在主循环工作背后）

selector 是一次独立 LLM 调用（flash，~1–2s）。若在回答前 `await` 它，每轮都 +1–2s。CC 的解法、也是我们采用的：**回合开始就 fire（不 await），让它和主模型的工作并行跑，等它好了再非阻塞地拼进对话，回合退出时清理。** 这就是预取（latency hiding）。

> Java 类比：`var f = supplyAsync(selector)` 回合开始 fire；之后每轮 `f.getNow(null)`（好了就用、没好就跳），而非 `f.get()`（阻塞）。CC 的 `settledAt` 轮询 = `getNow`。

## 2. 关键决策（带依据）

| ID | 决策 | 依据 |
|---|---|---|
| **D-NONBLOCK** | **非阻塞（A）**，**不** block 首调 | always-block 是业界点名的 indiscriminate-retrieval 反模式（[Self-RAG](https://www.emergentmind.com/topics/self-rag) / [Adaptive-RAG](https://blog.reachsumit.com/posts/2025/10/learning-to-retrieve/)）；no-tool 轮漏的那块由**路径1（B-1 index 每轮可见 + WHEN_TO_ACCESS → 模型自己 Read）**兜，等价 Self-RAG 的"模型决定检索"；非阻塞成本藏在工具轮工作背后，且 selector 自门控到 `[]` |
| **D-ATTACH** | 新结构化 **`RelevantMemoryMessage`** 类型 | 结构化 `path/content/mtime` 让 ②④⑥ 派生干净；与 CC 的 `relevant_memories` attachment 同构 |
| **D-WIRE** | kick-off 在 `query()` / consume 在 `get_messages` 钩子 / cancel 在 `query()` finally | `AgentLoop` 是纯生产者（`agent_loop.py:1-21`），engine 唯一能插手的每轮钩子就是它传的 `get_messages`（`query_engine.py:265`） |
| **D-MARKAFTER** | surface 后把每条 `record` 进 `file_read_state` | surfacing = 内容进 context ≈ 模型已 Read；登记后 file_read 去重 / edit-gate / 跨轮 consume-filter 三者统一；**先 filter 再 record 的次序 load-bearing**（CC `attachments.ts:2513`） |

## 3. CC ↔ mini-cc 映射（思想 1:1，仅消费点位置因结构而异）

| CC | mini-cc | 理由 |
|---|---|---|
| `startRelevantMemoryPrefetch`（loop entry） | `_start_memory_prefetch`（`query()` dispatch user 后） | query() 是回合入口，直接拿 user_text |
| 不 await 的 promise | `asyncio.create_task(coro)` | Python task = JS 未 await promise |
| `settledAt` | `task.done()` | "跑完没"的非阻塞探针 |
| `consumedOnIteration` | `PrefetchHandle.consumed` | "消费过没"，保证只拼一次 |
| 消费点（循环体 post-tools） | `_consume_prefetch_if_ready`（挂 `get_messages` 钩子，仅 `parent_id is None`） | AgentLoop 纯生产 → 只剩这个每轮钩子 |
| `filterDuplicateMemoryAttachments(readFileState)` | 过滤 `file_read_state._state` + mark-after | 同一个去重（⑥） |
| `yield` attachment + push toolResults | `dispatch(RelevantMemoryMessage(...))` | mini-cc 单写路径，下次 `api_view` 自然看到 |
| `using` + `Symbol.dispose` | `query()` 的 `finally: task.cancel()` | Python 无 `using`，finally 是清理保证 |

## 4. 接线（三处 seam，全在 QueryEngine，AgentLoop 不动）

```
query(user_text):
  dispatch(UserMessage(user_text))              # 已有
  self._start_memory_prefetch(user_text)        # 新：起 task，存 self._pending
  try: await run_loop(...)
  finally: self._cancel_prefetch()              # 新：Symbol.dispose 的 Python 替代

run_loop 的 get_messages 钩子（每轮 LLM 调用前）:
  get_messages = lambda: self._turn_pre_call(parent_id)
  _turn_pre_call(parent_id):
     await self._consume_prefetch_if_ready(parent_id)   # 新：非阻塞 poll + 注入
     return await self._prepare_messages(parent_id)     # 已有（保持纯净）
```

## 5. 实现伪代码（① 主体，含 ⑥ mark-after）

```python
@dataclass
class PrefetchHandle:
    task: asyncio.Task
    consumed: bool = False

def _start_memory_prefetch(self, user_text: str) -> None:
    if not _should_prefetch(user_text):   # gate：enabled + 非单词 + ④byte budget（②③④⑤待接）
        self._pending = None
        return
    recent_tools     = collect_recent_successful_tools(self.store.all())   # ③（暂 ()）
    already_surfaced = collect_surfaced(self.store.all())                  # ②（暂 frozenset()）
    coro = self._surface(user_text, recent_tools, already_surfaced)
    self._pending = PrefetchHandle(task=asyncio.create_task(coro))

async def _surface(self, query, recent_tools, already_surfaced) -> list[SurfacedMemory]:
    headers = await find_relevant_memories(           # Layer 1，已 ship
        query, get_auto_mem_path(),
        recent_tools=recent_tools, already_surfaced=already_surfaced)
    return [read_and_truncate(h) for h in headers]    # ⑤（暂：读全文不截断）

async def _consume_prefetch_if_ready(self, parent_id) -> None:
    if parent_id is not None:                         # sidechain 不注入 memory
        return
    h = self._pending
    if h is None or h.consumed or not h.task.done():  # 非阻塞 poll
        return
    mems = [m for m in h.task.result()                # ⑥ filter（mark 之前，用旧账本）
            if file_read_state._state.get(m.path) is None]
    if mems:
        await self._dispatch(RelevantMemoryMessage(memories=mems))   # B 结构化注入
        for m in mems:                                # ⑥ mark-after
            file_read_state._state.record(m.path, m.content, m.mtime_ms,
                                          offset=1, limit=m.line_count)
    h.consumed = True

def _cancel_prefetch(self) -> None:
    if self._pending and not self._pending.task.done():
        self._pending.task.cancel()
    self._pending = None
```

**取消干净的保证**：`find_relevant_memories` 的 best-effort 是 `except Exception`，**不吞** `CancelledError`（`BaseException`），所以 `task.cancel()` 干净传播 —— 这正是 Layer 1 特意写 `except Exception` 而非裸 `except` 的回报。

## 6. `RelevantMemoryMessage`（D-ATTACH，B 结构化类型）

```python
class SurfacedMemory(BaseModel):
    filename: str      # 相对名：selector / already_surfaced(②) 去重键
    path: str          # 绝对路径：file_read_state(⑥) 键
    content: str       # ⑤ 截断后的内容
    mtime_ms: int
    line_count: int    # 供 ⑥ record 的 limit
    header: str        # 固化的 freshness 头（创建时算一次，cache 稳定）

class RelevantMemoryMessage(Message):
    type: Literal["relevant_memory"] = "relevant_memory"
    memories: list[SurfacedMemory]
```
- 进 `LAYER_1_TYPES`（API 可见）。
- `to_langchain_single` → 单个 `HumanMessage`，content 里 **每条 memory 各一个 `<system-reminder>` block**（`{header}\n\n{content}`，对齐 CC `messages.ts:3708` 的 per-memory map：1 memory = 1 block）→ 经 `_merge_consecutive_human` 并入 user preamble（同 B-1）。
- **freshness 头固化**：`header = memory_header(filename, mtime_ms)` 在 `read_and_truncate`（创建 SurfacedMemory）时算一次存入，渲染直接用 `m.header` 不重算——否则 age 每天变 → 渲染字节漂移 → prompt-cache miss（CC `messages.ts:3711-3715`）。age-as-words 从 L4 提前到本层（`memdir/age.py`）。
- **persistence**：mini-cc persistence 是 **write-only**（`persistence.py:57`），不 replay，故无需反序列化器——B 相对 A 并不多出这块工作面。

## 7. 生命周期 & 不变量

- `self._pending: PrefetchHandle | None` 单字段：mini-cc 串行处理一轮到底（`store.py:140` serial 假设），同时只有一个 prefetch。
- consume 只在 main branch（`parent_id is None`）；sub-agent（sidechain）不注入 memory。
- no-tool 轮的"漏 memory"是**接受的代价**（D-NONBLOCK）：交给路径1（B-1 index + 强化 WHEN_TO_ACCESS 指令让模型该 Read 时 Read）。降低 A-miss 的杠杆是 prompt，不是 block。

## 8. ① 测试 outline（待写 TDD plan 时展开）

| 用例 | 验证 |
|---|---|
| gate：单词 query → 不起 prefetch（`self._pending is None`） | kick-off gate |
| consume：`task` 未 done → 不 dispatch RelevantMemoryMessage | 非阻塞 |
| consume：`task` done + 非空 → dispatch + 每条 `file_read_state.record` | 注入 + ⑥ mark-after |
| ⑥ filter：某 path 已在 file_read_state → 被滤、不注入 | consume-time dedup |
| sidechain：`parent_id != None` → 不注入 | 分支隔离 |
| 回合结束：`finally` 调用 → 未完成的 `task` 被 cancel | 清理保证 |

---

## 9. ②③④⑤⑥（待续，下一轮 brainstorm）

> - **② already_surfaced 派生**：`collect_surfaced(store.all())` 扫 `RelevantMemoryMessage` → ∪ `memories[].filename`（相对名，与 selector 去重键一致；绝对 `path` 留给 ⑥ file_read_state）；喂给 `find_relevant_memories` 的 `already_surfaced`。
> - **③ recent_tools 派生**：port `collectRecentSuccessfulTools` —— 扫 store 到上一个 user 轮边界，从 `AssistantMessage(ToolUseBlock)` 收工具名、从 `ToolResultMessage`（`output` 是否 `ToolErrorOutput`）判成败，返回"成功且没报错"的工具名。
> - **④ session 字节预算**：`collect_surfaced` 顺带 `Σ len(content)`；gate 里 `≥ MAX_SESSION_BYTES` → 这轮不 prefetch。
> - **⑤ 单文件截断 + freshness 固化**：`read_and_truncate(header)` —— 200 行 / 4KB 上限，超出附"truncated, use file_read"提示；产出 `SurfacedMemory`，并把 `memory_header(filename, mtime_ms)` 固化进 `.header`（见 §6）。
> - **⑥ file_read 去重**：consume 的 filter + mark-after（§5 已含 mark-after；filter 已含）。
