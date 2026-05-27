# L2: Mini-cc Memory MVP — 垂直切片 Spec

> **本文档目的**: L2 课的实施参考。每个模块给出角色、接口、关键约束、思考题。**不写实现代码**——你边写边学。
>
> **scope 哲学**: 垂直切片。CC 全部机制端到端最小可用，**唯一妥协**: cache 5 字段对齐。L3-L8 不再是"加新功能"，而是对 L2 已写组件的深度迭代。

---

## 0. 课级 overview

### 安装

CC 全部 memory 机制端到端最小可用版。

### 改变的线

```
课前: mini-cc 重启失忆
课后: butler 闭环
  - 跨会话记忆 + 重启不忘 (可存 + 可取)
  - 用户主动 /remember + 后台自动学习 (可写双路径)
  - 模型主动质疑老 memory (可久读时 3 层防御)
  - 周期性蒸馏 + 用户 /review (可久生命周期)
```

### 时长

10-15 小时。这是垂直切片项目，不是 3 小时 hello-world。

---

## 1. MVP 全功能架构图

```
[用户输入] ──→ [主 agent (mini-cc query loop)]
                          │
        ┌─────────────────┼─────────────────────────────┐
        ▼                 ▼                             ▼
  系统启动注入        每轮处理                       stopHooks
                                                        │
  ┌─────────────┐   ┌─────────────────┐    ┌──────────────────────────┐
  │ 通道 A      │   │ 通道 B           │    │ executeExtractMemories    │
  │ 行为指令    │   │ 索引 + 相关 mem  │    │ (forked agent, sandbox,   │
  │ ── system   │   │ ── messages[0]   │    │  互斥, 节流, 直接 drop)    │
  │   prompt    │   │   + attachment   │    │                           │
  │             │   │                  │    │ executeAutoDream          │
  │ - 4 type    │   │ - MEMORY.md      │    │ (3-gate: time/sess/lock)  │
  │ - WHEN_..   │   │ - selector top-N │    │                           │
  │ - TRUSTING. │   │ - 异步 prefetch  │    └──────────────────────────┘
  │ - DRIFT.    │   │ - <system-       │              │
  │             │   │   reminder>      │              │ fork query loop
  │             │   │   freshness      │              │ (同进程, 共享前缀)
  └─────────────┘   └─────────────────┘              ▼
                                            ┌──────────────────────────┐
                                            │ 后台 agent 用 Write tool  │
                                            │ 落 .md 到 memory dir      │
                                            │                           │
                                            │ /remember 也走这条路       │
                                            │ /review 也走这条路         │
                                            └──────────────────────────┘
                                                       │
                                                       ▼
                              ~/.minicc/projects/<slug>/memory/
                                  ├── MEMORY.md (索引)
                                  ├── user_role.md
                                  ├── feedback_*.md
                                  ├── ...
                                  └── .consolidate-lock (autoDream 用)

✗ Cache 5 字段对齐 (waived) — extraction 每次按全价计算
```

---

## 2. 4 属性 minimum-core 表

### 必须做

| 属性 | 模块 | 一句话 |
|---|---|---|
| **可存** | M1-M3 | 4 type 封闭存储 + 路径派生 + frontmatter 扫描 |
| **可取** | M4-M6 | 双通道注入 + LLM selector + 异步 prefetch |
| **可写** | M7-M9 | 用户 /remember + forked extract（简化版）|
| **可久** | M4 + M10-M12 | age-as-words + freshness note + autoDream + /review |

### 可妥协（用户已批）

| CC 有 | MVP 妥协 | 理由 |
|---|---|---|
| Cache 5 字段对齐 / cache_control marker / KV reuse | 不做 | DeepSeek 自动 cache 蹭得到就蹭，全价付得起 |

### 永不做

- Team memory
- KAIROS 模式
- `tengu_moth_copse` flag 切换
- `/btw` / SDK side_question / promptSuggestion
- GrowthBook flags（用 env + settings.json 替代）
- Beta header latches
- 多进程并发安全（mini-cc 单进程假设）

---

## 3. 12 模块 spec

### 可存 区域

#### M1: `memdir/types.py`

**角色**: 定义合法记忆形态。

**接口**:

```python
class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"

@dataclass(frozen=True)
class MemoryHeader:
    path: Path
    name: str
    description: str
    type: MemoryType
    mtime_ms: float
```

**约束**:
- 4 type 封闭集（不允许 unknown / fallback）
- frontmatter 解析**宽容**：缺字段给默认值，不抛异常
- `MemoryHeader` 只读

**Java 类比**: `MemoryType` 像 enum，`MemoryHeader` 像 record。

---

#### M2: `memdir/paths.py`

**角色**: 决定 memory 文件落地位置。

**接口**:

```python
def get_auto_mem_path() -> Path: ...
def is_auto_mem_path(p: Path) -> bool: ...   # sandbox 用
```

**约束**:
- env override (`MINICC_MEMORY_PATH`) → 默认派生路径
- 默认路径: `~/.minicc/projects/<sanitized-cwd>/memory/`
- 自动 mkdir

**思考题**:
1. 为什么需要 env override？哪些场景用得上？
2. sanitize 失败（路径含非法字符）你怎么处理？

---

#### M3: `memdir/scan.py`

**角色**: 把 .md 读回结构化对象 + 生成 manifest。

**接口**:

```python
def scan_memory_files(dir: Path) -> list[MemoryHeader]:
    """扫描所有 .md → MemoryHeader[].
    
    - 排除 MEMORY.md / .md.tmp / .md.bak
    - 按 mtime 倒序
    - 仅读前 30 行解析 frontmatter
    - 解析失败 skip + log warning, 不抛异常
    """

def format_memory_manifest(headers: list[MemoryHeader]) -> str:
    """格式: '- [type] filename (timestamp): description' 一行一个."""
```

**约束**:
- pyyaml 解析 frontmatter
- frontmatter 必须在文件最前面（`---\n...\n---`）
- 一个坏文件不能毒死整个 memory 系统

**思考题**:
1. 为什么只读前 30 行？
2. 解析失败 skip+log 而非 raise，原因是？

---

### 可取 区域

#### M4: `memdir/age.py`

**角色**: 把 mtime 转成 "N days ago" + freshness 警告。

**接口**:

```python
def memory_age_days(mtime_ms: float) -> int: ...
def memory_age(mtime_ms: float) -> str: ...
    # "today" | "yesterday" | "{N} days ago"

def memory_freshness_note(mtime_ms: float) -> str: ...
    # d <= 1 → ""
    # d > 1  → "<system-reminder>This memory is N days old. ...</system-reminder>\n"
```

**约束**:
- CC-faithful: 复刻 `memdir/memoryAge.ts:6-53`
- d=0 today / d=1 yesterday / d>=2 "{d} days ago"
- freshness note 只对 d>1 生成

**思考题**:
1. 为什么 "47 days ago" 而不是 "2026-03-20"？
2. 为什么 d<=1 不生成 warning？

---

#### M5: `memdir/selector.py`

**角色**: 用 LLM 选最相关的 top-N memory。

**接口**:

```python
async def find_relevant_memories(
    query: str,
    memory_dir: Path,
    already_surfaced: set[Path],
    *,
    max_results: int = 5,
) -> list[MemoryHeader]:
    """DeepSeek 单 shot sideQuery, 选最相关的 top-N."""
```

**约束**:
- DeepSeek 单 shot（不需要 fallback chain）
- prompt 嵌 manifest 作 candidate 列表
- 期望 JSON 输出 `{"selected_memories": ["filename1", ...]}`，prompt 引导 + 解析容错
- max_tokens=256
- 失败/解析错 → 返回空 list（best-effort，不 raise）
- 候选限 200（mtime 倒序前 200）

**思考题**:
1. selector 失败 fallback 是什么？为什么不是"返回全部"？
2. 用 embedding 相似度比 LLM 便宜，为什么 CC 没用？

---

#### M6: `memdir/injection.py`（双通道）

**角色**: 把 memory 注入到 system prompt（通道 A）和 messages（通道 B）。
**方案 Y（CC-faithful）**：MEMORY.md 索引走通道 B（messages[0]），**不放 system prompt**——与 CC `prependUserContext` 一致；通道 A 只放行为指令。

**接口**:

```python
def build_memory_section_channel_a(memory_dir: Path) -> str:
    """通道 A: 仅行为指令（TYPES/WHEN_TO_ACCESS/TRUSTING_RECALL/DRIFT）.
    注入 system prompt. 不含 MEMORY.md 索引——索引走通道 B-1."""

def build_memory_context(memory_dir: Path) -> str | None:
    """通道 B-1 内层(对照 CC getClaudeMds claudemd.ts:1185): 生成 memory 段 value——
    'Contents of {path} ({desc}):\n\n{truncated 索引}'. MEMORY.md 不存在/空 → None."""

def render_user_context(context: dict[str, str]) -> str:
    """通道 B-1 外层(对照 CC prependUserContext api.ts:461): 遍历 context dict 渲染
    '# {key}\n{value}' + <system-reminder> 包装. key 数据驱动可扩展(未来加
    gitStatus/userEmail 只需调用方往 dict 塞 key, 此函数零改动).
    调用方(boot)组装: {'memory': build_memory_context(...), 'currentDate': '...'}."""

async def build_memory_attachments_channel_b(
    relevant_memories: list[MemoryHeader],
) -> list[Attachment]:
    """通道 B-2: selector 选出的 relevant memory 包装成 attachment（每轮按需）."""
```

**通道 A 内容结构**:

```
# Persistent memory

You have a file-based memory at <memory_dir>.

## Memory types

<TYPES_SECTION 从 M10 引入>

## When to access memories

<WHEN_TO_ACCESS_SECTION 从 M10 引入>

## Before recommending from memory

<TRUSTING_RECALL_SECTION 从 M10 引入>
```

⚠️ **方案 Y**：通道 A **不再有** "## Current memory index"——索引移到下面通道 B-1（messages[0]）。理由：① system prompt 是跨会话稳定前缀，索引变化（`/remember`）不该破坏它；② 与 CC `prependUserContext` 一致（索引在 messages 区，不在 system）。

**通道 B-1 内容**（MEMORY.md 索引 → messages[0] context message，boot 注入一次）:
> `# memory` / `# currentDate` 是 `render_user_context` 遍历 context dict 的 key——**数据驱动不写死**；`Contents of {path} (...)` 是 `build_memory_context` 生成的 value（内层）。两层对照 CC 的 prependUserContext（外层遍历）+ getClaudeMds（内层 value）。

```
<system-reminder>
As you answer the user's questions, you can use the following context:
# memory
Contents of <绝对路径>/MEMORY.md (your persistent memory, persists across conversations):

<MEMORY.md 内容，truncate 到 200 行 / 25KB>
# currentDate
Today's date is <date>.

IMPORTANT: this context may or may not be relevant to your tasks. You should
not respond to this context unless it is highly relevant to your task.
</system-reminder>
```

**通道 B-2 每个 attachment 内容**（selector relevant memory，每轮按需）:

```
<system-reminder>This memory is N days old. ...</system-reminder>
<.md 文件实际内容>
```

**约束**:
- 通道 A（行为指令）启动时一次性注入 system prompt（cached，不每轮重算）
- 通道 B-1（MEMORY.md 索引）boot 时注入 messages 区一次（见 M9），不每轮重读文件
- 通道 B-2（relevant memory）是 attachment，每轮按需注入（配合 selector + prefetch）
- freshness note 只对 d>1 包 `<system-reminder>` tag
- B-1 截断 200 行/25KB（truncate_entrypoint），超过追加 WARNING

**思考题**:
1. 为什么把 MEMORY.md 索引放通道 B（messages[0]）而非通道 A（system prompt）？
   （cache 分层 tools→system→messages；CC 用 `prependUserContext`；注意 CC 的"跨用户 cache 共享"理由在 mini-cc 单用户**不成立**，但仍选 CC-faithful）
2. B-1 截断 200 行/25KB 是为了什么？超过会发生什么？
3. B-1 进 messages 区后，和真实用户输入连续两条 user message——DeepSeek-v4-pro 怎么处理？（见 M9 api_view 合并）

---

### 可写 区域

#### M7: `memdir/extract.py`（**简化版** forked agent）

**角色**: 每轮主 agent 完成后，后台 fork agent 提取新 memory。

**接口**:

```python
class MemoryExtractor:
    def __init__(self, throttle_turns: int = 3):
        self._in_progress: bool = False
        self._turns_since_last: int = 0
        self._throttle: int = throttle_turns
    
    async def execute(self, context, append_msg_fn):
        """每轮 stopHook 调一次. fire-and-forget."""
```

**简化版决策**（v.s. CC 完整版）:

| 保留 | 删除 |
|---|---|
| ✅ Forked agent（同进程新 query loop）| ❌ pendingContext / 单 slot 暂存 |
| ✅ Sandbox（Write/Edit 路径限于 memory dir）| ❌ Trailing run / isTrailingRun |
| ✅ 互斥（**扫最近 3 条** assistant message）| ❌ Cursor (lastMemoryMessageUuid) |
| ✅ 节流（turns_since_last counter, **默认 3**）| ❌ 细粒度遥测 |
| ✅ manifest 预注入 | ❌ Cache 5 字段对齐 |

**算法骨架**:

```
async def execute(context, append_msg_fn):
    1. 总开关: is_auto_memory_enabled() ? 否则 return
    2. 在跑 drop: self._in_progress ? 直接 return (NO stash)
    3. 互斥: 扫最近 3 条 assistant message 找 Write/Edit → memory dir
            找到 → return (不计入节流计数)
    4. 节流计数: turns_since_last += 1
                if turns_since_last < throttle: return
                turns_since_last = 0
    5. self._in_progress = True
       try:
           5a. manifest = format_memory_manifest(scan_memory_files(memory_dir))
           5b. prompt = build_extraction_prompt(memory_dir, manifest)
           5c. result = await run_forked_agent(
                   prompt_messages=[user(prompt)],
                   parent_context=context,
                   can_use_tool=create_sandbox(memory_dir),
                   max_turns=5,
               )
           5d. 通知主线程: append_msg_fn(memory_saved(extract_written_paths(result)))
       finally:
           self._in_progress = False
```

**Sandbox 实现**:

```
can_use_tool(tool_name, tool_input) -> bool:
    - Read / Grep / Glob → 允许
    - Bash → 拒绝（或仅允许只读子集 ls/cat/find/...）
    - Write / Edit → 仅当 file_path 在 memory_dir 内才允许
    - 其他工具 → 拒绝
```

**思考题**:
1. 简化版相比 CC 完整版，哪些场景会出问题？（pendingContext 删了 / cursor 删了的代价）
2. "扫最近 3 条" 而非 1 条，多了什么保护？还是会漏什么？
3. Forked agent 怎么"共享前缀"？mini-cc 的 query loop 接受这种 fork 吗？

---

#### M8: `commands/remember.py`

**角色**: 用户主动写 memory。

**接口**:

```python
async def remember_command(args: str, context) -> None:
    """/remember <内容> → 主 agent 用 Write tool 落盘"""
```

**约束**:
- 把 args 转成 user prompt 给主 agent
- prompt 引导主 agent 选合适的文件名 + frontmatter
- **不绕开主 agent** 直接写文件

**思考题**:
1. 为什么不直接在 slash command 里 fs.write？

---

#### M9: 主流程改动 `engine/query_engine.py` + `store.py` + 主 prompt

**改动 0 (B-0)**: 主 system prompt 加 `<system-reminder>` 通用协议声明（CC 对照 `prompts.ts:190`）。mini-cc 当前缺这条，否则模型不认得注入的 system-reminder 是系统元信息。

**改动 1**: boot 时 system prompt 加 memory 通道 A（**仅行为指令，不含索引**）

**改动 2 (方案 Y 核心)**: boot 时注入通道 B-1 context message 到 messages 区：
- `section = build_memory_context(memory_dir, current_date=...)`
- `if section:` → dispatch `UserMessage(section, is_synthetic=True, source="memory")`
- **不配对** Acknowledged assistant（靠改动 3 的 api_view 合并维持交替）
- 同时**去掉 task_state 现有的配对 Acknowledged**（统一改为不配对）

**改动 3 (方案 C)**: `store.py` `api_view` 加"合并连续 HumanMessage" post-pass：
- 现有 merge-assistant-by-turn_id 之后，加独立 pass
- 连续 HumanMessage → `"\n\n"` 拼接（按原始顺序，context 在前、输入在后）
- ToolMessage/AIMessage/SystemMessage 不参与 → tool_result 不被误合并
- 同步更新 `compact.py:_group_by_api_round` docstring

**改动 4**: 每轮 user prompt 进来时启动异步 prefetch（通道 B-2，不阻塞）

**改动 5**: tool 执行边界检查 prefetch 是否完成 → yield attachment messages（通道 B-2）

**改动 6**: 每轮完成后 stopHook 触发：
- `void extractor.execute(...)` (M7)
- `void auto_dream.execute(...)` (M11)

**注意**: stopHook 是 fire-and-forget。**不要 await**。

**为什么 boot 注入一次而非 CC 式每轮 prepend**：MEMORY.md 会话级不变，mini-cc 消息持久在 store（不像 CC 每轮重建列表），注入一次足够。context 固定在 messages 前部 → api_view 合并进第一轮 user → 与 CC 被 Anthropic 自动合并后的形态等价。

**方案 C 背景（为什么改 api_view 而非配对假 assistant）**：
DeepSeek-v4-pro **不支持连续同 role** 消息（Anthropic 会自动合并，DeepSeek 返回 400）。两种解法：
- **配对模式**：每条 synthetic user 配一条假 "Acknowledged" assistant → 往持久 store 塞假对话，污染 compact 轮次分析 / 历史回看
- **方案 C（选用）**：synthetic user 不配假 assistant，在 `api_view` 边界合并连续 Human → store 保留真实语义，合并隔离在 API 边界（= CC `normalizeMessagesForAPI` 哲学）
- 风险已排除：① tool_result 是 `ToolMessage(role=tool)` 不被误合并；② compact `_group_by_api_round` 消费 api_view 输出，看到的是合并后**单条 Human** preamble。

---

### 可久 区域

#### M10: `memdir/prompt_sections.py`

**角色**: 通道 A 引用的常量。

**内容**: 直接复刻 CC 的 `memoryTypes.ts:200-256`：

```python
TYPES_SECTION: str = """
## Memory types
<user> ...
<feedback> ... (must include Why + How to apply)
<project> ...
<reference> ...
"""

WHEN_TO_ACCESS_SECTION: str = """
## When to access memories
- When memories seem relevant...
- You MUST access...
- If user says ignore...
- Memory records can become stale over time. ...
"""

TRUSTING_RECALL_SECTION: str = """
## Before recommending from memory
A memory that names a function/file/flag is a claim it existed when written...
- If memory names file path: check exists.
- If memory names function or flag: grep for it.
- ...
"""

MEMORY_DRIFT_CAVEAT: str = "..."   # 嵌在 WHEN_TO_ACCESS 末尾
```

**约束**:
- header 文字关键（CC 注释里说 eval 测过）：
  - "When to access memories" — access 决策触发器
  - "Before recommending from memory" — 推荐前验证触发器
- 不要合并成一个 section

---

#### M11: `services/auto_dream.py`

**角色**: 周期性蒸馏 memory（去重 / 合并 / 清理）。

**接口**:

```python
class AutoDream:
    def __init__(self, min_hours: int = 24, min_sessions: int = 5):
        ...
    
    async def execute(self, context, append_msg_fn):
        """每轮 stopHook 调一次. fire-and-forget."""
```

**3-gate 顺序**（cost ascending）:

```
1. Time gate: hoursSince(lastConsolidatedAt) >= min_hours ?
              否则 return (1 个 stat 调用)

2. Session gate: count_sessions_with_mtime_after(lastConsolidatedAt) >= min_sessions ?
                 否则 return (扫 transcript 目录)

3. Lock gate: try_acquire_lock() ?
              失败 return (写 lock 文件 + verify)
```

**Lock 文件机制**: `<memory_dir>/.consolidate-lock`

```
body = process pid (string)
mtime = lastConsolidatedAt (acquire 时设为 now, 即"假装成功")

成功不更新 mtime (acquire 时已写)
失败 rollback: rewind mtime to priorMtime
```

**约束**:
- 跑 dream 时 fork 一个 agent，给它 consolidation prompt
- sandbox 同 extract（限 memory dir）
- 失败必须 rollback lock（让 24h 后能重试，crash 自动 backoff）

**思考题**:
1. 为什么 acquire 时立即写 mtime（而非"成功才写"）？
2. 3 gate 顺序为什么是 time → sessions → lock？

---

#### M12: `commands/review.py`

**角色**: 用户主动扫描 memory 找过期。

**接口**:

```python
async def review_command(args: str, context) -> None:
    """/review → 主 agent 扫 memory + 报告过期 / 冲突."""
```

**约束**:
- 把命令转成 user prompt：让主 agent 用 Read 扫 memory dir + 结合代码状态找问题
- 模型给出报告，用户决定是否手动 Edit / Delete
- 不要让 review 自动删除 memory（用户要保有控制权）

---

## 4. 测试 spec

| 测试 | 属性 | 验证什么 |
|---|---|---|
| T1: scan 空目录 | 可存 | 边界稳定 |
| T2: scan 3 个 sample | 可存 | 基础正确性 |
| T3: scan 排除 MEMORY.md | 可存 | 过滤规则 |
| T4: 通道 A 包含 4 type 定义 | 可取 | 行为指令注入 |
| T5: selector 选出相关 + 排除 already_surfaced | 可取 | selector + 去重 |
| T6: prefetch 异步不阻塞主流程 | 可取 | 异步性 |
| T7: attachment freshness note 只对 d>1 | 可取 + 可久 | freshness 阈值 |
| T8: `/remember` 落盘 | 可写 | 用户主动路径 |
| T9: extract agent 不能写 memory dir 外 | 可写 | sandbox |
| T10: extract 互斥（最近 3 条有 Write）| 可写 | 互斥 |
| T11: extract 节流（每 3 轮才跑 1 次）| 可写 | 节流 |
| T12: 重启 mini-cc 后能召回 memory | 可存 + 可取 | 跨会话闭环 |
| T13: age 标签 today / yesterday / 47 days ago | 可久 | age-as-words |
| T14: autoDream 3-gate 失败路径 | 可久 | gate 顺序 |
| T15: autoDream 失败 rollback lock | 可久 | crash backoff |
| T16: `/review` 列出 90+ 天的 memory | 可久 | 生命周期 |
| T17: api_view 合并连续 HumanMessage（ToolMessage 不被误合并）| 可取 | 方案C 边界合并 |
| T18: context message 三层包装（currentDate/全路径/不存在→None）| 可取 | 通道 B-1 |
| T19: task_state 去配对回归（compact preamble 多条 Human→合并单条）| 可取 | 方案C 回归 |

**通过标准**:
- 16 测试全 pass
- T12 重启循环 5 次稳定
- T9 sandbox 越界尝试都被拒绝
- T15 模拟 crash → 24h 内不重试

**Mock 边界**:
- 不 mock 文件系统（CC-faithful 语义点是 fs，mock 了测不出真问题）
- mock LLM 调用（成本 + 确定性）
- 不 mock 路径派生（测真实 sanitize 行为）
- T13 改 mtime 用 `os.utime`，**不 mock `time.time()`**

---

## 5. Demo（端到端）

```
$ uv run mini-cc
> 我喜欢用 uv 管理 Python 依赖, 测试用 pytest
mini-cc: 好的。

[3 轮后 stopHook 触发, extract agent 后台运行, 写入 user_pref_pkg.md]

> /quit

$ uv run mini-cc      # 第二天再启动 (mtime > 1day)
> 推荐一个 Python 包管理器
mini-cc: 你之前 (1 day ago) 提到喜欢 uv, 推荐 uv.

[selector 选了 user_pref_pkg.md, prefetch 异步注入了 attachment]

> /remember 我换工作了, 现在做 ML
mini-cc: 好的, 已记住.

[第 60 天后]

$ uv run mini-cc
> 我用什么框架?
mini-cc: 你之前 (60 days ago) 提到喜欢 Python + uv. 不过这条记忆已经
        2 个月前的了, 你可能换了技术栈, 要不要确认?
        [<system-reminder>This memory is 60 days old...</system-reminder> 触发了]

[第 80 天, 累计 5 个 session 后]
[autoDream 触发: time gate 通过 + session gate 通过 + lock acquire]
[forked agent 扫所有 transcript, 蒸馏 memory, 重写过期文件]

> /review
mini-cc: 发现 3 条 memory 引用的文件已不存在: ...
        建议 (Edit / Delete) ...
```

---

## 6. Critical Files

| 路径 | 操作 | 模块 |
|---|---|---|
| `src/mini_cc/memdir/__init__.py` | 新建 | - |
| `src/mini_cc/memdir/types.py` | 新建 | M1 |
| `src/mini_cc/memdir/paths.py` | 新建 | M2 |
| `src/mini_cc/memdir/scan.py` | 新建 | M3 |
| `src/mini_cc/memdir/age.py` | 新建 | M4 |
| `src/mini_cc/memdir/selector.py` | 新建 | M5 |
| `src/mini_cc/memdir/injection.py` | 新建 | M6 (build_memory_context + render_user_context 两层) |
| `src/mini_cc/memdir/truncate.py` | 新建 | M6 (D1: entrypoint 截断, 复刻 CC truncateEntrypointContent) |
| `src/mini_cc/memdir/extract.py` | 新建 | M7 |
| `src/mini_cc/memdir/prompt_sections.py` | 新建 | M10 |
| `src/mini_cc/services/auto_dream.py` | 新建 | M11 |
| `src/mini_cc/commands/remember.py` | 新建 | M8 |
| `src/mini_cc/commands/review.py` | 新建 | M12 |
| `src/mini_cc/engine/query_engine.py` | 修改 | M9 (boot 注入 context + 去配对 + stopHooks) |
| `src/mini_cc/engine/store.py` | 修改 | M9 改动3 (api_view 合并连续 Human, 方案C) |
| `src/mini_cc/prompts.py` | 修改 | M9 改动0 (B-0 system-reminder 协议) |
| `src/mini_cc/engine/sandbox.py` (如果不存在) | 新建 | M7 用的 can_use_tool |
| `tests/test_memdir_*.py` | 新建 | 16 个测试 |

---

## 7. 推进顺序（10-15h）

```
Phase 1 (4h): 可存 + 双通道注入骨架（行为指令 + 索引 messages[0]）
  M1, M2, M3, M10, M6.A（通道A行为指令）
  M6.B-1（build_memory_context）+ M9 改动0/2/3（B-0协议 + boot注入 + api_view合并连续Human）
  T1, T2, T3, T4, T17, T18, T12（重启召回）通过

Phase 2 (2h): 可久 信号
  M4 (age + freshness)
  T7, T13 通过

Phase 3 (3h): 可取 完整 (selector + prefetch + 通道 B-2 attachment)
  M5, M6.B-2, M9 改动4/5 (主流程 prefetch 消费)
  T5, T6 通过

Phase 4 (1h): /remember + 重启验证
  M8, M9 (slash 注册)
  T8, T12 通过

Phase 5 (3h): 自动 extract
  M7 (简化版) + M9 (stopHook 触发)
  T9, T10, T11 通过

Phase 6 (2-3h): autoDream + /review
  M11, M12, M9 (stopHook 触发)
  T14, T15, T16 通过

Phase 7 (1h): 端到端 demo
```

---

## 8. 学习自检

### 写代码前 — 应能脱口说出

1. 4 属性各自的最小核心模块？
2. 主 agent 启动时做了什么 memory 相关的事？
3. 主 agent 完成一轮后做了什么 memory 相关的事？
4. selector 失败时 fallback 是什么？为什么？
5. extract 互斥简化版怎么实现？为什么扫最近 3 条不是 1 条？
6. autoDream 3 gate 顺序为什么是 time → sessions → lock？
7. 双通道注入的两条通道分别在 API 请求的哪个段？
8. 为什么删 cache 优化没问题？
9. MEMORY.md 索引放 messages[0] 还是 system prompt？为什么？（cache 分层 + CC 跨用户共享理由在 mini-cc 是否成立）
10. context message 进 messages 区后和真实输入连续两条 user，DeepSeek-v4-pro 会怎样？方案 C 怎么解决？为什么不用配对假 assistant？

### 写完后 — 应能验证

1. 重启 5 次 demo 都稳定吗？
2. extract 在 sandbox 之外尝试写 → 真的被拒绝吗？
3. 60 天后的 demo 模型有没有质疑老 memory？
4. autoDream 模拟 crash 后 24h 内是不是不重试？
5. selector 给 garbage prompt → 是不是返回空 list（而非 raise）？

---

## 9. Not in Scope（永不做）

- Cache 5 字段对齐 / cache_control 标记 / KV reuse 优化
- Team memory
- KAIROS 模式
- `tengu_moth_copse` flag 切换
- `/btw` / SDK side_question / promptSuggestion
- GrowthBook flags
- Beta header latches
- 多进程并发安全

---

## 10. L3-L8 重新定义

L2 跑通后再决定 L3-L8 是否还需要。可能方向：

| 课 | 可能的深度迭代 |
|---|---|
| L3 | scan/manifest schema 严谨化（如 selector 表现不稳）|
| L4 | injection 截断策略 / 行为指令优化（如 token 失控）|
| L5 | selector 准确率优化（embedding fallback / few-shot）|
| L6 | extract 质量优化 / 加 cache（如成本太高）|
| L7 | 抗漂移效果验证 / 长期使用反馈 |
| L8 | autoDream 蒸馏质量 / capstone PR-review 集成 |

如果 L2 butler 已经"够用"，L3-L8 可以不做。

---

## 11. 附：4 个手工 sample（启动 demo 前先建好）

`~/.minicc/projects/<your-mini-cc-slug>/memory/`:

### `MEMORY.md`（索引）

```
- [User Role](user_role.md) — software engineer, prefers minimalist tools
- [Testing Policy](feedback_testing.md) — write integration tests with real DB
- [Mini-cc Project](project_minicc.md) — building memory-enabled butler
```

### `user_role.md`

```markdown
---
name: User Role
description: senior backend engineer, comfortable with Python and Java
type: user
---

User is a backend engineer with several years of Java experience, currently
learning Python. Prefers concise tools (uv, ruff over conda+black).
```

### `feedback_testing.md`

```markdown
---
name: Testing Policy
description: integration tests must hit real DB
type: feedback
---

Don't mock the database in integration tests.

Why: Got burned last quarter when mocked tests passed but the prod
migration failed.

How to apply: For DAO / repository layer tests, set up a real PostgreSQL
test container.
```

### `project_minicc.md`

```markdown
---
name: Mini-cc Project
description: building a personal butler agent on top of mini-cc
type: project
---

Mini-cc is a personal butler. Stack: Python 3.11 + LangChain + DeepSeek.
The L2 task is to add full CC-parity memory MVP that survives session restart.
```
