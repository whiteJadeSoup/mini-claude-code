# 记忆读写操作的 TUI 折叠展示（第②面）

> 设计日期：2026-06-02
> 范围：让 agent **主动**读/写/搜记忆文件时，TUI 显示语义化的折叠行
> （`Recalling/Wrote/Searched N memories`），复刻 Claude Code 的 inline 记忆展示。

---

## 1. 问题与目标

### 问题

mini-cc 当前对 agent 主动操作记忆文件**不做任何特殊展示**：`file_read`/`file_edit`
命中 memdir 时，和操作普通源码文件一样，逐条裸渲染成
`● file_read(user_nickname.md · 7 lines)`（见现状截图 #1）。用户无法一眼看出
「agent 正在/已经 读写了几条记忆」。

对照真 CC（截图 #2）：连续的记忆读写被折叠成**一行带计数**的语义行
`Recalling 10 memories…`（进行中）/ `Recalled 10 memories`（完成）。这条干净展示
来自 CC 的 `collapseReadSearch.ts`——一个「按路径分类 + 连续折叠」的渲染层。

### 目标 (Goals)

- G1：agent 主动 `file_read` 命中 memdir → 折叠显示 `Recalling/Recalled N memories`。
- G2：`file_edit`/`file_write` 命中 memdir → `Writing/Wrote N memories`。
- G3：`grep`/`glob`（`path` 落在 memdir）→ `Searching/Searched memories`。
- G4：连续的记忆操作折叠成一段、带准确计数；被非记忆顶层工具或 turn 末打断则分段。
- G5：对未来「并行执行工具」**前向兼容**——计数与分段恒正确（行序问题见 §6）。

### 非目标 (Non-Goals)

- 不动自动注入通道（`RelevantMemoryMessage` / `_recalled_markup`，即第①面）——它已工作。
- 不实现后台 forked 提取 + `memory_saved`（第③面，`extractMemories` 那套）。
- 不折叠普通（非记忆）文件读写——维持现有逐条渲染。
- 不实现 ctrl+o 展开 / 列出文件名（MVP）。
- 不嗅探 glob pattern / shell 命令里的记忆路径（CC 的 `isMemorySearch` 有，MVP 省）。

---

## 2. 架构（§1）

三个组件，依赖只从 UI 流向 domain：

```
① is_memory_path(resolved: str) -> bool          src/mini_cc/memdir/paths.py（新增）
   判定 resolved（已 resolve 的绝对路径）是否落在本会话 memdir 内（含 memdir 自身）。
   实现 = 抽取现有重复代码：
     - file_read/__init__.py:37  (_frozen_memory_staleness)
     - config.py:56              (safe_path)
   两处改为调用它（顺手 DRY，单独 refactor 提交）。

② classify_memory_op(name: str, args: dict) -> "read"|"write"|"search"|None
   read   = name=="file_read"                       且 is_memory_path(resolve(args["path"]))
   write  = name in {"file_edit","file_write"}      且 is_memory_path(resolve(args["path"]))
   search = name in {"grep","glob"} 且 args["path"] 存在 且 is_memory_path(resolve(path))
   resolve 一律走 config.safe_path()——它已做 resolve+sandbox；抛 ValueError → 返回 None
   （按非记忆处理）。保证「判定的路径 == 工具真正读写的路径」。

   注：config.safe_path(config.py:35-59) 有 memdir 白名单(L52-57)——memdir 内路径不抛错、
   正常返回 resolved，只有 cwd 和 memdir 双双之外才抛 ValueError。所以记忆路径能被正常
   resolve 并分类；越权路径（../../etc）被 safe_path 挡下 → classify 得 None。
   is_memory_path 本身是纯比较（不 resolve、不依赖 config），避免 config↔memdir 循环导入；
   resolve 的职责在调用方。

③ ToolStatus 记忆游程累加器                        src/mini_cc/consumers/tui/app.py
   复用既有 self._groups（sub-tool 折叠）的同构模式，新增一个 open-run 状态。
```

### 关键决策：游程锚定 `add_tool`（发出顺序），不锚定完成顺序

`agent_loop.py:115-205` 的当前时序（**串行**）：一个 LLM step 内，所有
`AssistantMessage(ToolUseBlock)` 先全部 yield（→ 全部 `add_tool`），**然后**逐个执行、
逐个 yield `ToolResultMessage`（→ 逐个 `complete_tool`）。即「发出顺序 == 完成顺序」。

未来改并行后「完成顺序 ≠ 发出顺序」。因此：

> **游程归属（谁和谁算一段、计数多少）在 `add_tool` 时即确定**（发出顺序在串行/并行下
> 都等于 agent 语义序列，正是 CC 折叠所依据的）。计数与分段恒正确；只有 ChatLog 里
> 行的先后在并行下可能乱——那是整个 TUI 共有的既存问题（所有工具都这样），等并行
> 落地时统一处理，不是本 feature 的债（见 §6）。

---

## 3. 游程状态机（§2）

```
                add_tool(memory op, parent_id=None)
                          │
            ┌─────────────┴──────────────┐
       无 open run                   有 open run
            │                            │
       开新 run                    并入（同类/异类都进）
       live: ● Recalling N…         计数 +1
            └─────────────┬──────────────┘
                          │
       ┌──────────────────┴───────────────────┐
   add_tool(非记忆顶层工具)              StatusMessage turn_end
       │  CLOSE + FLUSH                      │  CLOSE + FLUSH
       ▼                                     ▼
   ToolFlushed("● Recalled 3 memories") → ChatLog 一行；清 live row；
   随后非记忆工具照原逻辑走
```

规则：

- **跨 LLM step 合并**：只要中间无非记忆顶层工具打断，多步里的记忆读并成一段
  （这正是截图 #2「10 memories」一行的来源）。
- **打断只认**：非记忆顶层工具、turn 末。assistant 文本/思考**不**打断（MVP 简化，
  比 CC 略宽松，后续可收紧）。
- **sub-tool（`parent_id` 非空）不并入游程**，维持现有 `self._groups` 行为。
- **失败的记忆 op（红 ●）仍计入**游程计数——折叠的是「操作」不是「成功」。
- **flush 时机**：run 关闭时发一次。串行下此刻成员 op 均已完成；live row 期间由
  `_tick` 渲染 `● Recalling N memories…`，关闭时替换为 ChatLog 的 done 行。

---

## 4. 显示文案（§3）

动词表（对标 `collapseReadSearch.ts:982-1013`）：

| 类型 | 进行中(active) | 完成(done) | 计数 |
|---|---|---|---|
| read | Recalling | Recalled | 有（`N memories`） |
| write | Writing | Wrote | 有（`N memories`） |
| search | Searching | Searched | **无**（只 `memories`） |

**多类型拼接**（一段游程含多种）：顺序 `read · search · write`，` · ` 分隔，
**首段动词首字母大写、后续小写**（CC 同款）。
例：2 读 1 写 → `Recalled 2 memories · wrote 1 memory`。
单复数：`1 memory` / `N memories`。

**两种行的 markup**（贴 mini-cc 现有风格：`app.py:297` live、`:279` flushed）：

```
live（_tick，闪烁点）:  {prefix}[grey50]{b}[/grey50] [cyan]Recalling 3 memories…[/cyan]
done（ToolFlushed）  :  {prefix}[green]●[/green] [cyan]Recalled 3 memories[/cyan]
```

**纯函数化**：把「计数 → 文案」抽成纯函数
`memory_run_summary(read: int, write: int, search: int, *, active: bool) -> str`，
对标 CC 的纯函数 `getSearchReadSummaryText` 和 mini-cc 已有的纯 `_recalled_markup`
（`renderers.py:52`，注释明说可单测）。状态机只管计数与触发，文案逻辑独立单测。

---

## 5. 测试策略（§4）

```
# is_memory_path（纯，unit）
T1 happy  is_memory_path(memdir/'user_nickname.md')        → True
T2 edge   is_memory_path(cwd/'src/foo.py')                 → False
T3 edge   大小写/反斜杠变体（Win normcase）                  → True
T4 edge   is_memory_path(memdir 自身)                       → True

# classify_memory_op（纯，unit）
T5 happy  file_read→read / file_edit,file_write→write / grep,glob(path∈mem)→search
T6 edge   file_read(项目源码)                               → None
T7 edge   grep(无 path，搜 cwd)                             → None
T8 error  file_read(path='../../etc/passwd')，safe_path 抛错 → None

# memory_run_summary（纯，unit）
T9  multi   (read=2,write=1,search=0,active=False)          → "Recalled 2 memories · wrote 1 memory"
T10 search  (0,0,search=1,active=True)                      → "Searching memories"（无数字）
T11 single  (read=1,active=False)                           → "Recalled 1 memory"（单数）

# 游程状态机（ToolStatus，integration）
T12 happy   连续 3 memory read → complete → turn_end ⇒ 恰好 1 条 ToolFlushed，含 "3 memories"
T13 break   read,read,[非记忆工具],read ⇒ 2 条记忆 flush（2 + 1），顺序正确
T14 cross   两个 step 的 read 中间无非记忆工具 ⇒ 并成 1 段
T15 edge    sub-tool(parent_id) memory read ⇒ 不并入游程，走 _groups
T16 error   红 ● 的记忆 read ⇒ 仍计入游程计数

Mock 边界：
  get_auto_mem_path → tmp_path fixture（复用 test_file_read_memory.py 现成 fixture）
  ToolFlushed → 捕获 post_message 断言（不 mock，行为即测点）
  safe_path → 真实（路径 resolve 是 T8 测点，不能 mock 掉）
Pass：计数准确 · 每段恰好 1 次 flush · 串行下顺序保真 · T13 重复 50× 无 flake
```

---

## 6. 范围边界与已知取舍

- **sub-tool 不并入游程**：`parent_id` 非空（如未来 forked 提取 agent）的记忆读写维持
  现有 `_groups` 行为，避免和主游程状态打架。CC 里这俩也分开计。
- **并行下的行序**：未来工具并行执行后，完成顺序 ≠ 发出顺序，ChatLog 里记忆行与其它
  工具行的先后可能错位。这是**整个 TUI 共有的既存属性**（非记忆工具同样如此），本
  feature 的计数/分段正确性不受影响。等并行落地时统一治理，不在本次范围。
- **搜索判定从简**：只认 `args["path"]` 落 memdir，不嗅探 glob pattern / shell 命令
  （CC 的 `isMemorySearch` 三者都查）。agent 极少 grep memdir，YAGNI。
- **不列文件名**：CC 第②面折叠行也不列（靠 ctrl+o 展开，mini-cc 无此机制）。
  可选后续加 dim 文件名。

---

## 7. 落点（文件清单 + 提交划分）

**提交 1（refactor，零行为变更）**：
- `src/mini_cc/memdir/paths.py`：新增 `is_memory_path()`，导出加进 `memdir/__init__.py:__all__`。
- `file_read/__init__.py:37`、`config.py:56` 改为调用它。
- 跑 `test_file_read_memory.py` / `test_file_tools.py` 证明行为不变。

**提交 2（feature）**：
- `classify_memory_op()` + `memory_run_summary()`（位置：建议 `consumers/tui/` 内新小模块或
  `memdir/paths.py` 旁，由实现时定）。
- `src/mini_cc/consumers/tui/app.py`：`ToolStatus` 加 open-run 状态；改动点
  `add_tool`(:214)、`complete_tool`(:252)、`_tick`(:285)、`start_turn`(:200)/`end_turn`(:207) 清状态；
  turn_end 在 `renderers._render_status`(:60) 已有钩子可触发 flush。
- `tests/`：新增 `test_memory_op_display.py`（T1-T16）。

**不动**：`renderers._render_relevant_memory`、`_recalled_markup`（第①面保留）。

