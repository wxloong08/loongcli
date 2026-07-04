# loong-bench 路线图：给 Agent 做诚实评测

loong-bench 不只是跑分，目标是系统性回答一个问题：**loongcli 作为 agent 运行时（harness），哪里可靠、哪里有风险。**

核心原则：**任务准确率 = 模型能力(A) × harness 质量(B)。** 评测要瞄准 **B 敏感区**，否则测出来的是模型不是 harness。这条原则决定了下面所有取舍。

---

## 一、已验证的维度

| 维度 | 怎么测 | 结论（实测） | 工具 |
|---|---|---|---|
| 任务完成准确率 | SWE-bench 式 fail-to-pass（从 git fix commit 挖任务，worktree 隔离） | 现有任务**饱和**——prompt 泄露解法 + 单文件局部修复，测的是模型不是 harness | `run.py` `harvest.py` |
| compact 正确性 + 效率 | 低 `compact_threshold` 制造压缩压力 × 多任务 × 多次重复 | 70× 压缩频率下 **4/4 resolved，步数无系统增长**——compact 不损正确性也不损效率 | `run.py --compact-threshold` |
| 缓存稳健性 | 压力下读 `cache_hit/miss` | 命中率 **85–97%、miss 不暴涨**——cache-aware compact 扛住极端压力 | `run.py` + 结果 jsonl |
| 长程记忆可达性 | 第一轮埋不可猜代号，多代压缩后取用 | **10 次连续压缩 5/5 可达**，纯摘要被动保留（未动用检索兜底） | `compact_memory_probe.py` |
| 缓存局部性（recall 注入位置） | A/B 探针，唯一变量是注入位置 | position-1 命中 **53%** → 末尾 **99%**，input 成本降 96% | `cache_probe.py` |
| 召回注意力（不可观测量） | 行为代理：能否用上召回信息答对题 | 末尾注入不退化（5/6 → 6/6） | `recall_attention_probe.py` |
| compact 摘要的缓存成本 | A/B：删减后摘要 vs 喂完整历史 | 删减省 37% token 却贵 **22 倍**（命中 12% vs 98%）→ cache-aware 分流的依据 | `compact_cache_probe.py` |
| 压缩后契约保持 + 归档保真 | 合成 skill + 强制多次压缩 | 显眼硬规则摘要即可保留；归档段数 == 压缩次数 | `stress_compact.py` |
| skill 多阶段执行（正常） | 4 阶段流水线 skill，客观 oracle（链式 key + 校验码 + 阶段序） | 正常执行下 **5/5 全对**——流程顺序/状态传递/规则遵守可靠 | `skill_exec_probe.py` |
| skill 细节契约（压缩下） | 细节型契约（不起眼第 3 条）+ 多次压缩 + 重注入开/关对照 | **18/18 vs 18/18 都守约**——compact 对 skill 细节保留好；重注入未测出价值（skill 完整进摘要已保住，重注入冗余） | `skill_detail_probe.py` |
| skill 外部状态重读（压缩下） | skill 开头读外部黑名单文件、后续凭记忆 + 多次压缩 + 大清单藏目标，对照重注入开/关 | 5 变体均 **0 违约**——loongcli 靠「摘要保留工具结果原文 + 入口截断」结构性避开 Claude Code 真实犯过的「compact 丢黑名单」bug | `external_state_probe.py` |

**总判断**：loongcli 在能造出的任务上又准又稳，压不出稳定弱项。这不是评测失败，是个诚实结论——harness 质量已经够高，残余风险集中在下面的未覆盖维度。

---

## 二、待覆盖的缺口（按优先级）

1. **skill 执行能力** —— 5 层：加载 / 多阶段流程正确性 / 状态传递 / 规则遵守 / 压缩下保持。
   - **第 1–4 层已验证**：`skill_exec_probe` 正常执行下 5/5 全对（流程/状态传递/规则）。
   - **第 5 层（压缩下保持）已测**：细节型契约（不起眼第 3 条），重注入开/关 **18/18 vs 18/18 都守约**。根因：skill 工具结果完整进摘要 LLM 输入（不在清理名单），摘要保住了细节，重注入成冗余兜底。结论：compact 对 skill 细节保留好（强正向）；**重注入未被证伪也未被证明必要**（坐实 stress_compact 的判断），价值只在未来"skill 被纳入清理"场景。未排除的混淆：agent 可能靠前几轮自己产出的惯性守约。要真测出重注入价值，需让 skill 工具结果也被清理，或设计"压缩后才首次出现、无惯性"的细节。

2. **Plan Mode 状态机** —— 规划态只读约束是否真生效、计划是否被执行态遵守、状态转换是否合法。

3. **/goal 自主循环** —— 停滞检测、进度回归检测在长程自主任务上是否真的兜底。

（注：`/loop` 尚未实现，不在评测范围。）

这些恰恰是 loongcli 比"裸调模型"更有价值的高阶能力，最该被证明，但现有评测只覆盖"单轮任务 + 对话/compact"。

---

## 三、方法论纪律（最重要的沉淀）

评测本身比被评测的系统更容易出错。下面几条是踩坑换来的：

1. **每 cell 多次重复，n=1 不下结论。** agent 评测的步数/耗时方差极大（同任务同配置步数可差近一倍），单次差异基本是噪声。曾把一个 n=1 的"步数 +84%"当成弱项，n=4 一跑全证伪。

2. **要测压缩规模/满载，必须用不可回收内容撑上下文。** 工具结果有入口截断（单条 >8000 字符只留 2000 预览，单轮 30000 上限，见 `core/tool_result_manager.py`），读大文件撑不起上下文（实测读 6 个 9000 字符文件，峰值才 16K）。得用对话（user/assistant 消息）撑。

3. **测试场景的资源特征 ≠ 真实使用。** 评测时的缓存命中率（大量独立会话冷启动 + 灌料 + 故意破坏前缀，可低到 46%）不代表产品真实表现（单个长会话 90%+）。判断产品要用真实使用数据，评测成本要单列。

4. **设计任何测试前先问"我到底在测什么"。** 模糊 prompt 以为在测 harness、其实测的还是模型；低阈值以为在测满载、其实测的是压缩频率。最隐蔽的坑是测了 B 却以为在测 A。

5. **判断让位于测量，且敢被自己的数据推翻。** 测量在 DeepSeek 上便宜（几毛钱），没有理由拍脑袋。

---

## 四、工具清单（bench/）

| 脚本 | 作用 |
|---|---|
| `run.py` | 跑分器。worktree 隔离、fail-to-pass 预检、改 tests/ 判负、隔离 HOME、`--profile` / `--compact-threshold` / `--only` / `--keep-worktrees` |
| `harvest.py` | 从 git 历史挖候选任务（代码+测试同时变更的 commit），提取 test-only patch |
| `report.py` | 汇总 `results/*.jsonl`，按 profile 分组出 `task × profile` 对照矩阵 |
| `tasks.jsonl` | 任务集（id / base / fix / prompt / test_cmd / difficulty） |
| `cache_probe.py` | 缓存局部性 A/B（recall 注入位置） |
| `recall_attention_probe.py` | 召回注意力（行为代理） |
| `compact_cache_probe.py` | compact 摘要的缓存成本（22 倍发现） |
| `stress_compact.py` | 压缩后契约保持 + 归档保真 |
| `compact_memory_probe.py` | 长程记忆可达性（压缩后第一轮信息是否可达） |

---

## 五、诚实边界

- **任务规模受 API 成本限**：长程记忆测到 51K 上下文 / 10 次压缩，**不是百万级真满载**——绝对规模是受成本约束的近似。
- **样本仍偏小**：多数 cell n=3–5，方差大，结论是趋势性的而非统计严谨。完整化的第一步就是加大重复次数。
- **缺强模型对照**：当前只有 DeepSeek，无法用"强模型过、弱模型不过"区分模型瓶颈 vs harness 瓶颈。补一个强模型 profile（或用同模型的能力消融）是 oracle 升级的关键。
