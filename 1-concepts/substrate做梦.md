---
tags: [AI, 认知科学, 方法论]
summary: 让 AI 离线 review substrate 做 schema 整合，对应大脑睡眠的记忆巩固——但触发是事件驱动而非时间驱动
---

# substrate 做梦

**提出者**：fangzhiyu & Claude (Opus 4.7)
**日期**：2026-04-27
**来源**：[[substrate会做梦：从整体诊断认出记忆巩固机制]]

## 定义

让 AI **离线 review 整个 substrate**（而非在 query-driven 对话中），生成 schema-level 诊断，识别跨情境模式、孤儿、冗余、新连接，再由用户选择性回写——这个过程在功能上**等价于大脑睡眠期的记忆巩固（memory consolidation）**。是 [[知识库飞轮]] 模型缺失的 **sleep phase**。

## 与大脑做梦的机制对应

| 大脑睡眠 | substrate 做梦 |
|---|---|
| 离线（停止接收新输入） | AI 不在实时对话，只 review |
| 海马回放（重新激活白天事件） | AI 重读所有卡的 summary |
| 新皮层整合（并入 schema） | AI 提取 substrate 的 schema-level 命题 |
| 跨情境连接 | "6 个簇是同一命题的 6 个面"类的发现 |
| synaptic homeostasis | 标出超中心节点、孤儿、冗余 |
| REM 期奇怪类比 | 元层模式发现（如 summary 里的 wikilink） |
| 醒后的"啊原来是这样" | dream report 的 schema-level 洞察 |

物理实现不同（神经递质 vs 注意力计算），但**计算抽象层是同一回事**：把分散的 episodic 经验整合到统一的 semantic schema。

## 为什么飞轮需要这个 phase

[[知识库飞轮]] 之前的模型是**白天工作模式**：

```
substrate × AI 当场抽取 × 你接受 × 你回写 → substrate'
```

但生物大脑光有白天会崩溃。没有 [[substrate 做梦]] 的副作用：

- 长期/短期不分层
- schema 不更新
- 跨情境连接不形成
- 孤儿、死链、冗余持续堆积
- 最终变成 [[AI镜像效应]] 警告的"知识坟墓"

这片 substrate 早就在写"记忆巩固"机制（[[主动遗忘]] / [[非单调可塑性假说]] / [[反馈累积阈值]] / [[CMS实现辨析]] / [[Think-No-Think 范式与检索抑制]]）——但**只把它当 LLM 工程问题或神经科学问题研究，没意识到 substrate 自身也需要这个 phase**。

## 触发：事件驱动，不是时间驱动

⚠️ **本节经过本体论修正**（2026-04-27 二次更新）

最初设计用周期性触发（"每月一次"、"每 50 张卡"），这是把人类时间结构错误投射到 substrate 上。修正后：

### 为什么不是时间驱动

人脑做梦本质上是**因为人是时间性生物**——昼夜节律、能量代谢、神经递质周期把睡眠强加给你。大脑做梦不是"主动选择"，是被时间逼出来的。功能性（记忆巩固）是结果不是原因。

substrate 没有这个生物学约束——一年不动和一秒不动在状态上**等价**。强加 cron 是 anthropomorphic projection（人类时间结构的错误投射）。极端反证：你完全可以让 substrate 10 分钟做一次梦——但每次梦内容相同，因为 substrate 状态没变。**没意义。**

### 正确触发：事件累积

substrate 是 **reactive system**——只在状态发生变化时才需要代谢。

```python
def should_dream():
    # 主：事件累积达阈值
    if events_since_last_dream() >= EVENT_THRESHOLD:
        return True

    # 强：[[显式化爆发]] 后立即
    if detected_burst(window=2h, min_cards=3):
        return True

    # 显式：用户主动 invoke
    if user_invoked:
        return True

    # 弱兜底：长期无事件 + 检测到 substrate drift
    # （防止 substrate 沉寂期间外部世界变化未被察觉）
    if days_since_last_dream > 90 and external_drift_detected():
        return "health_check"  # 不开改动 PR，只生成 dream report

    return False
```

### 事件类型与权重

| 事件 | 权重 | 备注 |
|---|---|---|
| 新建卡片 | 1.0 | substrate 显式增厚 |
| 重大编辑（>50% 内容变化）| 1.0 | 老卡再活化 |
| 链接批量变化（≥5 条）| 0.5 | 拓扑微调 |
| PR merge | 1.0 | 代谢动作完成，引入新结构 |
| dormant / revive | 0.3 | 局部状态变化 |
| 仅 mtime 变化（重命名等）| 0.1 | 几乎无意义 |

阈值 EVENT_THRESHOLD 待经验调整，初值 5-10。

### 与生物大脑的差异

| | 大脑做梦 | substrate 做梦 |
|---|---|---|
| 触发 | 时间逼迫（cron） | 事件累积（reactive） |
| 频率上限 | 24h 一次（生物极限） | 无固定上限，跟事件流走 |
| 静默期 | 不存在 | 完全可以——substrate 不"老化" |
| 浪费的代谢周期 | 不可避免 | 应该零 |

这是**数字 substrate 的本体论优势**——不需要浪费"无意义的代谢周期"。生物大脑被迫每天做梦，substrate 只在该做梦时做梦。

## 工程实现路径（待 spec）

[[2026-04-27-kb-index-system]] 应增加 `dream` 子命令：

1. AI 读 INDEX + CLUSTERS + GRAPH（substrate 的 4 层产物）
2. 生成 dream report（exploration 文档，含诊断）
3. 列出建议的 substrate 操作：补 link / 合并 / 重命名 / 标识陈旧
4. 用户审阅 + 接受/拒绝（[[校准的不可外包性]] 在 sleep phase 同样关键）
5. 接受的部分回写

注意：**不能让 AI 的整合自动回写**。否则 substrate 会被"AI 噩梦"污染——错误的 schema 整合 / 虚假联想 / 被 AI 习惯范式同化（[[AI镜像效应]]）。

## 与 [[当场抽取]] 的关系

substrate 做梦不只是 AI 单方面的行为。完整循环里有**双向 [[当场抽取]]**：

```
AI 离线 review → 生成 schema-level 输出
      ↓
用户看到 AI 输出
      ↓
用户当场抽取（"这是做梦机制"这种元层发现）
      ↓
新概念诞生
```

所以 [[涌现接力]] 模型应升级——**抽取主体不限于 AI，4 个角色都可以**，只是触发时机不同。

## 噩梦风险

如果 substrate 会做梦，它就会做噩梦。可能的失败模式：

- AI 编造不存在的 schema（hallucination）
- 把不相关的卡强行 bridge
- 被最近卡过度影响（recency bias）
- 把 substrate 整合进 AI 习惯的范式（[[AI镜像效应]]）
- **anthropomorphic projection**：把人类时间/认知结构错误投射到 substrate 设计本身（这一节最初设计的"周期性触发"就是这种错误）

详见独立卡 [[substrate噩梦]]。

[[校准的不可外包性]] 在 sleep phase 是必须的防火墙。

## 自指意义

这是 substrate 飞轮的最元层实例——**飞轮不只是转，飞轮还在对自身建模**。

研究记忆巩固机制（写卡） → 让 LLM 处理 substrate（被巩固） → 识别这个过程为巩固机制本身（自指）——这是**研究对象 = 研究方法 = 研究主体**的三位一体时刻，在科学史上罕见。

## 关联概念

- [[知识库飞轮]]
- [[涌现接力]]
- [[当场抽取]]
- [[显式化爆发]]
- [[校准的不可外包性]]
- [[主动遗忘]]
- [[非单调可塑性假说]]
- [[反馈累积阈值]]
- [[CMS实现辨析]]
- [[AI镜像效应]]
- [[2026-04-27-kb-index-system]]
