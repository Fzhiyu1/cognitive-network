# Nested Learning 学习指南：从理解到复现 Hope

**讨论者**：fangzhiyu & Claude Opus 4.6
**日期**：2026年2月19日
**背景**：基于论文《Nested Learning: The Illusion of Deep Learning Architectures》（Ali Behrouz et al., Google Research, NeurIPS 2025），为 AI 应用工程师制定的学习与实践路线
**目标**：理解核心思想，小规模复现 Hope 架构
**设备**：48GB MacBook Pro（Apple Silicon，MPS 后端）

---

## 零、这篇论文在讲什么（30秒版本）

传统观点把"模型架构"和"优化器"当两件事。这篇论文说：**它们是同一个东西**——都是"关联记忆"，都在做"把信息压缩到参数里"这件事，只是工作频率不同。

基于这个统一视角，作者设计了 **Hope 架构**——一个能持续学习、不会灾难性遗忘的新模型。

---

## 零点五、为什么要学这个

### 我的背景

AI 应用工程师，目前在雷石（北京）。公司允许自由探索 AI 方向。对 AI 底层原理有浓厚兴趣，目标是成为超级个体——不只是会调 API，而是真正理解模型在做什么。

### 学习目标

纯粹的技术兴趣 + 能力积累。通过复现 Hope 来：
1. **深入理解**现代序列模型的本质（不只是会用，还知道为什么）
2. **掌握模型训练**的实战能力（从架构设计到训练调参）
3. **建立认知优势**——在别人只能调 API 的时候，你能自己设计和训练模型

---

## 一、论文全景地图

```
┌──────────────────────────────────────────────────────────────┐
│                    Nested Learning 论文结构                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  理论部分（理解即可，不需要手推）                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ §1-2  引言 + 预备知识                                │     │
│  │ §3    嵌套学习范式定义                                │     │
│  │ §4    优化器 = 关联记忆（Adam、SGD的新解释）          │     │
│  │ §5    现有架构 = 关联记忆（Attention、RNN的新解释）    │     │
│  │ §6    重新定义：ICL、预训练、持续学习                  │     │
│  └─────────────────────────────────────────────────────┘     │
│                          ↓ 理论指导实践                       │
│  工程部分（需要深入理解并实现）                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ §7    连续记忆系统（CMS）                  ← 要实现   │     │
│  │ §8    自修改 Titans + Hope 架构            ← 要实现   │     │
│  │ §9    实验                                 ← 要复现   │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  附录（参考即可）                                             │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ A  广义嵌套系统定义                                   │     │
│  │ B  Adam/AdaGrad 是关联记忆的完整证明                  │     │
│  │ C  归一化条件下 DGD 的闭式解推导                      │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、核心概念速查表

### 2.1 你必须理解的 5 个概念

#### 概念1：关联记忆（Associative Memory）

**一句话**：一个把"键"映射到"值"的东西。

**类比**：就像一个字典/哈希表，给一个 key 返回一个 value。但这个"字典"是用神经网络参数实现的，是"模糊"的、可以泛化的。

**数学**：M(key) → value，其中 M 的参数通过优化 `L(M(K); V)` 来学习。

**为什么重要**：论文的核心论点是——深度学习里的一切（MLP层、注意力、优化器的动量）本质上都是关联记忆。

#### 概念2：更新频率（Update Frequency）

**一句话**：一个组件多久更新一次参数。

**直觉**：
- Softmax注意力的"记忆"（KV cache）：每个 token 都更新 → 频率最高
- RNN 的隐藏状态：每个 token 更新 → 高频
- CMS 中的低频 MLP：每 2000 个 token 更新一次 → 低频
- 标准 MLP 层的权重：预训练后冻结 → 频率为 0

**为什么重要**：频率决定了组件在嵌套系统中的"层级"。高频 = 短期记忆，低频 = 长期记忆。这是 CMS 的设计基础。

#### 概念3：嵌套系统（Nested System）

**一句话**：多个不同频率的优化问题套在一起。

**举例**：训练一个带动量的 SGD
- 第1级（高频）：动量项在每个 batch 更新，把梯度压缩到自己的参数里
- 第2级（低频）：模型权重基于动量的输出进行更新

**类比**：就像公司组织架构——员工（高频）每天处理任务，经理（中频）每周汇总调整方向，CEO（低频）每季度做战略决策。不同层级处理不同时间尺度的信息。

#### 概念4：Delta 梯度下降（DGD）

**一句话**：比标准梯度下降多了一个"遗忘"机制。

**标准 GD**：`W_new = W_old - lr * 梯度 * 输入^T`
**DGD**：`W_new = W_old * (I - α * 输入 * 输入^T) - lr * 梯度 * 输入^T`

多出来的 `W_old * (I - α * x * x^T)` 就是**遗忘项**——在当前输入的方向上衰减旧权重，为新信息腾出空间。

**为什么重要**：这是 Hope 中自修改 Titans 的核心更新规则。

#### 概念5：自修改（Self-Modification）

**一句话**：模型不仅学习数据，还学习"怎么学习"。

**标准模型**：W_k, W_v, W_q 在预训练后冻结，遇到新上下文只能靠固定的投影来处理。

**自修改模型**：把 W_k, W_v, W_q 替换为记忆模块 M_k, M_v, M_q，它们在处理每个 token 时也在更新自己。而且每个记忆模块还能**生成自己的训练目标**（自己给自己出题）。

**为什么重要**：这是 Hope 区别于 Transformer 的关键创新。

---

### 2.2 你可以先跳过的概念

| 概念 | 论文位置 | 为什么可以跳 |
|------|---------|-------------|
| 近端梯度下降（Proximal GD）的理论 | §3 定义3-4 | 你只需要知道 GD 可以写成 argmin 形式，不需要推导 |
| Adam = 最优关联记忆的证明 | 附录 B | 纯理论，不影响实现 |
| Sherman-Morrison 引理推导 | 附录 C | DGD 的公式直接用结论就行 |
| Muon 优化器的 Newton-Schulz 推导 | §4.3 | 实现时调库即可 |
| 脑科学类比（Gamma/Beta/Theta波） | §1.1 | 有趣但不影响编码 |

---

## 三、Hope 架构详解（工程视角）

### 3.1 Hope 的整体结构

```
输入 x_t
    │
    ▼
┌──────────────────────────────────┐
│     自修改 Titans（核心序列模型）   │
│                                  │
│  记忆模块 M_k, M_v, M_q,        │
│  M_η（学习率）, M_α（遗忘门）     │
│                                  │
│  每个都是 2层MLP，用 DGD 更新     │
│  容量小，但学习规则复杂            │
└──────────────┬───────────────────┘
               │ 输出 o_t
               ▼
┌──────────────────────────────────┐
│     CMS（连续记忆系统）            │
│                                  │
│  MLP^(f_1) → MLP^(f_2) → ...    │
│  f_1 > f_2 > ... > f_k          │
│  高频块：快速适应，短期记忆         │
│  低频块：缓慢更新，长期知识         │
│  容量大，但学习规则简单            │
└──────────────┬───────────────────┘
               │
               ▼
           输出 y_t
```

### 3.2 自修改 Titans 的前向传播（伪代码）

```python
# 每个记忆模块是一个 2层 MLP + 残差连接
class MemoryModule:
    def __init__(self, d_in):
        self.W1 = nn.Linear(d_in, d_in)
        self.W2 = nn.Linear(d_in, d_in)
        self.act = nn.SiLU()  # 或其他激活函数

    def forward(self, x):
        return x + self.W1(self.act(self.W2(x)))  # 残差连接

# 自修改 Titans 的核心流程
class SelfModifyingTitans:
    def __init__(self):
        # 6 个记忆模块，分别负责 k, v, q, eta, alpha, memory
        self.M_k = MemoryModule(d)
        self.M_v = MemoryModule(d)
        self.M_q = MemoryModule(d)
        self.M_eta = MemoryModule(d)   # 生成学习率
        self.M_alpha = MemoryModule(d)  # 生成遗忘门
        self.M_mem = MemoryModule(d)    # 主记忆

    def forward(self, x_t, prev_states):
        # 1. 用各记忆模块生成 k, v, q, eta, alpha
        k_t = self.M_k(x_t)       # 不是固定的 W_k @ x_t !
        v_t = self.M_v(x_t)
        q_t = self.M_q(x_t)
        eta_t = self.M_eta(x_t)   # 数据依赖的学习率
        alpha_t = self.M_alpha(x_t) # 数据依赖的遗忘门

        # 2. 每个记忆模块生成自己的"潜在值"（自修改的关键）
        v_hat_k = self.M_k(v_t)    # M_k 给自己生成训练目标
        v_hat_v = self.M_v(v_t)
        # ... 其他模块类似

        # 3. 用 DGD + 权重衰减 更新每个记忆模块
        # M_□_new = M_□_old * (alpha * I - eta * k @ k^T) - eta * grad_L
        self.update_memory(self.M_k, k_t, v_hat_k, eta_t, alpha_t)
        self.update_memory(self.M_v, k_t, v_hat_v, eta_t, alpha_t)
        # ...

        # 4. 用主记忆生成输出
        o_t = self.M_mem(q_t)
        return o_t

    def update_memory(self, M, k, v_hat, eta, alpha):
        """DGD 更新规则"""
        # M_new = M_old * (alpha*I - eta * k @ k^T) - eta * grad
        # 具体实现见第四节
        pass
```

### 3.3 CMS 的前向传播（伪代码）

```python
class CMS:
    """连续记忆系统 - 顺序变体"""
    def __init__(self, num_levels, d_model, chunk_sizes):
        # chunk_sizes 例如 [1, 16, 128, 2048]
        # 频率从高到低：每1步更新, 每16步, 每128步, 每2048步
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(num_levels)
        ])
        self.chunk_sizes = chunk_sizes
        self.step = 0

    def forward(self, x):
        # 顺序通过每个 MLP 块
        out = x
        for level in self.levels:
            out = level(out)
        return out

    def should_update(self, level_idx):
        """第 level_idx 层是否应该在当前步更新"""
        return self.step % self.chunk_sizes[level_idx] == 0

    def update_parameters(self, level_idx, loss):
        """只更新到期的 MLP 块的参数"""
        if self.should_update(level_idx):
            # 对 self.levels[level_idx] 的参数执行梯度下降
            pass
```

### 3.4 Hope-Attention 变体（更简单的起步方案）

```
Hope-Attention = Softmax全局注意力（替代自修改Titans）+ CMS
```

如果自修改 Titans 太难实现，可以先做 Hope-Attention：用标准的多头注意力 + CMS。论文实验表明 Hope-Attention 已经优于标准 Transformer。

---

## 四、DGD 更新规则详解（工程实现级）

这是 Hope 中最核心的新公式，需要真正理解。

### 4.1 标准梯度下降 vs DGD

**标准 GD**（你熟悉的）：
```
W_{t+1} = W_t - η * ∇L
```
每次更新只看当前梯度，跟当前权重 W_t 的状态无关。

**DGD**（论文提出的）：
```
W_{t+1} = W_t * (I - α * x_t * x_t^T) - η * ∇_{y_t}L * x_t^T
```

拆开看两部分：
- **遗忘项** `W_t * (I - α * x_t * x_t^T)`：在输入 x_t 的方向上衰减权重
  - 直觉：遇到新数据时，先"腾出空间"
  - 当 α→0 时退化为标准 GD（没有遗忘）
- **写入项** `η * ∇_{y_t}L * x_t^T`：写入新信息
  - 这部分跟标准 GD 一样

### 4.2 PyTorch 实现思路

```python
def dgd_update(W, x_t, grad_y_L, alpha, eta):
    """
    Delta Gradient Descent 更新

    参数:
        W: 当前权重矩阵 [d_out, d_in]
        x_t: 当前输入（假设已归一化）[d_in]
        grad_y_L: 损失对输出的梯度 [d_out]
        alpha: 遗忘率标量
        eta: 学习率标量

    返回:
        W_new: 更新后的权重
    """
    # 遗忘项: W * (I - alpha * x @ x^T)
    # 等价于: W - alpha * (W @ x) @ x^T
    forget = alpha * torch.outer(W @ x_t, x_t)

    # 写入项: eta * grad_y_L @ x^T
    write = eta * torch.outer(grad_y_L, x_t)

    W_new = W - forget - write
    return W_new
```

### 4.3 在自修改 Titans 中的实际使用

```python
def update_memory_module(M_params, k_t, v_hat, eta_t, alpha_t):
    """
    更新记忆模块（2层MLP）的参数

    注意：实际中 M 是 2层MLP，不是单个矩阵
    需要对 MLP 的每一层分别应用 DGD

    简化版（将 MLP 视为单矩阵时）:
    M_new = M_old * (alpha * I - eta * k @ k^T) - eta * (M_old @ k - v_hat) @ k^T
    """
    # 使用 L2 回归损失时的梯度: (M @ k - v_hat)
    pred = M_forward(M_params, k_t)
    error = pred - v_hat

    # DGD 更新
    # 遗忘: alpha * I - eta * k @ k^T  (作用在 M 上)
    # 写入: -eta * error @ k^T
    pass
```

---

## 五、分块并行训练（Chunk-wise Training）

这是实现效率的关键。自修改 Titans 的更新是顺序的（每个 token 依赖上一个），直接实现会很慢。

### 5.1 核心思路

```
序列: [t1, t2, t3, t4, t5, t6, t7, t8, ...]
分块:  [--- chunk 1 ---] [--- chunk 2 ---] ...
       块末状态 S1 ──→  块初状态
```

1. 将输入序列分为大小 C 的块
2. 在每个块末尾，计算下一个块所需的所有状态（k, v, η, α 的生成）
3. **块内的梯度可以并行计算**（因为都基于同一个块初始状态）
4. 块间串行传递状态

### 5.2 两种目标函数的更新公式

**点积相似度**（更简单，先用这个）：
```
L(M; k, v) = -<Mk, v>
梯度 = v @ k^T
M_t = M_{t-1} * (α*I - η * k_t @ k_t^T) - η * v_hat @ k_t^T
```

**L2 回归损失**（更强但更复杂，后期升级）：
```
L(M; k, v) = ||Mk - v||^2
梯度 = (Mk - v) @ k^T
M_t = M_{t-1} * (α*I - η * k_t @ k_t^T) - η * (M_prev @ k_t - v_hat) @ k_t^T
```

---

## 六、复现路线图

### 开发环境

```
硬件：48GB MacBook Pro（Apple Silicon）
框架：PyTorch + MPS 后端
注意：.cuda() 全部换成 .to("mps")
      少数 MPS 不支持的算子会自动 fallback 到 CPU
      速度比 CUDA 慢 2-5 倍，但 48GB 内存比 RTX 4090 的 24GB 大一倍
```

### 算力实际需求（Mac 视角）

| 实验 | 参数量 | 能跑吗 | 预计时间 |
|------|--------|--------|---------|
| DGD 验证（toy 任务） | ~1M | 轻松 | 分钟级 |
| 基础 Hope（合成任务） | 10M | 轻松 | 分钟到小时 |
| Hope 形式语言识别 | 10-50M | 没问题 | 小时级 |
| Hope 小规模语言建模 | 50-100M | 没问题 | 半天到一天 |
| Hope 中等规模验证 | 50-100M | **够用** | 数小时到一天 |
| 200M+ 参数 | 200M+ | 能装下但慢 | 可能要几天 |
| 论文级别（760M-1.3B） | 760M+ | **跑不了** | 需要 GPU 集群 |

**结论**：100M 以内的实验全部可以在 Mac 上完成，足够验证架构正确性和做实际应用。

### Phase 1：跑通前置工作（1-2周）

- [ ] 阅读 Titans 论文（Hope 的前身）
- [ ] 找到 Titans 的开源实现（如有），跑通基础实验
- [ ] 用 PyTorch 实现一个简单的线性关联记忆（Hebbian 规则）
- [ ] 在 toy 任务上验证：记住一组 key-value 对

### Phase 2：实现核心组件（2-3周）

- [ ] 实现 DGD 更新规则（上面的 `dgd_update` 函数）
- [ ] 在简单回归任务上对比 SGD vs DGD
- [ ] 实现 2层MLP 记忆模块 `MemoryModule`
- [ ] 实现基础 Titans 记忆（用 DGD 更新记忆模块，不带自修改）

### Phase 3：实现自修改 Titans（2-3周）

- [ ] 将固定投影 W_k, W_v, W_q 替换为记忆模块
- [ ] 实现自修改机制（记忆模块生成自己的 v_hat）
- [ ] 实现分块并行训练（先用 chunk_size=64）
- [ ] 在小任务上验证：NIAH（大海捞针）

### Phase 4：实现 CMS（1-2周）

- [ ] 实现顺序 CMS（多个不同频率更新的 MLP 块）
- [ ] 实现更新调度器（哪个块在哪一步更新）
- [ ] 测试不同频率组合的效果

### Phase 5：组装 Hope（1-2周）

- [ ] Hope = 自修改 Titans + CMS
- [ ] 或先做 Hope-Attention = Softmax注意力 + CMS（更简单）
- [ ] 在形式语言识别任务上验证（应该达到 100%）

### Phase 6：扩展验证

- [ ] 小规模语言建模（WikiText-2 / TinyStories，50-100M 参数）
- [ ] 持续学习实验（类增量学习，验证 CMS 的抗遗忘能力）
- [ ] 对比：标准 Transformer vs Hope-Attention vs 完整 Hope
- [ ] 如果有 GPU 集群资源：尝试更大规模的实验

---

## 七、论文关键实验结果速查

### 7.1 消融实验（1.3B 参数 / 100B tokens）

告诉你每个组件有多重要：

| 模型变体 | 困惑度 | 推理准确率 | 启示 |
|---------|--------|----------|------|
| **Hope（完整）** | **12.24** | **58.1** | 基准 |
| 去掉 DGD | 13.41 | 56.5 | DGD 很重要 |
| 去掉动量 | 13.58 | 56.9 | 动量有帮助 |
| 去掉权重衰减 | 13.71 | 57.2 | 权重衰减有帮助 |
| 去掉 CMS | 13.04 | 57.3 | CMS 对长期记忆重要 |
| 去掉 k 的内部投影 | 13.77 | 56.9 | k 的自修改重要 |
| 去掉 v 的内部投影 | 13.90 | 55.1 | **v 的自修改最重要** |
| 去掉 q 的内部投影 | 12.19 | 57.4 | q 的自修改影响最小 |

**实现优先级**：v 的自修改 > DGD > k 的自修改 > CMS > 动量 > 权重衰减 > q 的自修改

### 7.2 关键实验结果

| 任务 | Hope 表现 | 对比 |
|------|----------|------|
| 形式语言识别 | **100% 全满分** | Transformer 在部分任务 0% |
| 大海捞针（NIAH） | 无注意力模型最佳 | 优于 Titans |
| Hope-Attention vs Transformer | **Hope-Attention 更好** | CMS 的加持 |
| BABILong（超长上下文） | **10M 长度仍保持性能** | GPT4 在 128K 后崩溃 |
| 语言建模 1.3B | 困惑度 14.39 / 准确率 58.04 | Titans: 15.60 / 56.82 |
| 持续学习（新语言翻译） | 几乎不遗忘 | ICL 灾难性遗忘 |
| MAD 合成基准 | **全面超越 Transformer** | 压缩、复制等任务 |

### 7.3 CMS 配置建议

- 层级越多 → 性能越好（但有边际递减）
- 最低频率 = 2K 是效率和性能的最佳平衡点
- 最低频率越高 → 长期记忆越弱

---

## 八、必读论文清单

按优先级排序：

| 优先级 | 论文 | 为什么要读 | 阅读深度 |
|--------|------|----------|---------|
| **P0** | **Titans** (Behrouz et al. 2025) | Hope 的直接前身，理解了 Titans 就理解了 Hope 的 70% | 精读 |
| P1 | Attention is All You Need (Vaswani 2017) | 如果你还没精读过 Transformer 原论文 | 精读 |
| P1 | DeltaNet (Yang et al.) | Delta 规则是 DGD 的灵感来源 | 核心部分 |
| P2 | Miras (Behrouz et al.) | 联想记忆统一框架 | 概览 |
| P2 | Linear Attention (Katharopoulos 2020) | 线性注意力 = 快速权重编程器 | 概览 |
| P3 | MAML (Finn et al. 2017) | 元学习的经典方法，理解"初始化知识转移" | 概念 |
| P3 | Adam 优化器 (Kingma & Ba 2015) | 理解 Adam 的数学 | 公式部分 |

---

## 九、可能遇到的坑和建议

### 9.1 工程实现上的坑

1. **记忆模块的初始化**：论文说 M_□,0 要通过元学习（meta-learning）跨所有序列优化。实际上就是让这些初始参数也参与预训练的梯度更新。
2. **分块训练的梯度截断**：块间状态传递时梯度是截断的（类似 TBPTT），不要试图跨块反传。
3. **数值稳定性**：DGD 的遗忘项 `(I - α * x * x^T)` 如果 α 太大会导致数值爆炸，需要小心调参。
4. **CMS 的不同频率更新**：需要自己实现更新调度器，标准的 PyTorch 训练循环不直接支持。

### 9.2 Mac MPS 后端注意事项

1. **设备设置**：`device = torch.device("mps")`
2. **不支持的算子**：遇到 MPS 不支持的操作，用 `.to("cpu")` 临时转移计算，完成后再转回
3. **内存优势**：48GB 统一内存意味着不存在"显存不够"的问题（100M 以内模型）
4. **速度预期**：比 RTX 3090 慢 2-5 倍，小实验可以接受，大规模训练不适合
5. **梯度检查点**：如果内存紧张可以用 `torch.utils.checkpoint` 用时间换空间

### 9.3 简化复现策略

如果时间有限，优先级：
1. **DGD 实现 + toy 验证**（核心创新，代码量小，几小时搞定）
2. **CMS 单独实现**（多频率 MLP 链，概念清晰）
3. **Hope-Attention = 标准注意力 + CMS**（比完整 Hope 简单很多，但已经很强）
4. 完整 Hope（最复杂，需要自修改 Titans + 分块训练 + CMS 全部到位）

---

## 十、论文核心思想精华（理解层面）

以下是论文理论部分的精华总结。你不需要推导，但理解这些会帮助你做出更好的实现决策。

### 10.1 统一视角：一切都是关联记忆

| 组件 | 键（Key） | 值（Value） | 记忆（Memory） |
|------|----------|------------|---------------|
| MLP 层 | 输入 x | 目标输出 | 权重 W |
| 反向传播 | 每层输入 x_{l-1} | 局部惊奇信号 δ_l | 权重 W_l |
| SGD 动量 | 梯度 | 梯度的全局属性 | 动量项 m |
| Adam | 梯度 | 梯度的方差 | 一阶/二阶矩 |
| 线性注意力 | k_t | v_t | 快速权重矩阵 M_t |
| Softmax注意力 | K (所有key) | V (所有value) | KV cache（非参数） |

### 10.2 预训练 = 超长上下文的 ICL

论文最有趣的观点之一：预训练不是一个特殊的过程，它就是上下文学习——只不过"上下文"是整个训练数据集。模型参数存储的就是对这个超长上下文的压缩。

### 10.3 灾难性遗忘 = 压缩的必然代价

网络容量有限，要记住新东西就必须遗忘旧东西。CMS 的解法不是消除遗忘，而是**让遗忘分层进行**——高频层快速遗忘旧的、适应新的；低频层保持旧知识。通过反向传播，被高频层遗忘的知识可以从低频层"恢复"。

### 10.4 循环模型 = MLP + 新计算层级

从 NL 视角，RNN/线性注意力等循环模型本质上就是 MLP 块加了一个高频更新的计算层级。所谓的"混合架构"（Transformer + RNN），其实就是给部分 MLP 块加了上下文学习能力。

---

## 附：学习进度追踪

| 阶段 | 状态 | 备注 |
|------|------|------|
| Phase 1: 前置工作 | ⬜ 未开始 | |
| Phase 2: 核心组件 | ⬜ 未开始 | |
| Phase 3: 自修改 Titans | ⬜ 未开始 | |
| Phase 4: CMS | ⬜ 未开始 | |
| Phase 5: 组装 Hope | ⬜ 未开始 | |
| Phase 6: 扩展验证 | ⬜ 未开始 | |
