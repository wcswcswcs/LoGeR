# LoGeR 中 TTT 层的原理、代码路径与先验引导的更新控制（超详细版）

> 这份文档的目标有两个：
>
> 1. 把 **LoGeR 中 TTT 层的工作原理** 讲清楚，而且尽量贴近当前代码，而不是只停留在论文的抽象描述。
> 2. 在 **不改变 TTT 基本结构** 的前提下，说明如何通过 **外部先验** 去控制 TTT 的更新行为，例如语义先验、几何先验、动态性先验、置信度先验等。
>
> 这份文档不再以 TTSA3R 为主要出发点，而是直接从 **LoGeR 的 TTT 代码本体** 出发，建立一个更一般、更稳的“先验引导更新”框架。这样做的好处是：
>
> - 逻辑更清楚；
> - 和当前代码更贴；
> - 后续改代码时更容易落地；
> - 不会被某篇 follow-up 的表面形式绑住。

---

## 0. 这份文档要回答什么问题

### 0.1 核心目标

这份文档聚焦两件事。

第一，LoGeR 里的 TTT 层到底在做什么，原理是什么，代码里是如何组织和调用的。

第二，如何在 **不改变 TTT 基本结构** 的前提下，引入外部先验去控制它的更新行为。

这里的“先验”不限定来源。它可以是：

- token-level 语义先验；
- 几何置信度；
- 区域属性，例如天空、树木、地面、建筑；
- overlap / boundary 类型；
- 动态性标签；
- 纹理质量；
- 视角稳定性；
- 外部 detector / segmentor 的标签或分数；
- head-level、layer-level 或 chunk-level 的额外控制量。

### 0.2 最终想得到的能力

如果这份文档足够清楚，最终我们应该能直接回答下面这些问题：

- 为什么某些 token 更应该被写进 TTT memory；
- 为什么某些 token 应该被弱化，甚至几乎不参与写入；
- 如何让天空、树木这类语义区域对 TTT 更新贡献更小；
- 如何让高价值的几何锚定区域贡献更大；
- 如何让某些 head 或某些 layer 的写入更保守；
- 如何在 chunk 整体质量较差时减少甚至跳过写入。

### 0.3 这份文档的组织原则

整篇文档遵循下面这条顺序：

1. 先把代码位置和执行路径说清楚；
2. 再解释 TTT 的原理；
3. 再解释当前代码到底在优化什么目标；
4. 再解释单个 token 如何影响 TTT 的 fast weights；
5. 再解释所有 token 如何聚合成一次 TTT update；
6. 最后再讨论如何把一般先验接到这条更新链上。

这样做是为了避免一个常见问题：**还没搞清楚 TTT 本身怎么更新，就开始讨论各种自适应权重**。那样很容易把“原始梯度贡献”“预聚合更新矩阵”“最终参数变化”混在一起。

### 0.4 阅读建议

如果你已经大致知道 TTT 是什么，但总觉得“代码里到底怎么更新 `w0 / w1 / w2`”还没想明白，建议按下面顺序读：

- 第 1 节：先知道代码位置；
- 第 3 节：先明确代码级目标；
- 第 4 节：单 token 如何影响 weights；
- 第 5 节：所有 token 如何聚合成一次 update；
- 第 8 节：先验如何接进来。

如果你只看一个章节，我建议先看 **第 4 节**。那一节是这份文档的技术核心。

---

## 1. LoGeR 中 TTT 的代码位置与功能介绍

这一节先只做“代码地图”，把关键位置钉死。先知道东西在哪里，后面原理和设计才不会飘。

### 1.1 `loger/models/ttt.py`：TTT 内核

LoGeR 的 TTT 内核主要在 `loger/models/ttt.py` 里，关键对象有四个。

#### 1.1.1 `TTTOperator`

位置：`ttt.py:12`

`TTTOperator` 是一个简单的 `namedtuple`，用来描述 TTT 在一次前向中执行哪些片段、是做 apply 还是 update。

它的字段是：

- `start`：片段起点；
- `end`：片段终点；
- `update`：这一段是否执行 update；
- `apply`：这一段是否执行 apply。

它本身不做数学计算，但它决定了 **同一段 token 序列先读 memory 还是先写 memory**。

在当前 LoGeR 里，真正使用的顺序是“先 apply，再 update”。后面在 `pi3.py` 里会看到具体定义。

---

#### 1.1.2 `zeropower_via_newtonschulz5(...)`

位置：`ttt.py:35-70`

这个函数非常关键。它不是一个普通的归一化函数，而是对 chunk 聚合后的更新矩阵做一种近似正交化 / zeroth power 变换。

从使用层面上，你可以先把它理解成：

> 它把“原始聚合更新矩阵”变成“真正准备写到 fast weights 上的方向”。

为什么这一步重要？因为它意味着：

- token 的原始贡献先被求和；
- 再一起经过这个 operator；
- 于是 token 对最终 applied update 的贡献就不再是线性可分的。

这也是为什么后面我们会反复强调：

> 当前最自然、最可解释的分析对象，是 token 对 **pre-zeropower 更新矩阵** 的贡献，而不是 token 对最终参数变化的精确归因。

换句话说，`zeropower` 是把“可线性拆解的 token 贡献”转换成“真正用于更新参数的整体方向”的关键分界线。

---

#### 1.1.3 `fast_weight_swish_glu_weight_norm_mini_batch_apply(...)`

位置：`ttt.py:76-175`

这是 TTT 的核心算子。它同时负责两件事：

- 根据当前 fast weights，对输入做 apply；
- 根据当前 token 序列，对 fast weights 做 update。

这个函数里真正发生了下面这些事情：

1. 接收当前的 `w0 / w1 / w2`；
2. 接收当前 chunk 的 `q / k / v`；
3. 接收 token-wise 的 `lr0 / lr1 / lr2`；
4. 根据 `TTTOperator` 决定先 apply 还是先 update；
5. 在 update 分支里构造三条分支的原始聚合矩阵；
6. 调用 `zeropower_via_newtonschulz5(...)` 得到 applied direction；
7. 可选融合 momentum；
8. 写回 `w0 / w1 / w2`；
9. 在 apply 分支里计算输出 `oi`；
10. 最后返回：
   - apply 输出 `output`
   - 更新后的 `w0 / w1 / w2`

这是我们后面讨论“单个 token 如何影响 TTT weights”的真正出发点。

---

#### 1.1.4 `FastWeightGluMLPMultihead.forward(...)`

位置：`ttt.py:178-318`

这个类是 TTT 模块的封装入口。它做的事情包括：

- 把输入整理成 token 序列；
- 通过 `to_qkv` 生成 `q / k / v`；
- 对 `q / k` 做归一化；
- 通过 `lr_fc` 得到 token-wise 的 `lr0 / lr1 / lr2`；
- 从 `info` 中取旧的 `w0 / w1 / w2`；
- 调用核心算子 `fast_weight_swish_glu_weight_norm_mini_batch_apply(...)`；
- 对输出做 `o_norm` 和 `c_proj`；
- 返回当前 TTT 的 apply 输出以及新的 fast weights。

从工程角度看，后面如果我们要把“先验引导更新”接进去，最自然的入口有两个：

- 进入 `lr0 / lr1 / lr2` 这条 token-wise 系数链；
- 或者在更底层直接改 token 对预聚合更新矩阵的贡献。

---

### 1.2 `loger/models/pi3.py`：TTT 如何接进主干

TTT 内核在 `ttt.py`，但 LoGeR 整体怎么用它，是在 `pi3.py` 里定义的。

#### 1.2.1 `self.ttt_layers` 的创建

位置：`pi3.py:220-251`

`pi3.py` 会根据 `ttt_insert_after` 创建多个 TTT layer：

```python
self.ttt_layers = nn.ModuleList([
    FastWeightGluMLPMultihead(...)
    for _ in self.ttt_insert_after
])
```

这一段很重要，因为它意味着：

> LoGeR 不是只有“一组全局 fast weights”，而是每个 TTT 插入层各自有一套 `w0 / w1 / w2` 状态。

也就是说，第 3 个插入层的 `w0/w1/w2` 和第 10 个插入层的 `w0/w1/w2` 是两套不同的 memory，它们不会共享。

因此后面文档中的很多量都必须带 layer index，不能偷懒写成全局一个对象。

---

#### 1.2.2 `apply-then-update` 的顺序定义

位置：`pi3.py:248-250`

`self.ttt_op_order` 被设置成：

```python
self.ttt_op_order = [
    TTTOperator(start=0, end=None, update=False, apply=True),
    TTTOperator(start=0, end=None, update=True, apply=False),
]
```

这说明当前 LoGeR 的 TTT 语义是：

> **先用已有 fast weights 读取 memory，再用当前 chunk 写入 fast weights。**

这和很多人直觉中的“先更新再读出”不一样，所以必须提前说明。

为什么这种顺序是合理的？  
因为如果你当前 chunk 先 update，再 apply，就等于“先把当前 chunk 的信息写进 memory，再立刻用这份 memory 去处理同一个 chunk”。而 LoGeR 现在的设计更偏向：

- 当前 chunk 先利用 **之前 chunk 累积下来的 memory**；
- 当前 chunk 的信息再写进去，供后面的 chunk 使用。

这更符合 chunk-wise streaming memory 的语义。

---

#### 1.2.3 decode 时如何调用 TTT

位置：`pi3.py:410-447`

在 `decode(...)` 里，当走到某个 TTT 插入层时，会做几件事：

1. 找到当前层在 `ttt_insert_after` 里的索引 `layer_idx`；
2. 从 `ttt_state` 中取出该层已有的 `w0 / w1 / w2`；
3. 构造 `info`，里面包括：
   - `ttt_op_order`
   - 旧的 `w0 / w1 / w2`
4. 调用：
   ```python
   ttt_output, output = self.ttt_layers[layer_idx](tokens_in, info)
   ```
5. 把 TTT apply 输出乘 `gate_scale`，再和当前 hidden 做残差相加：
   ```python
   update_term = ttt_output * gate_scale
   tokens_out = update_term + tokens_post
   ```
6. 把新的 `w0 / w1 / w2` 写进 `ttt_output_info`

从数据流上看，TTT 不是“旁路逻辑”，而是真实参与了主干前向，并且把跨 chunk 的 state 滚动了下去。

---

#### 1.2.4 `w0 / w1 / w2` 如何跨 chunk 传递和 reset

位置：`pi3.py:643-669`、`pi3.py:697-735`

在 `forward(...)` 的 window / chunk 逻辑中：

- `w0 / w1 / w2` 会作为列表按层维护；
- 每处理完一个 chunk，会把更新后的列表写回；
- 如果触发 `reset_every`，会把这些 fast weights 清空。

相关代码大意是：

```python
w0 = [None] * len(self.ttt_insert_after)
w1 = [None] * len(self.ttt_insert_after)
w2 = [None] * len(self.ttt_insert_after)
```

处理完一个 chunk 后：

```python
w0, w1, w2 = ttt_output_info["w0"], ttt_output_info["w1"], ttt_output_info["w2"]
```

触发 reset 时：

```python
w0 = [None] * len(self.ttt_insert_after)
w1 = [None] * len(self.ttt_insert_after)
w2 = [None] * len(self.ttt_insert_after)
```

这意味着：

- TTT 的“记忆”不是单步的；
- 它会跨 chunk 持续存在；
- reset 是一种显式打断 memory 的手段。

---

### 1.3 一个必须先讲清楚的事实：`turn_off_ttt` 不是 write gate

这一点特别重要。

在 `pi3.py` 中，`turn_off_ttt` 只会把 TTT residual 注回 hidden 的 `gate_scale` 置零。这里的 **residual** 指的是：

> TTT apply 输出经过一个门控因子之后，加回当前主干 hidden 表示的那条残差支路。

也就是说，`turn_off_ttt` 关掉的是：

- **“TTT 输出注回当前 hidden”** 这条路径。

但它并不会阻止：

- TTT layer 继续执行；
- `w0 / w1 / w2` 继续在 update 分支里被改写；
- 更新后的 fast weights 继续写回 state。

所以：

> `turn_off_ttt` 不是 memory write gate，而是 output injection gate。

这个结论对后面的方法设计非常关键，因为它说明：

> 如果要控制“TTT 写多少”，必须进入 `ttt.py` 的 update 路径；只改 `pi3.py` 外层 gate 是不够的。

---

## 2. TTT 的基本原理：它到底在学什么、记什么

这一节只讲原理，不讲新设计。

### 2.1 从显式 KV 记忆到参数化记忆

最直观的对比对象是 attention 里的 KV cache。

在普通 attention 里，历史记忆是显式存下来的：

- 每来一个 token，会生成新的 key 和 value；
- 之后 query 会显式去访问所有历史 key/value；
- 序列越长，历史缓存越大。

这种方式的好处是：

- 记忆保真；
- 信息不需要强行压缩；
- 局部和全局的访问都比较自然。

但坏处也很明显：

- 长序列时代价越来越大；
- KV cache 占用越来越高；
- 对极长视频流不太友好。

TTT 想做的是另一件事：

> 不再把历史以显式 KV 的方式全部存着，而是把历史压缩进一小组可更新的 fast weights 里。

于是，原来“查显式 KV”的过程，变成：

- 用当前 chunk 的信息去更新 fast weights；
- 后续再通过 fast weights 去读出记忆。

所以 TTT 的“记忆”不是一个越来越大的缓存，而是一组固定大小、但会持续变化的小参数。

你可以把它理解成：

- Attention / KV cache：**把每条历史都直接存着**
- TTT：**把历史的统计结构压缩进一个小网络的参数里**

LoGeR 之所以要这么做，是因为它要处理很长的视频序列。  
如果纯靠显式历史，代价太高；如果纯靠一个 RNN 式 state，又可能表达能力不够。  
TTT 试图在“容量固定”与“表达能力仍然较强”之间取一个平衡。

---

### 2.2 TTT 的两个动作：apply 与 update

TTT 只做两件事。

#### apply：读 memory

apply 的意思是：

> 在当前时刻，用已有的 fast weights 去读 memory。

如果把 fast-weight module 写成函数 $f_W(x)$，那么 apply 可以写成：

$$
o = f_W(q)
$$

其中：

- $q$ 是 query；
- $o$ 是 memory 对当前 query 的 readout。

在 LoGeR 里，这个输出会经过 `o_norm`、`c_proj` 等处理，再通过门控残差注回主干 hidden。

#### update：写 memory

update 的意思是：

> 用当前 chunk 的信息去改写 fast weights。

直觉上，它做的是：

- 让当前 memory 在输入 key 时，能更好地预测对应的 value；
- 这样以后遇到相似 query 时，就能从 memory 中读出更有用的信息。

所以：

- apply 更像“读记忆”
- update 更像“写记忆”

这两个动作是 TTT 的最基本语义。

---

### 2.3 在 LoGeR 中，TTT 不是 frame-wise，而是 chunk-wise

这点非常容易误解。

LoGeR 里的 TTT 不是“每帧单独更新一个 state token”。

LoGeR 做的是：

- 先把一个 chunk 的所有帧展开成一个 token 序列；
- 在每个 TTT layer 上，聚合这个 chunk 内所有 token 的贡献；
- 然后再对 fast weights 做一次 update。

所以，LoGeR 里 TTT 的更新粒度是：

> **一个 layer、一个 chunk，一次聚合式 update。**

而不是：

> 一帧一个 update。

这也是后面为什么“token contribution $\rightarrow$ chunk 聚合 $\rightarrow$ pre-zeropower 更新矩阵”这条链如此重要的原因。

如果一个 chunk 有 $T$ 帧，每帧有 $P$ 个 patch token，再加上 special tokens，那么总 token 数大约是：

$$
L \approx T \times (P + \text{special tokens per frame})
$$

LoGeR 当前实现里，每帧会额外拼进：

- 5 个 register tokens；
- 1 个 role token（从 3 个 role embeddings 中按 overlap 位置选一个）。

所以粗略地说，每帧是：

$$
P + 6
$$

个 token；整个 chunk 展平后大约是：

$$
L \approx T(P+6)
$$

TTT 更新就是基于这整个长度为 $L$ 的 token 序列做 chunk 聚合。

---

### 2.4 LoGeR 里 fast weights 的 function form

LoGeR 里的 TTT 不是一个简单的线性 map，而是一个小的 SwiGLU MLP。

它最核心的形式是：

$$
f_W(x)=\big(\mathrm{SiLU}(xW^{(0)}) \odot (xW^{(2)})\big)W^{(1)}
$$

这里：

- $W^{(0)}$ 对应 `w0`；
- $W^{(1)}$ 对应 `w1`；
- $W^{(2)}$ 对应 `w2`。

为了可读性，我们把中间量写出来：

$$
g = xW^{(0)}
$$

$$
h = xW^{(2)}
$$

$$
u = \mathrm{SiLU}(g) \odot h
$$

$$
f_W(x)=uW^{(1)}
$$

这意味着：

- `w0` 控制 gate 分支；
- `w2` 控制另一条 hidden / content 分支；
- `w1` 负责把两者融合后的 hidden 投到输出空间。

它比简单的线性 memory 更强，因为它允许 memory 不是一个单纯线性表，而是一个带门控的、可表达更复杂映射的小网络。

---

## 3. 当前代码到底对什么目标在更新 TTT

这一节回答一个根本问题：

> 当前 `ttt.py` 到底是在朝什么方向更新 fast weights？

### 3.1 当前代码里的目标信号是什么

当前代码里有一个非常重要的注释：

`Fixed objective: neg_dot_product (gradient ascent)`

这句话给了我们一个很强的信号：

> update 分支的本质目标，可以理解成：让当前 memory 对 key 的读出，和对应的 value 更对齐。

因此，对单个 token，最自然的数学抽象是：

$$
\mathcal S(W;k,v)=\langle f_W(k), v\rangle
$$

这里：

- $k$：当前 token 的 key；
- $v$：当前 token 的 value；
- $f_W(k)$：当前 memory 对这个 key 的读出；
- $\langle \cdot, \cdot \rangle$：点积。

这个式子表达的意思很简单：

> 如果当前 memory 对 key 的读出和 value 越对齐，点积越大；
> 那么更新就应该沿着让这个点积更大的方向去走。

这不是凭空猜一个目标，而是对当前代码注释和实现最自然的数学抽象。

---

### 3.2 “负点积 / 对齐目标”

如果你更习惯“loss”的写法，也可以把上面的目标写成：

$$
\mathcal L(W;k,v) = -\langle f_W(k), v\rangle
$$

然后做 gradient descent。

这和直接最大化 $\mathcal S$ 做 gradient ascent 是等价的：

$$
\max_W \mathcal S(W;k,v)
\quad\Longleftrightarrow\quad
\min_W \mathcal L(W;k,v)
$$

#### 为什么这里不是“负余弦相似度”

点积和余弦不是一回事。

$$
\langle y, v\rangle = \|y\|\,\|v\|\cos\theta
$$

所以点积同时受两部分影响：

- 方向是否一致；
- 向量范数大小。

除非先把两个向量都归一化，否则点积不等于余弦相似度。

因此，当前代码注释里的 `neg_dot_product` 更自然地对应：

- **负点积**
- 而不是 **负余弦相似度**

#### 为什么这一点很重要

因为这会影响你如何理解 TTT update 的本质：

- 如果是 cosine，更偏“方向对齐”
- 如果是 dot product，则既看方向，也看幅值

当前代码更接近后者。

---

### 3.3 apply 时的输出 vs update 时的预测

这里很容易混淆，所以单独写清楚。

#### update 时

当输入是 key $k$ 时：

$$
y_{\text{pred}} = f_W(k)
$$

这里的 $y_{\text{pred}}$ 不是最终输出给主干的 readout，而是：

> 当前 memory 对 value 的预测。

它被拿来和 $v$ 对齐，从而构造 update 信号。

#### apply 时

当输入是 query $q$ 时：

$$
o_{\text{apply}} = f_W(q)
$$

这里的 $o_{\text{apply}}$ 才是：

> TTT 真正输出给主干的 memory readout。

所以同一个 fast-weight function $f_W(\cdot)$ 有两种使用方式：

- $f_W(k)$：update 时的 value prediction；
- $f_W(q)$：apply 时的 memory output。

这两个记号一定要区分，否则后面推导时很容易把“用于构造梯度的读出”和“真正用于残差注入的输出”混在一起。

---

## 4. 单个 token 是怎么影响 TTT weights 的

这一节是全文最重要的技术核心。目标是回答：

> 单个 token 如何影响 `w0 / w1 / w2`？

为了把这一节讲清楚，我会先统一记号和 shape 约定，再分别推导 `w1 / w0 / w2`。

### 4.1 notation：先定义所有符号

这一节的符号以后会一直用到，所以先统一写清楚。

#### 索引

- $m$：第 $m$ 个 chunk；
- $l$：第 $l$ 个 TTT layer；
- $i$：当前 chunk 内第 $i$ 个 token；
- $h$：第 $h$ 个 head；
- $r\in\{0,1,2\}$：fast-weight 分支编号，分别对应 `w0 / w1 / w2`。

#### fast weights

$$
W_{m,l}^{(r,h)}
$$

表示：在处理第 $m$ 个 chunk 之前，第 $l$ 层、第 $h$ 个 head、分支 $r$ 的 fast weights。

为了便于推导，我们先省略 $m,l,h$ 这些索引，先只看单层、单头、单 token 的情况。  
等推导完成后，再把索引加回去。

#### 当前 token 的 key / value

$$
k,\quad v
$$

表示某个固定 token 在当前 head 上的 key / value 行向量。

#### 行向量约定（非常重要）

为了和当前代码中的 batched matmul 形式保持一致，本文默认 token 向量按 **行向量** 理解。

也就是说：

- $k$ 是 $1\times d$；
- $u$ 是 $1\times d_h$；
- $v$ 是 $1\times d$；
- $a$、$b$ 也是行向量。

在这个约定下，像

$$
u^\top v
$$

这样的式子不是标量内积，而是：

$$
(d_h\times 1)(1\times d)=d_h\times d
$$

也就是一个矩阵。  
换句话说，这里写出来的都是 **outer product**，只是为了和代码 shape 对齐，保留了这种写法。

---

### 4.2 从前向中间量开始

对单个 token、单个 head，当前 TTT 前向可以写成：

$$
g = kW^{(0)}
$$

$$
h = kW^{(2)}
$$

$$
u = \mathrm{SiLU}(g)\odot h
$$

$$
y=uW^{(1)}
$$

这里每个量的含义是：

- $k$：当前 token 的 key；
- $g$：gate 分支的 pre-activation；
- $h$：第二条 hidden / content 分支的线性输出；
- $u$：门控后的 hidden；
- $y$：当前 memory 对这个 key 的 value prediction。

这四个式子是后面所有梯度推导的出发点。

---

### 4.3 为什么 `w1 / w0 / w2` 的单 token 原始更新项长那个样子

这一节最重要的原则是：

> 下面写出来的式子，都是从前向和对齐目标用链式法则严格推出的“单 token 原始梯度贡献”。

它们还不是最终参数更新。

---

#### 4.3.1 对 `w1`

我们先看：

$$
y = uW^{(1)}
$$

目标是：

$$
\mathcal S = \langle y,v\rangle
$$

把 $y$ 代进去：

$$
\mathcal S = \langle uW^{(1)}, v\rangle
$$

如果用 trace / differential 的方式写，可以得到：

$$
\mathcal S = \mathrm{tr}(uW^{(1)}v^\top)
= \mathrm{tr}(v^\top uW^{(1)})
$$

于是对 $W^{(1)}$ 求导，得到：

$$
\frac{\partial \mathcal S}{\partial W^{(1)}} = u^\top v
$$

这个结果怎么理解？

- `w1` 是一个矩阵参数；
- 单个 token 要对它提出一个更新建议，这个建议也应该是一个矩阵；
- $u^\top v$ 正好是一个 rank-1 矩阵，也就是一个 outer product。

直觉上，它表示：

> 当前 token 希望把 `w1` 往“让 hidden $u$ 更容易产生 value $v$”的方向推。

---

#### 4.3.2 对 `w0`

现在看：

$$
g = kW^{(0)}
$$

同时又有：

$$
u = \mathrm{SiLU}(g)\odot h
$$

所以从目标 $\mathcal S$ 反传回来，会先得到对 $g$ 的梯度。为了不让记号太重，我们记：

$$
a = \frac{\partial \mathcal S}{\partial g}
$$

它在代码里对应 `dgate_before_act`。

然后由：

$$
g = kW^{(0)}
$$

可得：

$$
\frac{\partial \mathcal S}{\partial W^{(0)}} = k^\top a
$$

这就是单个 token 对 `w0` 的原始更新项。

直觉上，`w0` 是 gate 分支的参数，所以：

- $k$ 决定当前 token 的输入方向；
- $a$ 决定 gate 分支上“应该往哪里调”；
- 二者的 outer product 就构成了这一 token 对 gate 分支的更新建议。

---

#### 4.3.3 对 `w2`

同理，由：

$$
h = kW^{(2)}
$$

先记对 $h$ 的梯度为：

$$
b = \frac{\partial \mathcal S}{\partial h}
$$

它在代码里对应 `dhidden_before_mul`。

于是：

$$
\frac{\partial \mathcal S}{\partial W^{(2)}} = k^\top b
$$

这就是单个 token 对 `w2` 的原始更新项。

直觉上：

- `w2` 负责另一条 hidden / content 分支；
- $b$ 表示这条分支上希望调整的方向；
- 所以 $k^\top b$ 就是当前 token 对 content 分支的 rank-1 更新建议。

---

### 4.4 什么叫“原始更新贡献”

为了统一记号，我们把单个 token 对某一分支的原始更新矩阵记成：

$$
J_{m,l,i}^{(r,h)}
$$

它的语义是：

> 第 $m$ 个 chunk、第 $l$ 个 TTT layer、第 $i$ 个 token、在第 $h$ 个 head 上，
> 对分支 $r$ 的 fast weights 提出的单 token 原始更新矩阵。

于是三条具体形式就是：

$$
J_{m,l,i}^{(1,h)} = u_{m,l,i}^{(h)\top} v_{m,l,i}^{(h)}
$$

$$
J_{m,l,i}^{(0,h)} = k_{m,l,i}^{(h)\top} a_{m,l,i}^{(h)}
$$

$$
J_{m,l,i}^{(2,h)} = k_{m,l,i}^{(h)\top} b_{m,l,i}^{(h)}
$$

这里的“原始”指的是：

- 它还没有乘 token-wise coefficient $\eta$；
- 还没有和别的 token 求和；
- 还没有过 `zeropower`；
- 还没有过 renorm。

所以它是最干净、最容易解释、也最容易和 token 一一对应的对象。

---

### 4.5 原始更新贡献 vs 最终更新的区别

这一步必须讲透，因为后面所有“先验控制”都建立在这个区分上。

#### 原始更新贡献

原始更新贡献是：

$$
J_{m,l,i}^{(r,h)}
$$

它是单个 token 提出的原始梯度项，仍然是线性可分的。

#### 最终更新

代码里的最终更新不是简单地把这些 $J_i$ 直接加到参数上，而是：

1. 先乘 token-wise 系数 $\eta$；
2. 再在 chunk 内求和；
3. 再经过 `zeropower`；
4. 再可选加 momentum；
5. 最后 renorm。

因此最终参数更新可以写成：

$$
W_{m+1,l}^{(r,h)} - W_{m,l}^{(r,h)}
$$

但这个量已经不能再被精确地线性拆成“第 $i$ 个 token 贡献多少”。

原因很简单：一旦经过非线性 operator，就一般有：

$$
\mathcal U(A+B) \neq \mathcal U(A) + \mathcal U(B)
$$

所以：

> 我们不能直接说 token 对最终 $W_{m+1}-W_m$ 的精确贡献是多少；
> 我们最自然能分析和控制的，是 token 对 pre-aggregation / pre-zeropower 更新链的贡献。

---

## 5. 当前代码中，所有 token 的贡献是如何聚合成一次 TTT update 的

### 5.1 token-wise coefficient：$\eta$

当前代码中的 `lr0 / lr1 / lr2` 对应：

$$
\eta_{m,l,i}^{(r,h)}
$$

很多人看到它容易本能地把它叫成 learning rate，但更准确地说，它不是普通 optimizer 的全局学习率，而是：

> **每个 token、每个分支、每个 head 的 pre-zeropower 重加权系数。**

换句话说，它控制的是：

- 当前 token 在 chunk 聚合时占多大权重；
- 而不是最终参数范数会改变多少。

#### 它在代码里怎么来的

在 `FastWeightGluMLPMultihead.forward(...)` 中：

```python
lr = self.lr_fc(x.float())
lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

lr0, lr1, lr2 = rearrange(
    lr, "b l (lrs h d) -> lrs (b h) l d",
    lrs=3, h=self.num_heads
)
```

所以：

- `lr_fc` 从每个 token 的特征预测一组系数；
- `softplus` 保证它们是正值；
- 再拆成三条分支：
  - `lr0`
  - `lr1`
  - `lr2`

这就是 $\eta_{m,l,i}^{(r,h)}$ 的来源。

---

### 5.2 pre-zeropower 聚合矩阵

把所有 token 的原始贡献乘上 $\eta$ 后，对当前 chunk 求和，得到：

$$
\tilde G_{m,l}^{(r,h)}
=
\sum_{i\in C_m}
\eta_{m,l,i}^{(r,h)} J_{m,l,i}^{(r,h)}
$$

这里：

- $C_m$：第 $m$ 个 chunk 的 token 集合；
- $\tilde G_{m,l}^{(r,h)}$：当前 chunk 对 layer $l$、head $h$、branch $r$ 的 **pre-zeropower 聚合更新矩阵**。

这条式子是后面最重要的操作对象。

它的直觉含义是：

> 每个 token 先提出自己的原始更新建议 $J_i$；
> 再乘当前代码已有的 token 系数 $\eta_i$；
> 然后所有 token 加起来，形成这一次 chunk 的原始整体更新方向。

---

### 5.3 为什么当前代码不显式 materialize 每个 token 的 $J_i$

虽然从数学上我们可以定义每个 token 的 $J_i$，但当前代码并不会真的把它们一个个存成矩阵列表，而是直接通过 batched matmul 算 chunk 级聚合。

比如对 `w1` 分支，代码是：

```python
(hidden * lr1i).transpose(-1, -2) @ vi
```

它本质上就是：

$$
\sum_i \eta_i^{(1)} J_i^{(1)}
$$

只是用了一次 batched matmul 来完成。

这样做有三个原因：

1. 数学上完全等价；
2. 显式展开所有 $J_i$ 会占很多内存；
3. batched matmul 在 GPU 上更高效。

因此，后面如果要引入先验，最推荐的方式也是：

> 尽量在 token-wise 系数层面做隐式加权，再让代码继续走 batched aggregation；
> 不要真的把所有 $J_i$ materialize 出来。

---

### 5.4 post-zeropower applied direction

在得到 $\tilde G$ 之后，当前代码不会直接把它加到参数上，而是先经过方向变换 operator：

$$
G_{m,l,\mathrm{app}}^{(r,h)}
=
\mathcal U_{\mathrm{dir}}\big(\tilde G_{m,l}^{(r,h)}\big)
$$

这里：

- $G_{m,l,\mathrm{app}}^{(r,h)}$：真正准备写到参数上的 applied direction；
- $\mathcal U_{\mathrm{dir}}(\cdot)$：包含 `zeropower`，以及当前代码下的可选 momentum 融合。

最后参数才更新成：

$$
W_{m+1,l}^{(r,h)}
=
\mathrm{Renorm}\Big(
W_{m,l}^{(r,h)}
+
G_{m,l,\mathrm{app}}^{(r,h)}
\Big)
$$

所以整个更新链是：

$$
J_i \rightarrow \tilde G \rightarrow G_{\mathrm{app}} \rightarrow W_{m+1}
$$

这个链条最好牢牢记住，因为后面所有“控制更新”的设计，实际上都要插入到这条链上的某个位置。

---

## 6. 我们真正能控制什么：不是“最终参数变化归因”，而是“token contribution 到 update chain 的传播”

这一节把视角彻底摆正。

### 6.1 先说不能直接控制什么

我们不能直接精确控制：

$$
W_{m+1,l}^{(r,h)} - W_{m,l}^{(r,h)}
$$

的 token 级最终归因。

原因不是“数学上完全不可能分析”，而是：

- 当前 update 路径有非线性 operator；
- token 贡献先被聚合，再一起被变换；
- 最终结果不是 token 级线性叠加。

所以，“某个 token 最终究竟改了多少参数”不是最自然的控制对象。

---

### 6.2 我们真正可控的层级

当前最自然、最稳定的控制层级有两层。

#### 第一层：token 对 pre-zeropower 聚合矩阵的贡献

也就是控制：

$$
\tilde G_{m,l}^{(r,h)}
=
\sum_{i\in C_m}
\eta_{m,l,i}^{(r,h)} J_{m,l,i}^{(r,h)}
$$

里每个 token 的权重。

这对应的直觉是：

> 在这一整个 chunk 里，谁更应该主导更新方向？

#### 第二层：聚合矩阵经过 operator 后的块级 applied gain

也就是在：

$$
G_{m,l,\mathrm{app}}^{(r,h)}
$$

之后，再决定这一整块是更保守还是更激进地写入。

这对应的直觉是：

> 当前 chunk 的整体质量或整体可信度够不够高？  
> 这一整块值得写多少？

---

### 6.3 所以控制 TTT 更新的两个自然入口

这也给出后面方法设计最清楚的结构：

- **入口 A：token-wise contribution weighting**
- **入口 B：block-wise applied-direction gain**

这两个入口分别对应：

- “谁更主导当前 chunk 的更新结构”；
- “这一块最终整体偏转多少”。

这个分层非常重要，因为如果不分开，就会把“结构控制”和“整体保守性控制”混成一个东西。

---

## 7. 先验引导更新

这一节开始从“理解当前代码”切换到“如何设计新机制”。

### 7.1 不再限定先验来源

这里的“先验”不再限定于某种 temporal / spatial mask，而是一个更一般的概念。

它可以来自：

- 语义类别；
- 区域属性，例如天空、树木、地面、建筑；
- 几何置信度；
- overlap / boundary 类型；
- 动态性；
- 纹理质量；
- 视角稳定性；
- 外部 detector / segmentor 的标签；
- head-level、layer-level 或 chunk-level 的外部控制量。

关键思想是：

> 只要你能把某种先验转成一个对 token 或 chunk 的可信度 / 重要性估计，它就可以被接到 TTT 更新链上。

---

### 7.2 一个统一的先验接口

为了统一记号，我们定义一个 token prior：

$$
p_{m,l,i}^{(r,h)} \in [0,\infty)
$$

它的含义是：

> 第 $m$ 个 chunk、第 $l$ 个 layer、第 $i$ 个 token，在第 $h$ 个 head、分支 $r$ 上的更新权重修正项。

这里之所以允许取到 $[0,\infty)$，是因为：

- 如果只想抑制写入，可以把它限制在 $[0,1]$；
- 如果也允许增强写入，就不必强行把它截断在 1 以下。

这个定义足够一般：

- 你可以把它看成纯手工规则；
- 也可以看成外部模型给的 soft score；
- 也可以是多个先验相乘之后的结果。

---

### 7.3 先验可以有哪些粒度

为了避免后续设计一上来就把所有东西都挤进 token-level scalar，最好先把粒度拆清楚。

#### token-level prior

最细粒度的版本。每个 token 一个权重。

例如：

- 天空 patch 一个权重；
- 建筑 edge patch 一个权重；
- 动态目标上的 patch 一个权重。

#### token-type prior

按 token 类型给权重，例如：

- patch token
- register token
- role token

这种设计经常是必须的，因为 special token 的语义和 patch token 不一样。

#### semantic-class prior

按语义类别给权重，例如：

- 天空小权重；
- 树木中等偏小；
- 建筑和稳定结构较大。

#### head-level prior

对不同 head 给不同控制量。适合当你认为某些 head 更偏局部、某些更偏全局时。

#### layer-level prior

对不同 TTT 层施加不同写入策略。比如前层更保守，后层更开放。

#### chunk-level prior

对整个 chunk 给一个整体门控，例如当当前 chunk 质量较差时整体减小写入。

---

### 7.4 例子：语义先验抑制天空和树木

这是最直观的例子。

假设你已经有一个语义先验，知道哪些 token 属于天空，哪些属于树木，那么可以设：

- 天空 token：较小权重；
- 树木 token：中等偏小权重；
- 建筑、边缘稳定结构：正常甚至更大权重。

这么做的直觉原因可能是：

- 天空对全局几何锚定价值低；
- 树木往往纹理复杂、结构重复、随风摆动，记进 global memory 的收益未必高；
- 稳定建筑结构更适合作为长期几何记忆的一部分。

这里要特别强调：

> 这不是说天空或树木永远不该写入；
> 而是说，在“global memory 是固定容量”的前提下，它们的写入优先级可能应该更低。

---

## 8. 如何把一般先验接到 TTT 更新里

这一节给出一个统一公式。

### 8.1 token-wise prior 接到 $\eta$ 上

最自然的做法是：

$$
\beta_{m,l,i}^{(r,h)}
=
p_{m,l,i}^{(r,h)}\,\eta_{m,l,i}^{(r,h)}
$$

这里：

- $\eta$ 是当前代码已有的 token-wise coefficient；
- $p$ 是你额外引入的先验权重；
- $\beta$ 是把先验和当前代码已有权重合并后的新 token 系数。

这一步最稳，因为它不破坏原有 TTT 结构，只是在现有 token-wise coefficient 上再乘一个因子。

如果你想把多种先验叠起来，也完全可以，例如：

$$
p_{m,l,i}^{(r,h)}
=
p^{\mathrm{sem}}_{m,l,i}
\cdot
p^{\mathrm{geom}}_{m,l,i}
\cdot
p^{\mathrm{motion}}_{m,l,i}
\cdot
p^{\mathrm{head}}_{l,h}
\cdot
p^{\mathrm{layer}}_{l}
$$

然后再统一乘到 $\eta$ 上。

---

### 8.2 归一化后的结构控制

如果直接用 $\beta$ 聚合，整体尺度会比较难解释，因此更推荐再做一次 chunk 内归一化：

$$
\gamma_{m,l,i}^{(r,h)}
=
\frac{\beta_{m,l,i}^{(r,h)}}{\frac{1}{|C_m|}\sum_j \beta_{m,l,j}^{(r,h)}+\varepsilon}
$$

这里：

- $\gamma$ 控制的是当前 chunk 内部，谁更主导更新结构；
- 它更像“结构重排”或“routing”；
- 这一步不会直接告诉你最终写了多少，而是告诉你“谁在定义更新方向”。

为什么这一步合理？因为当前代码后面还有 `zeropower`。  
`zeropower` 会吃掉很多统一尺度，所以真正更关键的是：

> token 之间的**相对权重结构**，而不是整体统一放大多少。

这也是为什么我们先把 prior 接到 $\eta$ 上，再通过归一化得到 $\gamma$，而不是直接拿 prior 当最终写入强度。

---

### 8.3 pre-zeropower 聚合

于是 pre-zeropower 聚合矩阵变成：

$$
\tilde G_{m,l}^{(r,h)}
=
\sum_{i\in C_m}
\gamma_{m,l,i}^{(r,h)} J_{m,l,i}^{(r,h)}
$$

这条式子是整个“先验引导更新”框架的核心。

它的含义可以用一句话概括：

> **先验不是直接改最终参数，而是在 token 级别改“谁更主导当前 chunk 的更新结构”。**

---

### 8.4 如果还需要整体更保守，再引入 block gain

如果你觉得：

- token 级结构重加权还不够；
- 某些 chunk 或某些 layer 整体就应该少写；

那可以再引入 block-level gain：

$$
W_{m+1,l}^{(r,h)}
=
\mathrm{Renorm}\Big(
W_{m,l}^{(r,h)}
+
\lambda_{m,l}^{(r,h)} G_{m,l,\mathrm{app}}^{(r,h)}
\Big)
$$

这里：

- $\gamma$ 控当前 chunk 内部的更新结构；
- $\lambda$ 控这一整块最终 applied direction 的偏转幅度。

所以两者的功能不一样：

- `gamma`：谁来定义方向；
- `lambda`：方向整体偏转多少。

在实现上，你可以把 $\lambda$ 设成：

- 常数；
- chunk 级质量分数；
- 某种历史稳定性指标的函数；
- 或者干脆先不加，只先做 token prior。

如果只是为了引入语义先验，我建议第一版先只做 `p -> beta -> gamma -> tilde G`，不要一上来就把 `lambda` 也做得很复杂。

---

## 9. 先验设计时必须考虑的几个现实问题

### 9.1 special tokens 怎么办

不能默认 patch token 的先验直接适用于：

- register tokens；
- role tokens。

原因是这些 special tokens 的语义和 patch token 不一样。

LoGeR 当前每帧会拼进：

- 5 个 register tokens；
- 1 个 role token（来自 3 个 role embeddings 中的某一个）；
- 再加 patch tokens。

所以更稳的做法是：

- patch token 用 patch-specific prior；
- special tokens 单独指定 policy，例如：
  - 常数权重；
  - 单独类别权重；
  - 或者先不做强干预。

一个简单、稳妥的版本可以写成：

$$
p_{m,l,i}^{(r,h)} =
\begin{cases}
p^{\mathrm{patch}}_{m,l,i}, & i \in \text{patch tokens}\\
p^{\mathrm{reg}}_{l,r,h}, & i \in \text{register tokens}\\
p^{\mathrm{role}}_{l,r,h}, & i \in \text{role tokens}
\end{cases}
$$

---

### 9.2 不同分支 `w0 / w1 / w2` 是否应共用同一个先验

最简单的做法是共用一个 prior。好处是：

- 实现简单；
- 参数少；
- 先做原型更方便。

但从语义上，分支的角色并不完全相同：

- `w0` 更接近 gate 分支；
- `w2` 更接近另一条内容分支；
- `w1` 更接近最终投影到输出空间。

所以更精细的做法是 branch-specific prior。实际实现时可以先从共用开始，再决定是否拆开。

一个更一般的写法是：

$$
p_{m,l,i}^{(r,h)} = p_{m,l,i}^{\mathrm{shared}} \cdot p_{l,r,h}^{\mathrm{branch}}
$$

其中：

- $p^{\mathrm{shared}}$ 控 token 级信息；
- $p^{\mathrm{branch}}$ 控 branch/head/layer 的额外偏置。

---

### 9.3 先验是抑制写入，还是增强写入

这个问题需要提前想清楚。

#### 如果只想抑制写入

那可以限制：

$$
p_{m,l,i}^{(r,h)} \in [0,1]
$$

这时先验只会弱化 token 对更新的贡献。

这种设计最安全，因为它不会让任何 token 比当前代码“写得更猛”。

#### 如果也允许增强写入

那就可以允许：

$$
p_{m,l,i}^{(r,h)} > 1
$$

这时某些高价值 token 会比默认更新得更强。

这种设计更灵活，但也更容易放大噪声，所以通常建议在第一版中先只做 downweight。

---

### 9.4 reset 时先验状态要不要同步处理

如果你的设计里额外维护了：

- 历史模板；
- chunk 级状态；
- 其他跨 chunk 先验统计量；

那么 reset 时必须和 `w0 / w1 / w2` 一起同步处理。

否则会出现：

- fast weights 已经 reset；
- 但先验状态还在使用 reset 前的历史。

这在语义上会不一致。

如果你的 prior 完全来自当前 chunk，例如纯语义标签、纯几何置信度，那么 reset 不一定有额外处理。  
但只要你的 prior 用到了历史摘要，就必须同步 reset。

---

## 10. 复杂度与实现代价

### 10.1 当前 TTT 基线复杂度

如果记：

- $N_{\text{ttt}}$：TTT 插入层数；
- $H$：head 数；
- $L$：chunk 内 token 数；
- $d_h$：单个 head 的维度；

那么当前 TTT update 的主项复杂度大约可以理解成：

$$
O\big(N_{\text{ttt}}\,H\,L\,d_h^2\big)
$$

这里真正主导的是 chunk 聚合时的 batched matmul。  
另外，每个分支还有 `zeropower`，其代价 roughly 是每个 head/branch 一个 $d_h\times d_h$ 矩阵上的若干步迭代，所以可以理解成额外的：

$$
O(d_h^3)
$$

但在很多实际设置下，$L d_h^2$ 往往更显眼。

---

### 10.2 加 token-wise prior 的额外代价

如果你只是给每个 token 一个 scalar prior，再把它乘进已有的 $\eta$，额外代价通常不大，主要是：

- token-wise 的乘法；
- chunk 维上的归一化；
- 少量额外统计。

真正重的是另一种情况：

> 如果你显式 materialize 所有 token 的 $J_i$，内存和计算都会很贵。

原因是每个 $J_i$ 本身就是一个矩阵，shape 和对应分支参数一样。  
如果你真的把所有 token 的 $J_i$ 都存下来，内存会随：

$$
O(N_{\text{ttt}}\,H\,L\,d_h^2)
$$

线性增长，非常不划算。

---

### 10.3 推荐的实现原则

我更推荐下面三条：

1. 不显式存所有 $J_i$；
2. 尽量用隐式双线性 / Frobenius trick；
3. 先在少数 layer、少数 head 上试，不要一上来全开。

这里“隐式双线性 / Frobenius trick”的意思是：

- 你在数学上可以定义 $J_i$；
- 但实现时尽量不要把它 materialize 成一个个矩阵；
- 而是把 prior 直接乘进 token-wise scalar，再让 batched matmul 完成聚合。

这样最贴当前代码，也最省内存。

---

## 11. 一个最小、通用、和语义先验兼容的实现版本

这一节给出一版非常克制的最小方案。

### 11.1 输入

这版最小方案只需要：

- 每个 token 的语义标签或语义权重；
- special token 类型；
- 可选的 head-level / layer-level / chunk-level 常数控制量。

例如你已经有一个 segmentor，那么就能给每个 patch token 一个语义类别，再映射成 prior 权重。

---

### 11.2 中间量

这版方案里会出现的核心中间量有：

- $\eta$：当前代码已有的 token-wise coefficient；
- $p$：外部先验；
- $\beta$：先验和现有系数的乘积；
- $\gamma$：归一化后的 token-wise structure weight；
- $\tilde G$：pre-zeropower 聚合矩阵；
- $G_{\mathrm{app}}$：post-zeropower applied direction；
- $\lambda$：可选的 block-level gain。

---

### 11.3 更新链

它们之间的关系可以写成：

$$
p \rightarrow \beta \rightarrow \gamma \rightarrow \tilde G \rightarrow G_{\mathrm{app}} \rightarrow W_{m+1}
$$

这条链读起来就是：

1. 先验先作用到 token 系数；
2. token 系数决定谁更主导更新结构；
3. 再形成 chunk 级聚合矩阵；
4. 再通过当前代码已有的 operator 变成真正 applied direction；
5. 最后再写回 fast weights。

---

### 11.4 一个具体例子

例如你已经有语义先验：

- 天空 token：$p=0.2$；
- 树木 token：$p=0.4$；
- 建筑结构 token：$p=1.0$。

那么在最简单的版本里：

- 天空 token 对 $\tilde G$ 的贡献会被显著压低；
- 树木 token 也会被弱化；
- 建筑结构 token 保持默认贡献。

如果再加归一化，结果会是：

- 当前 chunk 的更新方向更多由建筑结构类 token 主导；
- 而不是被天空或树木主导。

这正是“语义先验引导 TTT 更新”的直观含义。

---

## 12. 结论

这份文档最想强调的一句话是：

> 在 LoGeR 中，控制 TTT 更新的正确落点不是直接操纵最终参数变化，而是
> **先验引导下的 token-wise 原始更新贡献重加权**，
> 必要时再加一个 block-level applied-direction gain。

更具体地说：

- 我们先通过当前前向和目标，定义单 token 对 fast weights 的原始更新矩阵 $J_{m,l,i}^{(r,h)}$；
- 再用现有的 $\eta$ 和新增先验 $p$ 去控制这些 token 贡献在 chunk 聚合里占多大权重；
- 最后保持 LoGeR 当前的 `zeropower + renorm` 更新链不变。

这样做的好处是：

- 逻辑清楚；
- 贴当前代码；
- 对语义先验、几何先验、动态先验都兼容；
- 后续改代码时也最容易逐步落地。

如果后面需要继续深化，最自然的下一步是：

1. 先做一个 **纯语义先验 downweight** 的最小原型；
2. 再决定是否引入 branch-specific prior；
3. 最后再考虑 chunk-level 的整体保守性控制，例如 block gain $\lambda$。
