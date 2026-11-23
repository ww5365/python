
## 整体说明

https://jalammar.github.io/illustrated-transformer/ 这篇博客为基础，总结计算公式。
Jay Alammar 的博客以可视化图解见长，对数学公式着墨不多。为了完整呈现 Transformer 的**计算过程数学公式**，以下结合原论文及严谨的学术资料，按**数据流顺序**系统梳理 Encoder 与 Decoder 的数学架构。

---

## 总体架构：数据流与公式概览

Transformer 采用 **Encoder-Decoder** 堆叠结构。Encoder 将输入序列 \(X = (x_1,\ldots,x_n)\) 映射为上下文表示 \(H^E\)，Decoder 以自回归方式基于 \(H^E\) 和已生成的前缀 \(y_{<t}\) 生成下一个 token \(y_t\)。核心公式链如下：

\[
\text{Embedding} \;\to\; \text{Positional Encoding} \;\to\; \underbrace{\text{Attention}(Q,K,V)}_{\text{缩放点积}} \;\to\; \text{Multi-Head} \;\to\; \text{Add \& Norm} \;\to\; \text{FFN}
\]

Decoder 在此基础上额外加入 **Masked Self-Attention** 与 **Cross-Attention**，后者以 Encoder 输出作为 \(K, V\)，以 Decoder 状态作为 \(Q\)。

---

## 输入表示：词嵌入 + 位置编码

设词嵌入矩阵为 \(E \in \mathbb{R}^{|V| \times d_{\text{model}}}\)，输入 token 序列的索引为 \((i_1,\ldots,i_n)\)，则嵌入向量矩阵为：

\[
X_{\text{emb}} = [E_{i_1}, \ldots, E_{i_n}]^\top \in \mathbb{R}^{n \times d_{\text{model}}}
\]

由于纯注意力无位置感知，需叠加**正弦位置编码** \(P \in \mathbb{R}^{n \times d_{\text{model}}}\)：

\[
\tilde{X} = X_{\text{emb}} + P
\]

位置编码的第 \(pos\) 个位置、第 \(2j\) / \(2j+1\) 维定义为：

\[
PE_{(pos, 2j)} = \sin\left(\frac{pos}{10000^{2j/d_{\text{model}}}}\right), \quad
PE_{(pos, 2j+1)} = \cos\left(\frac{pos}{10000^{2j/d_{\text{model}}}}\right)
\]

Decoder 端对目标序列做类似处理：将 \(Y_{0:s-1} = [\text{BOS}, Y_1, \ldots, Y_{s-1}]^\top\) 嵌入并加位置编码，得到 \(\tilde{F}_{0:s-1}\)。

---

## Encoder 层：数学计算过程

每个 Encoder 层包含 **两个子层**：多头自注意力 + 逐位置前馈网络。两个子层均采用 **残差连接 + LayerNorm**。

### 1. 自注意力（Scaled Dot-Product Attention）

对输入 \(\tilde{X} \in \mathbb{R}^{n \times d_{\text{model}}}\)，通过可学习投影矩阵生成 \(Q, K, V\)：

\[
Q = \tilde{X}W^Q,\quad K = \tilde{X}W^K,\quad V = \tilde{X}W^V
\]

其中 \(W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}\)，\(W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}\)。缩放点积注意力为：

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

缩放因子 \(\sqrt{d_k}\) 用于防止点积过大导致 softmax 梯度消失。

```text

**翻译：**

为什么要除以 \(\sqrt{d_h}\)？如果 \(q\) 和 \(k\) 的各分量相互独立、均值为 0、方差为 1，那么它们的点积就是 \(d_h\) 个这样的乘积之和，因此方差为 \(d_h\)。除以 \(\sqrt{d_h}\) 可以把方差重新变回 1，这样当每个头的维度变宽时，softmax 不会饱和——也就是不会把几乎全部权重都放在某一个 token 上，从而导致梯度消失。GPT-2 中 \(d_h=64\)，缩放因子是 8。

**原因总结：**

Transformer 中注意力分数是 \(q\cdot k\)。如果 \(q,k\) 各维度独立且方差为 1，那么点积的方差会随着维度 \(d_k\)（即每个头的维度 \(d_h\)）增大而增大，变成 \(d_k\)。点积过大时，softmax 容易进入饱和区：输出接近 one-hot，几乎把所有注意力集中到一个 token 上，梯度会变得很小，训练不稳定。

因此用：

\[
\frac{q\cdot k}{\sqrt{d_k}}
\]

进行缩放，使点积的方差回到约 1，让 softmax 输入保持在较合理范围，避免饱和和梯度消失。GPT-2 中每个头维度 \(d_h=64\)，所以除以 \(\sqrt{64}=8\)。

```




### 2. 多头注意力（Multi-Head Attention）

并行执行 \(h\) 个注意力头，每个头使用独立的投影 \(W_i^Q, W_i^K, W_i^V\)：

\[
\text{head}_i = \text{Attention}(\tilde{X}W_i^Q,\; \tilde{X}W_i^K,\; \tilde{X}W_i^V)
\]

\[
\text{MultiHead}(\tilde{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O
\]

通常取 \(d_k = d_v = d_{\text{model}}/h\)，使总计算量与单头全维度注意力相近。

### 3. Add & Norm（残差 + 层归一化）

设子层函数为 \(\text{Sublayer}(\cdot)\)，则输出为：

\[
X' = \text{LayerNorm}\big(\tilde{X} + \text{Sublayer}(\tilde{X})\big)
\]

对多头注意力子层：

\[
H^A = \text{LayerNorm}\big(\tilde{X} + \text{MultiHead}(\tilde{X})\big)
\]

### 4. 逐位置前馈网络（Position-wise FFN）

FFN 对每个位置独立应用两层线性变换 + 激活函数（原论文用 ReLU，后续常用 GELU）：

\[
\text{FFN}(x) = \phi(xW_1 + b_1)W_2 + b_2
\]

其中 \(W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}\)，\(W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}\)。再经 Add & Norm：

\[
H^E = \text{LayerNorm}\big(H^A + \text{FFN}(H^A)\big)
\]

**Encoder 整体**：将上述层重复 \(N\) 次，第 \(l\) 层输出可写为：

\[
H^{E,(l)} = \text{LayerNorm}\Big(H^{E,(l-1)} + \text{FFN}\big(\text{LayerNorm}(H^{E,(l-1)} + \text{MultiHead}(H^{E,(l-1)}))\big)\Big)
\]

最终 \(H^E = H^{E,(N)}\) 即为编码器输出，供 Decoder 的 Cross-Attention 使用。

---

## Decoder 层：数学计算过程

每个 Decoder 层包含 **三个子层**：Masked Multi-Head Self-Attention、Cross-Attention、FFN。同样均带残差 + LayerNorm。

### 1. Masked Self-Attention（因果掩码）

Decoder 自注意力必须保持自回归性：位置 \(u\) 只能关注 \(v \leq u\) 的位置。设 Decoder 层输入为 \(\tilde{F} \in \mathbb{R}^{s \times d}\)，先计算未掩码分数：

\[
A' = \frac{QK^\top}{\sqrt{d}} + M
\]

其中 \(M \in \mathbb{R}^{s \times s}\) 为掩码矩阵：

\[
m_{u,v} = \begin{cases}
0, & v \leq u \\
-\infty, & v > u
\end{cases}
\]

经 softmax 后，未来位置的权重趋近于零：

\[
A = \text{softmax}(A'),\quad H^{\text{mask}} = A V
\]

然后 Add & Norm：

\[
H^{\text{mask}} = \text{LayerNorm}\big(\tilde{F} + H^{\text{mask}}\big)
\]

### 2. Cross-Attention（编码器-解码器注意力）

Cross-Attention 让 Decoder 在生成每个 token 时“查看”整个编码器输出。**Query 来自 Decoder**，**Key 和 Value 来自 Encoder 输出 \(H^E\)**：

\[
Q = H^{\text{mask}} W_c^Q,\quad K = H^E W_c^K,\quad V = H^E W_c^V
\]

\[
H^{\text{cross}} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

经 Add & Norm：

\[
H^{\text{cross}} = \text{LayerNorm}\big(H^{\text{mask}} + H^{\text{cross}}\big)
\]

### 3. FFN + Add & Norm

与 Encoder 中完全相同，逐位置前馈网络后接残差与层归一化：

\[
H^D = \text{LayerNorm}\big(H^{\text{cross}} + \text{FFN}(H^{\text{cross}})\big)
\]

重复 \(N\) 层后得到最终 Decoder 隐藏状态 \(H^{D,(N)}\)。

---

## 输出层：从隐藏状态到概率分布

Decoder 最后一层的隐藏状态经**线性投影**映射到词表维度 \(|V|\)，再经 **softmax** 得到下一个 token 的概率分布：

\[
P(y_t \mid y_{<t}, X) = \text{softmax}\big(H^{D,(N)} W_{\text{out}} + b_{\text{out}}\big)
\]

训练目标为最大化目标序列的对数似然：

\[
\mathcal{L} = -\sum_{t=1}^{s} \log P(y_t \mid y_{<t}, X)
\]

---

## 一页公式速查表

| 组件 | 核心公式 |
|---|---|
| 位置编码 | \(PE_{(pos,2j)}=\sin(pos/10000^{2j/d_{\text{model}}})\)，\(PE_{(pos,2j+1)}=\cos(\cdot)\) |
| 缩放点积注意力 | \(\text{softmax}(QK^\top/\sqrt{d_k})V\) |
| 多头注意力 | \(\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O\) |
| Encoder 子层 | \(\text{LayerNorm}(x+\text{Sublayer}(x))\) |
| FFN | \(\phi(xW_1+b_1)W_2+b_2\) |
| Decoder 掩码 | \(M_{u,v}=0\;(v\leq u),\ -\infty\;(v>u)\) |
| Cross-Attention | \(Q\) 来自 Decoder，\(K,V\) 来自 Encoder 输出 |
| 输出概率 | \(\text{softmax}(H^D W_{\text{out}}+b_{\text{out}})\) |

以上公式体系完整覆盖了 Transformer **Encoder 的双子层结构**与 **Decoder 的三子层结构**，每一步均对应 Alammar 博客中可视化展示的计算环节。









