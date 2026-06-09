---
title: "FPN-Transformer: Hierarchical Sparse Attention via Feature Pyramid Networks"
date: "May 29th, 2026"
tags: ["LLM", "Efficiency", "WIP"]
teaser: "Can we use a feature pyramid to make attention sub-quadratic without sacrificing expressiveness?"
---

DISCLAIMER: this is still work in progress and this article will be further updated down the line as results come through, including sharing the code.

The transformer architecture has proven remarkably durable, but it carries a structural tax that becomes harder to ignore as context windows grow: the scaled dot-product attention mechanism scales quadratically in both compute and memory with respect to sequence length. For a sequence of $n$ tokens, computing the full attention matrix $QK^T$ costs $O(n^2 d)$ operations, where $d$ is the head dimension. At $n = 4096$ this is already uncomfortable on a single GPU. At $n = 1M$, which is where frontier models are now routinely operating, it is the dominant cost in the entire forward pass.

Existing workarounds tend to fall into one of two camps: fixed sparse patterns (local windows, strided global tokens) which are cheap but make hard assumptions about where relevant information lives, or linear attention approximations which are fast but replace the softmax kernel entirely and tend to hurt quality on tasks that depend on precise, content-dependent retrieval.

We propose a different approach. We use a feature pyramid network to build a coarse-to-fine representation of the key and query sequences, and then traverse this pyramid top-down to identify which regions of the sequence genuinely need fine-grained attention for each query. Regions that are irrelevant at coarse resolution are approximated using pooled value representations. Regions that matter are resolved at full resolution. The sparsity pattern is dynamic, input-dependent, and emerges naturally from the attention weights themselves.

## Feature Pyramid Attention

The core idea is borrowed from feature pyramid networks in computer vision, where a hierarchy of representations at decreasing spatial resolutions is used to efficiently reason about objects at multiple scales. Here we apply the same principle to the sequence dimension.

We build three separate pyramids, one each for queries, keys and values. At level $l$, each pyramid contains a downsampled version of the corresponding projection at resolution $n / 2^l$. The pyramid is constructed by a series of FPN blocks, each of which applies a strided depthwise convolution with fixed kernel width $k$ (we use $k=3$), followed by layer normalization, a SiLU activation, and a pointwise linear projection back to the head dimension $d_h$:

$$
    X^{(l)} = \text{FPNBlock}(X^{(l-1)}) \in \mathbb{R}^{B \times n/2^l \times d_h}
$$

The strided convolution halves the sequence length at each level, while the pointwise projection preserves the dimensionality. This is important: unlike feature pyramid networks in vision, we do not add channels to compensate for the reduction in sequence length. Instead, the pointwise projection learns to pack the most relevance-predictive information into $d_h$ dimensions at each level.

We build three separate FPNs rather than one shared network, even though Q, K and V all derive from the same residual stream. The reason is that the linear projections $W_Q$, $W_K$ and $W_V$ are trained to push these representations into very different subspaces optimized for their respective roles. By the time the FPN sees its input, the three tensors already live in distinct statistical spaces with different structure. A shared pyramid would have to simultaneously learn to summarize all three, which is a harder optimization problem with no real upside given the modest parameter count of each FPN.

The traversal proceeds top-down. Starting at the coarsest level $L$, we compute the coarse attention weights for all block pairs $(b, c)$:

$$
    a^{(L)}[b, c] = Q^{(L)}[b] \cdot K^{(L)}[c] / \sqrt{d_h}
$$

We then apply softmax over the full coarse sequence to obtain normalized attention weights. Any block pair whose weight falls below a threshold $\varepsilon$ is resolved at this coarse level: its contribution to the output uses $V^{(L)}[c]$, the pooled value representation of that region. Block pairs above the threshold are split into four child pairs at level $L-1$, each corresponding to a $2 \times 2$ subdivision of the original block pair, and the process recurses.

This gives us a mixed-resolution attention computation: fine-grained where the model is paying attention, coarse-grained everywhere else. Every region of the sequence contributes to every query output (there is no hard zeroing) which keeps gradients alive and reduces error accumulation across layers.

### Reformulating the softmax across resolutions

Because different block pairs are resolved at different pyramid levels, their attention logits are not directly comparable: a logit from a level-2 block summarizes 4 tokens, while a level-0 logit summarizes 1. We cannot simply threshold raw logits across levels and sum the resulting weights; they live on different scales.

The fix is to use the online softmax formulation from FlashAttention. For each query position $i$ we maintain a running triple $(m, s, o)$ where $m$ is the current maximum logit, $s$ is the running sum of $\exp(\text{logit} - m)$, and $o$ is the running weighted sum of value vectors. As we resolve block pairs at each level, we merge their contributions using the log-sum-exp rescaling trick:

$$
    m_{\text{new}} = \max(m, a^{(l)}[i, b])
$$
$$
    s \leftarrow s \cdot \exp(m - m_{\text{new}}) + \exp(a^{(l)}[i, b] - m_{\text{new}})
$$
$$
    o \leftarrow o \cdot \exp(m - m_{\text{new}}) + \exp(a^{(l)}[i, b] - m_{\text{new}}) \cdot V^{(l)}[b]
$$

The final output for query $i$ is $o / s$. This is the standard online softmax, applied to a set of logits that happen to come from different pyramid levels. The online accumulator doesn't care about the provenance of each logit, only its value, which is precisely what we need.

### The soft gate

During training we replace the hard threshold decision with a soft gate, making
the entire traversal differentiable. For each block pair $(b, c)$ at level $l$,
we define the descent gate as

$$
    g^{(l)}[b, c] = \sigma\left(\tau \cdot \left(\max_{j \in c} a^{(l)}[b, j] - \varepsilon\right)\right)
$$

where $\tau$ is a temperature parameter and $\sigma$ is the sigmoid function. A
value of $g^{(l)}$ close to 1 means the block pair is routed to the next finer
level, and close to 0 means it is resolved at the current level.

The contribution weight of each level is then a recursive product of descent
decisions. Defining level $L$ as the coarsest and level $0$ as full resolution:

$$
    w^{(L)}[b, c] = 1 - g^{(L)}[b, c]
$$
$$
    w^{(l)}[b, c] = \left(1 - g^{(l)}[b, c]\right) \prod_{m=l+1}^{L} g^{(m)}[b, c]
    \quad \text{for } 0 < l < L
$$
$$
    w^{(0)}[b, c] = \prod_{m=1}^{L} g^{(m)}[b, c]
$$

The finest level carries no $(1 - g^{(0)})$ factor: once we reach full
resolution there is nowhere left to descend, so the remaining probability mass
is assigned entirely to level $0$. One can verify that $\sum_{l=0}^{L}
w^{(l)}[b, c] = 1$ for any choice of gate values.

The final output for query block $b$ is then a weighted sum of value
representations across all levels:

$$
    o = \sum_{l=0}^{L}W^{(l)}\cdot A^{(l)}\cdot V^{(l)}
$$

## Complexity

We analyze the computational complexity of FPN-Transformer separately for the pyramid construction and the attention traversal.

### FPN cost

At each level $l$, the FPN applies the convolution of complexity bounded by $O(nd)$ and the pointwise projection, which dominates the computation with complexity bounded by $O(nd^2)$ (we have ommited the $1/2^l$ factor here).

Each FPN applies $L$ blocks to a sequence that is halved at each level. The total cost across all levels and can be summed as a geometric series:

$$
    \sum_{l=0}^{L} \frac{n}{2^l} \cdot d_h^2 = n d_h^2 \sum_{l=0}^{L} \frac{1}{2^l} \leq 2 n d_h^2
$$

The pyramid costs $O(n d_h^2)$ regardless of $L$, the same asymptotic order as the linear projections $W_Q$, $W_K$, $W_V$ themselves. The FPN is therefore never the bottleneck as its complexity w.r.t n is smaller than that of the attention mechanism, at least for long sequences.

### Attention traversal

The traversal cost depends on how many block pairs survive to each level.

In the worst case, every block pair at every level exceeds the threshold and is descended into. The total number of pairs processed sums as a geometric series dominated by the finest level, giving $O(n^2 d_h)$, identical to standard attention. This happens when attention is perfectly uniform, which is a degenerate case and arguably the one where sparse attention provides the least value in any scheme.

In the best case, attention is maximally sparse: each query attends to $O(1)$ key blocks regardless of $n$. The total number of resolved pairs is $O(n)$ and the traversal costs $O(n d_h)$, linear in sequence length.

In the realistic case, if each query attends to $O(n^\alpha)$ tokens for some $0 < \alpha < 1$, the traversal costs $O(n^{1 + \alpha} d_h)$. For $\alpha = 0.5$ this is $O(n^{1.5} d_h)$. This happens when roughly half the token are attended to, at maximum resolution.

Again, the point where the FPN overhead becomes negligible relative to the attention savings is whenever $n > d_h$, which holds for virtually all practical configurations, and reasonable sequence length. Because this mechanism can be retrofitted on top of exisiting pretrained models with no modifications, as will be discussed in the next section, we could even consider introducing a mechanism whereby the FPN is deactivated when a sequence is short enough to allow overcoming the cost of it.

## Retrofitting to pre-trained models

One of the more practical aspects of this approach is that it can be injected into an existing pre-trained transformer without retraining from scratch. The FPN modules are inserted alongside the frozen attention layers, and only the FPN weights are trained. The original QKVO projections remain unchanged.

The threshold $\varepsilon$ deserves particular attention as a runtime parameter. Because the attention weights it is applied to are post-softmax, they are guaranteed to lie in $[0, 1]$. This means $\varepsilon$ has a clean interpretation: it is the fraction of total attention mass a block must carry before it is worth computing at full resolution. Setting $\varepsilon = 1$ forces every block above the minimum to be resolved at full resolution, effectively bypassing the pyramid and recovering exact attention. Setting $\varepsilon = 0$ always resolves at the coarsest level available. In between, $\varepsilon$ is a continuous dial between quality and compute. This is a useful property at inference time: a production system could vary $\varepsilon$ dynamically based on estimated task complexity, using high resolution for complex reasoning tasks and coarser approximations for simpler completions.

### Which layers to modify

Not all transformer layers are equally good candidates for FPN injection. Attention behavior varies systematically with depth.

Early layers tend to attend locally, building positional and syntactic representations over short windows. The attention patterns here are already dense and short-range, which means the pyramid has little to prune and the approximation error would be concentrated precisely where the model is doing the most precise work. I think these layers are better left untouched.

Middle layers show mixed local and mid-range attention, with some heads beginning to specialize.

Late layers tend to exhibit the sparsest, most task-specific attention patterns, often with clear long-range dependencies and well-separated heads. These are the best candidates for FPN injection: the pyramid is most likely to find genuine block-level structure to exploit, and the approximation error lands in regions the model was already ignoring.

As a starting point we will try in this work injecting the last 25% to 33% of layers and profiling per-layer distillation loss before committing. If a given layer's FPN fails to approximate its exact attention within a reasonable tolerance, we will exclude it.

## Training procedure

We train the FPN weights in two phases, with the backbone frozen throughout. The QKVO projections are never updated.

### Phase 1: Bootstrap (distillation)

In the bootstrap phase, we deactivate the soft gate, and instead we compute the entire FPN and minimize the error on each FPN level independantly. That is we compute a loss against the true attention for each level, as if it had computed every single token.

$$
    \mathcal{L}_{\text{distill}} = \frac{1}{N_{\text{layers}}} \sum_l \left\| \hat{o}^{(l)} - o^{(l)}_{\text{exact}} \right\|_2^2
$$

This is a form of layer-wise knowledge distillation: the frozen model acts as a teacher and the FPN learns to replicate its attention behavior.
The bootstrap phase has a clean stopping criterion: monitor the per-layer distillation loss and stop when it plateaus. Layers whose loss does not converge below a tolerance should be excluded from the FPN injection.

Note that in practice, the "deep-supervision", that is applying the loss on coarser levels, might be too hard of a problem for the network, so we suggest adding some weighting to the loss at each level which follows the same geometric series $1/2^l$.

### Phase 2: Tuning (threshold annealing)

In the tuning phase, we activate the soft gate and linearly anneal the threshold $\varepsilon$ from 0 to its target value over the course of training. At the start of tuning, $\varepsilon = 0$ means the gate sigmoid is centered at 0 and the blend is approximately 50/50 between exact and approximate, a gentle introduction of sparsity. As training proceeds and $\varepsilon$ increases, the model is progressively pushed toward coarser approximations for low-weight regions.

We also anneal $\tau$ from a very soft value, e.g. 1 to a much sharper distribution with e.g. $\tau=10$.

The distillation loss remains active throughout, acting as a regularizer that keeps the FPN anchored to the teacher's behavior even as sparsity increases. We also expose a language modeling loss term weighted by $\lambda_{\text{lm}}$, which can be enabled for the tuning phase to provide an end-to-end signal, though we find distillation alone is sufficient for the experiments described here.

The soft gate ensures that gradients flow through the routing decision throughout tuning. The FPN does not receive an explicit gradient through the binary threshold comparison, but it does receive a gradient through the output: bad routing decisions produce high distillation loss, which backpropagates through the value readout and the FPN weights that produced the coarse representations. The model learns to route well implicitly, without needing a separate routing loss, in much the same way MoE routers learn without explicit supervision on which expert to use.

## Efficient implementation

Once the training is complete, we no longer want to use the softgat, and instead only compute the attention for the coarsest possible level for each block, as selected by the gate value at this level. This does imply that the inference path requires a true sparse traversal. This is non-trivial as this defeats the usual no-branching highly parallel approach of dense gemm. We therefore need to design the traversal carefully.

The block pair indices surviving each level form an explicit list. Using a bitmask here would still materialize all entries and multiply by zero, which saves no compute. Instead the surviving pairs are stored as a variable-length list of (query block, key block) tuples per level, and the matmul at each level operates only on the listed pairs. Block boundaries are aligned to powers of 2 to give regular structure within each level, allowing a single batched sparse matmul per level.

The pyramid is pre-built once at the start of each forward pass from the output of the linear projections, costing $O(n d_h^2)$ as analyzed above. At inference time with KV caching, the pyramid is updated incrementally: when a new token arrives, at most one block per pyramid level needs recomputing (the one that just completed its $2^l$-token span). With fixed-width depthwise convolutions, each such update costs $O(d_h \cdot k)$ per level, giving a total KV cache update cost of $O(L \cdot d_h \cdot k) = O(\log n \cdot d_h)$ per new token. The cache itself stores $K$ and $V$ at all pyramid levels, which converges towards $2\times$ the standard KV cache as $L$ grows, a fixed constant overhead.

The naive implementation of the dynamic list is obviosuly suboptimal and we will explore a custom CUDA kernel, most likely, to try and speed it up.

## Results

*Coming soon.*

## Discussion

The approach as described keeps the QKVO projections frozen throughout. This is practical and sufficient for a first experiment, but it does leave some performance on the table. The FPN approximation introduces a small bias into the attention output that the downstream layers were not trained to handle. It might be interesting to try co-finetuning the QKVO projections alongside the FPN weights, allowing the model to slightly adapt its internal representations to be more amenable to the hierarchical compression. This is straightforward to add as a stage 3 of the training procedure and is a natural direction for follow-up work.

A second limitation is that the threshold $\varepsilon$ is currently a fixed scalar shared across all layers, heads and input positions. In practice different heads specialize at different granularities, and the right threshold for a head that does broad document-level retrieval is very different from one that tracks local syntactic dependencies. Future work could train a small auxiliary network to predict the appropriate threshold per head per input, conditioned on a lightweight estimate of the task complexity, something as simple as the entropy of the coarse attention distribution. This would turn the threshold from a hyperparameter into a learned, input-adaptive routing policy, which is the natural endpoint of the design.
