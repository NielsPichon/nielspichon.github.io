---
title: "Attention Is All You Need — and Here's Why It Still Blows My Mind"
date: "Apr 14, 2026"
readTime: "4 min"
tag: "Transformers"
teaser: "Vaswani et al. killed the RNN and nobody fully appreciated it at the time. Re-reading this paper in 2026 feels like reading a founding document."
paper: 'Vaswani et al. (2017) — "Attention Is All You Need"'
---

I keep coming back to this paper. Not because it's new — it's almost a decade old now — but because every time I re-read it, I notice something I missed. There's a compactness to the original Transformer architecture that subsequent work has only obscured.

## What they actually did

The core idea is disarmingly simple: replace recurrence entirely with a mechanism that lets every position in a sequence attend to every other position simultaneously. No hidden states. No vanishing gradients. Just a matrix multiply asking: *how relevant is each token to every other token?*

The scaled dot-product attention is:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The $\sqrt{d_k}$ scaling prevents the dot products from blowing up as dimension grows — a small detail that matters a lot.

> "We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism."

![The Transformer architecture — encoder on the left, decoder on the right](/images/placeholder.svg)

The positional encoding is what I keep returning to. Without recurrence, you have no notion of order. Their solution — adding fixed sinusoidal patterns to embeddings — is almost embarrassingly simple and it just works.

## My honest take

What strikes me most is what this paper *destroyed*. RNNs, LSTMs, GRUs — entire research lineages built over a decade, made largely obsolete in one swing. Within 18 months we had BERT and GPT-1.

The quadratic complexity — $O(n^2)$ in sequence length — is the original paper's Achilles heel. Almost every major Transformer variant since has tried to address this. FlashAttention, sparse attention, linear attention... we're still fighting the same battle they identified in 2017.

### What I'd read next

*A Mathematical Framework for Transformer Circuits* by Elhage et al. — gives you tools to reason about what individual attention heads are actually doing, rather than treating them as black boxes.
