---
layout: post
title:  "Retrieval Augmented Generation and trying to get Legal Advice"
date:   2023-09-18 21:25:12 +0200
categories: AI NLP Legal
---

Ever wondered what the law says about whether it is legal to use AI to generate
pointless back story to write on some hard discount peanut butter jars' labels?
Well now is your chance to ask! Well... sort of.

I have been learning about Retrieval Augmented Generation (RAG) and wanted to
give it a try. And for once I wanted to use a topic for which the data is easily
accessible, and even more so, in a legal manner. From this aspect, the law texts
are indeed publicly available and (supposedly) comprehensive.
So I thought I'd try to create a small tool to generate legal advice.
Being a French citizen, I naturally started with the French law texts. But this
should apply to about any country's law.

## The gist of RAG

Essentially, pre-trained LLMs are good chatty chatters but not
really useful, on their own, on specific topics. This is because the
domain-specific knowledge either got diluted in the training data, was not
in the said training data, or is newer than the last training.
So one thing that could be done is, upon
receiving some question or query, is somehow retrieve a relevant document or
paragraph and feed its content as part of the query.

In the implementation I use a vector database to get the most relevant document.

Initially I started using PyMilvus, as I already had a nice wrapper from some
previous project (that I might write about at some point). But then I realized
it'd be more versatile to use a simple vectorizer in the form
of SentenceTransformers, without all the bells and whistle of Pymilvus (which
retrospectively does not seem to bring that much to the table, but I'm probably
missing the point given my current knowledge of what it does).

As of the the LLM I use LLama2 7B-chat directly from MetaAI's repo. Because I
want to be able to experiment on the cheap, on my home's GPU (RTX3080),
I find it strikes a good balance of precision and VRAM requirements. On that
said, it did force me to rewrite some of the code to avoid having torch DDP
throw a million errors on my home rig with a single GPU, and everything
running inside WSL2.

## Why RAG rather than fine-tuning?

While conceptually simple, RAG does bring in quite a few extra components.
One have to worry about data preparation for better retrieval, finding a
suitable vectorizer and distance metric choice, store the documents in an
efficient format, do some query engineering. Additionally, upon inference, in a
live context, one would have to handle talking to 2 databases or stores
(one for the vector embeddings and one for the source data), as well as handle
having at least 2 AI models (albeit one of them being relatively small).

All of this makes the RAG perhaps a bit more complex to implement than a simple
fine-tuning, where you end up with a single model and that's it. But it does
have a few advantages over a fine-tuned model:
* it is very economical. One does not have to fine-tune any model which can
quickly cost thousands of dollars for a single run (and you'd better pray it
works well enough the first time around). Instead, one can use a pre-trained
model, which is free, and only pay
for the vectorization of the documents, which is the same cost as regular
inference. And in most instances, with a relatively small model like LLama2-7B,
or ome other larger quantized model, it can even fit on consumer grade hardware.
* it is more flexible. One can easily add new documents to the database, or
remove some, without having to retrain the model. This is especially useful
when the documents are updated regularly, or when one wants to add some
documents on the fly, like in a chatbot. In our case it really is important as
the law texts are updated regularly. Also, a number of court cases end up
being more or less equivalent to a real law (like supreme court decisions in
the US for instance or "jurisprudence" in teh french system).
* in many instances, fine-tuned LLMs are very good at capturing styles, but not
so much as keeping their focus on just the content of
the fine-tuning dataset. This makes it easy for such model to start
extrapolating from prior knowledge rather than use the provided training data.
With RAG, the retrieval step itself often provides to match quality metric which
will ensure the model will see the relevant documents only and is specifically
asked to talk about. This significantly reduces the risk for hallucinations,
especially when dealing with models fine-tuned with RLHF, where instruction
following often a key training exercise.


To be continued...

## Efficient data storage

## Vectorization in French

## The legal advice generator in action

## Towards a nice (non-dev) user experience