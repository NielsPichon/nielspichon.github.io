---
layout: post
title:  "Retrieval Augmented Generation and trying to get Legal Advice"
date:   2023-09-18 21:25:12 +0200
categories: AI NLP Legal
---

Ever wondered what the law says about whether it is legal to use AI to generate pointless back story to write on some hard discount peanut butter jars' labels? Well now is your chance to ask! Well... sort of.

I have been learning about Retrieval Augmented Generation (RAG) and wanted to give it a try. And for once I wanted to use a topic for which the data is easily accessible, and even more so, in a legal manner. From this aspect, the law texts are indeed publicly available and (supposedly) comprehensive.
So I thought I'd try to create a small tool to generate legal advice. Being a French citizen, I naturally started with the French law texts. But this should apply to about any country's law.

## The gist of RAG

Essentially, pre-trained LLMs are good chatty chatters but not really useful, on their own, on specific topics. This is because the domain-specific knowledge either got diluted in the training data, was not in the said training data, or is newer than the last training. So one thing that could be done is, upon receiving some question or query, is somehow retrieve a relevant document or paragraph and feed its content as part of the query.

In the implementation, I use a vector database to get the most relevant document.

Initially I started using PyMilvus, as I already had a nice wrapper from some previous project (that I might write about at some point). But then I realized it'd be more versatile to use a simple vectorizer in the form of SentenceTransformers, without all the bells and whistle of Pymilvus (which retrospectively does not seem to bring that much to the table, but I'm probably missing the point given my current knowledge of what it does).

As of the the LLM I use LLama2 7B-chat directly from MetaAI's repo. Because I want to be able to experiment on the cheap, on my home's GPU (RTX3080), I find it strikes a good balance of precision and VRAM requirements. On that said, it did force me to rewrite some of the code to avoid having torch DDP throw a million errors on my home rig with a single GPU, and everything running inside WSL2.

## Why RAG rather than fine-tuning?

While conceptually simple, RAG does bring in quite a few extra components. One have to worry about data preparation for better retrieval, finding a suitable vectorizer and distance metric choice, store the documents in an efficient format, do some query engineering. Additionally, upon inference, in a live context, one would have to handle talking to 2 databases or stores (one for the vector embeddings and one for the source data), as well as handle
having at least 2 AI models (albeit one of them being relatively small).

All of this makes the RAG perhaps a bit more complex to implement than a simple fine-tuning, where you end up with a single model and that's it. But it does have a few advantages over a fine-tuned model:

* it is very economical. One does not have to fine-tune any model which can quickly cost thousands of dollars for a single run (and you'd better pray it
works well enough the first time around). Instead, one can use a pre-trained model, which is free, and only pay for the vectorization of the documents, which is the same cost as regular inference. And in most instances, with a relatively small model like LLama2-7B, or ome other larger quantized model, it can even fit on consumer grade hardware.
* it is more flexible. One can easily add new documents to the database, or remove some, without having to retrain the model. This is especially useful when the documents are updated regularly, or when one wants to add some documents on the fly, like in a chatbot. In our case it really is important as the law texts are updated regularly. Also, a number of court cases end up being more or less equivalent to a real law (like supreme court decisions in the US for instance or "jurisprudence" in the french system).
* in many instances, fine-tuned LLMs are very good at capturing styles, but not so much as keeping their focus on just the content of the fine-tuning dataset. This makes it easy for such model to start extrapolating from prior knowledge rather than use the provided training data. With RAG, the retrieval step itself often provides to match quality metric which will ensure the model will see the relevant documents only and is specifically asked to talk about. This significantly reduces the risk for hallucinations, especially when dealing with models fine-tuned with RLHF, where instruction following often is a key training exercise.


## Efficient data storage

A key design choice is that of the data storage. Ideally it should be both fast to query, require as little storage space as possible, and ideally convenient to use.

In this instance, the first step in making a good decision is to analyze the sort of data we are going to work with, and what we want to feed our final augmented chat model with. The French law "code books" are decided in Chapters, books, subsections with their own title and articles, the body of which being the actual law text. I believe that in most cases, knowing such context of an article is very important to understanding what a text of law should apply to. For instance, there is a whole text dedicated to Mayotte, an island with a specific jurisdiction. If the chat model was provided with the article's body without the book's title it may very well provide erroneous advice. Essentially, what we therefore want is to feed the chat model with a string for each article, containing the context of the article and it's body. So we need to store both.

Now this is well structured data (granted it mirrors the law books' own structure). So we can envision a simple SQL database with 2 tables, one for the context and one for the articles, having a many-to-one relationship between the articles and their context. I'm using SQLite and sqlalchemy (to handle everything easily in python) as an ORM here.

One advantage of this approach is that I can easily mirror the entities with `Pydantic` objects, which upon parsing the original pdf documents (as downloaded from the French justice department's website), will ensure that whatever data I extract always matches the database structure.

As of the vector database itself, I could use a fancy architecture, which would make retrieval faster, but given the data at hand is so small (a single book is around 3000 articles, in the case of the "code civil" for instance, so in total we might end up handling a few tens of thousands embeddings max) and should not grow in any significant proportion, a simpler architecture is much more convenient. As I mentioned earlier, I initially started the project using PyMilvus, which indeed handles all of the storing and retrieving data, in an efficient fashion, for us. But in practice, given I did not really care about advanced nearest neighbour estimation algorithms for the retrieval, using Sentence transformers with a brute force retrieval using cosine similarity, felt like a more versatile approach. This is something I might revisit later, should the need for optimization appear.

Because I did not use a premade database I had to choose my storage myself. I did not do anything fancy here, using a tuple of the embedding and the associated Article unique identifier from the SQLite database for each entry. This is in turn saved usin pytorch `torch.save` and `torch.load`. Now this has 2 main issues:

* The embeddings cannot be used a key to a dictionary as their hash is the python object `id`, which changes for each new tensor. This in turn means that 2 identical embeddings allocated as different objects in memory will not match one another if used as key. We therefore have to keep 2 ordered lists when doing the retrieval operations, using the index in the list of the closest embedding to that of the query (using cosine similarity in our case) to retrieve the article uid. This is really ugly and not robust to insertions and deletions. For instance, should I want to add an updated version of a law book later on, I can't know for sure that I am not duplicating an entry from this storage alone. I'll have to rely on the SQLite database for this, which is not ideal.
* Pytorch save and load functions where written with tensor data in mind. While it is very capable of storing arbitrary serializable data structures, given it is defaulting to using `pickle`, it is probably not where it is the most efficient. It also brings with it all the safety issues of pickle, for instance that an ill-intentioned actor could replace the storage file with malicious code which would run upon deserialization. I will definitely revisit this storage format, should this project go any further...

If we want to add a new law article, we first add it's context to the SQLite database if it does not already exist, and then add the article. Then we pass the article's text (as generated from the body and context), jointly with the SQLite article's uid to the vector database.

For retrieval, we get the closest embedding to that of the question/query, get the associated SQLite article uid and query it, before passing it to the chat model.

## Vectorization in French

I have been using this concept of vector database a lot. Essentially, what they are is a form of database which stores unstructured documents (such as plain text, images or audio) in a vector form. In this context, a vector is an array of number which captures the essence of the object, much like the embeddings produced by a neural network. Note that they do not need to allow the reconstruction of the source data. Instead they should be a unique encoding which minimises the distance in vector space with similar unstructured objects. In our context, what we want is some encoding which would map a law article to a vector close to that representing a question which answer is in the said article.

Finding the right vectorizer do do just is a non-trivial task, especially when it comes to french text. There are of course some very good models these days that produce quality embeddings (e.g. [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)), all of which can be found at the top of the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). However they oftentimes work well on small sentences only. And even when they don't very few actually handle french. There are a few multilingual models, with which I have had mixes results. The only model that has had a dedicated fine-tuning for french is the meme-named CamemBert. While this is indeed a fairly capable model, it is after all a fine-tuned version of Roberta, with a limited capability. And as such I have found it did not exhibit the subtleties required for semantic retrieval when dealing with the law (which is often written in non-sensical jargon).

To alleviate this issue, I resulted to using an extra layer of translation. I was lucky enough that Seamless4t just got released on a permissive licence by MetaAI. And this model does wonders. So now, before passing the text through the vectorizer, it is first translated. And at inference, the question also gets translated.

The good news is that allowed me to use SOTA vectorizers for semantic retrieval such as [`bge_reranker`](https://huggingface.co/BAAI/bge-base-en-v1.5). I'm using the base version for faster throughput and lower VRAM requirements. But the large version does perform better.

The downside obviously is an increased inference latency as well as longer time for populating the database.

## The legal advice generator in action

TBC...

## Towards a nice (non-dev) user experience