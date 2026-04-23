---
title: "From concept to Language - Mapping the English language."
date: "Nov 22, 2025"
readTime: "5 min"
tag: "Idea"
teaser: "How would you leverage AI to invent a new language?"
paper: "Own interest"
---

*Disclaimer: I am no linguist and this work is purely empirical*

At the core of any language is the will to convey ideas, efficiently and with as much nuance as contextually required. Sprinkle some social markers, history and cultural exchanges, biases, and other indirect communication concepts such as implied gradation, judgment or feelings, and you end up with very complex, yet fascinating, systems.

In the context of narrative work, to support immersion, it is common practice for writers to define a number of rules to enforce consistency in the naming of places, people, and sometimes inform scriptures and dialogues. If the goal is completeness, this usually requires tremendous amount of work, and often cannot be reused from people to people as this is often done through manual labour, in part to better capture the richness of the worlds created by the writer.

In this work we will attempt to create a language capable of communicating as many concepts as possible, in as simple a manner as possible, in an automated fashion.

### Associativity as first class citizen

At its core, the language generator we are creating relies on concept association. If we take the example of english, gradation is accomplished by taking an adjective, and adding the -er suffix: big $\rightarrow$ bigger. The Chinese language takes this concept a notch further: each ideogram corresponds to a base idea, that the language can combine to create more complex ideas. The phone for instance, is 电话 Diànhuà, the electric speech. Obviously this very quickly gets somewhat more abstract or poetic, and number of concepts, especially for imported goods or foreign names, are simply phonetic translations. But even then there is an attempt to preserve some meaning. For instance 法国 Faguo, France, is the country (国) of the law (法).

What we want is to take this concept and see how far we can push it. This implies a few steps, assuming we base our language on the modern English language:
* Map out the english language by breaking each word/concept into its core concepts.
* Identify "atomic" concepts, that is those that either derive from no other concept or from which the most amount of other concepts derive. We will also add here that oftentimes we will see concept "loops", that is that only exist as references to other concepts that in turn reference them, creating cycles. In this instance we will try to identify the concepts in the loop with the least amount of external references, and use it as an "atomic" concept.
* Assign individual phonemes to each atomic concept, and recreate every other word by association of phonemes. This step alone will allow us to create a language, but in all likelihood, we will end up with very long words which are not practical neither for writing nor speaking.
* The last step is systemic rules for contracting words. The goal here is to contract words enough such that they can then could be used in a piece of work or spoken in real life.

At this stage there some important observations that need to be made:
* The first one is that of cultural bias. The English language has its own limits and as initially stated, its words are the mirror of english culture, assumptions and biases. Some concept cannot simply expressed in English. An example I like to use is that of "gourmandise" in French, which can be best translated to "self-Indulgence with respect to food". But realistically this not a good translation as this misses the somewhat "positive" aspect and perhaps "cute" of the french word. As such, the new language will also reflect the same English biases.
* These same biases will be echoed in the choice of "phonemes" for each atomic concept, avoiding those that are hard to pronounce for a native English speaker or less common in the English language.
* The concept associativity exploration is not completely "free". There are some concepts that we have chosen to embed into the language from the start. These concepts are that of "action", "past", "future", "more", "less", "negation/absence of", "inverse of", "left/right/up/down". The reason for imposing such concepts as "atomic", is to guide the exploration, and often reduce the number of concepts needed to express another concept, by associativity. This can be seen as somewhat arbitrary, and in many ways it is, and probably is one more expression of the Western language biases. Similarly we restrict the number of concepts to be within 2 and 3, to better enforce concept recursive nature. Lastly we impose that any "subject" be explicitly named. For instance, I go, would be Niels go. As a consequence it is likely that a concept of "someone" will quickly be derived as a subject.

### LLMs to the rescue
English dictionaries, include more than 400k common words (450k for Merriam-Webster Unabridged if i am to trust Mistral's LeChat, 600k for Oxford's English Dictionary which also include older forms and regional words). And even if we were to reduce this to the 35k words estimated to be used at least once in their lifetime by the average English speaker, that remains way too may words for a single person to register, without going completely crazy that is.

As a makeshift solution, I have chosen to use a LLM. The choice of said LLM does not really matter as long as it allows JSON generation (which I find important for automation purposes). For this work we use a System Prompt with the following instructions:

```python
"""
You are a linguistic expert. Your task is to derive a concept from sub-concept(s), in order to progressively build a hierarchy. You can do so by splitting the concept into sub-concepts (sun = light + sky, tree = wood + plant, magic = light + spiritual, rain = sky + water) or by applying "modifiers" from this list: "action", "past", "future", "more", "less" "abscence of", "inverse of", "left", "right", "up", "down". For instance "casting" a spell is action + magic.

You will choose between 2 and 3 concepts.

You will answer using the following JSON schema:

{
	[
		"concept": { // name of the concept
			"sub_concepts": [
				"sub_concept_1",
				"sub_concept_2",
				...
			],
			"modifiers": [
				"modifier_1" // in "action", "past", "future", "more", "less" "abscence of", "inverse of", "left", "right", "up", "down"
				...
			]
		}
	]
}
"""
```

We then run this, providing several words on each query and then arranging them into a massive directed graph.

From there we measure the degree of each node and extract the nodes with highest degree. We also identify and loops.

Note that the prompt does not allow for "leaf" nodes to be formed, despite our initial intentions. This is because empirically we found out LLMs tend to favour this option too often which left us with too many "atomic" words.

Lastly, we "merge" words that have the name concept sub-concepts. These are invariably synonyms, and while they obviously have some nuances in English, these nuances don't seem to be easily captured by our system.

### Contraction rules
This part is tricky. We now have a full "dictionary". But many of the words are >4 syllables long which make them extremely impractical to pronounce.  We even see a number of words with repeating syllables, especially when it comes to gradation, where we tend to have the "more" or "less" phonemes repeated many times. We could decide to arbitrarily cut into the word but then we would end up having many identical words, which is not desired.

This is basically where I am at with this project. Hopefully I'll find the time to do more work on it some day.
