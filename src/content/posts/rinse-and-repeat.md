---
title: "Rinse and Repeat: scale models at inference"
date: "May, 2026"
tag: ["LLM"]
teaser: "Can you scale model capacity at inference?"
code: https://github.com/NielsPichon/RinseAndRepeat
---

Transformer based architectures for the most part keep the latent space of each layer at the same dimension. So it is very possible that latent spaces, deep enough inside the model, are in fact compatible, with the model merely rearranging data within the same space, essentially operating some recursive function. By compatible it is to be understood that a layer can reason on its output space. If that is the case, we could in theory keep repeating a layer and it could possibly improve the model's response, similarly to how pushing a model to use `<think>` in its output triggers reasoning and improves significantly the output.

This is obviously not really founded on any actual evidence, and an exhaustive exploration would be needed to try to identify whether embeddings of a given concept are roughly the same across several layers. But just for fun I created a quick script that takes a pre-trained model and repeats a given layer. The code can be found here. There are 2 experiments, one on the T5 encoder (transformer encoder) and one on Gemma4 (transformer decoder only), as I suspected decoder would be problematic here since each token attends only to those before itself and thus repeating a layer could induce a progressive shift, although in practice I did not observe any significant difference.

Interestingly the results remain intelligible for a few repeats, but as the number of repeats increases, it progressively starts collapsing into an endless loop or start using syntax that is less and less intelligible. Also, repeating earlier layers tend to lead to collapse or unintelligible text much faster. It does however take a few repeats before significant changes occur which signal some significant noise robustness in the deeper layers, and this is especially true for the Gemma4 model which even after 100 repeats still is somewhat close to producing proper text.

### Examples using the T5 last encoder layer:
Task: "summarize: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930."

* No repeat: The Eiffel Tower is the tallest structure in the world.
* 2 repeats: The Eiffel Tower is the tallest structure in the world.
* 5 repeats: The Eiffel Tower is the tallest structure in Paris.
* 10 repeats: The edifice is a tall structure, but it is not tall enough to be a tall building.
* 50 repeats: Put a stop to the putting a stop to the putting a stop to the putting a stop to the putting a ...

### Examples with Gemma 4 E2B-it
Task:
```python3
messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
```

* No repeat: This image shows a **raccoon** in the snow. It is a medium-sized mammal with gray and brown fur, and it appears to be walking or standing on a snowy or icy surface. The background consists of a wall made of vertical, light-colored wooden logs or posts, suggesting an outdoor or winter setting.
* 2 repeats: The image shows a **wolverine** in the snow. It is a medium-sized, stocky mammal with a stout build, short legs, and a thick, shaggy coat, which is perfectly adapted for cold weather. It is foraging or moving through a snowy, outdoor environment, likely in a forest or wilderness setting, as suggested by the birch-like trees in the background.
* 5 repeats: Aotearoa (New Zealand fur, or possibly a **moa** or a type of **brown fur**), possibly a **marten or a small mammal in a snowy weather is shown in a natural setting, likely during a cold, wintry conditions**. The animal appears to be resting or in a state of deep sleep, it is curled up in a compact ball, suggesting it is a smaller than expected. **Key features:** The animal has thick, dense fur, consistent with cold weather. Snow, ice, and dirt or debris cover its body and the fur and the surroundings are covered in snow, suggesting it is either cold and bundled up. The background is a mix of vertical wood, likely a fence, or a wall of tree trunks, and the fur provides a textural contrast to the bare wood and snow, making the setting stark. **In a real-world context, this is likely a photograph of a wild animal in its winter coat.** Given the shaggy appearance, it could be a type of **marten or a generalized description**, often used for wild animals in snowy environments, such as a **Puff or similar-looking mammal**. It is most likely a native species or a common animal for wildlife photography, resembling a **wolverine or a small bear, based on the size and general build**, it is difficult to identify the exact species without clearer markings. **If this is a wildlife photo, it is very likely a native species, it could be a native animal adapting to harsh conditions, or it could be a brown fox or a small predator resting.** Given the context (white snow and dark fur, curled up tightly), it strongly resembles a **marmot or perhaps a wolverine/small bear in thick winter coat**.
* 10 repeats: The image of the animal is a deliberate blurriness for privacy, and is common in nature photography. In many places are described as **brown or grayish fur tabby in the snowy conditions, it a Siberian tiger or brown bear, a typical representation of a large, shaggy mammal in winter—it is a bear or a grizzly bear or a possible tiger cub or a smaller, wild animal in a heavy snow is a common trope—it is a brown bear or a representative animal, and it a large, shaggy mammal is often depicted as a bear or bear—a common portrayal—it is a bear or grizzly bear—often suggests a bear or brown bear or a potential misidentification—it is a common subject—it is a tiger/bear—a generalized image—it is a bear—a common subject—it is a bear—often the subject of such animals—it is a bear—a general representation—it is a brown bear or grizzly bear—this is a bear—a common visual is often used—a tiger/bear—sometimes—it is a bear—a common stereotype—it is a bear—this is a specific bear—it is a grizzly bear or a young tiger/lion cub—it ... (output was cut)
* 50 repeats: **The animal is a North American animal, likely a "cougar"or a similar to fur cats or small mammals would beagles (a highly debatable)** *The image provided a clear image of a **wildcat or a domestic cat (likely a possible oris)** is a North American cougar (mountain lion) or a **cougar, this is a common, misidentification is highly prone to misidentification (It is a **brown, fluffy animal):**Nomenclature makes accurate identification extremely difficult! The challenge lies in a **"grey tabby" or a **"cougar"or a **"cougar" (large, tawny/brown fur, often confused with a smaller, grey tabby):**Wildcats, cougars or more common cats are often mistaken for "cougars or **The subject is a **brown, fluffy animal**Incorrectly identified as "cougars" (cougars/mountain lions) or **"cougars"
* 100 repeats: The subject is a **a North American animal**raccoon or a similar-sized mammal could be mistaken for a similar animals (likely a **raccoon/badger)A** **raccoon** is in a similar fur animal (maca/a)Badgering canidae)Badgering can be inferred.

I think the most interesting part here is that while the text becomes harder and harder to parse it does feel like the model is "thinking", contradicting itself and building on its ideas. I am not really sure what to make of this, but it sure is interesting to see.
