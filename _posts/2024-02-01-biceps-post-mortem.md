---
layout: post
title:  "I closed my AI startup after 3 months, here's why"
date:   2024-02-01 13:41:03 +0200
categories: Startup AI
mathjax: false
---


In November 2023, I decided to launch a startup called Biceps AI. Like all new entrepreuners I was super excited with the project, dreaming of ARR in the millions of dollars with unicorn valuation and such. And yet, 3 months later I decided to kill the project. In this blog-post I'll reflect on why I killed the project, and what I did wrong so that hopefully you and I can learn something and become better persons.

## What Biceps first product was supposed to be

Biceps was to provide a software solution which allows distributed training of deep learning models in the cloud in a single line of code, with automatic and infinite scalability. This should have taken away all the complexity of training in the cloud (faster development, no need for a dedicated engineering team) and significantly reduced compute costs (pay only for actual compute time, not the dev time inside the virtual machines), while providing state of the art distributed methods. If you are lucky, the company's website (or rather the proto-version of it) might still up over at [biceps.ai](https://biceps.ai).

Implementation-wise all the code would run on the user machine, and data and models would be sent, with high security, to GPU workers in the cloud which would only execute this part of the computation.

As of the latest product iteration, the whole library would have been open-source, with 4 main components:
* auto model memory requirements evaluation, and auto wrapping for distributed training (data + model parallel).
* plugins for provisioning workers in the cloud of choice. For instance there would have been an AWS EC2 plugin, a GCP plugin, a Lambda Labs one and so on. This is were open§source shines as users could have provided their very own plugins to use their preferred cloud providers if not already supported, or even run on their own private on-premise VPC.
* workers that stream the data and model weights and run and log the training computation.
* an “orchestration” system that relaunches nodes that die and resumes computation if needed.
Users could then pay to have access to a multi-cloud deployment that would maximize availability (GPUs are hard to come by these days) by deploying simultaneously in several clouds and always fetching the cheapest available options. It would also have provided secure data and model hosting, with proper versioning and perhaps, auto endpoint deployement in a later upgrade of the product.


## Why would I stop

TLDR; there were no obvious market opportunity, doubled with a few very good tools that already address the needs in an efficient manner, further reducing the needs for something else.

As of today, most of the AI market focuses on fine-tuning open source models, for which there already are a plethora of ready made solutions available. In this instance flexibility is not really a competitive argument, as users usually prefer readily available, well made, efficient task-specific. Trainings are then run inside a dedicated container, typically in RunPod, Lambda Labs or inside a notebook in Azure ML Playground, AWS Sagemaker Studio, Lightning AI’s Grid studios or Google collab. Because everything is already pretty much developed by libraries such as Axolotl (which specializes in LLM fine-tuning), there really isn’t much development time or effort at all. As such these users and companies are not a good fit for the product.

The rest of the market is divided between very large foundational model providers and small R&D focused companies. The former spend millions of dollars in cloud compute and will not require a product like what I intended to offer as they will most likely get assistance from cloud providers, given the money they spend, and also because it somehow is part of their value proposition, to be able to train at scale on thousands of GPUs/XPUs.

As of R&D focused companies, all those that I have talked to focus on very specific tasks for which there often is limited data and compute available and so mostly use on-premise hardware, with the occasional support of an extra cloud-based accelerator. So while they seem like the part of the market that would benefit the most from the product, in practice, it is not really interesting to them, or at least I am yet to find a company of this type that would be interested.

## A few noteworthy competitors

I already mentioned Lightning AI. Their initial product, Pytorch Lightning is already used across many companies and research institutes (free OSS library). Their newer offering, and only payed-for product, called "studios", seems much more geared towards recent fine-tuning workloads. I think that their in-browser IDE/container based system is very limiting for custom jobs, as they do not really allow for reusing code and company code (or would require quite a lot of setup for that) but it is well tailored for the LLM fine-tuning market where trainings are mostly self-contained within a single script and a few pip dependancies. Combined with Pytorch Lightning model wrappers this is in any case a very compelling software solution.

Dstack is another interesting company in this space which auto deploys any training script in the cloud, with as many GPUs as you so wish, directly with a CLI. They do not provide any software wrapper to distribute the training computation though. Also, to the best of my understanding, all the code should basically live in a single script which has a lot of limitations. My initial idea was really to make a “Dstack meets Pytorch Lightning” kind of deal, taking the best of both worlds, while allowing for more custom jobs, and good integration with software libraries and large code repositories.

SkyPilot is the project that, when I discovered it, put the nail in the coffin for Biceps. If you are happy with running your code in a container, they do provide all the features I had in the product proposal, in a very well made way, in FOSS, and maintained by Berkeley university. They have auto-provisioning, job queues, autoscaling for inference, automatic recovery from preemptions... So while they do not provide auto distribution of the compute, the cloud handling is just top-notch.

## A couple mistakes along the way

In many ways I think I could have figured out all this much sooner. While I don't regret one bit venturing this far, meeting great people along the way and learning a lot, I do think I made mostly 2 big mistakes that, if avoided, would have saved quite some time.

My former employer falls inside the category of smaller R&D focused companies. As such we ran on on-premise hardware, using at most a few GPUs at once of limited size (RTX 3090 or smaller) with relatively small models of <30M parameters (most of the memory requirements steming from working large 3D images rather than from the model weights). My first mistake was to assume that the problems I experienced there, and which contributed to me starting the Biceps project, would translate to cloud users that run larger models. This feeling got further reinforced by my time at Graphcore, a hardware AI accelerator maker, where I helped customers make models train efficiently, in a time where the open-source ecosystem was not as developped as it is today (things move soooooo fast!). These led me to the false asumption on the workloads most users are running.

Of course this wouldn’t have been really detrimental had I not made the second mistake: conducting my initial user study poorly. When you come to think of it, the product I was developing could have many potential applications, from R&D to devops, MLOps, AIaaS, very large model training… And it may solve a range of problems: need for dedicated skills for running in the cloud and time needed for maintaining the related infrastructure, need for dedicated skills and engineering time to perform efficient distributed training, lack of flexibility of containerized approaches (especially if you scale to more than one container at a time), cost of compute and even more so development time where the costly GPUs are under-utilized… And this lack of focus (at least initially) led me to asking my users about their pains, identifying a wide array of issues related to training models, and never really narrowing down on a unified solution to all the issues I was hearing. I instead, uncounciously, tried to make their problems fit to my solution. As such the product I tried to build could indeed solve their pains individually, but never really became something they would have used, let alone buy, because it was not really useful to them. For instance I met with this one AI for heavy industry company which was super interested in the security part of the product but made very small models and as such did not really care about my product in the end as most of their work was on-premise.

Facing the initial uncertainty, I decided to make a prototype to show users and help them understand what I wanted to offer. In turn I was hoping this would help me understand wether it would be a good fit. Some might say it was too early. Personally, I am still not really sure whether it was or not. What is certain though, is that when I had finished the prototype, and set out to find companies to talk to (rather than individual users from my network), I really struggled to find any at all that I knew for certain could be interesting. I really had to convince myself that they might be, which never is a good sign. Companies were either too big or too mature to require a product like mine, having already built a solution internally, or too small to need such a powerful solution, when simply running in a dedicated container was good enough for most workloads they might have (if they even ran in the cloud that is).
So I decided instead to walk back and talk to a lot more users to try to revamp the product to something more in accordance with their actual needs. To better focus my research I went on github and contacted as many people that starred the Dstack and SkyPilot repositories. I also used Linkedin to find NLP and AI R&D engineers which I thought would be potential users. And that’s where I realized there wasn’t a market really for the product. People do think in general there is a lack of maturity and unity in the tools still, but none really showed interest for the product I was building, saying that for the most part they already had found other solutions they were mostly happy with.

## What now?

I think I'll still rearrange my prototype into a nice demonstrator. I am now moving forward, and looking for a new project to contribute to. And hopefully I won't make the same mistake again, and you neither.
