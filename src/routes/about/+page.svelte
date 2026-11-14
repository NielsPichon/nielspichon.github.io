<script>
  import { Button } from '#lib/components/ui/button/index.js';
</script>

<svelte:head><title>About — Electron Avalanche</title></svelte:head>

<article class="fade-up mx-auto max-w-3xl px-5 py-16 sm:px-8 sm:py-24">
  <header class="border-b pb-10">
    <p class="mb-3 text-sm font-medium text-muted-foreground">About</p>
    <h1 class="text-4xl font-bold tracking-tight sm:text-5xl">Niels Pichon</h1>
    <p class="mt-4 text-lg text-muted-foreground">AI researcher & engineer</p>
  </header>
  <div class="space-y-6 py-10 text-[1.05rem] leading-8 text-muted-foreground">
    <p class="text-foreground">Ideas are like electrons: you need to shoot but one to trigger a whole cascade of ionization events.</p>
    <p>Hi! I am Niels, R&D AI engineer/scientist. I write about research ideas I pursue for fun, as well as papers I read and found interesting.</p>
    <p>I am currently working as the AI director, R&D engineer, and Associate Game-Director at Onibi, a small studio working on its first title. My recent research at Onibi involves diffusion for voxel-based 3D models (RF, data collection and curation, SFT, evaluation...), visual world understanding, as well as agents based on world graphs, LLM steering, and hybrid AI-procedural generation workflows for generating dynamic, in-lore NPC behaviour, quests, and dialogues.</p>
    <p>My interests span a broad range of fields, from computer vision and LLMs to music, games, graphic rendering, shaders, and procedural generation.</p>
    <p>If that sounds interesting, or you want to discuss work opportunities, let's chat!</p>
  </div>
  <div class="flex flex-wrap gap-2 border-t pt-8">
    <Button href="https://github.com/nielspichon" variant="outline" target="_blank" rel="noopener">GitHub</Button>
    <Button href="https://www.linkedin.com/in/niels-pichon-bb530510a/?locale=en-US" variant="outline" target="_blank" rel="noopener">LinkedIn</Button>
    <Button href="/Resume-NielsPichon.pdf" variant="outline" download>Resume</Button>
    <Button href="mailto:niels.pchn@gmail.com">Email me</Button>
  </div>
</article>
