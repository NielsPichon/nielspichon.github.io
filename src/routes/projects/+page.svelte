<script>
  import { Card } from '$lib/components/ui/card/index.js';
  let { data } = $props();
</script>

<svelte:head>
  <title>Projects — Electron Avalanche</title>
  <meta name="description" content="Projects by Niels Pichon." />
</svelte:head>

<div class="mx-auto max-w-3xl px-5 py-16 sm:px-8 sm:py-24">
  <header class="fade-up border-b pb-10">
    <p class="mb-3 text-sm font-medium text-muted-foreground">Selected work</p>
    <h1 class="text-4xl font-bold tracking-tight sm:text-5xl">Projects</h1>
    <p class="mt-4 text-lg leading-8 text-muted-foreground">Things I have built, researched, and explored.</p>
  </header>

  <section class="pt-10">
    {#if data.projects.length}
      <div class="space-y-4">
        {#each data.projects as project}
          <Card class="p-6">
            <p class="text-xs text-muted-foreground">{project.date}</p>
            <h2 class="text-xl font-semibold tracking-tight">{project.title}</h2>
            {#if project.description}<p class="leading-7 text-muted-foreground">{project.description}</p>{/if}
          </Card>
        {/each}
      </div>
    {:else}
      <Card class="items-center px-6 py-14 text-center shadow-none">
        <h2 class="text-lg font-semibold">No projects yet</h2>
        <p class="max-w-sm text-sm leading-6 text-muted-foreground">Projects will appear here as they are published.</p>
      </Card>
    {/if}
  </section>
</div>
