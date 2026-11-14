<script>
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import PostCard from '#lib/components/PostCard.svelte';
  import { Button } from '#lib/components/ui/button/index.js';
  import { Card } from '#lib/components/ui/card/index.js';
  let { data } = $props();

  const latestPost = $derived(data.posts[0]);
  const latestProject = $derived(data.projects[0]);
</script>

<svelte:head>
  <title>Electron Avalanche</title>
  <meta name="description" content="Cascading ideas in AI, computer vision, and related fields." />
</svelte:head>

<div class="mx-auto max-w-3xl px-5 py-16 sm:px-8 sm:py-24">
  <section class="fade-up border-b pb-12">
    <p class="mb-4 text-sm font-medium text-muted-foreground">Cascading Ideas</p>
    <h1 class="text-4xl font-bold tracking-tight sm:text-6xl">Electron Avalanche</h1>
    <p class="mt-5 max-w-2xl text-lg leading-8 text-muted-foreground">Cascading ideas in AI, computer vision, and related fields.</p>
  </section>

  <section class="pt-12">
    <div class="mb-6 flex items-center justify-between">
      <h2 class="text-lg font-semibold">Latest thought</h2>
      <Button href="/thoughts" variant="ghost" size="sm">All thoughts <ArrowRight class="size-4" /></Button>
    </div>
    {#if latestPost}
      <PostCard post={latestPost} />
    {:else}
      <Card class="px-6 py-10 text-center text-sm text-muted-foreground shadow-none">No thoughts yet.</Card>
    {/if}
  </section>

  <section class="pt-12">
    <div class="mb-6 flex items-center justify-between">
      <h2 class="text-lg font-semibold">Latest project</h2>
      <Button href="/projects" variant="ghost" size="sm">All projects <ArrowRight class="size-4" /></Button>
    </div>
    {#if latestProject}
      <a href={latestProject.href} class="group block rounded-xl focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
        <Card class="gap-3 p-6 transition-colors hover:bg-accent/50 hover:shadow-md">
          {#if latestProject.date}<p class="text-xs text-muted-foreground">{latestProject.date}</p>{/if}
          <h3 class="text-xl font-semibold tracking-tight group-hover:underline group-hover:underline-offset-4">{latestProject.title}</h3>
          {#if latestProject.description}<p class="leading-7 text-muted-foreground">{latestProject.description}</p>{/if}
        </Card>
      </a>
    {:else}
      <Card class="items-center px-6 py-12 text-center shadow-none">
        <h3 class="text-base font-semibold">No projects yet</h3>
        <p class="text-sm text-muted-foreground">The first project will appear here when it is published.</p>
      </Card>
    {/if}
  </section>
</div>
