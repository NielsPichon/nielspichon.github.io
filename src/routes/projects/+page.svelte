<script>
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import { Badge } from '#lib/components/ui/badge/index.js';
  import { Card } from '#lib/components/ui/card/index.js';
  let { data } = $props();

  const FILTERS = [
    { id: 'all', label: 'All' },
    { id: 'pro', label: 'Professional' },
    { id: 'personal', label: 'Personal' }
  ];
  let filter = $state('all');
  const shown = $derived(
    data.projects.filter((p) => filter === 'all' || (filter === 'pro') === p.pro)
  );
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
      <div class="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div class="inline-flex rounded-lg border bg-muted p-1" role="group" aria-label="Filter projects">
          {#each FILTERS as f}
            <button
              type="button"
              class="rounded-md px-3 py-1.5 text-sm font-medium transition-colors {filter === f.id
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'}"
              aria-pressed={filter === f.id}
              onclick={() => (filter = f.id)}
            >{f.label}</button>
          {/each}
        </div>
        <span class="text-sm text-muted-foreground">{shown.length} {shown.length === 1 ? 'project' : 'projects'}</span>
      </div>
      <div class="space-y-4">
        {#each shown as project, i (project.slug)}
          <a
            href={project.href}
            class="fade-up group block rounded-xl focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            style="animation-delay:{0.05 + i * 0.05}s"
          >
            <Card class="gap-2 p-6 transition-colors hover:bg-accent/50 hover:shadow-md">
              <div class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                {#if project.date}<span>{project.date}</span>{/if}
                {#if project.pro}<Badge>Professional</Badge>{/if}
              </div>
              <div class="flex items-start gap-4">
                <div class="min-w-0 flex-1">
                  <h2 class="text-xl font-semibold tracking-tight group-hover:underline group-hover:underline-offset-4">{project.title}</h2>
                  {#if project.description}<p class="mt-2 leading-7 text-muted-foreground">{project.description}</p>{/if}
                </div>
                <ArrowRight class="mt-1 size-5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-foreground" />
              </div>
            </Card>
          </a>
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
