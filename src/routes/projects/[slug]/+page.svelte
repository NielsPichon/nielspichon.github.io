<script>
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import { Badge } from '#lib/components/ui/badge/index.js';
  import { Button } from '#lib/components/ui/button/index.js';
  import GlassGallery from '#lib/components/project/GlassGallery.svelte';
  import OrganHologram from '#lib/components/project/OrganHologram.svelte';
  import ProjectVideo from '#lib/components/project/ProjectVideo.svelte';

  let { data } = $props();
  const meta = $derived(data.metadata);
  const aside = $derived(meta.aside);
</script>

<svelte:head>
  <title>{meta.title ?? 'Project'} — Electron Avalanche</title>
  {#if meta.description}<meta name="description" content={meta.description} />{/if}
</svelte:head>

<div
  class="mx-auto px-5 py-10 sm:px-8 sm:py-16 {aside
    ? 'grid max-w-7xl grid-cols-[minmax(0,1fr)] gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:gap-14 lg:py-0'
    : 'max-w-3xl'}"
>
  <article class="fade-up min-w-0 {aside ? 'lg:py-16' : ''}">
    <Button href="/projects" variant="ghost" size="sm" class="-ml-3 text-muted-foreground">
      <ArrowLeft class="size-4" /> All projects
    </Button>

    <header class="mt-8 border-b pb-10">
      <div class="mb-5 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        {#if meta.date}<span>{meta.date}</span>{/if}
        {#each meta.tags ?? [] as tag}<Badge>{tag}</Badge>{/each}
      </div>
      <h1 class="text-3xl font-bold tracking-tight text-balance sm:text-5xl">{meta.title}</h1>
      {#if meta.description}<p class="mt-5 text-lg leading-8 text-muted-foreground">{meta.description}</p>{/if}
    </header>

    <div class="post-body"><data.content /></div>
  </article>

  {#if aside}
    <!-- Sticky on desktop: the panel stays in view while the article scrolls past it. -->
    <aside class="fade-up order-first h-[70svh] min-h-80 lg:sticky lg:top-14 lg:order-none lg:h-[calc(100svh-3.5rem)] lg:py-8">
      {#if aside === 'organs'}
        <OrganHologram />
      {:else if aside === 'gallery'}
        <GlassGallery items={meta.gallery ?? []} aspect={meta.galleryAspect} hint={meta.galleryHint} instagram={meta.instagram} />
      {:else if aside === 'video'}
        <ProjectVideo src={meta.video} poster={meta.poster} label={meta.title} />
      {/if}
    </aside>
  {/if}
</div>
