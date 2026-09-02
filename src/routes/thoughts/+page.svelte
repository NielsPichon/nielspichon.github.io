<script>
  import PostCard from '$lib/components/PostCard.svelte';
  let { data } = $props();
</script>

<svelte:head>
  <title>Thoughts — Electron Avalanche</title>
  <meta name="description" content="Thoughts on AI, computer vision, and related fields." />
</svelte:head>

<div class="mx-auto max-w-3xl px-5 py-16 sm:px-8 sm:py-24">
  <header class="fade-up border-b pb-10">
    <p class="mb-3 text-sm font-medium text-muted-foreground">Writing</p>
    <h1 class="text-4xl font-bold tracking-tight sm:text-5xl">Thoughts</h1>
    <p class="mt-4 text-lg leading-8 text-muted-foreground">Ideas and notes on AI, computer vision, and related fields.</p>
  </header>

  <section class="pt-10">
    <div class="mb-6 flex items-center justify-between">
      <h2 class="text-lg font-semibold">All thoughts</h2>
      <span class="text-sm text-muted-foreground">{data.posts.length} posts</span>
    </div>
    <div class="space-y-4">
      {#each data.posts as post, i}
        <div class="fade-up" style="animation-delay:{0.05 + i * 0.05}s"><PostCard {post} /></div>
      {/each}
    </div>
  </section>
</div>
