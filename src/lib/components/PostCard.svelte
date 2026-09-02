<script>
  import ArrowRight from '@lucide/svelte/icons/arrow-right';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { Card } from '$lib/components/ui/card/index.js';
  let { post } = $props();
</script>

<a href="/posts/{post.slug}" class="group block rounded-xl focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
  <Card class="gap-3 p-6 transition-colors hover:bg-accent/50 hover:shadow-md">
    <div class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      <span>{post.date}</span><span aria-hidden="true">·</span><span>{post.readTime}</span>
      {#each post.tags ?? [] as tag}<Badge>{tag}</Badge>{/each}
    </div>
    <div class="flex items-start gap-4">
      <div class="min-w-0 flex-1">
        <h2 class="text-xl font-semibold tracking-tight text-card-foreground group-hover:underline group-hover:underline-offset-4">{post.title}</h2>
        {#if post.teaser}<p class="mt-2 leading-7 text-muted-foreground">{post.teaser}</p>{/if}
        {#if post.paper}<p class="mt-3 text-xs italic text-muted-foreground">Based on: {post.paper}</p>{/if}
      </div>
      <ArrowRight class="mt-1 size-5 shrink-0 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-foreground" />
    </div>
  </Card>
</a>
