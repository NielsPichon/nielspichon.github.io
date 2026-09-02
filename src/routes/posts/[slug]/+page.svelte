<script>
  import { onMount, tick } from 'svelte';
  import hljs from 'highlight.js/lib/core';
  import python from 'highlight.js/lib/languages/python';
  import ArrowLeft from '@lucide/svelte/icons/arrow-left';
  import ExternalLink from '@lucide/svelte/icons/external-link';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { Button } from '$lib/components/ui/button/index.js';

  let { data } = $props();
  let mermaid;
  let mermaidInitialized = false;
  hljs.registerLanguage('python', python);

  async function renderMermaidBlocks() {
    await tick();
    const blocks = document.querySelectorAll('.post-body pre code.language-mermaid, .post-body pre code.lang-mermaid');
    if (!blocks.length) return;

    if (!mermaid) {
      const module = await import('mermaid');
      mermaid = module.default;
    }
    if (!mermaidInitialized) {
      mermaid.initialize({
        startOnLoad: false,
        theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default'
      });
      mermaidInitialized = true;
    }

    let chartIdx = 0;
    for (const block of blocks) {
      const chart = block.textContent?.trim();
      const pre = block.closest('pre');
      if (!chart || !pre) continue;
      try {
        const id = `mermaid-${Date.now()}-${chartIdx++}`;
        const { svg } = await mermaid.render(id, chart);
        const container = document.createElement('div');
        container.className = 'mermaid-diagram';
        container.innerHTML = svg;
        pre.replaceWith(container);
      } catch (err) {
        console.error('Failed to render Mermaid chart:', err);
      }
    }
  }

  async function highlightCodeBlocks() {
    await tick();
    document.querySelectorAll('.post-body pre code').forEach((block) => {
      if (!block.classList.contains('language-mermaid') && !block.classList.contains('lang-mermaid')) {
        hljs.highlightElement(block);
      }
    });
  }

  async function processPostBody() {
    await renderMermaidBlocks();
    await highlightCodeBlocks();
  }

  onMount(processPostBody);
  $effect(() => {
    data.content;
    processPostBody();
  });

  const jsonLdScript = $derived.by(() => {
    const json = JSON.stringify(data.jsonLd).replace(/</g, '\\u003c');
    return `<script type="application/ld+json">${json}</scr` + `ipt>`;
  });
</script>

<svelte:head>
  <title>{data.metadata?.title ?? 'Post'} — Electron Avalanche</title>
  {@html jsonLdScript}
</svelte:head>

<article class="fade-up mx-auto max-w-3xl px-5 py-10 sm:px-8 sm:py-16">
  <Button href="/thoughts" variant="ghost" size="sm" class="-ml-3 text-muted-foreground">
    <ArrowLeft class="size-4" /> All thoughts
  </Button>

  <header class="mt-8 border-b pb-10">
    <div class="mb-5 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
      {#if data.metadata?.date}<span>{data.metadata.date}</span>{/if}
      {#if data.metadata?.readTime}<span aria-hidden="true">·</span><span>{data.metadata.readTime}</span>{/if}
      {#each data.metadata?.tags ?? [] as tag}<Badge>{tag}</Badge>{/each}
    </div>
    <h1 class="text-3xl font-bold tracking-tight text-balance sm:text-5xl">{data.metadata?.title}</h1>
    {#if data.metadata?.paper}
      <p class="mt-6 border-l-2 pl-4 text-sm italic leading-6 text-muted-foreground">{data.metadata.paper}</p>
    {/if}
    {#if data.metadata?.code}
      <div class="mt-6">
        <Button
          href={data.metadata.code}
          variant="outline"
          size="sm"
          target={data.metadata.code.startsWith('http') ? '_blank' : undefined}
          rel={data.metadata.code.startsWith('http') ? 'noopener noreferrer' : undefined}
        >View code <ExternalLink class="size-3.5" /></Button>
      </div>
    {/if}
  </header>

  <div class="post-body"><data.content /></div>

  {#if data.showCitation && data.citation}
    <footer class="mt-14 border-t pt-8">
      <p class="mb-3 text-sm font-medium">Liked what you read? Cite as</p>
      <p class="text-sm leading-6 text-muted-foreground">
        {data.citation.beforeUrl}<a class="break-all underline underline-offset-4 hover:text-foreground" href={data.citation.url}>{data.citation.url}</a>
      </p>
    </footer>
  {/if}
</article>
