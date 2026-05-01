<script>
  import { onMount, tick } from 'svelte';
  import hljs from 'highlight.js/lib/core';
  import python from 'highlight.js/lib/languages/python';

  let { data } = $props();
  let mermaid;
  let mermaidInitialized = false;

  hljs.registerLanguage('python', python);

  async function renderMermaidBlocks() {
    await tick();

    const blocks = document.querySelectorAll(
      '.post-body pre code.language-mermaid, .post-body pre code.lang-mermaid'
    );

    if (!blocks.length) {
      return;
    }

    if (!mermaid) {
      const module = await import('mermaid');
      mermaid = module.default;
    }

    if (!mermaidInitialized) {
      const isDarkTheme =
        document.documentElement.classList.contains('dark') ||
        document.body.classList.contains('dark');

      mermaid.initialize({
        startOnLoad: false,
        theme: isDarkTheme ? 'dark' : 'default'
      });
      mermaidInitialized = true;
    }

    let chartIdx = 0;
    for (const block of blocks) {
      const chart = block.textContent?.trim();
      const pre = block.closest('pre');

      if (!chart || !pre) {
        continue;
      }

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
      if (block.classList.contains('language-mermaid') || block.classList.contains('lang-mermaid')) {
        return;
      }
      hljs.highlightElement(block);
    });
  }

  async function processPostBody() {
    await renderMermaidBlocks();
    await highlightCodeBlocks();
  }

  onMount(() => {
    processPostBody();
  });

  $effect(() => {
    data.content;
    processPostBody();
  });
</script>

<svelte:head>
  <title>{data.metadata?.title ?? 'Post'} — Electron Avalanche</title>
</svelte:head>

<article class="post-page fade-up">
  <a href="/" class="back-btn">← All posts</a>

  <header class="post-header">
    <div class="post-header-meta">
      {#if data.metadata?.date}<span>{data.metadata.date}</span>{/if}
      {#if data.metadata?.readTime}<span>{data.metadata.readTime}</span>{/if}
      {#if data.metadata?.tags?.length}
        {#each data.metadata.tags as tag}
          <span class="post-tag">{tag}</span>
        {/each}
      {/if}
    </div>
    <h1 class="post-header-title">{data.metadata?.title}</h1>
    {#if data.metadata?.paper}
      <p class="post-header-paper">{data.metadata.paper}</p>
    {/if}
  </header>

  {#if data.metadata?.code}
    <p class="post-code-link">
      <a
        class="about-link"
        href={data.metadata.code}
        target={data.metadata.code.startsWith('http') ? '_blank' : undefined}
        rel={data.metadata.code.startsWith('http') ? 'noopener noreferrer' : undefined}
      >Code</a>
    </p>
  {/if}

  <div class="post-body">
    <data.content />
  </div>
</article>

<style>
  .post-page {
    max-width: calc(var(--max) + var(--page-pad) * 2);
    margin: 0 auto;
    padding: 0 var(--page-pad) 6rem;
  }

  .back-btn {
    display: block;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink3);
    padding: 1.5rem 0 0;
    transition: color 0.15s;
  }

  .back-btn:hover {
    color: var(--accent);
  }

  .post-header {
    padding: 2rem 0 2.5rem;
    border-bottom: 1px solid var(--bg2);
  }

  .post-header-meta {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink3);
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .post-tag {
    background: var(--accent-light);
    color: var(--accent);
    padding: 1px 7px;
    font-size: 9px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .post-header-title {
    font-family: var(--serif);
    font-size: clamp(2rem, 5vw, 3.2rem);
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-bottom: 1rem;
    font-weight: normal;
  }

  .post-header-paper {
    font-family: var(--body);
    font-size: 0.95rem;
    font-style: italic;
    color: var(--ink3);
    border-left: 2px solid var(--accent);
    padding-left: 0.8rem;
    margin-top: 1rem;
  }

  .post-code-link {
    margin: 0;
    padding: 1.25rem 0 0;
  }

  .post-body {
    padding-top: 2.5rem;
  }

  :global(.post-body p) {
    margin-bottom: 1.4rem;
    font-size: 1.1rem;
    line-height: 1.75;
    color: var(--ink2);
  }

  :global(.post-body p:first-child) {
    color: var(--ink);
    font-size: 1.18rem;
  }

  :global(.post-body ul),
  :global(.post-body ol) {
    margin: 0 0 1.4rem;
    padding-left: 1.5rem;
  }

  :global(.post-body li) {
    font-family: var(--body);
    font-size: 1.1rem;
    line-height: 1.75;
    color: var(--ink2);
    margin-bottom: 0.4rem;
  }

  :global(.post-body li > ul),
  :global(.post-body li > ol) {
    margin-top: 0.4rem;
    margin-bottom: 0.7rem;
    padding-left: 1.3rem;
  }

  :global(.post-body h2) {
    font-family: var(--serif);
    font-size: 1.5rem;
    color: var(--ink);
    margin: 2.5rem 0 0.8rem;
    letter-spacing: -0.01em;
    font-weight: normal;
  }

  :global(.post-body h3) {
    font-family: var(--mono);
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 2rem 0 0.6rem;
  }

  :global(.post-body blockquote) {
    border-left: 2px solid var(--accent);
    margin: 2rem 0;
    padding: 0.5rem 0 0.5rem 1.2rem;
    font-style: italic;
    color: var(--ink2);
    font-size: 1.15rem;
  }

  :global(.post-body code) {
    font-family: var(--mono);
    font-size: 0.82em;
    background: var(--bg2);
    padding: 1px 5px;
  }

  :global(.post-body pre) {
    background: var(--bg2);
    padding: 1.2rem;
    margin: 1.5rem 0;
    overflow-x: auto;
    font-family: var(--mono);
    font-size: 0.85rem;
    line-height: 1.6;
  }

  :global(.post-body pre code) {
    background: none;
    padding: 0;
  }

  :global(.post-body pre code.hljs) {
    color: oklch(30% 0.01 260);
  }

  :global(.post-body .mermaid-diagram) {
    background: var(--bg2);
    padding: 1.2rem;
    margin: 1.5rem 0;
    overflow-x: auto;
  }

  :global(.post-body .mermaid-diagram svg) {
    display: block;
    max-width: 100%;
    height: auto;
    overflow: visible;
  }

  :global(.post-body .mermaid-diagram svg foreignObject) {
    overflow: visible;
  }

  :global(:not(.dark) .post-body .mermaid-diagram svg text),
  :global(:not(.dark) .post-body .mermaid-diagram svg tspan),
  :global(:not(.dark) .post-body .mermaid-diagram svg .label),
  :global(:not(.dark) .post-body .mermaid-diagram svg foreignObject div) {
    fill: oklch(28% 0.01 260) !important;
    color: oklch(28% 0.01 260) !important;
  }

  :global(.dark .post-body .mermaid-diagram) {
    background: oklch(18% 0.008 80);
  }

  :global(.dark .post-body .mermaid-diagram svg .node rect),
  :global(.dark .post-body .mermaid-diagram svg .node circle),
  :global(.dark .post-body .mermaid-diagram svg .node ellipse),
  :global(.dark .post-body .mermaid-diagram svg .node polygon),
  :global(.dark .post-body .mermaid-diagram svg .cluster rect),
  :global(.dark .post-body .mermaid-diagram svg .labelBkg) {
    fill: oklch(30% 0.012 80) !important;
    stroke: oklch(62% 0.012 80) !important;
  }

  :global(:not(.dark) .post-body pre code.hljs .hljs-string) {
    color: oklch(42% 0.07 40);
  }

  :global(.dark .post-body pre code.hljs) {
    color: oklch(88% 0.01 80);
  }

  :global(.post-body img) {
    max-width: 100%;
    height: auto;
    margin: 2rem 0;
    display: block;
  }

  :global(.katex-display) {
    margin: 1.5rem 0;
    overflow-x: auto;
  }

  :global(.dark .post-body .katex),
  :global(.dark .post-body .katex-display) {
    color: oklch(96% 0.004 80);
  }
</style>
