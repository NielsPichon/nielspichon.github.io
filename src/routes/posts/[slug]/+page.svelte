<script>
  let { data } = $props();
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
      {#if data.metadata?.tag}<span class="post-tag">{data.metadata.tag}</span>{/if}
    </div>
    <h1 class="post-header-title">{data.metadata?.title}</h1>
    {#if data.metadata?.paper}
      <p class="post-header-paper">{data.metadata.paper}</p>
    {/if}
  </header>

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
</style>
