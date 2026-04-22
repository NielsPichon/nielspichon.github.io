<script>
  let { post } = $props();
</script>

<article class="post-item">
  <a href="/posts/{post.slug}" class="post-link">
    <div class="post-meta">
      <span>{post.date}</span>
      <span>{post.readTime}</span>
      {#if post.tag}<span class="post-tag">{post.tag}</span>{/if}
    </div>
    <h2 class="post-title">{post.title}</h2>
    {#if post.teaser}<p class="post-teaser">{post.teaser}</p>{/if}
    {#if post.paper}<p class="post-paper">↳ {post.paper}</p>{/if}
    <span class="post-arrow">→</span>
  </a>
</article>

<style>
  .post-item {
    padding: 1.8rem 0;
    border-bottom: 1px solid var(--bg2);
    position: relative;
    transition: border-color 0.2s;
  }

  .post-item:hover {
    border-bottom-color: var(--accent);
  }

  .post-link {
    display: block;
    color: inherit;
  }

  .post-meta {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.06em;
    color: var(--ink3);
    margin-bottom: 0.5rem;
    display: flex;
    gap: 1.2rem;
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

  .post-title {
    font-family: var(--serif);
    font-size: 1.6rem;
    line-height: 1.2;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin-bottom: 0.5rem;
    transition: color 0.15s;
    font-weight: normal;
  }

  .post-item:hover .post-title {
    color: var(--accent);
  }

  .post-teaser {
    font-family: var(--body);
    font-size: 1rem;
    color: var(--ink2);
    line-height: 1.6;
    max-width: 540px;
  }

  .post-paper {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.04em;
    color: var(--ink3);
    margin-top: 0.7rem;
    font-style: italic;
  }

  .post-arrow {
    position: absolute;
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0;
    transition: opacity 0.2s, transform 0.2s;
    color: var(--accent);
    font-size: 1.2rem;
  }

  .post-item:hover .post-arrow {
    opacity: 1;
    transform: translateY(-50%) translateX(4px);
  }
</style>
