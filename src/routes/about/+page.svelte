<svelte:head>
  <title>About — Electron Avalanche</title>
</svelte:head>

<div class="about-page fade-up">
  <header class="about-header">
    <h1 class="about-name">Niels <em>Pichon</em></h1>
    <p class="about-title">AI researcher & engineer</p>
  </header>

  <hr class="about-divider" />

  <div class="about-body">
    <p>
      I read AI papers the way some people read novels — compulsively, argumentatively, with a lot of
      margin notes. This blog is where I put those notes in order.
    </p>
    <p>
      The name comes from the physics phenomenon where a single electron triggers a cascade of
      ionization events. Feels right for how ideas in AI tend to propagate.
    </p>
    <p>
      I write about what I find genuinely interesting, not what's trending. Mostly transformers,
      scaling, and the occasionally unsettling implications of both.
    </p>
  </div>

  <div class="about-links">
    <a href="https://github.com/nielspichon" class="about-link" target="_blank" rel="noopener">GitHub</a>
  </div>
</div>

<style>
  .about-page {
    max-width: calc(var(--max) + var(--page-pad) * 2);
    margin: 0 auto;
    padding: 0 var(--page-pad) 6rem;
  }

  .about-header {
    padding: 3rem 0 2rem;
  }

  .about-name {
    font-family: var(--serif);
    font-size: clamp(2.5rem, 7vw, 4.5rem);
    line-height: 1.05;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
    font-weight: normal;
    color: var(--ink);
  }

  .about-name em {
    font-style: italic;
  }

  .about-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink3);
  }

  .about-divider {
    border: none;
    border-top: 1px solid var(--bg2);
    margin: 2rem 0;
  }

  .about-body p {
    margin-bottom: 1.3rem;
    font-size: 1.1rem;
    line-height: 1.75;
    color: var(--ink2);
  }

  .about-body p:first-child {
    color: var(--ink);
    font-size: 1.18rem;
  }

  .about-links {
    display: flex;
    gap: 1.5rem;
    margin-top: 2.5rem;
    flex-wrap: wrap;
  }

  .about-link {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--accent);
    padding-bottom: 1px;
    transition: opacity 0.15s;
  }

  .about-link:hover {
    opacity: 0.6;
  }
</style>
