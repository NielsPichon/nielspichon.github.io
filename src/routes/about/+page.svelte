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
      I have come to realise that it quickly gets easy to get out of touch with the
      broader litterature when you have your head down to a specific applied research project,
      especially as moving fast often means cutting corners.
      To avoid losing the plot, I have set as an exercise to myself to read and explain at least one paper a week here.
      I'll also share thoughts and ideas of research topics.
    </p>
    <p>
      The name comes from the physics phenomenon where a single electron triggers a cascade of
      ionization events. Feels right for how ideas in AI tend to propagate.
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
