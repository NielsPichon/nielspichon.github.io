<script>
  import { page } from '$app/state';

  let { dark = $bindable(false) } = $props();
</script>

<nav>
  <a href="/" class="nav-logo">Electron Avalanche</a>
  <div class="nav-links">
    <a href="/" class="nav-link" class:active={page.url.pathname === '/'}>Writing</a>
    <a href="/about" class="nav-link" class:active={page.url.pathname === '/about'}>About</a>
    <button class="nav-link" onclick={() => dark = !dark} aria-label="Toggle dark mode">
      {dark ? 'Light' : 'Dark'}
    </button>
  </div>
</nav>

<style>
  nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.6rem var(--page-pad);
    border-bottom: 1px solid var(--bg2);
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 100;
    transition: background 0.3s;
  }

  .nav-logo {
    font-family: var(--serif);
    font-size: 1.1rem;
    color: var(--ink);
    letter-spacing: -0.01em;
  }

  .nav-links {
    display: flex;
    gap: 2rem;
    align-items: center;
  }

  .nav-link {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--ink3);
    cursor: pointer;
    transition: color 0.15s;
    background: none;
    border: none;
    padding: 0;
  }

  .nav-link:hover, .nav-link.active {
    color: var(--accent);
  }
</style>
