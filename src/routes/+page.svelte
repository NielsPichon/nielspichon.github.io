<script>
  import SquiggleUnderline from '$lib/components/SquiggleUnderline.svelte';
  import PostCard from '$lib/components/PostCard.svelte';

  let { data } = $props();
</script>

<svelte:head>
  <title>Electron Avalanche</title>
</svelte:head>

<section class="home-hero fade-up">
  <p class="hero-eyebrow">AI Papers — My Take</p>
  <h1 class="hero-title">
    Electron <em>Avalanche</em>
  </h1>
  <SquiggleUnderline width={420} />
  <p class="hero-subtitle">
    Reading the papers that move AI forward, then telling you what I actually think about them.
  </p>
</section>

<section class="post-list">
  <p class="post-list-header">Essays</p>
  {#each data.posts as post, i}
    <div class="fade-up" style="animation-delay:{0.05 + i * 0.07}s">
      <PostCard {post} />
    </div>
  {/each}
</section>

<style>
  .home-hero {
    padding: calc(var(--page-pad) * 1.5) var(--page-pad) var(--page-pad);
    max-width: calc(var(--max) + var(--page-pad) * 2);
    margin: 0 auto;
  }

  .hero-eyebrow {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 1rem;
  }

  .hero-title {
    font-family: var(--serif);
    font-size: clamp(3rem, 8vw, 5.5rem);
    line-height: 1.05;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-bottom: 0.2rem;
    font-weight: normal;
  }

  .hero-title em {
    font-style: italic;
  }

  .hero-subtitle {
    font-family: var(--body);
    font-size: 1.1rem;
    color: var(--ink2);
    margin-top: 1.2rem;
    max-width: 480px;
    line-height: 1.6;
  }

  .post-list {
    max-width: calc(var(--max) + var(--page-pad) * 2);
    margin: 0 auto;
    padding: 0 var(--page-pad) 5rem;
  }

  .post-list-header {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink3);
    border-top: 1px solid var(--ink3);
    padding-top: 0.8rem;
    margin-bottom: 0.5rem;
  }
</style>
