<script>
  import '../app.css';
  import Nav from '$lib/components/Nav.svelte';
  import Footer from '$lib/components/Footer.svelte';
  import favicon from '$lib/assets/favicon.svg';

  let { children } = $props();
  let dark = $state(false);
</script>

<svelte:head>
  <link rel="icon" href={favicon} />
  <meta name="description" content="Electron Avalanche — A blog about AI" />
</svelte:head>

<div class:dark>
  <Nav bind:dark />
  <main>
    {@render children()}
  </main>
  <Footer />
</div>

<style>
  div {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background: var(--bg);
    transition: background 0.3s;
  }

  main {
    flex: 1;
  }
</style>
