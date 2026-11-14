<script>
  import '../app.css';
  import { ModeWatcher } from 'mode-watcher';
  import Nav from '#lib/components/Nav.svelte';
  import Footer from '#lib/components/Footer.svelte';

  let { children } = $props();
</script>

<ModeWatcher defaultMode="system" />

<svelte:head>
  <link rel="icon" href="/ElectronAvalancheIcon.ico" sizes="any" />
  <meta name="description" content="Electron Avalanche — A blog about AI" />
</svelte:head>

<div class="flex min-h-screen flex-col bg-background text-foreground transition-colors">
  <Nav />
  <main class="flex-1">
    {@render children()}
  </main>
  <Footer />
</div>
