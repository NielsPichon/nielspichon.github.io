<script>
  import { page } from '$app/state';
  import { mode, toggleMode } from 'mode-watcher';
  import Moon from '@lucide/svelte/icons/moon';
  import Sun from '@lucide/svelte/icons/sun';
  import { Button } from '#lib/components/ui/button/index.js';
</script>

<header class="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
  <nav class="mx-auto flex h-14 max-w-5xl items-center px-3 sm:px-8" aria-label="Main navigation">
    <a href="/" class="mr-auto text-sm font-semibold tracking-tight sm:mr-8">
      <span class="sm:hidden">EA</span><span class="hidden sm:inline">Electron Avalanche</span>
    </a>
    <div class="flex items-center sm:ml-auto sm:gap-1">
      <Button href="/thoughts" variant="ghost" size="sm" class={page.url.pathname.startsWith('/thoughts') || page.url.pathname.startsWith('/posts/') ? 'bg-accent' : ''}>Thoughts</Button>
      <Button href="/projects" variant="ghost" size="sm" class={page.url.pathname.startsWith('/projects') ? 'bg-accent' : ''}>Projects</Button>
      <Button href="/about" variant="ghost" size="sm" class={page.url.pathname === '/about' ? 'bg-accent' : ''}>About</Button>
      <Button variant="ghost" size="icon" onclick={toggleMode} aria-label="Toggle color theme">
        {#if mode.current === 'dark'}<Sun class="size-4" />{:else}<Moon class="size-4" />{/if}
      </Button>
    </div>
  </nav>
</header>
