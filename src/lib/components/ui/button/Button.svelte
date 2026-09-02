<script>
  import { cn } from '$lib/utils.js';
  let { class: className, variant = 'default', size = 'default', href, children, ...restProps } = $props();
  const variants = {
    default: 'bg-primary text-primary-foreground shadow-xs hover:bg-primary/90',
    outline: 'border border-input bg-background shadow-xs hover:bg-accent hover:text-accent-foreground',
    ghost: 'hover:bg-accent hover:text-accent-foreground',
    link: 'text-primary underline-offset-4 hover:underline'
  };
  const sizes = { default: 'h-9 px-4 py-2', sm: 'h-8 rounded-md px-3 text-xs', icon: 'size-9' };
  const classes = $derived(cn('inline-flex shrink-0 items-center justify-center gap-2 rounded-md text-sm font-medium whitespace-nowrap transition-colors outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50', variants[variant], sizes[size], className));
</script>

{#if href}
  <a {href} class={classes} {...restProps}>{@render children?.()}</a>
{:else}
  <button class={classes} {...restProps}>{@render children?.()}</button>
{/if}
