<script>
  import { cn } from '$lib/utils.js';
  let { class: className, children, ...restProps } = $props();
</script>

<span class={cn('inline-flex w-fit shrink-0 items-center justify-center rounded-md border border-transparent bg-secondary px-2 py-0.5 text-xs font-medium text-secondary-foreground transition-colors', className)} {...restProps}>{@render children?.()}</span>
