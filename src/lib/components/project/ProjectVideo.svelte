<script>
  import Film from '@lucide/svelte/icons/film';
  let { src = '', poster = '', label = 'Project animation' } = $props();

  let reducedMotion = $state(false);
  $effect(() => {
    reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
</script>

<div class="relative h-full w-full overflow-hidden rounded-xl border bg-black">
  {#if src}
    <!-- Decorative render loop; the article carries the content. -->
    <video
      class="h-full w-full object-cover"
      {src}
      poster={poster || undefined}
      autoplay={!reducedMotion}
      controls={reducedMotion}
      muted
      loop
      playsinline
      preload="metadata"
      aria-label={label}
    ></video>
  {:else}
    <div class="grid h-full w-full place-items-center bg-[radial-gradient(circle_at_50%_40%,#1d2a3a,#05070a)] text-white/50">
      <div class="flex flex-col items-center gap-3 text-sm">
        <Film class="size-8" />
        Animation coming soon
      </div>
    </div>
  {/if}
</div>
