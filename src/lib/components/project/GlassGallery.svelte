<script>
  /**
   * A receding stack of glass slides. Hovering (or focusing / tapping) a slide lifts it out of
   * the stack and shows its title and description in the top-left corner.
   * items: [{ title, description?, image?, animated?, href? }]
   * `image` is the still frame; `animated` (e.g. an animated WebP) plays while the slide is lifted.
   */
  let { items = [], aspect = '1 / 1', hint = 'Hover a slide', instagram = '' } = $props();

  let active = $state(-1);
  let touch = $state(false);
  $effect(() => {
    touch = window.matchMedia('(hover: none)').matches;
  });
  const hintText = $derived(touch ? hint.replace(/^Hover/, 'Tap') : hint);
  const shown = $derived(items[active]);
  const mid = $derived((items.length - 1) / 2);

  function onclick(event, i) {
    // Touch: the first tap lifts the slide, a second tap follows its link.
    if (active !== i) {
      active = i;
      event.preventDefault();
    }
  }
</script>

<div class="stage relative h-full w-full overflow-hidden">
  <div class="caption pointer-events-none absolute top-0 left-0 z-10 max-w-[70%] p-6 sm:p-8" aria-live="polite">
    {#if shown}
      {#key active}
        <div class="caption-in">
          <h3 class="text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">{shown.title}</h3>
          {#if shown.description}<p class="mt-2 text-sm leading-6 text-muted-foreground">{shown.description}</p>{/if}
        </div>
      {/key}
    {:else}
      <p class="caption-in font-mono text-xs tracking-[0.18em] text-muted-foreground uppercase">{hintText}</p>
    {/if}
  </div>

  <div class="scene" style="--n:{items.length}">
    <div class="rig">
      {#each items as item, i}
        <svelte:element
          this={item.href ? 'a' : 'button'}
          href={item.href}
          target={item.href?.startsWith('http') ? '_blank' : undefined}
          rel={item.href?.startsWith('http') ? 'noopener noreferrer' : undefined}
          type={item.href ? undefined : 'button'}
          class="slot"
          class:lifted={active === i}
          style="--z:{(mid - i).toFixed(2)}; --aspect:{aspect}"
          aria-label={item.title}
          onpointerenter={(e) => e.pointerType === 'mouse' && (active = i)}
          onpointerleave={(e) => e.pointerType === 'mouse' && active === i && (active = -1)}
          onfocus={() => (active = i)}
          onblur={() => active === i && (active = -1)}
          onclick={(e) => onclick(e, i)}
        >
          <span class="glass">
            {#if item.image}
              <img src={item.image} alt="" loading="lazy" draggable="false" />
              {#if item.animated && active === i}
                <!-- Mounted only while lifted, so the animation restarts from its first frame each time. -->
                <img class="animated" src={item.animated} alt="" draggable="false" />
              {/if}
            {:else}
              <span class="placeholder">{item.title?.[0] ?? '?'}</span>
            {/if}
          </span>
        </svelte:element>
      {/each}
    </div>
  </div>

  {#if instagram}
    <a
      href={instagram}
      target="_blank"
      rel="noopener noreferrer"
      class="absolute right-4 bottom-4 z-10 inline-flex items-center gap-2 rounded-full border bg-background/70 px-3 py-1.5 text-xs font-medium backdrop-blur transition-colors hover:bg-accent"
    >
      <svg viewBox="0 0 24 24" class="size-3.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="2" y="2" width="20" height="20" rx="5" /><circle cx="12" cy="12" r="4" /><circle cx="17.5" cy="6.5" r="0.5" fill="currentColor" />
      </svg>
      More on Instagram
    </a>
  {/if}
</div>

<style>
  .stage {
    container-type: size;
    --glass-edge: rgb(0 0 0 / 0.14);
    --glass-fill-a: rgb(255 255 255 / 0.55);
    --glass-fill-b: rgb(255 255 255 / 0.12);
    --glass-shadow: rgb(15 23 42 / 0.25);
  }
  :global(.dark) .stage {
    --glass-edge: rgb(255 255 255 / 0.28);
    --glass-fill-a: rgb(255 255 255 / 0.16);
    --glass-fill-b: rgb(255 255 255 / 0.03);
    --glass-shadow: rgb(0 0 0 / 0.6);
  }

  .scene {
    position: absolute;
    inset: 0;
    perspective: 1600px;
    perspective-origin: 50% 35%;
    /* Slide size and spacing scale with the panel and the number of slides. */
    --w: min(40cqw, 34cqh);
    --gap: min(calc(70cqw / max(var(--n) - 1, 1)), 13cqw);
  }

  .rig {
    position: absolute;
    left: 52%;
    top: 70%;
    transform-style: preserve-3d;
    transform: rotateX(-16deg) rotateY(-48deg);
  }

  .slot {
    position: absolute;
    width: var(--w);
    aspect-ratio: var(--aspect);
    transform-style: preserve-3d;
    transform: translate(-50%, -50%) translateZ(calc(var(--z) * var(--gap)));
    cursor: pointer;
    outline: none;
  }
  /* Keep the hover target under the cursor once the slide has risen. */
  .slot.lifted::before {
    content: '';
    position: absolute;
    inset: -112% 0 100% 0;
  }

  .glass {
    position: absolute;
    inset: 0;
    display: block;
    padding: 6%;
    border-radius: 14px;
    border: 1px solid var(--glass-edge);
    background: linear-gradient(135deg, var(--glass-fill-a), var(--glass-fill-b));
    box-shadow:
      inset 0 1px 0 rgb(255 255 255 / 0.35),
      0 30px 60px -30px var(--glass-shadow);
    backdrop-filter: blur(6px) saturate(140%);
    -webkit-backdrop-filter: blur(6px) saturate(140%);
    transition:
      transform 0.55s cubic-bezier(0.2, 0.8, 0.2, 1),
      box-shadow 0.55s ease,
      border-color 0.3s ease;
  }
  .glass::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    background: linear-gradient(115deg, transparent 30%, rgb(255 255 255 / 0.28) 45%, transparent 60%);
    opacity: 0.6;
    transition: opacity 0.4s ease;
  }
  .slot.lifted .glass {
    /* Rise clear of the stack so the whole frame is visible. */
    transform: translateY(-110%);
    border-color: color-mix(in oklab, var(--glass-edge), white 40%);
    box-shadow:
      inset 0 1px 0 rgb(255 255 255 / 0.5),
      0 50px 80px -30px var(--glass-shadow);
  }
  .slot.lifted .glass::before {
    opacity: 1;
  }
  .slot:focus-visible .glass {
    border-color: var(--ring);
  }

  .glass img,
  .placeholder {
    position: relative;
    display: block;
    width: 100%;
    height: 100%;
    border-radius: 7px;
    object-fit: cover;
    user-select: none;
  }
  .placeholder {
    display: grid;
    place-items: center;
    font-size: calc(var(--w) * 0.25);
    font-weight: 700;
    color: rgb(255 255 255 / 0.7);
    background: linear-gradient(160deg, #6d7cff, #ff8a6b);
  }
  .glass img.animated {
    position: absolute;
    inset: 6%;
    width: 88%;
    height: 88%;
  }

  .caption-in {
    animation: caption-in 0.35s ease both;
  }
  @keyframes caption-in {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .glass {
      transition: none;
    }
    .caption-in {
      animation: none;
    }
  }
</style>
