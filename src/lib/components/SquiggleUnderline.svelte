<script>
  let { width = 300 } = $props();
</script>

<svg
  viewBox="0 0 {width} 12"
  height="12"
  width={width}
  style="overflow:visible; display:block; max-width:100%"
  aria-hidden="true"
>
  <path
    d="M 0 8 Q {width*0.1} 2 {width*0.2} 8 Q {width*0.3} 14 {width*0.4} 8 Q {width*0.5} 2 {width*0.6} 8 Q {width*0.7} 14 {width*0.8} 8 Q {width*0.9} 2 {width} 8"
    fill="none"
    stroke="var(--accent)"
    stroke-width="2"
    stroke-linecap="round"
    stroke-dasharray={width * 1.5}
    style="stroke-dashoffset:{width * 1.5}; animation: drawLine 1s ease 0.4s forwards"
  />
</svg>
