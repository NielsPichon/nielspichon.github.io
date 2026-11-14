<script>
  import { onMount } from 'svelte';
  import Pause from '@lucide/svelte/icons/pause';
  import Play from '@lucide/svelte/icons/play';
  import { ATTRIBUTION, GROUPS, STRUCTURES } from '#lib/hologram/anatomy.js';

  // UI mirror of the render loop state. The loop owns the truth; these update only on change.
  let groupIndex = $state(0);
  let available = $state([]); // structure ids present in the loaded model
  let autoplay = $state(true);
  let activeStructure = $state('');
  let pinnedStructure = $state('');
  let backend = $state('');
  let loading = $state(true);
  let failed = $state(false);

  /** @type {HTMLDivElement} */
  let container;
  let api = {};

  const visibleStructures = $derived(
    STRUCTURES.filter((st) => st.group === GROUPS[groupIndex].id && available.includes(st.id))
  );

  function selectGroup(i) {
    autoplay = false;
    pinnedStructure = '';
    // Before the model has loaded, the choice is kept in state and applied once it is ready.
    if (api.setGroup) api.setGroup(i);
    else groupIndex = i;
  }

  function pin(id) {
    pinnedStructure = pinnedStructure === id ? '' : id;
  }

  onMount(() => {
    let disposed = false;
    let cleanup = () => {};

    (async () => {
      try {
        const [THREE, TSL, { GLTFLoader }, { DRACOLoader }, { BODY_COLOR, IDLE_COLOR, MODEL }] = await Promise.all([
          import('three/webgpu'),
          import('three/tsl'),
          import('three/addons/loaders/GLTFLoader.js'),
          import('three/addons/loaders/DRACOLoader.js'),
          import('#lib/hologram/anatomy.js')
        ]);
        if (disposed) return;

        const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        // WebGPURenderer uses WebGPU when available and transparently falls back to a WebGL2 backend.
        const renderer = new THREE.WebGPURenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x000000, 0);
        await renderer.init();
        if (disposed) {
          renderer.dispose();
          return;
        }
        backend = renderer.backend?.isWebGPUBackend ? 'WebGPU' : 'WebGL2';
        renderer.domElement.className = 'block h-full w-full touch-none';
        container.prepend(renderer.domElement);

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 30);

        // ---- Holographic material --------------------------------------------------------------
        const { uniform, float, abs, sin, smoothstep, normalView, positionViewDirection, positionWorld, screenCoordinate, time } = TSL;
        const uScan = uniform(0.0);
        const uScanWidth = uniform(0.03);
        const materials = [];

        function holo(hex, intensity) {
          const uColor = uniform(new THREE.Color(hex));
          const uIntensity = uniform(intensity);
          const facing = abs(normalView.dot(positionViewDirection));
          const fresnel = float(1).sub(facing).pow(2.2);
          const scanlines = sin(screenCoordinate.y.mul(1.4).sub(time.mul(8))).mul(0.5).add(0.5);
          const band = float(1).sub(smoothstep(float(0), uScanWidth, abs(positionWorld.y.sub(uScan))));
          const flicker = sin(time.mul(31)).mul(0.025).add(0.975);
          const glow = fresnel.mul(1.35).add(0.05).mul(scanlines.mul(0.4).add(0.6)).add(band.mul(0.45)).mul(uIntensity).mul(flicker);

          const material = new THREE.MeshBasicNodeMaterial({
            transparent: true,
            depthWrite: false,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending
          });
          material.colorNode = uColor.mul(glow);
          material.opacityNode = float(1);
          materials.push(material);
          return { material, uColor, uIntensity };
        }

        // ---- Body -------------------------------------------------------------------------------
        const loader = new GLTFLoader();
        // Draco-compressed meshes; the decoder (from three/examples) is served from /draco/.
        const draco = new DRACOLoader().setDecoderPath('/draco/');
        loader.setDRACOLoader(draco);
        const groupColor = Object.fromEntries(GROUPS.map((g) => [g.id, new THREE.Color(g.color)]));
        const white = new THREE.Color('#ffffff');
        const idle = new THREE.Color(IDLE_COLOR);

        async function loadBody() {
          const gltf = await loader.loadAsync(MODEL);
          const root = gltf.scene;
          scene.add(root);

          const byName = {};
          root.traverse((o) => {
            if (o.isMesh) byName[o.name] = o;
          });

          const skin = holo(BODY_COLOR, 0.38);
          const bone = holo(BODY_COLOR, 0.22);
          const pedestal = holo(BODY_COLOR, 0.9);
          if (byName.skin) byName.skin.material = skin.material;
          if (byName.skeleton) byName.skeleton.material = bone.material;

          const bodyBox = new THREE.Box3().setFromObject(byName.skin ?? root);
          const bodySize = bodyBox.getSize(new THREE.Vector3());
          for (const r of [0.2, 0.24, 0.28]) {
            const g = new THREE.RingGeometry(r * bodySize.y, r * bodySize.y + 0.004, 128);
            g.rotateX(-Math.PI / 2);
            const ring = new THREE.Mesh(g, pedestal.material);
            ring.position.y = bodyBox.min.y;
            root.add(ring);
          }

          const runtime = STRUCTURES.filter((st) => byName[st.id]).map((st) => {
            const color = groupColor[st.group].clone();
            if (st.node) color.lerp(white, 0.45);
            const mat = holo(IDLE_COLOR, 0.2);
            byName[st.id].material = mat.material;
            return { ...st, object: byName[st.id], mat, color, intensity: 0.2 };
          });

          // Frame each region from the bounds of its (compact) structures.
          const frames = {};
          for (const g of GROUPS) {
            const box = new THREE.Box3();
            for (const r of runtime) if (r.group === g.id && r.frame !== false) box.expandByObject(r.object);
            frames[g.id] = box;
          }
          frames.overview = bodyBox;

          available = runtime.map((r) => r.id);
          return { root, runtime, frames, skin, bone, pedestal };
        }

        const body = await loadBody();
        if (disposed) return;
        loading = false;

        // ---- Timeline ------------------------------------------------------------------------
        const STEP = reducedMotion ? 2.6 : 1.5;
        let gi = 0;
        let stepIndex = 0;
        let stepClock = 0;
        let introTime = 0;

        const current = () => body.runtime.filter((r) => r.group === GROUPS[gi].id);

        function restartCycle() {
          stepIndex = 0;
          stepClock = 0;
          activeStructure = current()[0]?.id ?? '';
        }
        function setGroup(i) {
          gi = (i + GROUPS.length) % GROUPS.length;
          groupIndex = gi;
          restartCycle();
        }
        api.setGroup = setGroup;
        setGroup(groupIndex);

        // ---- Camera & interaction ------------------------------------------------------------
        const target = new THREE.Vector3();
        body.frames.overview.getCenter(target);
        let dist = 4;
        let yaw = -0.5;
        let pitch = 0.1;
        let dragging = false;
        let lastX = 0;
        let lastY = 0;
        let idleSpin = 0;

        const canvas = renderer.domElement;
        const onDown = (e) => {
          dragging = true;
          lastX = e.clientX;
          lastY = e.clientY;
          canvas.setPointerCapture(e.pointerId);
        };
        const onMove = (e) => {
          if (!dragging) return;
          yaw -= (e.clientX - lastX) * 0.008;
          pitch = Math.min(0.9, Math.max(-0.5, pitch + (e.clientY - lastY) * 0.005));
          lastX = e.clientX;
          lastY = e.clientY;
          idleSpin = 0;
        };
        const onUp = () => (dragging = false);
        canvas.addEventListener('pointerdown', onDown);
        canvas.addEventListener('pointermove', onMove);
        canvas.addEventListener('pointerup', onUp);
        canvas.addEventListener('pointercancel', onUp);

        const resize = () => {
          const { clientWidth: w, clientHeight: h } = container;
          if (!w || !h) return;
          renderer.setSize(w, h, false);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        };
        const ro = new ResizeObserver(resize);
        ro.observe(container);
        resize();

        // Only render while on screen.
        let visible = true;
        const io = new IntersectionObserver(([entry]) => (visible = entry.isIntersecting));
        io.observe(container);

        const tmpColor = new THREE.Color();
        const tmpCenter = new THREE.Vector3();
        const tmpSize = new THREE.Vector3();

        /** Distance at which `box` fits the viewport whatever the orbit angle. */
        function fitDistance(box, vertical, horizontal) {
          box.getSize(tmpSize);
          const half = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
          const width = Math.max(tmpSize.x, tmpSize.z) / 2;
          return Math.max((tmpSize.y / 2 / half) * vertical, (width / (half * camera.aspect)) * horizontal);
        }

        // ---- Loop ----------------------------------------------------------------------------
        let prev = performance.now();
        let scanPhase = 0;
        let shellFade = 1;

        renderer.setAnimationLoop(() => {
          const now = performance.now();
          // The tour runs on wall-clock time; easing uses a capped step so slow frames don't overshoot.
          const elapsed = Math.min(0.5, (now - prev) / 1000);
          const dt = Math.min(0.05, elapsed);
          prev = now;
          if (!visible) return;

          introTime += elapsed;
          const intro = introTime < 1.8; // open on a full-body overview, then dive into the first region
          const group = GROUPS[gi];
          const list = current();

          // Advance through the structures of the active region; move on after a full pass.
          if (!intro && !pinnedStructure && !loading) {
            stepClock += elapsed;
            if (stepClock > STEP) {
              stepClock = 0;
              stepIndex++;
              if (stepIndex >= list.length) {
                if (autoplay) setGroup(gi + 1);
                else restartCycle();
              } else activeStructure = list[stepIndex].id;
            }
          }
          const highlighted = pinnedStructure || activeStructure;

          const k = 1 - Math.exp(-dt * 6);
          for (const r of body.runtime) {
            const inGroup = !intro && r.group === group.id;
            const goal = intro ? 0.35 : inGroup ? (r.id === highlighted ? 2.4 : 0.5) : 0.12;
            r.intensity += (goal - r.intensity) * k;
            r.mat.uIntensity.value = r.intensity;
            tmpColor.copy(inGroup ? r.color : idle);
            r.mat.uColor.value.lerp(tmpColor, k);
          }

          // Camera: ease towards the region, orbit slowly unless the user is dragging.
          const box = intro ? body.frames.overview : body.frames[group.id];
          const kc = 1 - Math.exp(-dt * 2.2);
          box.getCenter(tmpCenter);
          target.lerp(tmpCenter, kc);
          const goalDist = intro ? fitDistance(box, 1.08, 1.08) : fitDistance(box, 1.8, 1.15);
          dist += (goalDist - dist) * kc;

          // Fade the body shell when zoomed in so it does not wash out the organs.
          shellFade += ((intro ? 1 : 0.22) - shellFade) * kc;
          body.skin.uIntensity.value = 0.38 * shellFade;
          body.bone.uIntensity.value = 0.22 * Math.max(shellFade, 0.45);
          body.pedestal.uIntensity.value = 0.9 * shellFade;

          if (!dragging && !reducedMotion) {
            idleSpin = Math.min(1, idleSpin + dt * 0.5);
            yaw += dt * 0.22 * idleSpin;
          }
          camera.position.set(
            target.x + Math.sin(yaw) * Math.cos(pitch) * dist,
            target.y + Math.sin(pitch) * dist,
            target.z + Math.cos(yaw) * Math.cos(pitch) * dist
          );
          camera.lookAt(target);

          // The scan band sweeps down through the framed region.
          box.getSize(tmpSize);
          scanPhase = (scanPhase + dt * 0.35) % 1;
          uScan.value = box.max.y + 0.02 - scanPhase * (tmpSize.y + 0.04);
          uScanWidth.value = 0.01 + tmpSize.y * 0.02;

          renderer.render(scene, camera);
        });

        cleanup = () => {
          renderer.setAnimationLoop(null);
          ro.disconnect();
          io.disconnect();
          canvas.removeEventListener('pointerdown', onDown);
          canvas.removeEventListener('pointermove', onMove);
          canvas.removeEventListener('pointerup', onUp);
          canvas.removeEventListener('pointercancel', onUp);
          scene.traverse((o) => o.geometry?.dispose());
          materials.forEach((m) => m.dispose());
          draco.dispose();
          renderer.dispose();
          canvas.remove();
        };
      } catch (err) {
        console.error('Hologram failed to start:', err?.stack ?? err);
        failed = true;
      }
    })();

    return () => {
      disposed = true;
      cleanup();
    };
  });
</script>

<div
  bind:this={container}
  class="holo relative h-full w-full overflow-hidden rounded-xl border border-white/10 bg-[#04070c] text-white"
  role="img"
  aria-label="Holographic body showing radiotherapy regions of interest and their organs at risk"
>
  <div class="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_50%_80%,rgba(76,201,255,0.12),transparent_60%)]"></div>

  {#if failed}
    <div class="absolute inset-0 grid place-items-center p-8 text-center text-sm text-white/60">
      This visualisation needs WebGPU or WebGL2, which your browser does not seem to provide.
    </div>
  {:else}
    <div class="pointer-events-none absolute top-3 left-4 font-mono text-[10px] tracking-[0.2em] text-cyan-200/70 uppercase">
      RT · regions of interest{#if backend}<span class="ml-2 text-white/35">{backend}</span>{/if}
    </div>

    {#if loading}
      <div class="pointer-events-none absolute inset-0 grid place-items-center font-mono text-xs tracking-[0.2em] text-cyan-200/60 uppercase">
        <span class="animate-pulse">Loading anatomy…</span>
      </div>
    {/if}

    <ul class="pointer-events-auto absolute top-9 left-4 max-w-[60%] space-y-0.5 font-mono text-[11px] leading-5 [text-shadow:0_1px_6px_#000]">
      {#each visibleStructures as st (st.id)}
        {@const on = (pinnedStructure || activeStructure) === st.id}
        <li>
          <button
            type="button"
            class="flex items-center gap-2 text-left transition-colors {on ? 'text-white' : 'text-white/45 hover:text-white/80'}"
            onclick={() => pin(st.id)}
            aria-pressed={pinnedStructure === st.id}
          >
            <span
              class="inline-block size-1.5 shrink-0 rounded-full transition-transform {on ? 'scale-150' : ''}"
              style="background:{GROUPS[groupIndex].color}; box-shadow: 0 0 {on ? 8 : 0}px {GROUPS[groupIndex].color}; opacity:{st.node ? 0.7 : 1}"
            ></span>
            {st.name}
          </button>
        </li>
      {/each}
    </ul>

    <div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-3 pt-8">
      <div class="flex flex-wrap items-center justify-center gap-1.5">
        {#each GROUPS as g, i}
          <button
            type="button"
            onclick={() => selectGroup(i)}
            class="rounded-full border px-3 py-1 text-xs transition-colors"
            style="border-color:{i === groupIndex ? g.color : 'rgba(255,255,255,0.15)'}; color:{i === groupIndex ? g.color : 'rgba(255,255,255,0.65)'}; background:{i === groupIndex ? g.color + '1f' : 'transparent'}"
            aria-pressed={i === groupIndex}
          >{g.name}</button>
        {/each}
        <button
          type="button"
          onclick={() => (autoplay = !autoplay)}
          class="grid size-7 place-items-center rounded-full border border-white/15 text-white/70 hover:text-white"
          aria-label={autoplay ? 'Pause tour' : 'Resume tour'}
        >
          {#if autoplay}<Pause class="size-3.5" />{:else}<Play class="size-3.5" />{/if}
        </button>
      </div>
      <a
        href={ATTRIBUTION.href}
        target="_blank"
        rel="noopener noreferrer"
        class="mt-2 block text-center text-[10px] text-white/35 hover:text-white/60"
      >{ATTRIBUTION.label}</a>
    </div>
  {/if}
</div>
