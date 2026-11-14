import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-static';
import { mdsvex } from 'mdsvex';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import { visit } from 'unist-util-visit';

function rehypeEscapeMathLt() {
	return (tree) => {
		visit(tree, 'text', (node, _index, parent) => {
			if (parent?.properties?.className?.includes('mrel')) {
				node.value = node.value.replaceAll('<', '&lt;').replaceAll('>', '&gt;');
			}
		});
	};
}

// SvelteKit 3 takes its configuration through the Vite plugin (svelte.config.js is no longer read).
export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			extensions: ['.svelte', '.md'],
			preprocess: [
				mdsvex({
					extensions: ['.md'],
					remarkPlugins: [remarkMath],
					rehypePlugins: [
						[rehypeKatex, { output: 'html' }],
						rehypeEscapeMathLt,
						[rehypeHighlight, { ignoreMissing: true }]
					]
				})
			],
			appDir: 'app',
			adapter: adapter({
				pages: 'build',
				assets: 'build',
				fallback: '404.html'
			}),
			paths: { base: '' }
		})
	]
});
