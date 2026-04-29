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
		  node.value = node.value
			.replaceAll('<', '&lt;')
			.replaceAll('>', '&gt;');
		}
	  });
	};
  }
const config = {
  extensions: ['.svelte', '.md'],
  preprocess: [
    mdsvex({
      extensions: ['.md'],
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        [rehypeKatex, { output: 'html' }],
        rehypeEscapeMathLt,           // ← add this
        [rehypeHighlight, { ignoreMissing: true }]
      ]
    })
  ],
  kit: {
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: '404.html'
    }),
    paths: { base: '' }
  }
};

export default config;
