import { error } from '@sveltejs/kit';

export const prerender = true;

const modules = import.meta.glob('/src/content/posts/*.md');

/** @type {import('./$types').EntryGenerator} */
export async function entries() {
  return Object.keys(modules).map((path) => ({
    slug: path.split('/').pop().replace('.md', '')
  }));
}

/** @type {import('./$types').PageLoad} */
export async function load({ params }) {
  const key = `/src/content/posts/${params.slug}.md`;
  if (!modules[key]) throw error(404, 'Post not found');

  const post = await modules[key]();
  return {
    content: post.default,
    metadata: post.metadata
  };
}
