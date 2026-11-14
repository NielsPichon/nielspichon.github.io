import { error } from '@sveltejs/kit';

export const prerender = true;

const modules = import.meta.glob('/src/content/projects/*.md');

/** @type {import('./$types').EntryGenerator} */
export function entries() {
  return Object.keys(modules).map((path) => ({ slug: path.split('/').pop().replace('.md', '') }));
}

/** @type {import('./$types').PageLoad} */
export async function load({ params }) {
  const key = `/src/content/projects/${params.slug}.md`;
  if (!modules[key]) throw error(404, 'Project not found');
  const project = await modules[key]();
  return { content: project.default, metadata: project.metadata ?? {} };
}
