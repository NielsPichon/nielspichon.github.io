import { error } from '@sveltejs/kit';
import readingTime from 'reading-time/lib/reading-time';

export const prerender = true;

const modules = import.meta.glob('/src/content/posts/*.md');
const rawModules = import.meta.glob('/src/content/posts/*.md', {
  query: '?raw',
  import: 'default'
});

function getMarkdownBody(markdown) {
  if (typeof markdown !== 'string') return '';
  return markdown.replace(/^---[\s\S]*?---\s*/, '');
}

function getReadTimeLabel(markdown) {
  const minutes = Math.max(1, Math.ceil(readingTime(getMarkdownBody(markdown)).minutes));
  return `${minutes} min`;
}

function normalizeCodeUrl(metadata) {
  const raw = metadata?.code;
  if (typeof raw !== 'string') return undefined;
  const trimmed = raw.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function normalizeTags(metadata) {
  const rawTags = metadata?.tags ?? metadata?.tag;

  if (Array.isArray(rawTags)) {
    return rawTags.filter((value) => typeof value === 'string' && value.trim().length > 0);
  }

  if (typeof rawTags === 'string' && rawTags.trim().length > 0) {
    return [rawTags.trim()];
  }

  return [];
}

/** @type {import('./$types').EntryGenerator} */
export async function entries() {
  return Object.keys(modules).map((path) => ({
    slug: path.split('/').pop().replace('.md', '')
  }));
}

/** @type {import('./$types').PageLoad} */
export async function load({ params }) {
  const key = `/src/content/posts/${params.slug}.md`;
  if (!modules[key] || !rawModules[key]) throw error(404, 'Post not found');

  const [post, raw] = await Promise.all([modules[key](), rawModules[key]()]);
  return {
    content: post.default,
    metadata: {
      ...post.metadata,
      tags: normalizeTags(post.metadata),
      readTime: post.metadata?.readTime ?? getReadTimeLabel(raw),
      code: normalizeCodeUrl(post.metadata)
    }
  };
}
