export const prerender = true;

function toTimestamp(value) {
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? 0 : parsed;
}

/** @type {import('./$types').PageLoad} */
export async function load() {
  const modules = import.meta.glob('/src/content/posts/*.md', { eager: true });

  const posts = Object.entries(modules).map(([path, mod]) => {
    const slug = path.split('/').pop().replace('.md', '');
    const { title, date, tag, teaser, paper, readTime } = mod.metadata ?? {};
    return { slug, title, date, tag, teaser, paper, readTime };
  });

  posts.sort((a, b) => {
    const byDate = toTimestamp(b.date) - toTimestamp(a.date);
    if (byDate !== 0) return byDate;

    return a.slug.localeCompare(b.slug);
  });

  return { posts };
}
