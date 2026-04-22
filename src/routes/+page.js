export const prerender = true;

/** @type {import('./$types').PageLoad} */
export async function load() {
  const modules = import.meta.glob('/src/content/posts/*.md', { eager: true });

  const posts = Object.entries(modules).map(([path, mod]) => {
    const slug = path.split('/').pop().replace('.md', '');
    const { title, date, tag, teaser, paper, readTime } = mod.metadata ?? {};
    return { slug, title, date, tag, teaser, paper, readTime };
  });

  posts.sort((a, b) => new Date(b.date) - new Date(a.date));

  return { posts };
}
