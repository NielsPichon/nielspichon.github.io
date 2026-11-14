/**
 * Projects are markdown files in `src/content/projects/`. Frontmatter:
 *   title, date, description, order (lower = first), unlisted?,
 *   pro?: true for professional work (used by the Projects page filter),
 *   aside?: 'organs' | 'gallery' | 'video'   (omit for a plain blog layout)
 *   gallery?: [{ title, description?, image?, href? }]   (aside: gallery)
 *   instagram?: url                                        (aside: gallery)
 *   video?: url, poster?: url                              (aside: video)
 */
const modules = import.meta.glob('/src/content/projects/*.md', { eager: true });

/** @type {Array<{slug: string, title: string, date: string, description?: string, href: string, order: number, pro: boolean}>} */
export const projects = Object.entries(modules)
  .filter(([, mod]) => mod.metadata?.unlisted !== true)
  .map(([path, mod]) => {
    const slug = path.split('/').pop().replace('.md', '');
    const { title, date, description, order, pro } = mod.metadata ?? {};
    return { slug, title, date, description, order: Number(order ?? 999), pro: pro === true, href: `/projects/${slug}` };
  })
  .sort((a, b) => a.order - b.order || a.slug.localeCompare(b.slug));
