export const prerender = true;

function toTimestamp(value) {
  if (typeof value !== 'string') return 0;

  const trimmed = value.trim();

  // Treat month/year dates as the end of the month so newer posts
  // without a specific day do not get pushed behind earlier dated posts.
  const monthYearMatch = trimmed.match(/^([A-Za-z]+)\s*,?\s*(\d{4})$/);
  if (monthYearMatch) {
    const monthName = monthYearMatch[1];
    const year = Number(monthYearMatch[2]);
    const monthIndex = new Date(`${monthName} 1, 2000`).getMonth();

    if (!Number.isNaN(monthIndex)) {
      return new Date(year, monthIndex + 1, 0, 23, 59, 59, 999).getTime();
    }
  }

  const yearOnlyMatch = trimmed.match(/^(\d{4})$/);
  if (yearOnlyMatch) {
    return new Date(Number(yearOnlyMatch[1]), 11, 31, 23, 59, 59, 999).getTime();
  }

  const parsed = Date.parse(trimmed);
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
