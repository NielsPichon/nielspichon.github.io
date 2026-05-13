export const prerender = true;
import readingTime from 'reading-time/lib/reading-time';

function getMarkdownBody(markdown) {
  if (typeof markdown !== 'string') return '';
  return markdown.replace(/^---[\s\S]*?---\s*/, '');
}

function getReadTimeLabel(markdown) {
  const minutes = Math.max(1, Math.ceil(readingTime(getMarkdownBody(markdown)).minutes));
  return `${minutes} min`;
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

/** When true, the post is omitted from the home page listing but `/posts/[slug]` still works. */
function isUnlisted(metadata) {
  const value = metadata?.unlisted;
  return value === true || value === 'true';
}

function toTimestamp(value) {
  if (typeof value !== 'string') return 0;

  const trimmed = value.trim();

  // Treat month/year dates as the first of the month so exact dates within
  // the same month sort ahead of them, while still ranking above earlier months.
  const monthYearMatch = trimmed.match(/^([A-Za-z]+)\s*,?\s*(\d{4})$/);
  if (monthYearMatch) {
    const monthName = monthYearMatch[1];
    const year = Number(monthYearMatch[2]);
    const monthIndex = new Date(`${monthName} 1, 2000`).getMonth();

    if (!Number.isNaN(monthIndex)) {
      return new Date(year, monthIndex, 1).getTime();
    }
  }

  const yearOnlyMatch = trimmed.match(/^(\d{4})$/);
  if (yearOnlyMatch) {
    return new Date(Number(yearOnlyMatch[1]), 11, 31, 23, 59, 59, 999).getTime();
  }

  const normalized = trimmed.replace(/(\d+)(?:st|nd|rd|th)\b/g, '$1');
  const parsed = Date.parse(normalized);
  return Number.isNaN(parsed) ? 0 : parsed;
}

/** @type {import('./$types').PageLoad} */
export async function load() {
  const modules = import.meta.glob('/src/content/posts/*.md', { eager: true });
  const rawModules = import.meta.glob('/src/content/posts/*.md', {
    eager: true,
    query: '?raw',
    import: 'default'
  });

  const posts = Object.entries(modules)
    .filter(([, mod]) => !isUnlisted(mod.metadata))
    .map(([path, mod]) => {
      const slug = path.split('/').pop().replace('.md', '');
      const { title, date, teaser, paper, readTime } = mod.metadata ?? {};
      const tags = normalizeTags(mod.metadata);
      return {
        slug,
        title,
        date,
        tags,
        teaser,
        paper,
        readTime: readTime ?? getReadTimeLabel(rawModules[path])
      };
    });

  posts.sort((a, b) => {
    const byDate = toTimestamp(b.date) - toTimestamp(a.date);
    if (byDate !== 0) return byDate;

    return a.slug.localeCompare(b.slug);
  });

  return { posts };
}
