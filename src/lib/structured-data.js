export const SITE_ORIGIN = 'https://nielspichon.github.io';

const MONTHS = {
  january: 0,
  february: 1,
  march: 2,
  april: 3,
  may: 4,
  june: 5,
  july: 6,
  august: 7,
  september: 8,
  october: 9,
  november: 10,
  december: 11
};

/** @param {string} dateStr */
function parsePostDateParts(dateStr) {
  const normalized = dateStr.trim().replace(/(\d+)(st|nd|rd|th)\b/gi, '$1');
  const match = normalized.match(/^([A-Za-z]+)(?:\s+(\d{1,2}))?,?\s+(\d{4})$/);
  if (!match) return undefined;

  const [, monthName, dayStr, yearStr] = match;
  const monthIndex = MONTHS[monthName.toLowerCase()];
  if (monthIndex === undefined) return undefined;

  const month =
    monthName.charAt(0).toUpperCase() + monthName.slice(1).toLowerCase();
  const year = parseInt(yearStr, 10);
  const day = dayStr ? parseInt(dayStr, 10) : undefined;

  return { year, month, day };
}

/** @param {string | undefined} dateStr */
export function parsePostDateToIso(dateStr) {
  if (typeof dateStr !== 'string' || !dateStr.trim()) return undefined;

  const parts = parsePostDateParts(dateStr);
  if (!parts) return undefined;

  const day = parts.day ?? 1;
  return new Date(Date.UTC(parts.year, MONTHS[parts.month.toLowerCase()], day))
    .toISOString()
    .slice(0, 10);
}

/** APA-style parenthetical date, e.g. "(2023, June 1)" or "(2026, May)". */
function formatCitationDateParenthetical(dateStr) {
  if (typeof dateStr !== 'string' || !dateStr.trim()) return '';

  const parts = parsePostDateParts(dateStr);
  if (!parts) return '';

  if (parts.day !== undefined) {
    return `(${parts.year}, ${parts.month} ${parts.day})`;
  }

  return `(${parts.year}, ${parts.month})`;
}

/**
 * @param {{ slug: string; metadata: Record<string, unknown> }} input
 * @returns {{ text: string; url: string; beforeUrl: string }}
 */
export function buildPostCitation({ slug, metadata }) {
  const url = `${SITE_ORIGIN}/posts/${slug}`;
  const title = typeof metadata?.title === 'string' ? metadata.title : 'Untitled';
  const datePart = formatCitationDateParenthetical(
    typeof metadata?.date === 'string' ? metadata.date : undefined
  );
  const author = 'Pichon, N.';
  const publication = 'Electron Avalanche Blog';

  const beforeUrl = datePart
    ? `${author} ${datePart}. ${title}. ${publication}. `
    : `${author}. ${title}. ${publication}. `;

  return { text: `${beforeUrl}${url}`, url, beforeUrl };
}

/**
 * @param {{ slug: string; metadata: Record<string, unknown> }} input
 * @returns {Record<string, unknown>}
 */
export function buildBlogPostingJsonLd({ slug, metadata }) {
  const url = `${SITE_ORIGIN}/posts/${slug}`;
  const datePublished = parsePostDateToIso(
    typeof metadata?.date === 'string' ? metadata.date : undefined
  );
  const dateModifiedSource =
    typeof metadata?.dateModified === 'string'
      ? metadata.dateModified
      : typeof metadata?.updated === 'string'
        ? metadata.updated
        : typeof metadata?.date === 'string'
          ? metadata.date
          : undefined;
  const dateModified = parsePostDateToIso(dateModifiedSource);

  return {
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline: typeof metadata?.title === 'string' ? metadata.title : 'Untitled',
    description: typeof metadata?.teaser === 'string' ? metadata.teaser : '',
    author: {
      '@type': 'Person',
      name: 'Niels Pichon'
    },
    ...(datePublished && { datePublished }),
    ...(dateModified && { dateModified }),
    publisher: {
      '@type': 'Organization',
      name: 'Electron Avalanche'
    },
    mainEntityOfPage: {
      '@type': 'WebPage',
      '@id': url
    },
    url,
    inLanguage: 'en-US'
  };
}
