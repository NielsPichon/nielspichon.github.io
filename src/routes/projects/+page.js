import { projects } from '#lib/data/projects.js';

export const prerender = true;

export function load() {
  return { projects };
}
