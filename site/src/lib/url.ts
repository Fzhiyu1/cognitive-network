export function withBasePath(pathname: string): string {
  const base = import.meta.env.BASE_URL ?? "/";
  const normalizedBase = base === "/" ? "" : base.replace(/\/$/, "");
  const normalizedPath = pathname.startsWith("/") ? pathname : `/${pathname}`;

  return `${normalizedBase}${normalizedPath}` || "/";
}

export function slugifyHeading(heading: string): string {
  return heading
    .trim()
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, "")
    .replace(/\s+/g, "-");
}
