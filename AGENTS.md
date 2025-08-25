# Repository Guidelines

## Instructions for the agent

- Read files before modifying them. The human developers can edit the files in-between the agent edits. 
- Do not revert the changes made by the human developers since you last read the file (unless asked to do so)

## Project Structure & Module Organization
- Source: Markdown pages at `index.md`, `about.md`, and posts in `_posts/` (use `YYYY-MM-DD-title.md`).
- Layouts/partials: `_includes/` (Liquid) and styles in `_sass/`.
- Assets: `assets/` and `pics/` for images; avoid committing large binaries elsewhere.
- Config: `_config.yml` (site settings) and `CNAME` (custom domain).
- Output: `_site/` is generated; do not edit or commit it.

## Build, Test, and Development Commands
- `bundle install`: install Ruby gems from `Gemfile`.
- `bundle exec jekyll serve --livereload`: local dev server at `http://127.0.0.1:4000`.
- `bundle exec jekyll build`: produce the static site in `_site/`.
- Optional: `JEKYLL_ENV=production bundle exec jekyll build` to mirror GitHub Pages.

## Coding Style & Naming Conventions
- Markdown: start each page/post with YAML front matter; wrap at ~100 chars where reasonable.
- Posts: filename `YYYY-MM-DD-slug.md`; title-cased `title:` in front matter.
- Liquid: 2‑space indentation; prefer includes in `_includes/` for repeated HTML.
- Styles: SCSS partials in `_sass/`; group variables/mixins at the top; compile via Jekyll.
- Media: place images in `pics/` or `assets/` and reference with relative paths.

## Testing Guidelines
- Build must succeed without errors: `bundle exec jekyll build`.
- Manually verify pages locally via `jekyll serve`; check links, code blocks, and image paths.
- Prefer local images over external hotlinks; use alt text.
- Optional link check (if available): `bundle exec htmlproofer ./_site`.

## Commit & Pull Request Guidelines
- Commits: concise, present‑tense imperative (e.g., "Add post: Transformer notes").
- Scope one change per commit when practical; include relevant paths in the body if helpful.
- Branches: short, kebab‑case names (e.g., `post/nvcc-walkthrough`, `style/header-tweak`).
- PRs: include a summary, screenshots for visual changes, and link related issues.
- CI/CD: GitHub Pages builds from the default branch; do not commit `_site/`.

## Configuration & Security Tips
- Keep `_config.yml` authoritative (e.g., `url`, `baseurl`, analytics toggles); test with production builds.
- Preserve `CNAME` for the custom domain.
- Do not commit secrets; environment-specific values should come from repository settings, not files.

