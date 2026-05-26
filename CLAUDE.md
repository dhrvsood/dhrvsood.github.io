# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
bundle install

# Run local dev server (http://localhost:4000)
bundle exec jekyll serve

# Build static site to _site/
bundle exec jekyll build
```

Requires Ruby, Bundler, and Jekyll. The `_site/` directory is the compiled output — never edit it directly.

## Architecture

This is a Jekyll static site using the [portfolYOU](https://github.com/YoussefRaafatNasry/portfolYOU) theme, deployed to GitHub Pages.

**Content layers:**

- `pages/` — top-level site pages (index, experience, resume, projects, etc.). Each has YAML front matter with `weight` for navbar ordering.
- `_projects/` — individual project entries as Markdown files. Front matter drives the card display (`name`, `tools`, `image`, `description`, `external_url`).
- `_data/` — YAML files that power dynamic page content:
  - `work_timeline.yml` / `volunteer_timeline.yml` — experience entries rendered by `_includes/about/work_timeline.html` and `volunteer_timeline.html`
  - `skills.yml` — badge list on the home page; `color` maps to Bootstrap badge variants (`primary`, `secondary`, `danger`, `success`, `info`, `warning`)
  - `social-media.yml` — footer social links
- `_includes/` — Liquid partials; `_includes/about/` holds the timeline and skills partials
- `_layouts/` — `default.html` is the base shell (navbar + footer + scripts); `page.html` and `post.html` extend it
- `_sass/` — SCSS source; `portfolYOU.scss` is the entry point that imports all partials; `_theme.scss` / `_theme-dark.scss` handle light/dark theming; `_variables.scss` has site-wide design tokens
- `assets/css/style.scss` — imports `portfolYOU.scss` to produce the compiled stylesheet

**Resume page** (`pages/resume.md`): uses a `doc_id` front matter variable pointing to a Google Doc ID. The iframe fetches a PDF export from Google Docs via the Mozilla pdf.js viewer — updating the resume means changing `doc_id` to the new document's ID.

**Navbar ordering** is controlled by the `weight` front matter on each page (lower = earlier in nav). Pages listed under `nav_exclude` in `_config.yml` are hidden from the navbar.

**Adding content:**
- New experience entry: add an object to `_data/work_timeline.yml` or `_data/volunteer_timeline.yml`
- New project: add a Markdown file to `_projects/` with required front matter
- New skill: add an entry to `_data/skills.yml`
- New top-level page: add a file to `pages/` with layout, title, permalink, and weight front matter
