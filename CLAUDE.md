# CLAUDE.md — gh-pages branch

## Branch Purpose

This branch (`gh-pages`) is **exclusively** for the project website. It is deployed via GitHub Pages. The ML pipeline code lives on `main` — do not port or reference that code here.

## Repository Structure

```
/
├── index.html      # Main (and currently only) page
├── styles.css      # Global styles
├── assets/         # Static assets (fonts, icons, JS, etc.)
├── image/          # Images referenced by the site
│   └── Intro_Pipe.jpg
└── README.md       # Project README (from main branch; background only)
```

## Website Status

The site is in early development. Current content in `index.html`:
- Introduction section
- Methods section
- Results section (performance table is a TODO placeholder)
- Conclusion section

## Development Guidelines

- All website work goes in this branch. Do not commit source code, notebooks, or data files here.
- Keep the site to static HTML/CSS/JS — no build tools or frameworks unless explicitly introduced.
- `styles.css` uses a simple centered-column layout (`max-width: 100ch`) with the system UI font.
- When adding images, place them in `image/`. For other static assets (scripts, icons), use `assets/`.
- The site deploys automatically to GitHub Pages on push — verify changes look correct locally before pushing.

## Project Background (for website content context)

The project evaluates whether LLMs can identify the 18 NTDS-defined trauma complications from clinical notes, replacing labor-intensive manual chart review. The pipeline uses DeepSeek R1 (`us.deepseek.r1-v1:0`) and Amazon Titan embeddings (`amazon.titan-embed-text-v2:0`) via AWS Bedrock. Preliminary results on 20 samples show 0.857 overall sensitivity. See `README.md` for full details.
