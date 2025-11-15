# Documentation Guide

This document explains how to build, preview, and deploy the Triton-Augment documentation.

## Structure

```
README.md                 # Home page (shared with GitHub)
docs/
├── installation.md       # Installation guide
├── quickstart.md        # Quick start tutorial
├── float16.md           # Float16 support
├── batch-behavior.md    # Batch behavior guide
├── contrast.md          # Contrast implementation details
├── auto-tuning.md       # Auto-tuning guide
├── api-reference.md     # API reference
├── requirements.txt     # Docs dependencies
└── stylesheets/
    └── extra.css        # Custom CSS
```

**Note**: The home page uses the root `README.md` to avoid duplication between GitHub and documentation site.

## Building the Documentation

### Install Dependencies

```bash
# From project root
pip install -r docs/requirements.txt
```

### Preview Locally

```bash
# Start local server (with auto-reload)
mkdocs serve

# Open in browser
# http://127.0.0.1:8000
```

The server will auto-reload when you edit documentation files.

### Build Static Site

```bash
# Build HTML files
mkdocs build

# Output is in site/ directory
# Open site/index.html in browser to preview
```

## Deploying to GitHub Pages

### Option 1: Automatic (Recommended)

```bash
# Build and deploy in one command
mkdocs gh-deploy

# This will:
# 1. Build the site
# 2. Push to gh-pages branch
# 3. GitHub will serve it automatically
```

Your docs will be available at: `https://<username>.github.io/<repo>/`

### Option 2: Manual

```bash
# Build locally
mkdocs build

# Copy site/ contents to your web server
# Or commit site/ to gh-pages branch manually
```

## Documentation Workflow

### Adding a New Page

1. Create new `.md` file in `docs/`
2. Add entry to `mkdocs.yml` navigation
3. Preview with `mkdocs serve`
4. Commit and deploy

Example:

```yaml
# mkdocs.yml
nav:
  - Home: index.md
  - Getting Started:
      - Installation: installation.md
      - Quick Start: quickstart.md
      - Your New Page: new-page.md  # Add here
```

### Updating API Reference

The API reference (`docs/api-reference.md`) is manually written. To add new functions:

1. Add function documentation to `api-reference.md`
2. Follow the existing format (code examples + parameter descriptions)
3. Add cross-references to relevant guides

For auto-generated API docs (future):
- Use `mkdocstrings` plugin (already configured)
- Add docstring references: `::: triton_augment.functional`

### Writing Style

- **Use code examples**: Show, don't just tell
- **Add admonitions**: `!!! note`, `!!! warning`, `!!! tip`
- **Include cross-references**: Link related pages
- **Keep it concise**: Short paragraphs, clear headings
- **Use tables**: For comparisons and parameter lists

### Markdown Extensions

Available extensions (configured in `mkdocs.yml`):

```markdown
# Admonitions (call-out boxes)
!!! note "Optional Title"
    This is a note

!!! warning
    This is a warning

# Code blocks with syntax highlighting
```python
import triton_augment as ta
```

# Tabbed content
=== "Tab 1"
    Content 1

=== "Tab 2"
    Content 2

# Tables
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
```

## Theme Customization

### Colors

Edit `mkdocs.yml`:

```yaml
theme:
  palette:
    primary: indigo  # Change to: red, pink, purple, etc.
    accent: indigo
```

### Logo and Favicon

Add files:
- `docs/images/logo.png`
- `docs/images/favicon.ico`

Update `mkdocs.yml`:

```yaml
theme:
  logo: images/logo.png
  favicon: images/favicon.ico
```

### Custom CSS

Edit `docs/stylesheets/extra.css` for custom styles.

## Troubleshooting

### Build Errors

```bash
# Clear cache
rm -rf site/

# Rebuild
mkdocs build
```

### Missing Dependencies

```bash
# Reinstall
pip install -r docs/requirements.txt --upgrade
```

### gh-deploy Issues

```bash
# Check git status
git status

# Ensure you're on main branch
git checkout main

# Try force deploy
mkdocs gh-deploy --force
```

## Best Practices

1. **Test locally** before deploying (`mkdocs serve`)
2. **Keep README short** - link to full docs
3. **Update docs** when adding features
4. **Use examples** - code speaks louder than words
5. **Cross-reference** - link related pages
6. **Version docs** - note when features were added

## Continuous Integration (Optional)

Add GitHub Actions to auto-deploy docs:

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.x
      - run: pip install -r docs/requirements.txt
      - run: mkdocs gh-deploy --force
```

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [Markdown Guide](https://www.markdownguide.org/)

