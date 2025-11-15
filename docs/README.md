# Triton-Augment Documentation

This folder contains the complete documentation for Triton-Augment.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Preview locally (with auto-reload)
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

## Documentation Files

- `index.md` - Home page (copied from root README.md)
- `installation.md` - Installation guide
- `quickstart.md` - Quick start tutorial
- `float16.md` - Float16 support guide
- `batch-behavior.md` - Batch augmentation patterns
- `contrast.md` - Contrast implementation details
- `auto-tuning.md` - Auto-tuning guide
- `api-reference.md` - Complete API reference

## Building and Deploying

### Build locally
```bash
mkdocs build
# Output in site/ directory
```

### Deploy to GitHub Pages
```bash
mkdocs gh-deploy
```

## Editing Documentation

1. Edit `.md` files in this folder
2. Preview changes: `mkdocs serve`
3. Commit changes
4. Deploy: `mkdocs gh-deploy` (optional)

## Style Guide

- Use code examples for every feature
- Add admonitions for important info (!!! note, !!! warning)
- Link to related pages
- Keep pages focused (< 500 lines)
- Test examples before committing

See `DOCUMENTATION.md` for complete guide.

## MkDocs Configuration

The site is configured via `../mkdocs.yml`:
- **Theme**: Material (modern, dark mode, search)
- **Plugins**: mkdocstrings (API docs), search
- **Extensions**: Admonitions, syntax highlighting, tables

## Resources

- [MkDocs](https://www.mkdocs.org/)
- [Material Theme](https://squidfunk.github.io/mkdocs-material/)
- [Writing Guide](DOCUMENTATION.md)

