# Documentation Structure

## Overview

```
triton-augment/
│
├── README.md                    # SHORT (150 lines) - Quick start only
│   ├── Features & installation
│   ├── 30-second example
│   └── Links to full docs →
│
├── docs/                        # FULL DOCUMENTATION
│   ├── index.md                # Home page
│   ├── installation.md         # Detailed setup
│   ├── quickstart.md          # Tutorial
│   ├── float16.md             # Float16 guide
│   ├── batch-behavior.md      # Batch patterns
│   ├── contrast.md            # Contrast details
│   ├── auto-tuning.md         # Performance tuning
│   ├── api-reference.md       # Complete API
│   ├── requirements.txt       # MkDocs deps
│   ├── stylesheets/
│   │   └── extra.css          # Custom styles
│   └── DOCUMENTATION.md       # Maintenance guide
│
└── mkdocs.yml                  # MkDocs config
```

## Content Flow

```
GitHub README (150 lines)
    ↓
    Quick pitch & example
    ↓
    Link to full docs →
                        ↓
                    docs/ folder
                        ↓
                    MkDocs site
                        ↓
                    GitHub Pages
```

## Page Purposes

| Page | Purpose | Length |
|------|---------|--------|
| **README.md** | Quick start & pitch | ~150 lines |
| **docs/index.md** | Welcome & overview | ~200 lines |
| **docs/installation.md** | Setup guide | ~150 lines |
| **docs/quickstart.md** | Tutorial | ~100 lines |
| **docs/float16.md** | Float16 guide | ~200 lines |
| **docs/batch-behavior.md** | Batch patterns | ~150 lines |
| **docs/contrast.md** | Contrast details | ~200 lines |
| **docs/auto-tuning.md** | Performance guide | ~250 lines |
| **docs/api-reference.md** | Complete API | ~400 lines |

**Total**: ~1,800 lines of organized documentation (vs 679 lines in one file)

## User Journey

### New User
```
1. Land on GitHub → README.md
2. See quick example → Try it
3. Want details → Click docs link
4. Read installation.md → Set up
5. Read quickstart.md → Learn basics
6. Bookmark api-reference.md → Use as reference
```

### Returning User
```
1. Google "triton augment float16"
2. Land on docs/float16.md directly
3. Find what they need
4. Done!
```

### Contributor
```
1. Fork repo
2. Read CONTRIBUTING.md
3. Edit specific doc page
4. Submit PR
5. Easy review (small diff)
```

## Benefits

### ✅ For Users
- **Quick start**: README gets you running in 2 minutes
- **Deep dive**: Docs provide complete information
- **Searchable**: Find what you need fast
- **Professional**: Looks like production library

### ✅ For Maintainers
- **Organized**: Each topic in its own file
- **Scalable**: Easy to add new pages
- **Maintainable**: Small, focused files
- **Reviewable**: Clear git diffs

### ✅ For Contributors
- **Clear structure**: Know where to add content
- **Easy editing**: Edit specific pages
- **Preview locally**: See changes before PR
- **Low friction**: Small, focused PRs

## Documentation Standards

### Each Page Should Have

1. **Clear title** (H1)
2. **Brief intro** (2-3 sentences)
3. **Code examples** (show, don't tell)
4. **Cross-references** (link related pages)
5. **Admonitions** (notes, warnings, tips)

### Style Guide

```markdown
# Use H1 for page title
## Use H2 for major sections
### Use H3 for subsections

# Code blocks with language
```python
import triton_augment as ta
```

# Admonitions for important info
!!! warning "Important"
    This is critical information

!!! note
    This is a helpful note

# Links to other pages
See the [Installation Guide](installation.md) for details.

# Tables for comparisons
| Feature | Value |
|---------|-------|
| Speed | Fast |
```

## MkDocs Features Used

### Theme: Material
- Modern, clean design
- Dark/light mode
- Mobile-responsive
- Fast search

### Plugins
- `search`: Client-side search
- `mkdocstrings`: Auto-generate from docstrings (ready)

### Extensions
- `admonition`: Call-out boxes
- `pymdownx.highlight`: Syntax highlighting
- `pymdownx.tabbed`: Tabbed content
- `tables`: Markdown tables

## Deployment Options

### Option 1: GitHub Pages (Recommended)
```bash
mkdocs gh-deploy
# Docs at: https://yuhezhang-ai.github.io/triton-augment/
```

### Option 2: ReadTheDocs
1. Connect GitHub repo
2. Add `.readthedocs.yml` config
3. Auto-deploy on push

### Option 3: Self-Hosted
```bash
mkdocs build
# Deploy site/ folder to web server
```

## Maintenance Workflow

### Adding New Feature

1. **Code**: Implement feature
2. **Docstring**: Add/update docstrings
3. **API Reference**: Update `docs/api-reference.md`
4. **Guide**: Add guide page if complex
5. **Quick Start**: Update example if relevant
6. **README**: Mention in README if major

### Fixing Bugs

1. **Code**: Fix bug
2. **Tests**: Add regression test
3. **Docs**: Update if behavior changed
4. **Changelog**: Document fix

### Major Version

1. **All docs**: Review and update
2. **Examples**: Ensure they work
3. **Migration guide**: Add if breaking changes
4. **README**: Update version badges

## Best Practices

### ✅ DO
- Keep README short (< 200 lines)
- Link to docs for details
- Use code examples
- Add cross-references
- Test docs locally before deploy
- Keep pages focused (< 500 lines each)

### ❌ DON'T
- Put everything in README
- Duplicate content across pages
- Use vague descriptions
- Forget to update docs with code changes
- Deploy without testing

## Quick Reference

```bash
# Install
pip install -r docs/requirements.txt

# Preview
mkdocs serve
# → http://127.0.0.1:8000

# Build
mkdocs build
# → site/ folder

# Deploy
mkdocs gh-deploy
# → GitHub Pages

# Clean
rm -rf site/
```

## Success Metrics

✅ **README is short** (~150 lines)  
✅ **Docs are organized** (9 focused pages)  
✅ **Easy to navigate** (clear structure)  
✅ **Professional look** (Material theme)  
✅ **Searchable** (built-in search)  
✅ **Mobile-friendly** (responsive)  
✅ **Fast** (static site)  
✅ **Maintainable** (small, focused files)  

## Result

**Before**: 1 file, 679 lines, hard to navigate  
**After**: 11 files, organized, professional, easy to use

**Your library now has world-class documentation!** 🚀

