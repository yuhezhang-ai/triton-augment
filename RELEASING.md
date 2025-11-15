# Release Process

This document describes the process for releasing new versions of triton-augment to PyPI.

## Prerequisites

- Maintainer access to the PyPI project
- PyPI API token configured in `~/.pypirc`
- All tests passing
- CHANGELOG.md updated

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes, backwards compatible

## Release Steps

### 1. Update Version

Update version in **two places**:

```bash
# Update version in pyproject.toml
vim pyproject.toml  # Change version = "X.Y.Z"

# Update version in triton_augment/__init__.py
vim triton_augment/__init__.py  # Change __version__ = "X.Y.Z"
```

### 2. Update CHANGELOG

Add release notes to `CHANGELOG.md`:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature descriptions

### Changed
- Changed feature descriptions

### Fixed
- Bug fix descriptions
```

### 3. Commit Changes

```bash
git add pyproject.toml triton_augment/__init__.py CHANGELOG.md
git commit -m "Release version X.Y.Z"
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin main
git push origin vX.Y.Z
```

### 4. Clean and Build

```bash
# Clean previous builds
make clean

# Or manually:
rm -rf dist/ build/ *.egg-info

# Build distribution packages
python -m build
```

### 5. Test on Test PyPI (Optional but Recommended)

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ triton-augment==X.Y.Z --no-cache-dir

# Verify it works
python -c "import triton_augment as ta; print(ta.__version__)"
```

### 6. Upload to Production PyPI

```bash
# Upload to production PyPI
python -m twine upload dist/*

# Verify on PyPI
# Visit: https://pypi.org/project/triton-augment/X.Y.Z/
```

### 7. Create GitHub Release

1. Go to https://github.com/yuhezhang-ai/triton-augment/releases
2. Click "Draft a new release"
3. Select the tag `vX.Y.Z`
4. Add release notes (copy from CHANGELOG.md)
5. Attach build artifacts (optional)
6. Click "Publish release"

## PyPI Configuration

Create or update `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_API_TOKEN_HERE
```

## Common Issues

### Package Already Exists Error

PyPI doesn't allow overwriting published versions. If you need to fix an issue:
1. Increment the patch version (e.g., 0.1.0 → 0.1.1)
2. Document the fix in CHANGELOG.md

### Missing Files in Package

Verify package contents before uploading:

```bash
# List contents of the wheel
unzip -l dist/triton_augment-X.Y.Z-py3-none-any.whl

# Check that triton_augment/kernels/ is included
unzip -l dist/triton_augment-X.Y.Z-py3-none-any.whl | grep kernels
```

### Cached Installation Issues

Users may have cached versions. Recommend:

```bash
pip install triton-augment==X.Y.Z --no-cache-dir
```

## Rollback

If a release has critical issues:
1. **Cannot delete** from PyPI (per their policy)
2. Release a new patch version with the fix
3. Mark the broken version in CHANGELOG.md
4. Optionally: Use `yank` to discourage new installations

```bash
# Yank a release (makes it invisible but doesn't delete)
twine yank triton-augment X.Y.Z -r pypi
```

## Checklist

Before releasing, verify:

- [ ] All tests pass: `pytest tests/`
- [ ] Version updated in `pyproject.toml` and `__init__.py`
- [ ] CHANGELOG.md updated with release notes
- [ ] Changes committed and tagged
- [ ] Package built: `python -m build`
- [ ] Package verified: `unzip -l dist/*.whl | grep kernels`
- [ ] (Optional) Tested on Test PyPI
- [ ] Uploaded to PyPI: `twine upload dist/*`
- [ ] GitHub release created
- [ ] Announcement posted (if major release)

## Tools

Install required tools:

```bash
pip install build twine
```

## Questions?

Contact the maintainers or open an issue.

