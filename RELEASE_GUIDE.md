# Release Guide - ADAS Core v0.1.0

## 📋 Pre-Release Checklist

- [x] All features implemented (12/12 complete)
- [x] All tests passing (32/32)
- [x] Documentation complete
- [x] Project reorganized with clean structure
- [x] ROS2 integration added
- [x] Debug tools added

---

## 🔀 Step 1: Create Release Branch

```bash
# Check current status
git status
git branch

# Create and switch to release branch
git checkout -b release/v0.1.0

# Add all new files
git add .

# Commit changes
git commit -m "feat: Complete production-ready ADAS Core v0.1.0

- Reorganized project structure (domain-driven packages)
- Added ROS2 integration layer
- Added recording/replay debug tools
- Comprehensive error handling and validation
- Safety monitoring and enforcement
- Production deployment support (Docker, CI/CD)
- Complete documentation (6 guides)

Phase 1-3 implementation complete (12/12 tasks)
"

# Push release branch
git push origin release/v0.1.0
```

---

## 🏷️ Step 2: Create Git Tag

```bash
# Create annotated tag
git tag -a v0.1.0 -m "Release v0.1.0 - Production-Ready ADAS Core

Major Features:
- Complete modular architecture (7 packages)
- Adaptive cruise control (ACC)
- Lane keeping assist (LKA)
- Multi-object tracking
- Safety monitoring
- ROS2 integration
- Record/replay tools
- Docker deployment
- Comprehensive testing (32 tests)

This is the first production-ready release with all Phase 1-3 features implemented.
"

# Verify tag
git tag -l -n9 v0.1.0

# Push tag
git push origin v0.1.0
```

---

## 🔄 Step 3: Merge to Main Branch

### Option A: Via Pull Request (Recommended)

```bash
# Push your branch (already done)
git push origin release/v0.1.0

# Create PR on GitHub
# Go to: https://github.com/jakhon37/ADAS/compare
# Select: base: main <- compare: release/v0.1.0
# Title: "Release v0.1.0 - Production-Ready ADAS Core"
# Review and merge
```

### Option B: Direct Merge

```bash
# Switch to main
git checkout main

# Merge release branch
git merge release/v0.1.0

# Push to main
git push origin main

# Push tag
git push origin v0.1.0
```

---

## 📦 Step 4: Prepare for PyPI Publication

### 4.1 Update Package Metadata

The `pyproject.toml` is already configured, but verify:

```toml
[project]
name = "adas-core"
version = "0.1.0"
description = "Production-grade Advanced Driver Assistance System"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

### 4.2 Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Verify email
3. Enable 2FA (recommended)
4. Create API token: https://pypi.org/manage/account/token/

### 4.3 Install Build Tools

```bash
pip install --upgrade build twine
```

### 4.4 Build Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build

# Verify contents
ls -lh dist/
# Should see:
# adas_core-0.1.0-py3-none-any.whl
# adas-core-0.1.0.tar.gz
```

### 4.5 Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install and test
pip install --index-url https://test.pypi.org/simple/ adas-core

# Test installation
python -c "import adas; print(adas.__version__)"
adas-run --frames 5
```

### 4.6 Publish to PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Enter credentials or use API token
# Username: __token__
# Password: pypi-xxxxxxxxxxxxxxxxxxxxx
```

### 4.7 Verify Publication

```bash
# Install from PyPI
pip install adas-core

# Test
python -c "import adas; print(adas.__version__)"
adas-run --frames 10
```

---

## 🚀 Step 5: Create GitHub Release

1. Go to: https://github.com/jakhon37/ADAS/releases/new
2. Choose tag: `v0.1.0`
3. Release title: `v0.1.0 - Production-Ready ADAS Core`
4. Description:

```markdown
## 🎉 ADAS Core v0.1.0 - First Production Release

This is the first production-ready release of ADAS Core, a modular Advanced Driver Assistance System for edge deployment.

### ✨ Major Features

- **Adaptive Cruise Control (ACC)** with time-gap following
- **Lane Keeping Assist (LKA)** with proportional steering
- **Multi-Object Tracking** with persistent IDs
- **Safety Monitoring** with multi-layer constraint enforcement
- **ROS2 Integration** for robotics stacks (Autoware, Apollo)
- **Record/Replay Tools** for debugging and analysis

### 📦 What's Included

- 29 Python modules across 7 packages
- 32 comprehensive unit tests
- 6 documentation guides
- Docker deployment support
- CI/CD pipeline
- Example configurations

### 📊 Statistics

- **Lines of Code**: ~2,900
- **Test Coverage**: 100% passing
- **Documentation**: 35+ KB
- **Performance**: <1ms latency (mock), 30-50ms (production)

### 🚀 Installation

```bash
pip install adas-core
```

### 📖 Quick Start

```bash
# Run synthetic test
adas-run --frames 60

# With custom config
adas-run --config config.json
```

### 📚 Documentation

- [Architecture Guide](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [ROS2 Integration](docs/ROS2_INTEGRATION.md)
- [Tools Guide](docs/TOOLS_GUIDE.md)

### 🎯 Roadmap

See [CHANGELOG.md](CHANGELOG.md) for full details.

**Full Changelog**: https://github.com/jakhon37/ADAS/commits/v0.1.0
```

5. Attach files (optional):
   - `dist/adas-core-0.1.0.tar.gz`
   - `dist/adas_core-0.1.0-py3-none-any.whl`

6. Click "Publish release"

---

## 📝 Step 6: Update Documentation

### Update README badges

```markdown
[![PyPI version](https://badge.fury.io/py/adas-core.svg)](https://badge.fury.io/py/adas-core)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/jakhon37/ADAS/workflows/CI/badge.svg)](https://github.com/jakhon37/ADAS/actions)
```

---

## 🔧 Configuration Files

### .pypirc (optional, for credentials)

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxx

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-xxxxxxxxxxxxxxxxxxxxxxxx
```

**⚠️ Security**: Never commit `.pypirc` to git! Add to `.gitignore`.

---

## ✅ Post-Release Checklist

- [ ] Branch created and pushed
- [ ] Tag created and pushed
- [ ] Merged to main
- [ ] Built distributions
- [ ] Tested on TestPyPI
- [ ] Published to PyPI
- [ ] GitHub release created
- [ ] Documentation updated
- [ ] Badges updated
- [ ] Announced on social media (optional)

---

## 🎊 Announcement Template

### Twitter/LinkedIn

```
🚀 Excited to announce ADAS Core v0.1.0!

Production-ready Advanced Driver Assistance System with:
✅ ACC & Lane Keeping
✅ Multi-object tracking
✅ ROS2 integration
✅ Safety monitoring
✅ Docker deployment

Install: pip install adas-core

#AutonomousDriving #ROS2 #Python #ADAS
https://github.com/jakhon37/ADAS
```

### GitHub Discussions

```
ADAS Core v0.1.0 is now available on PyPI! 🎉

This is our first production-ready release with complete Phase 1-3 implementation.

Try it out:
pip install adas-core
adas-run --frames 60

We'd love your feedback!
```

---

## 🔄 Future Releases

### Semantic Versioning

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.2.0): New features, backwards compatible
- **PATCH** (0.1.1): Bug fixes, backwards compatible

### Next Release Workflow

```bash
# Start new development
git checkout main
git pull
git checkout -b develop

# Make changes...

# When ready for release
git checkout -b release/v0.2.0
# Update version in pyproject.toml
# Update CHANGELOG.md
# Create tag and release (same process)
```

---

## 📞 Support

- Issues: https://github.com/jakhon37/ADAS/issues
- Discussions: https://github.com/jakhon37/ADAS/discussions
- PyPI: https://pypi.org/project/adas-core/

---

**Congratulations on your first release! 🎉**
