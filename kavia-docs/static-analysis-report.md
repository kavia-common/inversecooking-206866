# Static analysis report — inversecooking-206866

Date: 2026-01-29

## Environment

- Workspace: `inversecooking-206866`
- Python: 3.12.3

## Tools executed

### 1) Syntax compilation check
Command:
- `python -m compileall -q src`

Result:
- ✅ Passed (no syntax errors reported)

---

### 2) Linting (ruff)
Installation:
- `python -m pip install -q ruff`

Command:
- `ruff check src --output-format=concise`

Result:
- ❌ **Found 50 errors** (39 auto-fixable with `--fix`)

High-level categories:

1. **Unused imports (F401)** (most common)
   - `src/args.py`: unused `os`
   - `src/build_vocab.py`: unused `numpy`, `re`
   - `src/data_loader.py`: unused `torchvision.transforms`, `nltk`, `json`, `build_vocab.Vocabulary`
   - `src/model.py`: unused `random`, `numpy`, `pickle`, `os`, unused `MultiheadAttention` import
   - `src/modules/encoder.py`: unused torchvision model imports (resnet/vgg/inception) + unused `random`, `numpy`
   - `src/utils/metrics.py`: unused `sys`, `time`, `math`, `torch.nn.functional`
   - `src/utils/tb_visualizer.py`: unused `numpy`, `ntpath`, `time`, `scipy.misc.imresize`
   - `src/sample.py`: unused `build_vocab.Vocabulary`

2. **Star import / undefined name risks**
   - `src/build_vocab.py:9`: `from tqdm import *` (F403)
   - `src/build_vocab.py:185, 294`: `tqdm` may be undefined (F405)

3. **Style/correctness suggestions**
   - `src/build_vocab.py`: membership test should use `not in` (E713)
   - `src/modules/transformer_decoder.py:446`: avoid `== True` (E712)
   - `src/utils/output_utils.py:67`: bare `except:` (E722)
   - `src/modules/utils.py`: ambiguous variable name `l` (E741)

4. **Notebook lint**
   - `src/demo.ipynb`: unused imports; imports not at top of cell (E402); multiple statements per line (E702)

---

### 3) Type checking (mypy)
Installation:
- `python -m pip install -q mypy`

Command:
- `mypy src --ignore-missing-imports`

Result:
- ❌ Fails early with:
  - `Source file found twice under different module names: "utils" and "modules.utils"`

Notes:
- This typically happens when the project isn't structured as packages (missing `__init__.py`) and the type checker maps the same file into multiple module names depending on import paths.
- Common resolutions: add `__init__.py`, use `--explicit-package-bases`, or set `MYPYPATH`.

---

### 4) Type checking (pyright)
Installation:
- `python -m pip install -q pyright`

Command:
- `pyright src`

Result:
- ❌ Reports **84 errors and 3 warnings**, dominated by **unresolved imports**:
  - `torch`, `torchvision`, `lmdb`, `nltk`, `tensorboardX`, `fairseq`, `scipy.misc`, etc.

Dependency root cause:
- `pip install -r requirements.txt` fails on Python 3.12 because:
  - `torch==0.4.1` has no compatible wheels for Python 3.12
  - many historical versions have `Requires-Python <3.12`
- As a result, pyright cannot resolve core imports and produces cascading type errors.

## Recommendations (next actions)

1. **Decide target runtime stack**
   - Option A (historical reproducibility): run static analysis in a Python 3.6/3.7 environment with old torch (0.4.1).
   - Option B (modernization): update dependencies to a modern torch/torchvision compatible with Python 3.12, then re-run pyright/mypy to get meaningful results.

2. **Fix ruff issues**
   - Many are safe cleanup:
     - remove unused imports
     - replace `from tqdm import *` with `from tqdm import tqdm`
     - change `if not word in x` → `if word not in x`
     - replace `== True` with direct truthiness
     - replace bare `except:` with `except Exception:` (or narrower)
     - rename ambiguous `l` variables

3. **Make module structure explicit for type checkers**
   - Add `__init__.py` under `src/`, `src/modules/`, `src/utils/` (or configure pyright/mypy to treat `src` as the import root).
   - Consider adding `pyproject.toml` with tool config (`ruff`, `pyright`, possibly `mypy`).

## Commands to reproduce

From repo root:

- Syntax:
  - `python -m compileall -q src`

- Ruff:
  - `python -m pip install -q ruff`
  - `ruff check src --output-format=concise`

- Mypy:
  - `python -m pip install -q mypy`
  - `mypy src --ignore-missing-imports`

- Pyright:
  - `python -m pip install -q pyright`
  - `pyright src`
