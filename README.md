# SymOp

**SymOp** is a symbolic framework for quantum optics and photonic systems.

It enables modeling of non-orthogonal photonic modes, symbolic operator transformations, and device-based simulation workflows.

---

## 📚 Documentation

Full documentation is available at:
👉 https://symop.readthedocs.io/en/latest/

---

## 🚀 Installation

**From PyPI (stable release)**

```bash
pip install symop
```

**From GitHub (latest master branch)**

```bash
pip install git+https://github.com/tqsd/symop.git
```

**From GitHub (Specific branch)**

```bash
pip install git+https://github.com/tqsd/symop.git@branch-name
```

**From GitHub (Specific commit)**

```bash
pip install git+https://github.com/tqsd/symop.git@<commit-hash>
```

---

## 🛠️ Development

**Setup:**

```bash
git clone https://github.com/tqsd/symop.git
cd symop
pip install -e .
# or
pip install -e .[dev]
```

**Common Tasks**
```bash
make format       # format code (ruff)
make lint         # lint code (ruff)
make typecheck    # static typing (mypy)
make contracts    # import structure checks
make check        # run all checks
```

**Testing**
```bash
make test         # run pytest
make test-all     # run full tox suite
```

**Coverage**
```bash
make coverage         # generate coverage reports
make docs-coverage    # include coverage in docs
```

**Documentation**
```bash
make docs-html   # build docs
make docs-live   # live docs server
make docs-clean  # clean docs build
```

---

## 🧠 Overview

SymOp focuses on:

* Symbolic representations of quantum optical systems
* Non-orthogonal mode handling
* Device-based simulation workflows

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{sekavcnik2026symbolic,
  title={Symbolic Quantum State Representation and its Simulation},
  author={Sekavcnik, Simon and Noetzel, Janis},
  journal={arXiv preprint arXiv:2603.11824},
  year={2026}
}
```

or refer to:
https://arxiv.org/abs/2603.11824

---

## ⚠️ Status

This is an early-stage research project. APIs may change.

---

## 📜 License

Apache-2.0
