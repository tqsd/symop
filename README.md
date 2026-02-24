# Anonymized Supplementary Material

This archive contains an anonymized snapshot of the code and notebook used
to generate the Hong-Ou-Mandel simulation results in the submitted manuscript.

All identifying information has been removed to preserve the double-blind review
process. This archive is intended solely to allow reviewers to reproduce the simulation
and inspect the operator-algebraic implementation.

## Contents
- `symop_proto/` - Minimal symbolic operator framework used in the simulation
- `examples/` - Example scripts, containing the Hong-Ou-Mandel simulation
- `pyproject.toml` - Python package requirements

## How to Run
Tested with Python 3.13.7, but any standard Python >3.9 environment should work.
```bash
pip install .
```

To run the example notebook:
```bash
pip install jupyter # Required to view ".ipynb" notebooks
# pip install tqdm # Required to run the example
cd examples
jupyter notebook
```

## About example

The notebook executes the full HOM workflow:
1. Constructing non-orthogonal temporal modes  
2. Applying the 50/50 beamsplitter map  
3. Computing the coincidence probability  
4. Reproducing the HOM dip shown in the manuscript  

## Notes

- This is a simplified, self-contained version of the full project aimed only at
  reproducibility during review.
- After acceptance, a full, non-anonymized version will be released.

