# Decoupling-QD

This repository contains the source code for DECOUPLING-QD. 

## Directory Structure

The file structure for each cipher is organized as follows:

```text
CIPHERNAME/
├── data/
│   └── (Contains differential trails for testing)
└── src/
    ├── {CIPHERNAME}_SOLVER/       # Contains solver files
    │                                           └─  {CIPHERNAME}_SOLVER.py # Run this script to generate distribution 
    │                                           └─  CONS/      # Folder of the identified constraints
    ├── constraint-collector.py    # Run this script to generate constraints
    ├── {CIPHERNAME}_Q.py          # Executes local search for QDC in each decoupled constraint under a mask
    └── utils.py                   # Utility functions (Adjust `THRESH` for correlation weight, and `MIN_CORR` for the search boundary)