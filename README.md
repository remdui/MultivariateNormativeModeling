# Thesis Project

Name: Remy Duijsens

Subject: Multivariate Normative Models Using Variational Auto-Encoders: A Study on Covariate Embedding and
Robustness to Site-Variance using Gen R Data

Under supervision of: C. Lofi (TU Delft), R. Muetzel (Erasmus MC), and H. Schnack (Erasmus MC)

Start date: 14/10/24

## Requirements
- Python >= 3.12
- Poetry >= 1.8.4

## Installation
You can install the package locally using `poetry`:

```console
$ poetry install
```

## Project Structure
```
./
│
├── config/
│   ├── .gitkeep
│   └── config_001.yml
│
├── data/
│   │
│   ├── processed/
│   │   └── .gitkeep
│   │
│   └── raw/
│       └── .gitkeep
│
├── docs/
│   ├── inference.md
│   ├── project_setup.md
│   └── training.md
│
├── logs/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── output/
│   └── .gitkeep
│
├── scripts/
│   ├── run_inference.sh
│   ├── run_training.sh
│   └── setup_env.sh
│
├── src/
│   │
│   ├── inference/
│   │
│   ├── preprocessing/
│   │   └── data_loader.py
│   │
│   ├── training/
│   │
│   └── .gitkeep
│
├── tests/
│   └── .gitkeep
│
├── .gitignore
├── LICENSE
├── README.md
```
