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

```bash
poetry install --no-root
```

## How to use
Train the model using `poetry`:

```bash
poetry run python src/training/train.py
```

## Code Quality
Run the linter using `poetry`:

```bash
poetry run pylint
```

## Testing
Run the unit tests using `poetry`:

```bash
poetry run pytest
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
