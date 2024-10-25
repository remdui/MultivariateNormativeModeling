# Master Thesis - Remy Duijsens

---
## General Information

### Project Information

- Author: `Remy Duijsens`
- University: `Delft University of Technology`
- Faculty: `Electrical Engineering, Mathematics, and Computer Science`
- Master: `Computer Science`
- Track: `Artificial Intelligence`
- In collaboration with: `Erasmus MC`
- Subject: `Multivariate Normative Models Using Variational Auto-Encoders: A Study on Covariate Embedding and
Robustness to Site-Variance using Gen R Data`
- Under supervision of: `C. Lofi (TU Delft), R. Muetzel (Erasmus MC), and H. Schnack (Erasmus MC)`

### Literature

Literature and research documents can be found in the `docs/literature` directory.

- The thesis document can be found at [`docs/literature/thesis.pdf`](docs/literature/thesis.pdf)
- The thesis proposal can be found at [`docs/literature/thesis_proposal.pdf`](docs/literature/thesis_proposal.pdf)
- Papers that are relevant to this project can be found at [`docs/literature/papers.md`](docs/literature/papers.md)
- Other documents can be found at [`docs/other`](docs/other)

Relevant documentation for the code can be found in the `docs/code` directory.

### Code Repository

This repository contains the code for the master thesis project. The code is structured as follows:

- The `config` directory contains configuration files required to run the project.
- The `data` directory contains raw and processed data (not included in the repository due to privacy reasons).
- The `docker` directory contains required files to run the project in a Docker container.
- The `docs` directory contains documentation and literature.
- The `logs` directory contains log files.
- The `models` directory contains saved models.
- The `output` directory contains output files (e.g., predictions, visualizations, etc.).
- The `scripts` directory contains shell scripts.
- The `src` directory contains the source code.
- The `tests` directory contains unit tests.

Please find more information on installing, running, and testing the project in the sections below.

---

## Requirements

This project requires the following dependencies installed:
- Python >= 3.12
- Poetry >= 1.8.4

All dependencies are listed in the `pyproject.toml` file.

Alternatively, Docker can be used to run the project. The Dockerfile is provided in the repository. This requires Docker to be installed on the host machine.

---

## Setup

### Directly on Host Machine
You can install the package locally using `poetry`:

```bash
poetry install --no-root
```

### Using Docker
You can build the Docker image using the provided Dockerfile:

```bash
docker build -t master-thesis .
```

---

## Usage

### Entry Point

The entry point of the project is the `main.py` file. Run the project using `poetry`:

```bash
poetry run python src/main.py --config [CONFIG_PATH] --mode [MODE] [OPTIONS]
```

### Arguments

The `main.py` file accepts the following required arguments:
- `--config`: Path to the configuration file. Default: `config/config_001.yml`.
- `--mode`: Mode to run the project in. Options: `train`, `inference`, `validate`. Default: `train`.

The `main.py` file accepts the following optional arguments:
- `--checkpoint`: Path to a checkpoint to load.
- `--checkpoint-interval`: Interval to save checkpoints. Default: `1`.
- `--data_dir`: Directory to save data.
- `--debug`: Debug mode flag.
- `--device`: Device to run the project on. Options: `cpu`, `cuda`.
- `--log_dir`: Directory to save logs.
- `--log_level`: Log level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- `--model_dir`: Directory to save models.
- `--num_workers`: Number of workers for data loading.
- `--output_dir`: Directory to save output.
- `--seed`: Seed for reproducibility.
- `--verbose`: Verbosity flag.

Command line arguments override settings in the provided configuration file.


Check the following files for more information:
- [`docs/code/project_setup.md`](docs/code/project_setup.md) for information on how to set up the project.
- [`docs/code/training.md`](docs/code/training.md) for information on how to train the model.
- [`docs/code/inference.md`](docs/code/inference.md) for information on how to run inference.


### Using Docker

You can run the project in a Docker container after building the image (see the `Setup` section above):

```bash
docker run -v /path/to/data:/data \
           -v /path/to/output:/output \
           -v /path/to/logs:/logs \
           -v /path/to/models:/models \
           -v /path/to/config:/config \
           master-thesis --config [CONFIG_PATH] --mode [MODE] [OPTIONS]
```
---

## Code Quality
### Static Code Analysis

- PyLint is used for static code analysis. Docs for PyLint can be found [here](https://pylint.pycqa.org/en/latest/). Run PyLint using `poetry`:

    ```bash
    poetry run pylint ./src
    ```
- Ruff can be used for additional static code analysis. Docs for Ruff can be found [here](https://docs.astral.sh/ruff/). Run Ruff using `poetry`:

    ```bash
    poetry run ruff check ./src
    ```

### Static Type Checking

- MyPy is used for static type checking. Docs for MyPy can be found [here](https://mypy.readthedocs.io/en/stable/). Run MyPy using `poetry`:

    ```bash
    poetry run mypy ./src
    ```

### Import Sorting
- isort is used for import sorting. Docs for isort can be found [here](https://pycqa.github.io/isort/). Run isort using `poetry`:

    ```bash
    poetry run isort --diff ./src
    ```

  To apply the changes to the files, use without the `--diff` flag:

    ```bash
    poetry run isort ./src
    ```


### Code Formatting
- Black (PEP8 compliant) is used for code formatting. Docs for Black can be found [here](https://black.readthedocs.io/en/stable/). Run Black using `poetry`:

    ```bash
    poetry run black --check --diff ./src
    ```

  To apply the changes to the files, use without the `--check` and `--diff` flags:

    ```bash
    poetry run black ./src
    ```

- Ruff (more aggressive, use with caution\[!]) can be used for additional code formatting. Docs for Ruff can be found [here](https://docs.astral.sh/ruff/). Run Ruff using `poetry`:

    ```bash
    poetry run ruff format --check --diff ./src
    ```

    To apply the changes to the files, use without the `--check` and `--diff` flags:

    ```bash
    poetry run ruff format ./src
    ```

### Unused Code Cleaning
- Autoflake can be used to remove unused imports and variables. Docs for Autoflake can be found [here](https://github.com/PyCQA/autoflake). Run Autoflake using `poetry`:

    ```bash
    poetry run autoflake --remove-all-unused-imports --remove-unused-variables --in-place --check -r .
    ```

    To apply the changes to the files, use without the `--check` flag:

    ```bash
    poetry run autoflake --remove-all-unused-imports --remove-unused-variables --in-place -r .
    ```

### Docstring Formatting

- Docstringsformatter is used to format docstrings. Docs for Docstringsformatter can be found [here](https://pydocstringformatter.readthedocs.io/en/latest/). Run Docstringsformatter using `poetry`:

    ```bash
    poetry run pydocstringformatter ./src
    ```

  To apply the changes to the files, use the `--write` flag:

    ```bash
    poetry run pydocstringformatter ./src --write
    ```

---
## Testing
Run the unit tests using `poetry`:

```bash
poetry run pytest
```
---
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact
For questions please contact me at
- Email: [remyduijsens@gmail.com](mailto:remyduijsens@gmail.com)
- LinkedIn: [Remy Duijsens](https://nl.linkedin.com/in/remyduijsens)
- GitHub: [remdui](https://github.com/remdui)

---
