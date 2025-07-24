# Multivariate Normative Modeling

[![CI-Pipeline](https://github.com/remdui/MasterThesis/actions/workflows/ci_pipeline.yml/badge.svg)](https://github.com/remdui/MasterThesis/actions/workflows/ci_pipeline.yml)

---
## General Information


### Code Repository

This repository contains the code for the master thesis project. The code is structured as follows:

- The `config` directory contains configuration files required to run the project.
- The `data` directory contains raw and processed data (not included in the repository due to privacy reasons).
- The `docker` directory contains required files to run the project in a Docker container.
- The `docs` directory contains documentation and literature.
- The `experiments` directory contains input and output artifact per experiment.
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
- Python >= 3.11.3 (matching the latest version on Snellius HPC)
- Poetry >= 1.8.4

All dependencies [README.md](README.md) are listed in the `pyproject.toml` file.

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
docker build -t masterthesis:latest -f ./docker/Dockerfile .
```

---

## Legacy Setup

In environments where `poetry` is not available and where Docker is not an option, the project can be run using `pip` and `virtualenv`. The following steps can be used to set up the project:

- Create a requirements.txt file using `poetry` from the `pyproject.toml` file:

    ```bash
    poetry export --with=dev --with=test --without-hashes --without-urls | awk '{ print $1 }' FS=';' > requirements.txt
    ```

- Create a virtual environment using `virtualenv`:

    ```bash
    virtualenv venv && source venv/bin/activate
    ```

- Install the dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

- Replace all `poetry run python` commands with `python` in all subsequent commands.


---

## Usage

### Entry Point

The entry point of the project is the `main.py` file. Run the project using `poetry`:

```bash
poetry run python src/main.py --config [CONFIG_PATH] --mode [MODE] [OPTIONS]
```

You can also run the project in a Docker container after building the image (see the `Setup` section above):

```bash
docker run --rm \
           -v ./data:/data \
           -v ./output:/output \
           -v ./logs:/logs \
           -v ./models:/models \
           -v ./config:/config \
           masterthesis:latest \
           --config [CONFIG_PATH] \
           --mode [MODE] \
           [OPTIONS]
```

### Arguments

The `main.py` file accepts the following required arguments:
- `--config`: Path to the configuration file. Default: `config/config_default.yml`.
- `--mode`: Mode to run the project in. Options: `train`, `inference`, `validate`, `tune`. Default: `train`.

The `main.py` file accepts the following optional arguments:
- `--checkpoint`: Path to a checkpoint to load.
- `--checkpoint-interval`: Interval to save checkpoints.
- `--data_dir`: Directory to save data.
- `--debug`: Debug mode flag.
- `--device`: Device to run the project on. Options: `cpu`, `cuda`.
- `--log_dir`: Directory to save logs.
- `--log_level`: Log level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- `--models_dir`: Directory to save models.
- `--num_workers`: Number of workers for data loading.
- `--output_dir`: Directory to save output.
- `--skip-preprocessing`: Skip preprocessing pipeline flag.
- `--seed`: Seed for reproducibility.
- `--verbose`: Verbosity flag.

Command line arguments override settings in the provided configuration file.


Check the following files for more information:
- [`docs/code/project_setup.md`](docs/code/project_setup.md) for information on how to set up the project.
- [`docs/code/training.md`](docs/code/training.md) for information on how to train the model.
- [`docs/code/inference.md`](docs/code/inference.md) for information on how to run inference.


### Datasets

Datasets must be provided in the `data` directory.

This project supports the following data types:
- **Tabular data** in the following formats: `csv`, `rds`.

- **Image data** in the following formats: `jpg`, `jpeg`, `png`, `bmp`, `tiff`.
  Images should be placed in a directory with the same name as the dataset file (e.g., `dataset/<image_name>.jpg`). If a test split is provided, the test split directory must be named `test` (e.g., `dataset/test/<image_name>.jpg`).

The preprocessing pipeline is automatically applied to the datasets. The configuration file contains settings for the preprocessing pipeline, such as split sizes, transformations, and shuffling.
Processed datasets are saved in the `data/processed` directory.

The internal file format is `hdf`, but can be changed to `csv` in the configuration file for debugging purposes (not recommended for large datasets). To view HDF files, use the `hdfview` tool, see [here](https://www.hdfgroup.org/download-hdfview/).

To skip the preprocessing pipeline, use the `--skip-preprocessing` flag.

### Configuration

The project configuration is defined in the `config` directory. The configuration file is in `YAML` format. The default configuration file is `config/config_default.yml`.

**Important:** Copy this file to create a custom configuration. The default configuration will be **overwritten** each time the application is run.

The configuration file contains the following sections:

- `dataset`: Configuration for the dataset.
- `model`: Configuration for the model.
- `train`: Configuration for the training process.
- `inference`: Configuration for the inference process.
- `validation`: Configuration for the validation process.
- `meta`: Meta information for the project.
- `general`: General configuration for the project.
- `system`: System configuration for the project.

The configuration is validated against the schema before a task is run. Missing values are filled with default values from the default configuration file.

### Output Files

The application generates different types of artifacts during the execution of the project.

- **Logs**: Logs are saved in the `logs` directory, unless a custom log directory is provided. Logs are saved in the `log` format.
- **Models**: Models are saved in the `models` directory, unless a custom models directory is provided. Models are represented by their state dictionary and are saved in the `pt` format. Checkpoints are stored in the `checkpoints` subdirectory.
- **Output**: Output files are saved in the `output` directory, unless a custom output directory is provided. Examples of output files are metrics, visualizations, and predictions.
- **Processed Data**: Processed data is saved in the `data/processed` directory. Processed data is saved in the `csv` format.

Output files are not tracked in the repository. Output files and folders can be safely deleted.
Additionally, all input and output files are copied to a timestamped experiment folder to keep track of experiments and to enable reproducability.

### Scripts

The `scripts` directory contains various scripts:

- The data scripts can be used to generate artificial data for testing purposes.
- The slurm script are used to run the project on the (Snellius) HPC.
- The sync script can be used to synchronize the required files to the HPC (requires established SSH connection).

---

## Contributing

### Dependency Management

`Poetry` is used for dependency management. Docs for Poetry can be found [here](https://python-poetry.org/docs/).


- Add a new dependency using `poetry`:

    ```bash
    poetry add [DEPENDENCY]
    ```

- To update a dependency, use the `poetry` command:

    ```bash
    poetry update [DEPENDENCY]
    ```

- To remove a dependency, use the `poetry` command:

    ```bash
    poetry remove [DEPENDENCY]
    ```

If the `.pyproject.toml` is manually updated, make sure to run the following command to update the lock file and to install the new dependencies:

```bash
poetry lock && poetry install
```


### Pre-Commit Hooks

Pre-commit hooks are used to ensure consistent and validated code style and quality. Docs for Pre-Commit can be found [here](https://pre-commit.com/). Pre-commit hooks are defined in the `.pre-commit-config.yaml` file.

- Install the pre-commit hooks using `poetry`:

    ```bash
    poetry run pre-commit install
    ```

- Run the pre-commit hooks manually using `poetry`:

    ```bash
    poetry run pre-commit run --all-files
    ```

**Note:** Code quality can also be manually checked before committing changes using the code quality tools described below in the `Code Quality` section.


### Git Workflow

- Create a new branch for a new feature or bug fix:

    ```bash
    git checkout -b feature/feature_name
    ```

- Commit changes to the branch:

    ```bash
    git add .
    git commit -m "Commit message"
    ```

- Push the branch to the remote repository:

    ```bash
    git push origin feature/feature_name
    ```

- Create a pull request on GitHub and assign reviewers.


### Code Style

- `PEP8` standard for python code is used as code style guide. Docs for PEP8 can be found [here](https://pep8.org/).
- Code style is enforced using the pre-commit hooks and CI/CD pipelines. Manual checks are available using the code quality tools described below in the `Code Quality` section.
- Type annotations are used to enforce type checking. Docs for type annotations can be found [here](https://docs.python.org/3/library/typing.html).


### Code Quality

Code quality is enforced using the bundled code quality tools and unit tests.

Different levels of code quality checks are available:
- Manual checks are available using the code quality tools described below in the `Code Quality` section and the testing tools in the `Testing` section.
- `Pre-commit hooks` are used to validate local changes before committing.
- Remote CI/CD pipelines are used to ensure code quality and to run tests. The CI/CD pipeline is set up using GitHub Actions. The pipeline can be found in the `.github/workflows` directory.
- Model validation is performed using the `validate` mode in the `main.py` file. This mode can be used to validate the model using a validation dataset.

---

## Code Quality
### Static Code Analysis

- PyLint is used for static code analysis. Docs for PyLint can be found [here](https://pylint.pycqa.org/en/latest/). Settings for PyLint can be found in the `.pylintrc` file. Run PyLint using `poetry`:

    ```bash
    poetry run pylint ./src
    ```
- Ruff can be used for additional static code analysis. Docs for Ruff can be found [here](https://docs.astral.sh/ruff/). Settings for Ruff can be found in the `.ruff` file. Run Ruff using `poetry`:

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

- Ruff (more aggressive, use with caution\[!]) can be used for additional code formatting. Docs for Ruff can be found [here](https://docs.astral.sh/ruff/). Settings for Ruff can be found in the `.ruff` file. Run Ruff using `poetry`:

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

### Upgrade Python Syntax (>=3.12 compatible)

- PyUpgrade can be used to upgrade Python syntax. Docs for PyUpgrade can be found [here](https://github.com/asottile/pyupgrade). Run PyUpgrade using `poetry`:

    ```bash
    poetry run pyupgrade --py312-plus
    ```

### Code Security

- Bandit is used for code vulnerability and security checks. Docs for Bandit can be found [here](https://bandit.readthedocs.io/en/latest/). Settings for Bandit can be found in the `.bandit` file. Run Bandit using `poetry`:

    ```bash
    poetry run bandit -r .
    ```

### Code Metrics

- `Radon` is used for code metrics. Docs for Radon can be found [here](https://radon.readthedocs.io/en/latest/). Various metrics can be calculated using Radon.

  - **Cyclomatic Complexity**: Measures the complexity of the code. Run Radon using `poetry`:

      ```bash
      poetry run radon cc ./src
      ```

  - **Maintainability Index**: Measures the maintainability of the code. Run Radon using `poetry`:

      ```bash
      poetry run radon mi ./src
      ```

  - **Halstead Metrics**: Measures the complexity of the code. Run Radon using `poetry`:

      ```bash
      poetry run radon hal ./src
      ```

  - **Raw Metrics**: Measures the raw metrics of the code. Run Radon using `poetry`:

      ```bash
      poetry run radon raw ./src
      ```

- `Xenon` is used for automated code complexity checks. Xenon uses Radon under the hood. Docs for Xenon can be found [here](https://xenon.readthedocs.io/en/latest/). Run Xenon using `poetry`:

    ```bash
    poetry run xenon --max-absolute B --max-modules B --max-average A ./src
    ```

  Meaning of the flags:
    - **Max Absolute**: Maximum absolute complexity.
    - **Max Modules**: Maximum complexity per module.
    - **Max Average**: Maximum average complexity.

---

## Testing

This code base uses different levels of testing to ensure code quality and functionality.

- **Unit testing**: Unit tests are used to test individual components of the code base. Unit tests are written using `pytest`. Docs for pytest can be found [here](https://docs.pytest.org/en/stable/).

Run the unit tests using `poetry`:
  ```bash
  poetry run pytest
  ```
- **Code coverage reports**: Code coverage reports are generated using `pytest-cov` (wrapper for Coverage). Docs for pytest-cov can be found [here](https://pytest-cov.readthedocs.io/en/latest/).

Run the code coverage reports using `poetry`:
  ```bash
  poetry run pytest --cov=src --cov-report=term --cov-report=html --cov-report=xml --cov-fail-under=80
  ```

- **Property-based testing**: Property-based testing is used to test the code base against a wide range of scenarios. Property-based tests are written using `hypothesis`. Docs for Hypothesis can be found [here](https://hypothesis.readthedocs.io/en/latest/). These tests are automatically executed together with pytest unit tests.

Hypothesis test statistics can be shown using the following command:
  ```bash
  poetry run pytest --hypothesis-show-statistics
  ```

---

## Projects Using This Framework

| Type of Project | Title | Author(s) | Link                                           |
|-----------------|--------|-----------|------------------------------------------------|
| MSc Thesis | *VAE-based Multivariate Normative Modeling: An Investigation of Covariate Modeling Methods* | Remy Duijsens | [View Project](docs/projects/Thesis_Remy_Duijsens.pdf) |


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
