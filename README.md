# 📊 Data Hustler

Welcome to **Data Hustler**! This project is dedicated to analyzing and forecasting time series data.

## 🚀 Getting Started

1. **Clone the Repository:** 🖥️

   ```bash
   git clone https://github.com/7uperior/data-hustler.git
   cd data-hustler
   ```

2. **Install Dependencies:** 📦

   Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed. Then, run:

   ```bash
   poetry install --with dev
   ```

3. **Set Up Pre-commit Hooks:** 🔍

   Install pre-commit hooks to maintain code quality:

   ```bash
   poetry run pre-commit install
   ```

4. **Pull Data with DVC:** 📥

   Retrieve the latest data:

   ```bash
   dvc pull
   ```

## 🛠️ Development Workflow

- **Before Committing Changes:** ✅

  Run pre-commit hooks on all files:

  ```bash
  poetry run pre-commit run --all-files
  ```

- **Save and Push Changes:** 🚀

  ```bash
  git add .
  git commit -m "Your descriptive commit message here"
  ```
  or
  ```bash
  git commit -m "Your descriptive commit message here" --no-verify
  ```
  *(Use `--no-verify` if pre-commit hooks are blocking your commit and you need to bypass them.)*

  ```bash
  git push origin main
  dvc push
  ```

## 🌿 Branch Naming Convention

To maintain a clean Git history, avoid pushing directly to the `main` branch. Instead:

- **Create a new branch** for your work using the naming convention:
  ```
  tgnick+timestamp
  ```
  - `tgnick`: Your Telegram username
  - `timestamp`: Current Unix epoch time (can be obtained from [Epoch Converter](https://www.epochconverter.com))

*Example:*  
If your username is `fdww` and the current epoch time is `1732447944`, your branch name should be:
```
fdww1732447944
```

## 📂 Project Structure

```plaintext
.
├── README.md                 # Project documentation
├── cmf                       # CMF-specific code and data
│   ├── data                  # Raw and processed data
│   ├── models                # Stored models
│   ├── notebooks             # Jupyter notebooks for analysis
│   └── scripts               # Python scripts for CMF
├── jane_street               # Jane Street-specific code and data
│   ├── data                  # Raw and processed data
│   ├── models                # Stored models
│   ├── notebooks             # Jupyter notebooks for analysis
│   └── scripts               # Python scripts for Jane Street
├── poetry.lock               # Poetry lockfile for dependency management
├── pyproject.toml            # Poetry configuration file
├── shared_scripts            # Shared scripts used across projects
```

- **`cmf/` and `jane_street/`**: Contain data, models, notebooks, and scripts specific to each project.
- **`shared_scripts/`**: Holds scripts shared across projects.
- **`poetry.lock` and `pyproject.toml`**: Manage project dependencies.

## ⚙️ Linter Management (Ruff)

If you make changes to the linter (`ruff`):

1. **Regenerate Dependencies:**

   ```bash
   rm -rf poetry.lock
   poetry lock
   poetry install --with dev
   ```

2. **Reinstall Pre-commit Hooks:**

   ```bash
   poetry run pre-commit install
   ```

3. **Run Pre-commit Hooks:**

   ```bash
   poetry run pre-commit run --all-files
   ```

4. **Check and Fix Pre-commit Errors:**

   To identify and fix errors, run:

   ```bash
   poetry run ruff check . --fix
   ```