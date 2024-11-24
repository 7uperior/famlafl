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

### 1. 🌿 Create a New Branch

TL;DR;
like `git checkout -b fdww1732453744`

To maintain a clean Git history, avoid pushing directly to the `main` branch. Instead, create a new branch using the naming convention:

```
tgnick+timestamp
```

- **`tgnick`**: Your Telegram username.
- **`timestamp`**: Current Unix epoch time (can be obtained from [Epoch Converter](https://www.epochconverter.com)).

#### Example:
If your Telegram username is `fdww` and the current epoch time is `1732447944`, your branch name should be:
```
fdww1732447944
```

### 2. ✅ Run Pre-commit Hooks Before Committing

Ensure your code passes pre-commit checks:

```bash
poetry run pre-commit run --all-files
```

### 3. 🚀 Save and Push Changes

1. Stage your changes:
   ```bash
   git add .
   ```

2. Commit your changes:
   ```bash
   git commit -m "commit message"
   ```
   If you encounter issues with pre-commit hooks, you can bypass them temporarily:
   ```bash
   git commit -m "commit message" --no-verify

   ```

3. Push your changes to your branch:
   ```bash
   git push origin {YOUR_BRANCH_NAME_IN_FORMAT:tgnicktimestamp}
   ```

4. Push updated data to DVC storage:
   ```bash
   dvc push
   ```

### 4. 🗑️ Deleting or Changing Files Tracked by DVC

If you delete a file (e.g., a zip archive) that is tracked by both Git and DVC:

1. **Delete the File Locally:**
   ```bash
   rm path/to/your/file
   ```

2. **Update DVC to Reflect Changes:**
   Re-add the directory to update the `.dvc` files:
   ```bash
   dvc add cmf/data
   dvc add jane_street/data
   ```

3. **Stage the Changes in Git:**
   ```bash
   git add cmf/data.dvc jane_street/data.dvc
   ```

4. **Commit the Changes:**
   ```bash
   git commit -m "Remove file from project"
   ```

5. **Push Changes to Remote Repositories:**
   ```bash
   git push origin your-branch-name
   dvc push
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