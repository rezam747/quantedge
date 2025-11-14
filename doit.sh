#!/bin/bash

# Exit on error
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Virtual environment setup
VENV_DIR="$SCRIPT_DIR/.venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    # Install development dependencies
    pip install doit pytest black isort flake8 mypy
else
    # Activate existing virtual environment
    source "$VENV_DIR/bin/activate"
fi

# Create .vscode directory if it doesn't exist
VSCODE_DIR="$SCRIPT_DIR/.vscode"
mkdir -p "$VSCODE_DIR"

# Create/update launch.json
cat > "$VSCODE_DIR/launch.json" << 'EOL'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${env:PYTHONPATH}"
            }
        },
        {
            "name": "Python: Tests",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/.venv/bin/pytest",
            "args": [
                "-v",
                "--cov=src",
                "--cov-report=term-missing",
                "-s"
            ],
            "console": "integratedTerminal",
            "justMyCode": true,
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src:${env:PYTHONPATH}"
            }
        }
    ]
}
EOL

# Create/update settings.json
cat > "$VSCODE_DIR/settings.json" << 'EOL'
{
    "python.pythonPath": ".venv/bin/python",
    "python.analysis.extraPaths": ["${workspaceFolder}/src"],
    "python.analysis.autoImportCompletions": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.banditEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.path": "isort",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "--rootdir=${workspaceFolder}",
        "--import-mode=importlib"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/*.pyo": true,
        "**/*.pyd": true,
        "**/.mypy_cache": true,
        "**/.pytest_cache": true
    }
}
EOL

# Create/update .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "# Project environment variables" > .env
    echo "PYTHONPATH=$SCRIPT_DIR/src" >> .env
fi

# If no arguments, show available tasks
if [ $# -eq 0 ]; then
    echo "No task specified. Available tasks:"
    echo "  unit_tests  - Run all unit tests"
    echo "  install     - Install project dependencies"
    echo "  format      - Format code using black and isort"
    echo "  lint        - Lint code using flake8"
    echo "  typecheck   - Run type checking using mypy"
    echo ""
    echo "Example: ./doit.sh unit_tests"
    exit 0
fi

# Run the specified task
echo "Running task: $1"
case "$1" in
    unit_tests)
        python -m pytest tests/
        ;;
    install)
        pip install -r requirements.txt
        ;;
    format)
        black src/ tests/
        isort src/ tests/
        ;;
    lint)
        flake8 src/ tests/
        ;;
    typecheck)
        mypy src/
        ;;
    *)
        echo "Error: Unknown task '$1'"
        exit 1
        ;;
esac
