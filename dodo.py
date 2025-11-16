from pathlib import Path
import subprocess
import sys
from doit.tools import CmdAction

def task_unit_tests():
    """Run all unit tests using pytest"""
    return {
        'actions': [
            'python -m pytest src/',
        ],
        'verbosity': 2,
        'task_dep': ['install']
    }

def task_install():
    """Install project dependencies"""
    return {
        'actions': ['pip install -r requirements.txt'],
        'verbosity': 2
    }

def task_format():
    """Format code using black and isort"""
    return {
        'actions': [
            'black src/ tests/',
            'isort src/ tests/'
        ],
        'verbosity': 2,
        'task_dep': ['install']
    }

def task_lint():
    """Lint code using flake8"""
    return {
        'actions': ['flake8 src/ tests/'],
        'verbosity': 2,
        'task_dep': ['install']
    }

def task_typecheck():
    """Run type checking using mypy"""
    return {
        'actions': ['mypy src/'],
        'verbosity': 2,
        'task_dep': ['install']
    }
