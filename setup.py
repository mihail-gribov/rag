#!/usr/bin/env python3
"""
arXiv RAG System Setup Script
Initializes the project structure and dependencies using uv
"""

import os
import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories for the project"""
    directories = [
        "papers",      # Downloaded PDF files
        "chroma_db",   # Vector database storage
        "output",      # Generated markdown files
        "log",         # Application logs
        "metadata",    # Document metadata
        "tests"        # Test files
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies using uv"""
    try:
        # Check if uv is available
        subprocess.check_call(["uv", "--version"], stdout=subprocess.DEVNULL)
        print("Using uv for dependency management")
        
        # Install dependencies using uv
        subprocess.check_call(["uv", "sync"])
        print("Dependencies installed successfully with uv")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv not found, falling back to pip")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e", "."
            ])
            print("Dependencies installed successfully with pip")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            return False
    return True


def create_env_file():
    """Create .env file from template if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy template to .env
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("Created .env file from template")
        print("Please edit .env file and add your OpenAI API key")
    elif env_file.exists():
        print(".env file already exists")
    else:
        print("Warning: env.example template not found")


def setup_gitignore():
    """Create .gitignore file for Python project"""
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# uv
.venv/

# Project specific
papers/*.pdf
chroma_db/
output/*.md
log/*.log
metadata/*.json
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("Created .gitignore file")


def main():
    """Main setup function"""
    print("Setting up arXiv RAG System with uv...")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    if not install_dependencies():
        print("Setup failed during dependency installation")
        return False
    
    # Create .env file
    print("\n3. Setting up environment configuration...")
    create_env_file()
    
    # Create .gitignore
    print("\n4. Creating .gitignore...")
    setup_gitignore()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Activate uv environment: uv shell")
    print("3. Test the setup: python -c 'import arxiv, openai, chromadb'")
    print("4. Run: python cli.py --help")
    print("\nuv commands:")
    print("- uv sync: install dependencies")
    print("- uv add package-name: add new dependency")
    print("- uv shell: activate virtual environment")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else True)
