import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
logger = logging.getLogger(__name__)

project_name = "winequality"
base_dir = Path.cwd()

list_of_files = [
    ".github/workflows/.gitkeep",
    base_dir / f"src/{project_name}/__init__.py",
    base_dir / f"src/{project_name}/components/__init__.py",
    base_dir / f"src/{project_name}/utils/__init__.py",
    base_dir / f"src/{project_name}/utils/common.py",
    base_dir / f"src/{project_name}/logging/__init__.py",
    base_dir / f"src/{project_name}/config/__init__.py",
    base_dir / f"src/{project_name}/config/configuration.py",
    base_dir / f"src/{project_name}/pipeline/__init__.py",
    base_dir / f"src/{project_name}/entity/__init__.py",
    base_dir / f"src/{project_name}/constants/__init__.py",
    base_dir / "config/config.yaml",
    base_dir / "params.yaml",
    base_dir / "schema.yaml",
    base_dir / "app.py",
    base_dir / "main.py",
    base_dir / "Dockerfile",
    base_dir / "requirements.txt",
    base_dir / "setup.py",
    base_dir / "research/trails.ipynb",
    base_dir / "templates/index.html",
    base_dir / "test.py"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent
    filename = filepath.name
    
    if filedir != "":
        filedir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating directory: {filedir} for the file {filename}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logger.info(f"Creating empty file: {filepath}")

    else:
        logger.info(f"{filename} already exists")
