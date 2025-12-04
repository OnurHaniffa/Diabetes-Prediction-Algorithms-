#!/usr/bin/env bash
# setup.sh – create & activate a virtual env, install common libs

set -e  # stop if any command fails

VENV_DIR=".venv"
PYTHON_BIN="python3"

echo ">>> Creating virtual environment..."
$PYTHON_BIN -m venv "$VENV_DIR"

echo ">>> Activating venv..."
source "$VENV_DIR/bin/activate"

echo ">>> Upgrading pip..."
python -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo ">>> requirements.txt found – installing from it..."
    pip install -r requirements.txt
else
    echo ">>> No requirements.txt found – installing common ML/data libs..."
    pip install numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel python-dotenv requests black isort pytest
fi

echo ">>> DONE. To activate later, run:"
echo "source .venv/bin/activate"
