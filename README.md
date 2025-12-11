# ASL-Interpreter-CNN
## Antonia Junod 
## Brendon Peters

## Setup
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt

## Training
python -m scripts.train_baseline
python -m scripts.train_transfer

## Evaluation
python -m scripts.evaluate_model

