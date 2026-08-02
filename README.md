# Badminton Shot Predictor

This project trains a PyTorch-based neural network to predict badminton shot types from sequences of prior rally events. The model uses rally-level features such as shot type, player, and landing coordinates to learn patterns from historical badminton data.

## What the project does

- Loads training and validation CSV files from the project root
- Builds a sequence dataset where each rally is treated as a time series
- Trains a transformer-style model in [main.py](main.py)
- Prints predicted versus actual shot types for a validation batch at the end of training

## Project structure

- [main.py](main.py) — training pipeline, model definition, and evaluation loop
- [data_cleaning/data.py](data_cleaning/data.py) — dataset class used by the training script
- [data_cleaning/data_filter.py](data_cleaning/data_filter.py) — optional helper to create a simplified training CSV from the raw data
- CSV files in the project root — training and validation datasets used by the model
- [requirements.txt](requirements.txt) — Python dependencies

## Setup

1. Create and activate a Python environment.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script from the project root:
   ```bash
   python main.py
   ```

## Notes

- The script will use GPU acceleration if available; otherwise it will fall back to Apple Silicon MPS or CPU.
- The training data files are expected to be present in the project root.
- The files in the data cleaning folder are optional helpers and are not required to run the main training workflow.
