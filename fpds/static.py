"""
Module to set some global variables for the project
"""

import os


LOCAL: bool = True  # True if the scripts are to be run locally
KAGGLE_PATH: str = 'data/' if LOCAL else '/kaggle/input/schneider-hackaton-2022/'
MODEL_PATH: str = 'models/' if LOCAL else '/kaggle/working/'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

IMAGE_SHAPE: tuple = (332, 332, 3)
NUM_CLASSES: int = 3
