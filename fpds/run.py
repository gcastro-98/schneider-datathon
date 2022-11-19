"""
Main module which drives the execution of the whole program
"""

import numpy as np
import pandas as pd
import os
import time
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow import keras


def main(sentence: str) -> None:
    """
    Print the passed sentence as a test function to preview the generated
    documentation

    Parameters
    ----------
    sentence: str
        Sentence which will be printed

    Returns
    -------
    None: nothing, just a print is done

    """
    print(sentence)
