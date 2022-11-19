"""
Implement the different (naive) blending possibilities
"""

from typing import List
import numpy as np


def average_blending(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Blend each of the models' predictions with the average

    Parameters
    ----------
    predictions_list: List[np.ndarray]
        List of the arrays corresponding to the classes probabilities inferences (for each model)

    Returns
    -------
        Predicted class according to the average of the models' probabilities for each sample
    """
    predictions_arr: np.ndarray = np.stack(predictions_list, axis=0)
    blended_probs = np.mean(predictions_arr, axis=0)
    return np.argmax(blended_probs, axis=1)


def weighted_blending(predictions_list: List[np.ndarray], weights_list: List[float]) -> np.ndarray:
    weights = np.array(weights_list)
    weights = weights / weights.sum(axis=0)
    predictions_arr: np.ndarray = np.stack(predictions_list, axis=0)
    blended_probs = np.average(predictions_arr, axis=0, weights=weights)
    return np.argmax(blended_probs, axis=1)
