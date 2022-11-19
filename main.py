import pandas as pd
import numpy as np


def generate_predictions(models_serialized: bool = True) -> np.ndarray:
    if not models_serialized:
        pass
        # TODO: run 3 times the cnn
        # TODO: run 2 times the transformer

    # TODO: look for all the folders and files in models (if it is and h5 starting by 'vit


def prepare_submission(predictions: np.ndarray) -> None:  # .json
    return pd.DataFrame(predictions, columns=['target']).to_json('predictions.json')
