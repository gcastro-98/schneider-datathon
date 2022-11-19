import pandas as pd
import numpy as np
from fpds.blending import weighted_blending


model_1_df: pd.DataFrame = pd.read_csv('predictions/vit_0.764.csv', index_col=0)
model_2_df: pd.DataFrame = pd.read_csv('predictions/vit_0.705.csv', index_col=0)
_weights = [0.764, 0.705]

# def infer_cnn_

# def

def generate_predictions(models_serialized: bool = True) -> np.ndarray:
    if not models_serialized:
        pass
        # TODO: run 3 times the cnn
        # TODO: run 2 times the transformer

    # TODO: look for all the folders and files in models (if it is and h5 starting by 'vit


blended_preds: np.ndarray = weighted_blending([df.iloc[:, :3].values for df in (model_1_df, model_2_df)],
                                              weights_list=_weights)
print(blended_preds)


def prepare_submission(predictions: np.ndarray) -> None:
    pass
    # prepare_submission(blended_preds)

# def baseline_blending() -> np.ndarray:
#     blending_average = (model_a_preds + model_b_preds)/2
#     blending_weighted_average = (weights[0] * model_a_preds + weights[1] * model_b_preds) / 2


def _format_submission(predictions: np.ndarray) -> None:  # .json
    return pd.DataFrame(predictions, columns=['target']).to_json('predictions/predictions.json')
