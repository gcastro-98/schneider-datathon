import pandas as pd
import numpy as np
from glob import glob
import os
# from fpds import cnn, transformer
from sklearn.metrics import f1_score
from fpds.blending import average_blending


LOCAL: bool = True
SERIALIZED_MODELS_PATH: str = 'data/' if LOCAL else '/kaggle/input/serialized-models/'
OUTPUT_PATH: str = 'predictions' if LOCAL else '/kaggle/working'


def infer_transformers_probabilities(_train_df, _test_df, in_train: bool = False) -> None:
    # we generate the inference .csv
    for _model_path in glob(f'{SERIALIZED_MODELS_PATH}/*.hdf5'):
        _model = transformer.build_model()
        _model.load_weights(_model_path)
        _model_name: str = os.path.basename(_model_path)
        # in-sample f1 predictions
        _y_hat = transformer.use_model_for_inference(_model, dataframe=_train_df)
        train_f1_score: float = f1_score(_train_df['label'].values, np.argmax(_y_hat, axis=1), average='macro')
        val_f1_score: float = float(_model_name.lstrip('vit_').rstrip('hdf5').rstrip('.'))

        # finally we create the dataframe
        if in_train:
            _y_hat = transformer.use_model_for_inference(_model, dataframe=_test_df)
        pred_df = pd.DataFrame(_y_hat, columns=['class_0', 'class_1', 'class_2'])
        pred_df = pd.concat(
            [pred_df, pd.DataFrame([train_f1_score for _ in range(len(pred_df))], columns=['train_f1'])],
            axis=1)
        pred_df = pd.concat(
            [pred_df, pd.DataFrame([val_f1_score for _ in range(len(pred_df))], columns=['val_f1'])], axis=1)

        if in_train:
            predictions_csv_name: str = f'{OUTPUT_PATH}/train_{_model_name.rstrip("hdf5")}csv'
        else:
            predictions_csv_name: str = f'{OUTPUT_PATH}/{_model_name.rstrip("hdf5")}csv'

        pred_df.to_csv(predictions_csv_name, index=None)


def infer_cnn_probabilities(_train_df, _test_df, in_train: bool = False) -> None:
    from tensorflow.keras.models import load_model
    # we generate the inference .csv
    for _model_path in glob(f'{SERIALIZED_MODELS_PATH}/cnn*'):
        _model = load_model(_model_path)
        _model_name: str = os.path.basename(_model_path)
        # in-sample f1 predictions
        _y_hat = cnn.use_model_for_inference(_model, dataframe=_train_df)
        train_f1_score: float = f1_score(_train_df['label'].values, np.argmax(_y_hat, axis=1), average='macro')
        val_f1_score: float = float(_model_name.lstrip('cnn_'))

        # finally we create the dataframe
        if not in_train:
            # then the dataframe the inference for the test set (for the submission)
            _y_hat = cnn.use_model_for_inference(_model, dataframe=_train_df)

        pred_df = pd.DataFrame(_y_hat, columns=['class_0', 'class_1', 'class_2'])
        pred_df = pd.concat(
            [pred_df, pd.DataFrame([train_f1_score for _ in range(len(pred_df))], columns=['train_f1'])], axis=1)
        pred_df: pd.DataFrame = pd.concat([pred_df, pd.DataFrame(
            [val_f1_score for _ in range(len(pred_df))], columns=['val_f1'])], axis=1)
        if in_train:
            predictions_csv_name: str = f'{OUTPUT_PATH}/train_{_model_name}.csv'
        else:
            predictions_csv_name: str = f'{OUTPUT_PATH}/{_model_name}.csv'
        pred_df.to_csv(predictions_csv_name, index=None)


def prepare_submission() -> None:
    preds_list = []
    for inference_ in glob(f'{OUTPUT_PATH}/*vit*.csv'):
        inference_df = pd.read_csv(inference_)[['class_0', 'class_1', 'class_2']]
        arr_ = inference_df.values
        preds_list.append(arr_)
        print(arr_.shape)
    blended_preds: np.ndarray = average_blending(preds_list)
    _format_submission(blended_preds)


def _format_submission(predictions: np.ndarray) -> None:  # .json
    pd.DataFrame(predictions, columns=['target']).to_json('predictions.json')


prepare_submission()
