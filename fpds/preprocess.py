import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from PIL import Image

KAGGLE_PATH: str = '/kaggle/input/schneider-hackaton-2022/'
IMAGE_SHAPE: tuple = (332, 332, 3)

# preprocessing data
train_df: pd.DataFrame = pd.read_csv(f"{KAGGLE_PATH}/train.csv")
train_df = train_df[['example_path', 'label']]
train_df['example_path'] = KAGGLE_PATH + train_df['example_path']
test_df: pd.DataFrame = (KAGGLE_PATH + pd.read_csv(f"{KAGGLE_PATH}/test.csv")['example_path']).to_frame()


def load_images_as_normalized_dataset(images_path: pd.Series) -> tf.data.Dataset:
    """
    Load, in memory as tensorflow Dataset, all the images from the passed paths (as pandas.Series).

    Parameters
    ----------
    images_path: pd.Series
        series of the images' path

    Returns
    -------
        tf.data.Dataset: a tensorflow dataset of the images

    """
    _from_path_to_tensor: callable = lambda x: tf.image.per_image_standardization(
        tf.convert_to_tensor(Image.open(x)))
    _tensor_list: List[Tensor] = [_from_path_to_tensor(_path) for _path in images_path.values]

    return tf.data.Dataset.from_tensors(_tensor_list)


train_dataset: tf.data.Dataset = load_images_as_normalized_dataset(train_df['example_path'])
test_dataset: tf.data.Dataset = load_images_as_normalized_dataset(test_df['example_path'])