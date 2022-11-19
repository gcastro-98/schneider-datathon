"""
CNN implementation: pre-trained EfficientNet model fine-tuned with the dataset
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.data import Dataset
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
from typing import Tuple, List
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping  # not finally used... TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.preprocessing import OneHotEncoder
import tensorflow_addons as tfa
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os


LOCAL: bool = True  # True if the scripts are to be run locally
KAGGLE_PATH: str = 'data/' if LOCAL else '/kaggle/input/schneider-hackaton-2022/'
MODEL_PATH: str = 'models/' if LOCAL else '/kaggle/working/'
if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

IMAGE_SHAPE: tuple = (332, 332, 3)
NUM_CLASSES: int = 3


def deprecated(function: callable) -> callable:
    """
    Simple decorator to indicate function deprecation.
    """
    print(DeprecationWarning("This function is not needed"))
    return function


def load_images_as_normalized_dataset(images_path: pd.Series, scale: bool = False) -> Dataset:
    """
    Load, in memory as tensorflow Dataset, all the images from the passed paths (as pandas.Series).

    Parameters
    ----------
    images_path: pd.Series
        series of the images' path
    scale: bool
        if True, then images are also scaled

    Returns
    -------
        tf.data.Dataset: a tensorflow dataset of the images

    """
    _from_path_to_tensor: callable = lambda x: tf.convert_to_tensor(Image.open(x))
    if scale:
        _from_path_to_tensor: callable = lambda x: tf.image.per_image_standardization(_from_path_to_tensor(x))
    _tensor_list: List[Tensor] = [_from_path_to_tensor(_path) for _path in images_path.values]

    return Dataset.from_tensors(_tensor_list)


def define_augmentation_layer() -> Sequential:
    """
    Implement a data augmentation model which can be used later on
    as model hidden layer. It contains several of the well-known
    data augmentation techniques.

    Returns
    -------
        Sequential: model which given a Tensor representing an image,
        apply a data augmentation technique (rotation, translation...)

    """

    img_augmentation = Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )

    return img_augmentation


@deprecated
def resize_images(dataset: Dataset, image_size: tuple) -> Dataset:
    """
    Given a (tensorflow) Dataset containing 3-axis Tensor representing
     images, resize each of them to the desired image size.

    Parameters
    ----------
    dataset: Dataset
        Dataset with the original images as Tensors

    image_size: tuple
        2-tuple (of integers) indicating the image size

    Returns
    -------
        Dataset: dataset with the resized images

    """
    assert len(image_size) == 2, "The image size is 2-tuple!"
    return dataset.map(lambda image: tf.image.resize(image, image_size))


@deprecated
def _sample_generator(features: tf.data.Dataset, label: pd.Series) -> Tuple[Tensor, int]:
    """
    Return a generator to ease the task to create a tensorflow Dataset from it,
    which should contain for each sample a tuple of (features, label))

    Parameters
    ----------
    features: Dataset
        tensorflow Dataset with the images as Tensor for each sample
    label: pd.Series
        column with the labels (0 to ``NUM_CLASSES`` - 1) for each sample

    Returns
    -------
        sample of (feature, label) [is a generator]

    """
    for i_, feature_ in enumerate(features):
        label = label.iloc[i_]  # integer
        label = tf.one_hot(label, NUM_CLASSES)
        yield feature_, label


def build_model(backbone: str, optimizer, dropout_rate: float = 0.,
                augmentation: bool = False, batch_normalization: bool = False) -> Model:
    """
    Compile and return the (pre-trained) model according to the selected backbone
    and the rest of hyperparameters

    Parameters
    ----------
    backbone: str
        It can be 'B2', 'B3', 'B4'
    optimizer
        The optimizer for the gradient descent
    dropout_rate: float
        Rate of cells dropout
    augmentation: bool
        if True, data augmentation will be used to train the model
    batch_normalization: bool
        if True, the images per batch are normalized (their distribution)

    Returns
    -------
        Pre-trained model compiled (last layers to be trained)

    """
    # transfer learning
    inputs = layers.Input(shape=IMAGE_SHAPE)
    if augmentation:
        x = define_augmentation_layer()(inputs)
    else:
        x = inputs

    if backbone == 'B2':
        model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
    elif backbone == 'B3':
        model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
    elif backbone == 'B4':
        model = EfficientNetB3(include_top=False, input_tensor=x, weights="imagenet")
    else:
        raise ValueError(f"Backbone {backbone} not implemented!")

    # freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)

    if batch_normalization:
        x = layers.BatchNormalization()(x)

    x = layers.Dropout(dropout_rate, name="top_dropout")(x)
    fcc_layer = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")
    outputs = fcc_layer(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(n_epochs: int, backbone: str, optimizer, features_arr: np.ndarray,
                labels_arr: np.ndarray, **kwargs) -> Model:
    """
    Train the model and serialize it (encoding the obtained [macro] F1 score)

    Parameters
    ----------
    n_epochs: int
        Number of epochs
    backbone: str
        It can be 'B2', 'B3', 'B4'
    optimizer
        The optimizer to use for gradient descent
    features_arr: np.ndarray
        Features array, hence of shape (n_samples, 332, 332, 3)
    labels_arr: np.ndarray
        Labels array, hence of shape (n_samples, NUM_CLASSES)
    kwargs: dict
        Other additional keyword arguments

    Returns
    -------
        None, just the model is serialized
    """
    # best known hyperparameters by the authors
    batch_size: int = kwargs.get('batch_size', 8)
    test_size: float = kwargs.get('test_size', .2)
    dropout_rate: float = kwargs.get('dropout_rate', 0.)
    # unlike the transformer, this CNN was not greatly improved with augmentation
    augmentation: bool = kwargs.get('augmentation', False)
    batch_normalization: bool = kwargs.get('batch_normalization', False)

    x_train, x_test, y_train, y_test = train_test_split(features_arr, labels_arr, test_size=test_size, shuffle=True)
    model = build_model(backbone, optimizer, dropout_rate, augmentation, batch_normalization)
    _ = model.fit(epochs=n_epochs, x=x_train, y=y_train, validation_data=(x_test, y_test),
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                  batch_size=batch_size)

    f1_value: float = f1_score(np.argmax(y_test), np.argmax(model.predict(x_test)), average='macro')
    print(f"F1 Score", f1_value)

    model.save(f"{MODEL_PATH}/cnn_{np.round(f1_value, 3)}")
    return model


def run_train_and_save(backbone: str = 'B3') -> Model:
    """
    Drive the preprocessing of the training set (and its splits for testing purposes),
    the transfer learning of an EfficientNet and the serialization of the model
    (with its F1 score encoded).

    Parameters
    ----------
        backbone: str
            It can be B2, B3, B4; different backbone versions of the EfficientNet.
            More details at: https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet

    Returns
    -------
        None: nothing but it saves a serialization of the trained model

    """

    train_df: pd.DataFrame = pd.read_csv(f"{KAGGLE_PATH}/train.csv")
    train_df = train_df[['example_path', 'label']]
    train_df['example_path'] = KAGGLE_PATH + train_df['example_path']
    labels_train: np.ndarray = train_df['label'].values
    train_dataset: Dataset = load_images_as_normalized_dataset(train_df['example_path'])

    # however the Efficientnet is able to handle it
    ds_train: Dataset = train_dataset

    # cosine schedule for the learning rate combined with Adam (or AdamW if weight_decay > 0)
    scheduler = CosineDecay(initial_learning_rate=1e-3, decay_steps=1000)
    optimizer = tfa.optimizers.AdamW(learning_rate=scheduler, weight_decay=0.0)

    features_arr = np.array([d.numpy() for d in ds_train])[0, :, :, :]
    labels_arr: np.ndarray = OneHotEncoder().fit_transform(labels_train.reshape(-1, 1)).toarray()

    # we could perform a K-fold cross validation
    # from sklearn.model_selection import KFold
    # _i: int = 0  # validation index
    # for train_index, test_index in KFold(n_splits=NUM_FOLDS).split(features_arr):
    #     _i += 1
    #     x_train, x_test = features_arr[train_index], features_arr[test_index]
    #     y_train, y_test = labels_arr[train_index], labels_arr[test_index]

    # we could just do a 1-fold split
    return train_model(50, backbone, optimizer, features_arr, labels_arr)


def use_model_for_inference(model, dataframe: pd.DataFrame = None) -> np.ndarray:
    """
    Given the trained model, retrieve the array of class' probabilities

    Parameters
    ----------
    model
        Trained model
    dataframe
        Dataframe to be use for the inference

    Returns
    -------
        np.ndarray: array of predictions, hence of shape (n_samples, 3)

    """
    if dataframe is None:
        test_df: pd.DataFrame = (KAGGLE_PATH + pd.read_csv(f"{KAGGLE_PATH}/test.csv")['example_path']).to_frame()
        test_dataset: Dataset = load_images_as_normalized_dataset(test_df['example_path'])
    else:
        test_dataset: Dataset = load_images_as_normalized_dataset(dataframe['example_path'])
    features_arr = np.array([d.numpy() for d in test_dataset])[0, :, :, :]
    return model.predict(features_arr)


if __name__ == '__main__':
    SERIALIZED_MODELS_PATH: str = 'models'
    _train_df = pd.read_csv(KAGGLE_PATH + 'train.csv')
    _model = load_model(f'{SERIALIZED_MODELS_PATH}/cnn_0.714')
    _train_df['example_path'] = KAGGLE_PATH + _train_df['example_path']
    _y_hat = use_model_for_inference(_model, dataframe=_train_df)
    print("F1 score", f1_score(_train_df['label'].values, np.argmax(_y_hat, axis=1), average='macro'))
