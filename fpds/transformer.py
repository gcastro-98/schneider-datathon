"""
Transformer implementation: pre-trained ViT model fine-tuned with the dataset
"""

import pandas as pd
import glob
from warnings import filterwarnings
import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import vit

filterwarnings('ignore')
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES: int = 3

# Path to IMAGES(.png) files
LOCAL: bool = True
KAGGLE_PATH: str = 'data' if LOCAL else '/kaggle/input/schneider-hackaton-2022'
TRAIN_PATH = f'{KAGGLE_PATH}/train_test_data/train'
TEST_PATH = f'{KAGGLE_PATH}/train_test_data/test'


def data_augment(image: tf.image) -> tf.image:
    """
    Based on random probabilities, perform different kind of augmentations on an image.

    Parameters
    ----------
    image: tf.image
        Image corresponding to the set.

    Returns
    -------
    tf.image: An augmented image. 
        
            
    """
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if p_spatial > .75:
        image = tf.image.transpose(image)
        
    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # rotate 90º
        
    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)
        
    return image


def _create_data_generator(preprocessing_f: callable = data_augment):
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255, samplewise_center=True, samplewise_std_normalization=True,
        validation_split=0.2, preprocessing_function=preprocessing_f)


def create_generators(train_df, preprocessing_f: callable = data_augment):
    # Create the Data Generator pipeline. It uses the data augmentation as a preprocessing function.
    datagen = _create_data_generator(preprocessing_f)

    # The train generator. 80% of our training set.
    train_gen = datagen.flow_from_dataframe(dataframe=train_df,
                                            directory=TRAIN_PATH,
                                            x_col='example_path',
                                            y_col='label',
                                            subset='training',
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # The validation generator. 20% of training set.
    valid_gen = datagen.flow_from_dataframe(dataframe=train_df,
                                            directory=TRAIN_PATH,
                                            x_col='example_path',
                                            y_col='label',
                                            subset='validation',
                                            batch_size=BATCH_SIZE,
                                            seed=1,
                                            color_mode='rgb',
                                            shuffle=False,
                                            class_mode='categorical',
                                            target_size=(IMAGE_SIZE, IMAGE_SIZE))

    return train_gen, valid_gen


def build_model() -> tf.keras.Sequential:
    vit_model = vit.vit_b32(
            image_size=IMAGE_SIZE,
            activation='softmax',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            classes=3)
    
    # Finetune the model. Create a sequential model using the MODEL | 3 outputs.
    # noinspection PyDeprecation
    model = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(11, activation=tfa.activations.gelu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(NUM_CLASSES, 'softmax')
        ],
        name='vision_transformer')

    return model


def train_model(train_gen, valid_gen, lr: float = 1e-4, epochs: int = 10):
    # we compile the model
    model = build_model()

    learning_rate = lr

    # Creates the optimizer using rectified ADAM.
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=['accuracy'])

    # Dynamic Learning Rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.2, patience=2, verbose=1,
        min_delta=1e-4, min_lr=1e-6, mode='max')
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=1e-4, patience=5,
        mode='max', restore_best_weights=True, verbose=1)

    # checkpoint. Save the model.
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model2.hdf5', monitor='val_accuracy', verbose=1,
        save_best_only=True, save_weights_only=True, mode='max')

    # our callbacks
    callbacks = [early_stopping, reduce_lr, checkpointer]

    model.fit(
        x=train_gen,
        steps_per_epoch=train_gen.n // train_gen.batch_size,  # training step size
        validation_data=valid_gen,
        validation_steps=valid_gen.n // valid_gen.batch_size,  # validation step size
        epochs=epochs, callbacks=callbacks)

    return model


def run_train_and_save():
    df_train = pd.read_csv('../input/schneider-hackaton-2022/train.csv', dtype='str')
    # drop tabular information
    df_train.drop(inplace=True, columns=['latitude', 'longitude', 'year'])
    # Get the name of the images
    df_train.example_path = df_train.example_path.apply(lambda x: x.split('/')[-1])
    train_gen, valid_gen = create_generators(df_train, preprocessing_f=data_augment)
    return train_model(train_gen, valid_gen)


def use_model_for_inference(model, dataframe: pd.DataFrame = None):
    if dataframe is None:
        test_path = '../input/schneider-hackaton-2022/train_test_data/test'
        test_images = glob.glob(test_path + '/*.png')  # Get all the images names
        test_df = pd.DataFrame(test_images, columns=['example_path'])  # Create the test.
    else:
        test_df = dataframe

    datagen = _create_data_generator()
    # Test generator. No labels available.
    test_gen = datagen.flow_from_dataframe(dataframe=test_df,
                                           x_col='example_path',
                                           y_col=None,
                                           batch_size=BATCH_SIZE,
                                           seed=1,
                                           color_mode='rgb',
                                           shuffle=False,
                                           class_mode=None,
                                           target_size=(IMAGE_SIZE, IMAGE_SIZE))

    probabilities_hat = model.predict(test_gen, steps=test_gen.n // test_gen.batch_size + 1)
    # predicted_test_classes = np.argmax(probabilities_hat)
    return probabilities_hat


if __name__ == '__main__':
    _train_df = pd.read_csv('data/train.csv')
    _model = build_model()
    _model.load_weights('data/vit_0.764.hdf5')
    _y_hat = use_model_for_inference(_model, dataframe=_train_df)
    import numpy as np
    from sklearn.metrics import f1_score
    print(np.argmax(_y_hat))
    print("F1 score", f1_score(_train_df['target'].values, np.argmax(_y_hat)))
