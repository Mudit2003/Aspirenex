from util import train_length , validation_length
import numpy as np
from tensorflow import keras 


def trainModel(model, x_train, y_train, train_input_len, train_label_len, x_val, y_val, valid_input_len, valid_label_len):
    """
    Trains the OCR model using the provided training and validation data.

    Parameters
    ----------
    model : keras.models.Model
        The OCR model to be trained.
    x_train : numpy.ndarray
        The training images.
    y_train : numpy.ndarray
        The training labels.
    train_input_len : numpy.ndarray
        The lengths of the input sequences for training.
    train_label_len : numpy.ndarray
        The lengths of the label sequences for training.
    x_val : numpy.ndarray
        The validation images.
    y_val : numpy.ndarray
        The validation labels.
    valid_input_len : numpy.ndarray
        The lengths of the input sequences for validation.
    valid_label_len : numpy.ndarray
        The lengths of the label sequences for validation.

    Returns
    -------
    keras.models.Model
        The trained OCR model.

    Notes
    -----
    This function trains the OCR model using the specified training and validation data.
    It uses early stopping based on validation loss to prevent overfitting.
    """
    epochs = 50
    early_stopping_patience = 10

    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        x=(x_train, y_train, train_input_len, train_label_len),
        y=np.zeros([len(x_train)]),  # Placeholder y values, not used due to custom loss
        validation_data=([x_val, y_val, valid_input_len, valid_label_len], np.zeros([len(x_val)])),
        epochs=epochs,
        batch_size=128,
        callbacks=[early_stopping]
    )

    print(*history)

    return model
