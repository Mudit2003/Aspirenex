from keras import backend as K
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow import keras
from tensorflow.keras import layers
from util import img_height, img_width
import Util

def ctc_loss(args):
    """
    Computes the CTC (Connectionist Temporal Classification) loss.

    Parameters
    ----------
    args : tuple
        A tuple containing the following elements:
        - labels (tensor): The ground truth labels for the input data.
        - y_pred (tensor): The predicted labels from the model.
        - input_length (tensor): The lengths of the input sequences.
        - label_length (tensor): The lengths of the label sequences.

    Returns
    -------
    tensor
        The computed CTC loss.
    """
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def buildModel(characters):
    """
    Builds a model for Optical Character Recognition (OCR) using a pre-trained ResNet50 base and custom layers.

    Parameters
    ----------
    characters : list
        A list of unique characters in the dataset, used to define the output layer size.

    Returns
    -------
    keras.models.Model
        The compiled OCR model.
    """
    # Inputs to the model
    input_img = layers.Input(shape=(img_height, img_width, 1), name="image")
    labels = layers.Input(name="label", shape=(Util.max_length,))
    input_length = layers.Input(name='input_length', shape=(1,))
    label_length = layers.Input(name='label_length', shape=(1,))

    # Convert grayscale image to RGB format
    x = layers.Lambda(lambda x: keras.backend.repeat_elements(x, 3, axis=-1))(input_img)

    # Load the pre-trained ResNet50 model without the top classification layers
    vgg_base = ResNet50(weights='imagenet', include_top=False, input_tensor=x)

    # Freeze the ResNet base layers to prevent them from being updated during training
    for layer in vgg_base.layers:
        layer.trainable = False

    # Add custom layers on top of the ResNet base
    x = vgg_base.output

    # Reshape the output tensor to be 3D for the LSTM layers
    x = layers.Reshape((-1, x.shape[-1]))(x)

    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNN
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)

    # Output layer
    y_pred = layers.Dense(len(characters) + 1, activation="softmax", name="output")(x)

    loss_out = layers.Lambda(ctc_loss, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])

    # Define the model
    model = keras.models.Model(inputs=[input_img, labels, input_length, label_length], 
                               outputs=loss_out,
                               name="ocr_model_resnet50")

    # Compile the model
    opt = keras.optimizers.Adam()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    
    return model

# Example usage
# model = buildModel(characters)
# model.summary()
