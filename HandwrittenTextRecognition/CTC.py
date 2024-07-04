import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class CTCLayer(layers.Layer):
    """
    Custom Keras layer for Connectionist Temporal Classification (CTC) loss.

    This layer calculates the CTC loss and adds it to the layer's losses.

    Parameters
    ----------
    name : str, optional
        The name of the layer. If not provided, a default name will be assigned.

    Attributes
    ----------
    loss_fn : function
        The CTC loss function from Keras backend.

    Methods
    -------
    call(y_true, y_pred)
        Computes the CTC loss and adds it to the layer's losses.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from tensorflow.keras import layers
    >>> from tensorflow.keras.models import Model
    >>> 
    >>> class CTCLayer(layers.Layer):
    >>>     def __init__(self, name=None):
    >>>         super().__init__(name=name)
    >>>         self.loss_fn = keras.backend.ctc_batch_cost
    >>> 
    >>>     def call(self, y_true, y_pred):
    >>>         batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
    >>>         input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
    >>>         label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')
    >>> 
    >>>         input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
    >>>         label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')
    >>>         loss = self.loss_fn(y_true, y_pred, input_length, label_length)
    >>>         self.add_loss(loss)
    >>>         return y_pred
    >>>
    >>> # Example usage
    >>> ctc_layer = CTCLayer(name="ctc_loss")
    >>> y_true = tf.random.uniform((32, 10), maxval=10, dtype=tf.int32)
    >>> y_pred = tf.random.uniform((32, 20, 10), maxval=10, dtype=tf.float32)
    >>> output = ctc_layer(y_true, y_pred)
    """

    def __init__(self, name=None):
        """
        Initializes the CTCLayer.

        Parameters
        ----------
        name : str, optional
            The name of the layer. If not provided, a default name will be assigned.
        """
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        """
        Computes the CTC loss and adds it to the layer's losses.

        Parameters
        ----------
        y_true : tensor
            The ground truth labels for the input data.
        y_pred : tensor
            The predicted labels from the model.

        Returns
        -------
        tensor
            The predictions (y_pred) are returned unmodified.
        """
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
