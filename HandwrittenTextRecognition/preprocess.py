from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from HandwrittenTextRecognition.util import train_images_dir, test_images_dir, validation_images_dir
import cv2
import numpy as np
import os
import Util
from util import img_height, img_width

def cleanData(train_csv, validation_csv, test_csv):
    """
    Cleans the data by removing NaN values and unreadable entries, and converts text to lowercase.

    Parameters
    ----------
    train_csv : pandas.DataFrame
        DataFrame containing the training data.
    validation_csv : pandas.DataFrame
        DataFrame containing the validation data.
    test_csv : pandas.DataFrame
        DataFrame containing the test data.

    Returns
    -------
    train_csv : pandas.DataFrame
        Cleaned training data.
    validation_csv : pandas.DataFrame
        Cleaned validation data.
    test_csv : pandas.DataFrame
        Cleaned test data.

    """
    train_csv = train_csv.dropna()
    validation_csv = validation_csv.dropna()
    test_csv = test_csv.dropna()

    train_csv = train_csv[train_csv['IDENTITY'] != 'UNREADABLE']
    validation_csv = validation_csv[validation_csv['IDENTITY'] != 'UNREADABLE']
    test_csv = test_csv[test_csv['IDENTITY'] != 'UNREADABLE']

    train_csv['IDENTITY'] = train_csv['IDENTITY'].str.lower()
    validation_csv['IDENTITY'] = validation_csv['IDENTITY'].str.lower()
    test_csv['IDENTITY'] = test_csv['IDENTITY'].str.lower()

    return train_csv, validation_csv, test_csv

def tokenizing(train_csv):
    """
    Tokenizes the labels in the training data.

    Parameters
    ----------
    train_csv : pandas.DataFrame
        DataFrame containing the training data.

    Returns
    -------
    None

    """
    max_length = max([len(label) for label in train_csv['IDENTITY'].values])
    tokenizer = Tokenizer(num_words=max_length, char_level=True)
    tokenizer.fit_on_texts(train_csv['IDENTITY'].values)
    word_index = tokenizer.word_index

    Util.tokenizer, Util.max_length = tokenizer, max_length

    train_sequences = tokenizer.texts_to_sequences(train_csv['IDENTITY'].values)

def imagePreprocessing(train_csv, test_csv, validate_csv):
    """
    Preprocesses the images and labels for training, validation, and test datasets.

    Parameters
    ----------
    train_csv : pandas.DataFrame
        DataFrame containing the training data.
    test_csv : pandas.DataFrame
        DataFrame containing the test data.
    validate_csv : pandas.DataFrame
        DataFrame containing the validation data.

    Returns
    -------
    list
        A list containing the preprocessed training, validation, and test datasets along with their respective labels and lengths.

    """
    tokenizing(train_csv)

    images = train_csv['FILENAME'].values
    labels = train_csv['IDENTITY'].values

    def preprocess_single_sample(image_path, label, TEST=False):
        """
        Preprocesses a single image and label.

        Parameters
        ----------
        image_path : str
            The file path of the image.
        label : str
            The label associated with the image.
        TEST : bool, optional
            Whether the image is from the test dataset (default is False).

        Returns
        -------
        img : numpy.ndarray
            The preprocessed image.
        label : numpy.ndarray
            The preprocessed label.

        """
        if TEST:
            img = cv2.imread(os.path.join(validation_images_dir, image_path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(train_images_dir, image_path), cv2.IMREAD_GRAYSCALE)
            
        img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) / 255  # normalization step
        
        label_sequence = Util.tokenizer.texts_to_sequences([label])
        label = pad_sequences(label_sequence, maxlen=Util.max_length, padding='post')[0]
        
        return img, label

    train_length = 9000
    validation_length = 2000
    test_length = 2000

    indices = np.arange(len(train_csv))
    np.random.shuffle(indices)

    test_indices = np.arange(len(test_csv))
    np.random.shuffle(test_indices)

    x_train = []
    y_train = []
    train_label_len = []
    for i in range(train_length):
        image_name = train_csv.iloc[indices[i], 0]
        label = train_csv.iloc[indices[i], 1]
        train_label_len.append(len(label))
        
        img, label = preprocess_single_sample(image_name, label)
        img = np.expand_dims(img, axis=2)
        x_train.append(img)
        y_train.append(label)
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    train_label_len = np.array(train_label_len)

    x_val = []
    y_val = []
    valid_label_len = []
    for i in range(train_length, train_length + validation_length):
        image_name = train_csv.iloc[indices[i], 0]
        label = train_csv.iloc[indices[i], 1]
        valid_label_len.append(len(label))
        
        img, label = preprocess_single_sample(image_name, label)
        img = np.expand_dims(img, axis=2)
        x_val.append(img)
        y_val.append(label)

    x_val = np.array(x_val)
    y_val = np.array(y_val)
    valid_label_len = np.array(valid_label_len)

    x_test = []
    y_test = []
    for i in range(test_length):
        image_name = train_csv.iloc[test_indices[i], 0]
        label = train_csv.iloc[test_indices[i], 1]
        
        img, _ = preprocess_single_sample(image_name, label)
        img = np.expand_dims(img, axis=2)
        x_test.append(img)
        y_test.append(label)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train_input_len = np.ones([train_length, 1]) * 48
    valid_input_len = np.ones([validation_length, 1]) * 48
    valid_output = np.zeros([validation_length])

    return [x_train, y_train, train_label_len, train_input_len, x_test, y_test, x_val, y_val, valid_label_len, valid_input_len]
