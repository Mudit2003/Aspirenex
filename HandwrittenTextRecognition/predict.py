import numpy as np
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2 
import matplotlib.pyplot as plt
import Util 

max_length, tokenizer = Util.max_length, Util.tokenizer 

def predict(model):
    """
    Sets up a prediction model from the trained OCR model for making predictions.

    Parameters
    ----------
    model : keras.models.Model
        The trained OCR model.

    Returns
    -------
    None
    """
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="output").output
    )
    prediction_model.summary()

def decode_predictions(pred):
    """
    Decodes predictions from the OCR model.

    Parameters
    ----------
    pred : numpy.ndarray
        The predictions made by the OCR model.

    Returns
    -------
    list
        A list of decoded texts corresponding to the predictions.
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results:
        decoded = tokenizer.sequences_to_texts([res.numpy()])
        output_text.append(decoded)
    return output_text

def testPrediction(prediction_model, x_test, y_test):
    """
    Tests the OCR model predictions against the ground truth labels and computes evaluation metrics.

    Parameters
    ----------
    prediction_model : keras.models.Model
        The prediction model setup from the trained OCR model.
    x_test : numpy.ndarray
        The test images for prediction.
    y_test : list
        The ground truth labels for the test images.

    Returns
    -------
    None
    """
    preds = prediction_model.predict(x_test)
    pred_texts = decode_predictions(preds)

    correct = 0
    correct_char = 0
    total_char = 0
    test_length = len(pred_texts)

    for i in range(test_length):
        pr = pred_texts[i][0].replace(' ', '')  # Extract the predicted word from the list
        tr = y_test[i]
        total_char += len(tr)
        for j in range(min(len(tr), len(pr))):
            if tr[j] == pr[j]:
                correct_char += 1
        if pr == tr:
            correct += 1

    accuracy = (correct / test_length) * 100
    acc = (correct_char * 100 / total_char)

    print(f'Correct characters predicted : {acc:.2f}%')
    print(f'Correct words predicted: {accuracy:.2f}%')

    for i in range(test_length):
        pred_texts[i][0] = pred_texts[i][0].replace(' ', '')

    precision = precision_score(y_test, pred_texts, average='weighted')
    recall = recall_score(y_test, pred_texts, average='weighted')
    f1 = f1_score(y_test, pred_texts, average='weighted')

    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')

    num_images = 25
    num_rows = int(np.ceil(num_images / 4))
    num_cols = 4

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(15, 5))

    for i in range(num_images):
        img = x_test[i]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        title = f"Prediction: {pred_texts[i][0]}"
        ax[i // num_cols, i % num_cols].imshow(img, cmap="gray")
        ax[i // num_cols, i % num_cols].set_title(title)
        ax[i // num_cols, i % num_cols].axis("off")

    # Hide any empty subplots
    for i in range(num_images, num_rows * num_cols):
        ax[i // num_cols, i % num_cols].axis("off")

    plt.tight_layout()
    plt.show()

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(16):
        img = x_test[i]
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        title = f"Prediction: {pred_texts[i][0]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
    plt.show()
