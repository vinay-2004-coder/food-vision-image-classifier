"""
Utility functions used throughout the Food Vision project.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import datetime
import os
import zipfile

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# -------------------------
# Image loading & preprocessing
# -------------------------
def load_and_prep_image(filename, img_shape=224, scale=True):
    """Loads and preprocesses an image for model prediction."""
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    return img / 255. if scale else img


# -------------------------
# Confusion matrix plotting
# -------------------------
def make_confusion_matrix(
    y_true,
    y_pred,
    classes=None,
    figsize=(10, 10),
    text_size=15,
    norm=False,
    savefig=False
):
    """Plots a labeled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes is not None else np.arange(n_classes)

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)" if norm else f"{cm[i, j]}"
        plt.text(
            j, i, value,
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size
        )

    if savefig:
        fig.savefig("confusion_matrix.png")

    plt.show()


# -------------------------
# Prediction visualization
# -------------------------
def pred_and_plot(model, filename, class_names):
    """Runs prediction on an image and plots the result."""
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


# -------------------------
# TensorBoard callback
# -------------------------
def create_tensorboard_callback(dir_name, experiment_name):
    """Creates a TensorBoard callback for experiment tracking."""
    log_dir = f"{dir_name}/{experiment_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# -------------------------
# Training history plots
# -------------------------
def plot_loss_curves(history):
    """Plots training and validation loss/accuracy curves."""
    epochs = range(len(history.history["loss"]))

    plt.plot(epochs, history.history["loss"], label="training_loss")
    plt.plot(epochs, history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    plt.plot(epochs, history.history["accuracy"], label="training_accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def compare_historys(original_history, new_history, initial_epochs):
    """Compares training metrics before and after fine-tuning."""
    acc = original_history.history["accuracy"] + new_history.history["accuracy"]
    loss = original_history.history["loss"] + new_history.history["loss"]

    val_acc = original_history.history["val_accuracy"] + new_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"] + new_history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(initial_epochs - 1, linestyle="--", label="Start Fine-tuning")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(initial_epochs - 1, linestyle="--", label="Start Fine-tuning")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.show()


# -------------------------
# Dataset utilities
# -------------------------
def unzip_data(filename):
    """Unzips a compressed dataset file."""
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


def walk_through_dir(dir_path):
    """Prints the number of files in each directory."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


# -------------------------
# Evaluation metrics
# -------------------------
def calculate_results(y_true, y_pred):
    """Computes accuracy, precision, recall and F1-score."""
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
