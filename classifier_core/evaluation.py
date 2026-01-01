import os
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def save_plot(filename):
    """
    Saves current plot to filename.
    """
    # Ensure directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

def plot_loss_curves(history, save_dir=None):
    """
    Returns separate loss curves for training and validation metrics.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.figure()
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if save_dir:
        save_plot(os.path.join(save_dir, "loss_curve.png"))
    else:
        plt.show()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    if save_dir:
        save_plot(os.path.join(save_dir, "accuracy_curve.png"))
    else:
        plt.show()

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(15, 15), text_size=10, save_dir=None):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.
    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0]

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes), # create enough axis slots for each class
           yticks=np.arange(n_classes), 
           xticklabels=labels, # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
    
    if save_dir:
        save_plot(os.path.join(save_dir, "confusion_matrix.png"))
    else:
        plt.show()

def evaluate_model(model, test_dataset, class_names, save_dir=None, plot_conf_mat=False):
    """
    Evaluates the model on test dataset and prints classification report.
    """
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    print("Generating predictions...")
    model_preds_probs = model.predict(test_dataset)
    model_preds = tf.argmax(model_preds_probs, axis=1)
    
    # Extract labels from dataset
    y_true = []
    for text, label in test_dataset.unbatch():
        y_true.append(tf.argmax(label, axis=0)) # Convert one-hot to index
    y_true = np.array(y_true)

    print(classification_report(y_true, model_preds, target_names=class_names))
    
    if save_dir or plot_conf_mat:
         make_confusion_matrix(y_true, model_preds, classes=class_names, save_dir=save_dir)
         
    return test_loss, test_accuracy
