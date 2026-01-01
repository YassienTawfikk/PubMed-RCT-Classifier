import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score


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
    Plots and saves training curves for loss and accuracy.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss and accuracy
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax[0].plot(epochs, loss, label='training_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(epochs, accuracy, label='training_accuracy')
    ax[1].plot(epochs, val_accuracy, label='val_accuracy')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    
    if save_dir:
        save_plot(os.path.join(save_dir, "training_curves.png"))
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

def plot_class_distribution(y_true, class_names, save_dir=None):
    """
    Plots the distribution of true classes in the test set.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=[class_names[i] for i in y_true], order=class_names)
    plt.title("Class Distribution in Test Set")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    
    if save_dir:
        save_plot(os.path.join(save_dir, "class_distribution.png"))
    else:
        plt.show()

def plot_per_class_metrics(report_df, save_dir=None):
    """
    Plots per-class Precision, Recall, and F1-score from the classification report dataframe.
    """
    # Filter for classes only (exclude accuracy, macro avg, weighted avg)
    metrics_df = report_df.iloc[:-3, :].copy() 
    
    metrics_df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(12, 6))
    plt.title("Per-Class Classification Metrics")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    
    if save_dir:
        save_plot(os.path.join(save_dir, "per_class_metrics.png"))
    else:
        plt.show()

def plot_pr_curves(y_true, y_pred_probs, class_names, save_dir=None):
    """
    Plots Precision-Recall curves for each class.
    """
    n_classes = len(class_names)
    
    # One-hot encode y_true for per-class binary evaluation
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
        ap = average_precision_score(y_true_one_hot[:, i], y_pred_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP={ap:.2f})')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best")
    plt.grid(alpha=0.2)
    
    if save_dir:
        save_plot(os.path.join(save_dir, "pr_curves.png"))
    else:
        plt.show()

def evaluate_model(model, test_dataset, class_names, save_dir=None, plot_conf_mat=False):
    """
    Evaluates the model on test dataset and prints classification report.
    Generates comprehensive plots and metrics csv if save_dir is provided.
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

    # Classification Report
    report_dict = classification_report(y_true, model_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df)
    
    if save_dir:
        # Save CSV
        csv_path = os.path.join(save_dir, "classification_metrics_report.csv")
        report_df.to_csv(csv_path)
        print(f"Classification report saved to {csv_path}")
        
        # Plots
        make_confusion_matrix(y_true, model_preds, classes=class_names, save_dir=save_dir)
        plot_class_distribution(y_true, class_names, save_dir=save_dir)
        plot_per_class_metrics(report_df, save_dir=save_dir)
        plot_pr_curves(y_true, model_preds_probs, class_names, save_dir=save_dir)
    elif plot_conf_mat:
         make_confusion_matrix(y_true, model_preds, classes=class_names)
         
    return test_loss, test_accuracy

