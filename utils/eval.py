from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix,  ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def plot_confusion_matrix(model: Model, x_test: np.ndarray, y_test: np.ndarray, title: str, save_path: str, knn: bool = False):
    # Predict on test set
    y_pred_cnn = model.predict(x_test)
    if knn:
        y_pred_cnn_classes = y_pred_cnn
    else:
        y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)  # Convert predictions to class labels

    # Compute confusion matrix
    cm_cnn = confusion_matrix(y_test, y_pred_cnn_classes)
    acc_tf = accuracy_score(y_test, y_pred_cnn_classes)

    # Display confusion matrix with class names
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for ' + title)
    plt.text(0.5, 1.10, f'Accuracy: {acc_tf:.2f}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.savefig(save_path)
    plt.show()