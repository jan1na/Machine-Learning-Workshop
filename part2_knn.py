from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Class names from CIFAR-10 dataset
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Flatten the images for KNN (since KNN doesn't accept 3D data like CIFAR-10)
X_train_flat = X_train.reshape(-1, 32 * 32 * 3)  # 32x32x3 for RGB
X_test_flat = X_test.reshape(-1, 32 * 32 * 3)

# TODO: Change the number of neighbors to experiment with different values
n_neighbors = 1  # Default value, change this to experiment with KNN

# Train KNN on CIFAR-10
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_flat, y_train.flatten())

# KNN prediction and accuracy
y_pred_knn = knn.predict(X_test_flat)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy on CIFAR-10 with {n_neighbors} neighbors: {acc_knn:.2f}")

# Compute confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Display confusion matrix with class names
disp = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for KNN (n_neighbors={n_neighbors})')
plt.show()

