from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import cifar10

from utils.eval import plot_confusion_matrix

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

plot_confusion_matrix(knn, X_test_flat, y_test, f'KNN Classifier (n_neighbors={n_neighbors})',
                      './plots/confusion_matrix_knn.png', knn=True)
