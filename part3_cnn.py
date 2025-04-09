from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Class names from CIFAR-10 dataset
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the images to [0, 1] range
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# TODO: Experiment with different learning rates
learning_rate = 0.001  # Default learning rate (feel free to change this)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes
])

# Compile the model with Adam optimizer and the set learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on CIFAR-10
model.fit(X_train, y_train_cat, epochs=10, batch_size=64, validation_data=(X_test, y_test_cat), verbose=2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Neural Network Accuracy on CIFAR-10 with learning rate {learning_rate}: {test_acc:.2f}")

# Predict on test set
y_pred_cnn = model.predict(X_test)
y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)  # Convert predictions to class labels

# Compute confusion matrix
cm_cnn = confusion_matrix(y_test, y_pred_cnn_classes)

# Display confusion matrix with class names
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for CNN with learning rate {learning_rate}')
plt.show()


