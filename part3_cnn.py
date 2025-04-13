import visualkeras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from utils.eval import plot_confusion_matrix

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
# TODO: Experiment with different architectures
model = Sequential(
    [Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     MaxPooling2D(pool_size=(2, 2)),
     Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     MaxPooling2D(pool_size=(2, 2)),
     Flatten(),
     Dense(units=128, activation='relu'),
     Dense(units=10, activation='softmax')  # 10 output classes
     ])
visualkeras.layered_view(model, legend=True, to_file='./plots/cnn_model_architecture.png')

# Compile the model with Adam optimizer and the set learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on CIFAR-10
model.fit(X_train, y_train_cat, epochs=10, batch_size=64, validation_data=(X_test, y_test_cat), verbose=2)

# Evaluate the model
plot_confusion_matrix(model, X_test, y_test, 'Convolutional Neural Network', './plots/confusion_matrix_cnn.png')
