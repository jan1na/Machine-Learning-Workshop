import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from utils.eval import plot_confusion_matrix
import visualkeras

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Resize images to 96x96 to fit MobileNetV2 input
x_train = tf.image.resize(x_train, (96, 96)) / 255.0
x_test = tf.image.resize(x_test, (96, 96)) / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Use MobileNetV2 as a base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
base_model.trainable = False  # Freeze the base model

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # TODO: Try other layer sizes
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),  # TODO: Try changing the learning rate
    loss='categorical_crossentropy', metrics=['accuracy'])
visualkeras.layered_view(model, legend=True, to_file='./plots/mobile_net_v2_model_architecture.png')

# Train the model
# TODO: Try different batch sizes or epochs
model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=5, batch_size=64)

# Evaluate the model
plot_confusion_matrix(model, x_test, y_test, 'Transfer Learning Model', './plots/confusion_matrix_tl.png')
