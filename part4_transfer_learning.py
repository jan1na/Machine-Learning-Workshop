import tensorflow
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from utils.eval import plot_confusion_matrix
import visualkeras

# TODO: Play around with hyperparameters
TRAIN_SIZE = 10000
TEST_SIZE = 2000
BATCH_SIZE = 16
IMAGE_SIZE = 96  # Input size for MobileNetV2
DENSE_LAYER_SIZE = 128  # Size of the dense layer after the base model
TRAIN_EPOCHS = 5  # Number of epochs for training


# Load CIFAR-10 and reduce dataset size for speed on laptop
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = x_train[:TRAIN_SIZE], y_train[:TRAIN_SIZE]
x_test, y_test = x_test[:TEST_SIZE], y_test[:TEST_SIZE]


# Preprocessing function for tf.data pipeline
def preprocess(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """
    Preprocess the input data for the model. Resizes the images to 96x96 and normalizes them to [0, 1]. Converts labels
    to one-hot encoding.

    :param x: Input image tensor.
    :param y: Input label tensor.
    """
    x = tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0  # slightly smaller than 96x96
    y = tf.one_hot(tf.squeeze(y), depth=10)
    return x, y


# Prepare datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).shuffle(5000).batch(BATCH_SIZE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(BATCH_SIZE)

# Load MobileNetV2 without top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False  # Freeze the base model

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(DENSE_LAYER_SIZE, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Visualize model architecture
visualkeras.layered_view(model, legend=True, to_file='./plots/mobile_net_v2_model_architecture.png')

# Train model
model.fit(train_ds, validation_data=test_ds, epochs=TRAIN_EPOCHS)

# Evaluate with confusion matrix
x_test_resized = tf.image.resize(x_test, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
plot_confusion_matrix(model, x_test_resized, y_test, 'Transfer Learning Model', './plots/confusion_matrix_tl.png')
