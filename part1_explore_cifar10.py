import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# Normalize the images to [0, 1] range
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Plot some of the images to visualize CIFAR-10
def plot_images(images, labels, num=16):
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {class_names[labels[i][0]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

plot_images(X_train, y_train, num=16)
