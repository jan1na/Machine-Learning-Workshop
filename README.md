# Machine-Learning-Workshop

## Workshop Overview

In this hands-on workshop, you will:
- Use the **K-Nearest Neighbors (KNN)** algorithm to classify images from the **CIFAR-10 dataset**.
- Use a **Convolutional Neural Network (CNN)** to improve classification performance.
- Tune hyperparameter to improve the performance.

You'll also visualize your models' performance using **confusion matrices**.

Files Overview
- ```part1_explore_cifar10.py```: Visualize the CIFAR-10 dataset.
- ```part2_knn.py```: Implement the K-Nearest Neighbors algorithm for image classification.
- ```part3_cnn.py```: Build and train a Convolutional Neural Network for image classification.
- ```part4_tranfer_learning.py```: Use transfer learning with a pre-trained model MobileNetV2 trained on ImageNet.

Presentation Slides: ```Machine_Learning_Workshop.pdf```

## Getting Started
1. Install **Python 3.12** from [here](https://www.python.org/downloads/).
2. Download **Jetbrains PyCharm** IDE from [here](https://www.jetbrains.com/pycharm/download/) or using Jetbrains Toolbox.
3. Install the IDE.
4. Open the IDE and get the **student license** from [here](https://www.jetbrains.com/community/education/#students).
5. Click on "Code" in the GitHub Repository and **copy the ssh link** (if you have an ssh-key pair for GitHub) or **download the zip file**.
6. If you copied the link and have an ssh-key open Pycharm and use the "Get from Version Control" option to **clone the repository**. If you downloaded the zip file, extract it and **open the folder in Pycharm**.
7. Create a **new virtual environment** by clicking in the bottom right corner of the IDE and selecting "Add New Interpreter", then "Add Local Interpreter..."
8. Now a new window is opened where you can create a Virtualenv Environment. You only need to select the **base interpreter** (Python 3.12) and click on "OK". 
9. Now **open the terminal** in python, which can be found in the bottom left corner of the IDE.
10. In the terminal, type the following command to **install the required packages**:
```bash
pip install -r requirements.txt
```
11. Don't worry if "keras" and "cifar10" in ```from tensorflow.keras.datasets import cifar10``` are highlighted red.

## Hints
### How to connect to GitHub with SSH
Check out the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).
### Where to find the python interpreter to create the virtual environment
#### Windows:
usually the python interpreter is located in the following path:
```angular2html
C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python312\python.exe
```
or
```angular2html
C:\Benutzer\<IhrBenutzername>\AppData\Local\Programs\Python\Python312\python.exe
```

#### macOS:
Use the terminal to find the path of the python interpreter:
```bash
which python3.12
```
It is usually located in:
```angular2html
/usr/local/bin/python3.12
```

#### Linux:
Use the terminal to find the path of the python interpreter:
```bash
which python3.12
```
It is usually located in:
```angular2html
/usr/bin/python3.12
```
