# %%
import matplotlib.pyplot as plt
from skimage.feature import hog
from mlxtend.data import loadlocal_mnist
import numpy as np
from sklearn import svm
import random

# 1. mengeluarkan data set image
train_images, train_labels = loadlocal_mnist(images_path='images/mnist-dataset/train-images-idx3-ubyte',
                                             labels_path='images/mnist-dataset/train-labels-idx1-ubyte')


test_images, test_labels = loadlocal_mnist(images_path='images/mnist-dataset/t10k-images-idx3-ubyte',
                                             labels_path='images/mnist-dataset/t10k-labels-idx1-ubyte')


plt.imshow(train_images[60].reshape(28,28), cmap='gray')
plt.show()

# 2. ekstraksi fitur hog ke setiap gambar
x_train = []
for i in train_images.reshape(-1,28,28):
    img, _ = hog(i, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True)
    x_train.append(img)

x_train = np.array(x_train)
y_train = np.array(train_labels)

x_tes = []
for i in test_images.reshape(-1,28,28):
    img, _ = hog(i, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True)
    x_tes.append(img)

x_tes = np.array(x_tes)
y_tes = np.array(test_labels)

# 3. svm proses
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_tes)


def sampletest():

    random_indices = random.sample(range(len(test_images)), 5)
    plt.figure(figsize=(15, 3))

    for i, index in enumerate(random_indices, 1):
        plt.subplot(1, 5, i)
        plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y_tes[index]}")

    plt.show()

if __name__ == "__main__":
    sampletest()