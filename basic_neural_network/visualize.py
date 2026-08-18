import matplotlib.pyplot as plt

from neural_network import load_images, load_labels


# Load training data
X_train = load_images("data/train-images-idx3-ubyte.gz")
y_train = load_labels("data/train-labels-idx1-ubyte.gz")


# Select one image
index = 0

image = X_train[index]
label = y_train[index]


# Display the image
plt.imshow(image, cmap="gray")
plt.title(f"Actual Digit: {label}")
plt.axis("off")
plt.show()