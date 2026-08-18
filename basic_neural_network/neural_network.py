import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD MNIST IMAGES
# ============================================================

def load_images(filename):
    with gzip.open(filename, "rb") as f:

        magic, num_images, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        if magic != 2051:
            raise ValueError("Invalid MNIST image file")

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        images = data.reshape(
            num_images,
            rows,
            cols
        )

        return images


# ============================================================
# 2. LOAD MNIST LABELS
# ============================================================

def load_labels(filename):
    with gzip.open(filename, "rb") as f:

        magic, num_labels = struct.unpack(
            ">II",
            f.read(8)
        )

        if magic != 2049:
            raise ValueError("Invalid MNIST label file")

        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        return labels


# ============================================================
# 3. MNIST FILE LOCATIONS
# ============================================================

train_images_file = "data/train-images-idx3-ubyte.gz"
train_labels_file = "data/train-labels-idx1-ubyte.gz"

test_images_file = "data/t10k-images-idx3-ubyte.gz"
test_labels_file = "data/t10k-labels-idx1-ubyte.gz"


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("Loading MNIST dataset...")

X_train = load_images(train_images_file)
y_train = load_labels(train_labels_file)

X_test = load_images(test_images_file)
y_test = load_labels(test_labels_file)


print("\nOriginal dataset:")
print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)


# ============================================================
# 5. PREPROCESSING
# ============================================================

print("\nPreprocessing data...")


# Convert pixel values from 0-255 to 0-1

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0


# Flatten 28x28 images into 784 values

X_train = X_train.reshape(
    X_train.shape[0],
    784
)

X_test = X_test.reshape(
    X_test.shape[0],
    784
)


print("After preprocessing:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print(
    "Pixel range:",
    X_train.min(),
    "to",
    X_train.max()
)


# ============================================================
# 6. NEURAL NETWORK
# ============================================================

class NeuralNetwork:

    # --------------------------------------------------------
    # Constructor
    # --------------------------------------------------------

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size
    ):

        # Input layer → Hidden layer
        # 784 → 128

        self.W1 = (
            np.random.randn(
                input_size,
                hidden_size
            ) * 0.01
        )


        # Hidden layer → Output layer
        # 128 → 10

        self.W2 = (
            np.random.randn(
                hidden_size,
                output_size
            ) * 0.01
        )


        # Biases

        self.b1 = np.zeros(
            (1, hidden_size)
        )

        self.b2 = np.zeros(
            (1, output_size)
        )


        print("\nNeural network created!")

        print("W1:", self.W1.shape)
        print("b1:", self.b1.shape)

        print("W2:", self.W2.shape)
        print("b2:", self.b2.shape)


    # ========================================================
    # RELU ACTIVATION
    # ========================================================

    def relu(self, x):

        return np.maximum(
            0,
            x
        )


    # ========================================================
    # SOFTMAX
    # ========================================================

    def softmax(self, x):

        # Numerical stability

        exp_x = np.exp(
            x - np.max(
                x,
                axis=1,
                keepdims=True
            )
        )

        return (
            exp_x /
            np.sum(
                exp_x,
                axis=1,
                keepdims=True
            )
        )


    # ========================================================
    # FORWARD PROPAGATION
    # ========================================================

    def forward(self, X):

        # Input → Hidden

        self.z1 = (
            np.dot(
                X,
                self.W1
            )
            + self.b1
        )


        # ReLU

        self.a1 = self.relu(
            self.z1
        )


        # Hidden → Output

        self.z2 = (
            np.dot(
                self.a1,
                self.W2
            )
            + self.b2
        )


        # Softmax

        self.a2 = self.softmax(
            self.z2
        )


        return self.a2


    # ========================================================
    # CROSS-ENTROPY LOSS
    # ========================================================

    def compute_loss(
        self,
        y_true,
        y_pred
    ):

        m = y_true.shape[0]


        # Probability assigned to correct class

        correct_class_probs = y_pred[
            np.arange(m),
            y_true
        ]


        # Avoid log(0)

        correct_class_probs = np.clip(
            correct_class_probs,
            1e-12,
            1.0
        )


        # Cross-entropy loss

        loss = -np.mean(
            np.log(
                correct_class_probs
            )
        )


        return loss


    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    def backward(
        self,
        X,
        y_true,
        y_pred
    ):

        m = X.shape[0]


        # Output layer gradient

        dz2 = y_pred.copy()

        dz2[
            np.arange(m),
            y_true
        ] -= 1

        dz2 /= m


        # Gradient for W2

        dW2 = np.dot(
            self.a1.T,
            dz2
        )


        # Gradient for b2

        db2 = np.sum(
            dz2,
            axis=0,
            keepdims=True
        )


        # Gradient flowing into hidden layer

        da1 = np.dot(
            dz2,
            self.W2.T
        )


        # ReLU derivative

        dz1 = (
            da1 *
            (self.z1 > 0)
        )


        # Gradient for W1

        dW1 = np.dot(
            X.T,
            dz1
        )


        # Gradient for b1

        db1 = np.sum(
            dz1,
            axis=0,
            keepdims=True
        )


        return (
            dW1,
            db1,
            dW2,
            db2
        )


    # ========================================================
    # UPDATE PARAMETERS
    # ========================================================

    def update_parameters(
        self,
        dW1,
        db1,
        dW2,
        db2,
        learning_rate
    ):

        self.W1 -= (
            learning_rate * dW1
        )

        self.b1 -= (
            learning_rate * db1
        )

        self.W2 -= (
            learning_rate * dW2
        )

        self.b2 -= (
            learning_rate * db2
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, X):

        probabilities = self.forward(
            X
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        return predictions


    # ========================================================
    # ACCURACY
    # ========================================================

    def accuracy(
        self,
        X,
        y
    ):

        predictions = self.predict(
            X
        )

        accuracy = np.mean(
            predictions == y
        )

        return accuracy


    # ========================================================
    # TRAINING
    # ========================================================

    def train(
        self,
        X,
        y,
        epochs=5,
        batch_size=64,
        learning_rate=0.1
    ):

        loss_history = []

        num_samples = X.shape[0]


        print("\nStarting training...")

        print(
            "Training samples:",
            num_samples
        )

        print(
            "Epochs:",
            epochs
        )

        print(
            "Batch size:",
            batch_size
        )

        print(
            "Learning rate:",
            learning_rate
        )


        # ====================================================
        # EPOCH LOOP
        # ====================================================

        for epoch in range(epochs):

            # Shuffle training data

            indices = np.random.permutation(
                num_samples
            )

            X_shuffled = X[
                indices
            ]

            y_shuffled = y[
                indices
            ]


            total_loss = 0.0
            num_batches = 0


            # =================================================
            # MINI-BATCH LOOP
            # =================================================

            for start in range(
                0,
                num_samples,
                batch_size
            ):

                end = min(
                    start + batch_size,
                    num_samples
                )


                X_batch = X_shuffled[
                    start:end
                ]

                y_batch = y_shuffled[
                    start:end
                ]


                # Forward propagation

                y_pred = self.forward(
                    X_batch
                )


                # Calculate loss

                loss = self.compute_loss(
                    y_batch,
                    y_pred
                )


                # Backpropagation

                dW1, db1, dW2, db2 = self.backward(
                    X_batch,
                    y_batch,
                    y_pred
                )


                # Update weights

                self.update_parameters(
                    dW1,
                    db1,
                    dW2,
                    db2,
                    learning_rate
                )


                total_loss += loss
                num_batches += 1


            # =================================================
            # AVERAGE LOSS
            # =================================================

            average_loss = (
                total_loss /
                num_batches
            )


            loss_history.append(
                average_loss
            )


            # =================================================
            # TRAINING ACCURACY
            # =================================================

            accuracy_samples = min(
                10000,
                num_samples
            )


            train_accuracy = self.accuracy(
                X[:accuracy_samples],
                y[:accuracy_samples]
            )


            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {average_loss:.4f} "
                f"| Accuracy: "
                f"{train_accuracy * 100:.2f}%"
            )


        return loss_history


# ============================================================
# 7. CREATE NEURAL NETWORK
# ============================================================

network = NeuralNetwork(
    input_size=784,
    hidden_size=128,
    output_size=10
)


# ============================================================
# 8. TRAIN NETWORK ON ALL 60,000 IMAGES
# ============================================================

# Use ALL 60,000 MNIST training images

X_train_small = X_train
y_train_small = y_train


loss_history = network.train(
    X_train_small,
    y_train_small,
    epochs=5,
    batch_size=64,
    learning_rate=0.1
)


print("\nTraining completed!")


# ============================================================
# 9. FINAL TRAINING ACCURACY
# ============================================================

print("\nCalculating training accuracy...")


training_accuracy = network.accuracy(
    X_train_small,
    y_train_small
)


print(
    f"Training Accuracy: "
    f"{training_accuracy * 100:.2f}%"
)


# ============================================================
# 10. TEST ACCURACY
# ============================================================

print("\nCalculating test accuracy...")


test_accuracy = network.accuracy(
    X_test,
    y_test
)


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 11. SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")


sample_size = 10

sample_images = X_test[
    :sample_size
]

sample_labels = y_test[
    :sample_size
]


sample_predictions = network.predict(
    sample_images
)


for i in range(sample_size):

    print(
        f"Image {i + 1}: "
        f"Predicted = {sample_predictions[i]}, "
        f"Actual = {sample_labels[i]}"
    )


# ============================================================
# 12. VISUALIZE TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        len(loss_history) + 1
    ),
    loss_history,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training Loss"
)

plt.grid(
    True
)

plt.show()


# ============================================================
# 13. VISUALIZE SAMPLE PREDICTIONS
# ============================================================

plt.figure(
    figsize=(12, 5)
)


for i in range(10):

    plt.subplot(
        2,
        5,
        i + 1
    )


    # Convert 784 values back to 28x28

    image = sample_images[i].reshape(
        28,
        28
    )


    plt.imshow(
        image,
        cmap="gray"
    )


    plt.title(
        f"Pred: {sample_predictions[i]}\n"
        f"Actual: {sample_labels[i]}"
    )


    plt.axis(
        "off"
    )


plt.tight_layout()

plt.show()


# ============================================================
# 14. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 50)

print("MNIST NEURAL NETWORK PROJECT COMPLETE")

print("=" * 50)

print(
    f"Training Accuracy: "
    f"{training_accuracy * 100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final Loss: "
    f"{loss_history[-1]:.4f}"
)

print("=" * 50)