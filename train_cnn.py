def train_simple_cnn():
    # Import libraries inside the function 
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    # 1. Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Preprocess images: reshape and normalize (0–1)
    x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255.0

    # 3. Convert labels to one‑hot vectors (10 classes: 0–9)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # 4. Build a simple CNN model
    model = Sequential()

    # First conv + pooling
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))

    # Second conv + pooling
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    # Flatten feature maps to a 1D vector
    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    # Output layer (10 neurons for 10 digits)
    model.add(Dense(10, activation="softmax"))

    # 5. Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6. Train the model
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    # 7. Evaluate on test data
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", test_acc)

    # 8. Save the trained model
    model.save("model/mnist_cnn.h5")
    print("Model saved as model/mnist_cnn.h5")


if __name__ == "__main__":
    train_simple_cnn()
