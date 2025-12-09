def train_linear_model():
    # Import libraries inside the function 
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Flatten, Dense
    from tensorflow.keras.utils import to_categorical
    import os

    # 1. Load the MNIST dataset (handwritten digits 0-9)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Scale pixel values from [0, 255] to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 3. Convert labels to one-hot vectors (10 classes)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 4. Build a very simple linear model
    #    Flatten: 28x28 image -> 784 vector
    #    Dense: 10 outputs for digits 0-9 with softmax
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(10, activation="softmax")
    ])

    # 5. Compile model with optimizer, loss and metrics
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6. Train the model
    print("Training started...")
    model.fit(x_train, y_train, epochs=5)
    print("Training finished.")

    # 7. Evaluate the model on test data
    loss, acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", acc)

    # 8. Save the trained model
    os.makedirs("model", exist_ok=True)
    model.save("model/mnist_linear.h5")
    print("Model saved as model/mnist_linear.h5")


if __name__ == "__main__":
    train_linear_model()
