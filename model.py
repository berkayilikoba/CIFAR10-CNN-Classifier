from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout

def create_model(input_shape=(32,32,3), num_classes=10):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=3, activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model
