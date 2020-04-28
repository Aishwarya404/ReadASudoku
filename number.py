import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import print_summary
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def processImages(x_train, x_test):
    # rehsape the dataset, convert to float and bring it to decimal values.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # show a random image
    plt.imshow(x_train[11111].reshape(28, 28), cmap='Greys')
    plt.show()
    return x_train, x_test

def cnnModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5,5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'mae'])
    filepath = "Number.h5"
    return model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("train shape: ", x_train.shape)
print("test_shape: ", x_test.shape)
(x_train, x_test) = processImages(x_train, x_test)
model = cnnModel()
print_summary(model)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=100)
model.save("Number.h5")