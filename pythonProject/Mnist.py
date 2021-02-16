from keras.datasets import mnist
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.utils import to_categorical
from keras.models import load_model
import h5py


# utility function for showing images
def show_imgs(x_test, n):
    plt.figure(figsize=(15, 2))  # figsize: width and height in inches
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)  # nrows, ncols, index
        plt.imshow(x_test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# display an image
image_index = 7777  # You may select anything up to 60,000
print(train_labels[image_index])  # The label is 8
plt.imshow(train_images[image_index], cmap='Greys')

# process the data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images2 = test_images

test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images.astype('float32') / 255

test_labels2 = test_labels

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

if (os.access('mnist_model.h5', os.R_OK)):
    model = load_model('nmist_model.h5')
    print('Model has been trained already')

else:
    # create the model
    model = models.Sequential()

    model.add(layers.Convolution2D(filters=10, kernel_size=(5, 5), padding='valid', activation='relu',
                                   input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(layers.Convolution2D(filters=10, kernel_size=(3, 3), padding='valid', activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train and evaluate the images
    model.fit(train_images, train_labels, epochs=5, batch_size=128)
    model.save('nmist_model.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print(model.summary())

# predicted probabilities and values for the test data.
predicted_prob = model.predict(test_images)
predicted_classes = np.argmax(predicted_prob, axis=1)

correct = 0
correctIndex = []
error = []
errorIndex = []

numberError = np.zeros((9))

for n in range(len(predicted_classes)):
    predicted = predicted_classes[n]
    actual = test_labels2[n]

    if predicted != actual:
        difference = abs(predicted_prob[n, predicted] - predicted_prob[n, actual])
        error.append(difference)
        errorIndex.append(n)
        numberError[test_labels2[n] - 1] += 1;

    elif predicted_prob[n, predicted] > .99:

        correct += 1;
        if len(correctIndex) < 4:
            correctIndex.append(n)

sort = np.array(errorIndex)[np.argsort(np.array(error)[:])]

worstErrorIndex = sort[-4:]

worstImages = np.array(test_images2[sort[-4:]])

show_imgs(worstImages, 4)

show_imgs(test_images2[correctIndex], 4)

print(numberError)

true = len(predicted_classes[predicted_classes == test_labels2]);
false = len(predicted_classes[predicted_classes != test_labels2]);

print()

accuracy = (true) / len(predicted_classes)

print("Correct".rjust(23), "Incorrect".rjust(10))

print("n = {f}".format(f=len(predicted_classes)).rjust(12), "{a}".format(a=true).rjust(10),
      "{a}".format(a=false).rjust(10))

print()

print("Accuracy:".rjust(12), "{a:.3f}".format(a=accuracy).rjust(10))


















