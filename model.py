import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# import keras
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

print('hello')
'Load the raw data'
def load_samples():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for sample in reader:
            samples.append(sample)
    return samples

'Function to extract images and steering angles from raw data'
def load_data(samples):
    images = []
    steering_angles = []
    for sample in samples:
        for i in range(3):
            source_path = sample[i]
            filename = source_path.split('/')[-1]
            image_path = './data/IMG/' + filename
            image = cv2.imread(image_path)
            images.append(image)

        correction = 0.1
        steering_angle_center = float(sample[3])
        steering_angle_left = steering_angle_center + correction
        steering_angle_right = steering_angle_center - correction
        steering_angles.append(steering_angle_center)
        steering_angles.append(steering_angle_left)
        steering_angles.append(steering_angle_right)
    return images, steering_angles

'Function to augment the images and steering angles'
def augment_data(images, steering_angles):
    augmented_images = []
    augmented_steering_angles = []
    for image, steering_angle in zip(images, steering_angles):
        augmented_images.append(image)
        augmented_steering_angles.append(steering_angle)
        flipped_image = cv2.flip(image, 1)
        flipped_steering_angle = steering_angle * -1.0
        augmented_images.append(flipped_image)
        augmented_steering_angles.append(flipped_steering_angle)
    return augmented_images, augmented_steering_angles

'Plot the training and validation loss for each epoch'
def plot_history(history_object, save_path, title):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(os.path.join(save_path + title + '.png'))
    plt.show()

'Simple Regression Model'
def simple_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

'LeNet Model'
def lenet_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

'NVIDIA End-to-End Model'
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))       # Layer 1
    model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))       # Layer 2
    model.add(Convolution2D(filters=48, kernel_size=(3, 3), strides=(1, 1), activation='relu'))       # Layer 3
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))       # Layer 4
    model.add(Flatten())                                                                              # Layer 5
    model.add(Dense(1024), activation='relu')                                                         # Layer 6
    model.add(Dense(128), activation='relu')                                                          # Layer 7
    model.add(Dense(10), activation='relu')                                                           # Layer 8
    model.add(Dense(1))                                                                               # Layer 9
    return model

print('hello')
samples = load_samples()
print(len(samples))
images, steering_angles = load_data()
augmented_images, augmented_steering_angles = augment_data(images, steering_angles)
x_train = np.array(augmented_images)
y_train = np.array(augmented_steering_angles)

model = simple_model()
history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1)
model.save('simple_model.h5')
plot_history(history_object, './images/', 'simple_model')

exit()

model = lenet_model()
history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1)
model.save('lenet_model.h5')
plot_history(history_object, './images/', 'lenet_model')

model = nvidia_model()
history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10, verbose=1)
model.save('nvidia_model.h5')
plot_history(history_object, './images/', 'nvidia_model')



