import csv
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
EPOCHS = 6

def load_samples():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for sample in reader:
            if float(sample[3])!=0:
                samples.append(sample)
    return samples

def load_data(samples):
    images = []
    steering_angles = []
    for sample in samples:
        for i in range(1):     # 1 is for center camera.  Change to 3 to get left and right cameras as well.
            source_path = sample[i]
            filename = source_path.split('/')[-1]
            image_path = './data/IMG/' + filename
            image = cv2.imread(image_path)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        steering_angle_center = float(sample[3])
        steering_angles.append(steering_angle_center)
        # steering_angles.append(steering_angle_center + 0.2)
        # steering_angles.append(steering_angle_center - 0.2)
    return images, steering_angles

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

def generator(samples, batch_size=BATCH_SIZE):
    while True:                                                # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, steering_angles = load_data(batch_samples)
            images, steering_angles = augment_data(images, steering_angles)
            x_train = np.array(images)
            y_train = np.array(steering_angles)
            yield x_train, y_train

'NVIDIA End-to-End Model'
def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))       # Layer 1
    model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))       # Layer 2
    model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))       # Layer 3
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))       # Layer 4
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))       # Layer 5
    model.add(Flatten())                                                                              # Layer 6
    model.add(Dense(1164, activation='relu'))                                                         # Layer 7
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))                                                          # Layer 8
    model.add(Dense(10, activation='relu'))                                                           # Layer 9
    model.add(Dense(1))                                                                               # Layer 10
    model.compile(optimizer='adam', loss='mse')
    return model

'Plot the training and validation loss for each epoch'
def plot_history(history_object, save_path, title):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model Mean Squared Error Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.savefig(os.path.join(save_path + title + '.png'))
    # plt.show()

samples = load_samples()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = nvidia_model()
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples)/BATCH_SIZE,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples)/BATCH_SIZE,
                                     epochs=EPOCHS,
                                     verbose=2)
model.save('model.h5')
plot_history(history_object, './images/', 'model')
