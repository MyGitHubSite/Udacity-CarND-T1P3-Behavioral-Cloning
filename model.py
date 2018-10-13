import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32    # set the batch size
EPOCHS = 5         # set the numer of training epochs

# Function to load csv records containing image paths and steering angles.
def load_samples():
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for sample in reader:
            # Only load records where the steering angle is not zero.
            if float(sample[3])!=0:
                samples.append(sample)
    return samples

# Function to load the images and steering angles into lists
def load_data(samples):
    images = []
    steering_angles = []
    for sample in samples:
        # We could load center, left, and right camera images but I am only loading the center image.
        for i in range(1):     # 1 is for center camera.  Change to 3 to get left and right cameras as well.
            source_path = sample[i]
            filename = source_path.split('/')[-1]
            image_path = './data/IMG/' + filename
            image = cv2.imread(image_path)
            # convert the image from BGR to RGB and add to the images list.
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # grab the steering angle and add to the steering angles list.
        steering_angle_center = float(sample[3])
        steering_angles.append(steering_angle_center)
        # steering_angles.append(steering_angle_center + 0.2)
        # steering_angles.append(steering_angle_center - 0.2)
    return images, steering_angles

# Function to augment the data
def augment_data(images, steering_angles):
    augmented_images = []
    augmented_steering_angles = []

    for image, steering_angle in zip(images, steering_angles):
        # add existing data
        augmented_images.append(image)
        augmented_steering_angles.append(steering_angle)
        # flip the image and steering angle
        flipped_image = cv2.flip(image, 1)
        flipped_steering_angle = steering_angle * -1.0
        # add new data to the images and steering angles list.
        augmented_images.append(flipped_image)
        augmented_steering_angles.append(flipped_steering_angle)
    return augmented_images, augmented_steering_angles

# Function to generate batches of data for training model
def generator(samples, batch_size=BATCH_SIZE):
    while True:                                                # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images, steering_angles = load_data(batch_samples)                  # Load original data
            images, steering_angles = augment_data(images, steering_angles)   # load augmented data
            x_train = np.array(images)
            y_train = np.array(steering_angles)
            yield x_train, y_train

# Define the training model - this is similar to the NVIDIA End-to-End Model
def nvidia_model():
    model = Sequential()
    # normalize the data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    # crop the image so that not need information is removed.
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
    model.add(Dense(1))                                                                               # Output
    model.compile(optimizer='adam', loss='mse')
    return model

samples = load_samples()                                                       # get the sample records
train_samples, validation_samples = train_test_split(samples, test_size=0.2)   # split data into training and validation
train_generator = generator(train_samples, batch_size=BATCH_SIZE)              # generate training batches
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)    # generate validation batches

# set the model
model = nvidia_model()
# train the model
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=len(train_samples)/BATCH_SIZE,
                                     validation_data=validation_generator,
                                     validation_steps=len(validation_samples)/BATCH_SIZE,
                                     epochs=EPOCHS,
                                     verbose=2)
# save the model
model.save('model.h5')

