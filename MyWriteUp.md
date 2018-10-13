# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


![Image1](/images/Original_Image.png)  
![Image2](/images/Flipped_Image.png) 
![Image3](/images/Cropped_Image.png)  

The first step in the model is top load the records from the cdv file that contain the paths to the images and the associated steering angle of an image.  I discarded all images and steering angles where the steering angle was equal to zero.  This helped balance out the dataset so that the model could more easily learn how to turn.  There were enough images and angles where the steering angle was close to zero that I did not exclude too many images for certain parts of the track (such as the bridge where it has a long straightaway). 

    'Load the raw data'
    def load_samples():
        samples = []
        with open('./data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for sample in reader:
                if float(sample[3])!=0:
                    samples.append(sample)
        return samples

Here is an image showing the steering angles before and after excluding zero steering angles.

![Image1](/images/Data_Plots.png) 

As part of the generator function, the next function loads the images from disk into batches.  The images are read in as BGR images but are converted to RGB.  Each associated steering angle is read in as well in the same order as the images.  There is data available for three cameras (center, left and right).  However, I only used the center camera data as I felt I would introduce too much training noise from the other cameras.  Fortunately, the center camera data was high quality for train and I did not need the left and right camera data.

    def load_data(samples):
        images = []
        steering_angles = []
    
        for i, sample in enumerate(samples):
            # print('Reading Sample: '+str(i)+' of '+str(num_samples), end='\r')
            for j in range(1):
                source_path = sample[j]
                filename = source_path.split('/')[-1]
                image_path = './data/IMG/' + filename
                image = cv2.imread(image_path)
                images.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

            steering_angle_center = float(sample[3])
            steering_angles.append(steering_angle_center)
            #steering_angles.append(steering_angle_center + 0.2)
            #steering_angles.append(steering_angle_center - 0.2)    
        return images, steering_angles
        
 Because the track tends to have a lot of left turns the data for right turns is much smaller than left.  In order for the model to have enough data to learn right turns quickly I created an augmented set of data using the flipped images and steering angles.  This resulted in the model having much smoother right turns.
 
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

I used a generator function to deliver data to the model in batches of 32.  Within the generator function the images and steering angles are loaded as well as augmented.  There is a shuffle routine before the batches are generated that makes a trained model a little different each time, even if all things stay the same.

    def generator(samples, batch_size=32):
        while True:                                                
            sklearn.utils.shuffle(samples)
            for offset in range(0, len(samples), batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images, steering_angles = load_data(batch_samples)
                images, steering_angles = augment_data(images, steering_angles)
                x_train = np.array(images)
                y_train = np.array(steering_angles)
                yield x_train, y_train

For my model I chose an architecture similar to the Nvidia End-to-End model.  I tried out a simple regression model as well as the LeNet model but found the Nvidia model higher quality.  Within the model I start by normalizing the images.  This helps with the gradient calculacations.  Next I crop the images so that information that is not needed to deleted from the data that goes to the model.  One other thing I did was add a dropout layer to help prevent model overfitting.  The dropout did not have much of an impact but I left it in anyway as it didn't hurt.  I used Adam optimization and a Mean Square Error loss function.

    def nvidia_model():
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((50, 20), (0, 0))))
        model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))     
        model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))     
        model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))     
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))     
        model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))     
        model.add(Flatten())                                                                            
        model.add(Dense(1164, activation='relu'))                                                       
        model.add(Dropout(0.4))
        model.add(Dense(100, activation='relu'))                                                        
        model.add(Dense(10, activation='relu'))                                                         
        model.add(Dense(1))                                                                             
        model.compile(optimizer='adam', loss='mse')
        return model

To start the training process I load the samples and generate the training and validation batches for the model in batches of 32.
    
    samples=load_samples()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

Finally, I train my model over 6 epochs.  I ran my model over many more epochs but found the validation stopped decreasing after about 5-6 epochs so I felt I only needed 6 epochs for a quality model.

    model = nvidia_model()
    history_object = model.fit_generator(train_generator, 
                                         steps_per_epoch=len(train_samples)/32, 
                                         validation_data=validation_generator, 
                                         validation_steps=len(validation_samples)/32, 
                                         epochs=20, 
                                         verbose=2)
    plot_history(history_object, './images/', 'samplemodel')
    model.save('samplemodel.h5')

Here is a plot of training anf validation over 20 epochs and you can see the validation starting to get worse after 5 epochs.

![Image3](/images/samplemodel.png)

---

To train the model I used the command:

    python model.py

The output of the model is saved to a file called: model.h5.

To automously drive the car in the simulator I opened the simulator in automous mod and run the command:

    python drive.py model.h5

Once I was satisfied the trained model was able to safely drive the car around the track I generated a video of the run around the track two times using the command:

    python drive.py model.h5 model

This created a folder of images of my automoous run in the simulator.

To make a video of the automous run I ran the command:

    python video.py model

This created an mp4 cideo called model.mp4.

For my submission, I included the following files:

    model.py
    model.h5
    model.mp4
    mywriteup.pdf
  
  
  
  




