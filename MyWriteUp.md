# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

### Data Exploration

The provided car simulator is able to capture data from three cameras (center, left and right) along with the steering angle as the car drives around a track.  The steering angle is most closely associated with the center camera.  To use the left and right camera data, adjustments to the steering angle would have to be made.

Here is an example of left, center, and right camera images of the track.

![Left Image](/images/Left_Image.png) 
![Center Image](/images/Center_Image.png) 
![Right Image](/images/Right_Image.png) 

We were given the options of creating our own simulator data, using the simulation data provided, or a combination of both.  My attempt was to use the provided data and supplement or replace, if needed.  Ultimately, I felt I did not need to create my own simulator data as a I found a suitable way to use the provided data.

---

### Description of the Model:

The first step in the model was to load the records from the csv file that contained the paths to the images and associated steering angles.  I discarded all images and steering angles where the steering angle was equal to zero.  This helped balance out the dataset so that the model could more easily learn how to turn.  There were enough images and angles where the steering angle was close to zero that I did not exclude too many images for certain parts of the track (such as the bridge where it has a long straightaway). 

    def load_samples():
        samples = []
        with open('./data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for sample in reader:
                if float(sample[3])!=0:  # Do not include samples where steering angle=0
                    samples.append(sample)
        return samples

Here is an image showing the steering angles before and after excluding zero steering angles.

![Data Plots](/images/Data_Plots.png) 

The original data contained 8060 center camera images.  After removing images where the associated steering angle was 0 I had 3650 images.  Each image had a shape of: (160, 320, 3).

As part of a data generation function (described later), the next function loads the images from disk into batches.  The images are read in as BGR images but are converted to RGB.  Each associated steering angle is read in as well in the same order as the images.  There is data available for three cameras (center, left and right).  However, I only used the center camera data as I felt I would introduce too much training noise from the other cameras.  Fortunately, the center camera data was high enough quality for training that I did not need the left and right camera data.

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
            # steering_angles.append(steering_angle_center + 0.2)
            # steering_angles.append(steering_angle_center - 0.2)    
        return images, steering_angles
        
Because track one tends to have a more left turns than right the data for right turns is smaller than left.  In order for the model to have enough data to quickly learn right turns I created an augmented set of data using the flipped images and steering angles.  This resulted in the model having more right turn data for training and ultimately smoother right turns in the simulator.
 
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

![Original Image](/images/Original_Image.png)  
![Flipped Image](/images/Flipped_Image.png) 

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

For my model I chose a Convolutional Neural Netowrk architecture similar to the Nvidia End-to-End model.  I tried out a simple regression model as well as the LeNet model but found the Nvidia model higher quality.  Within the model I started by normalizing the images.  This helped with the gradient calculations.  Next, I cropped the images so that unhelpful information was discarded.  

Here is an example of a cropped image where a portion of the top and bottom have been removed.

![Original Image](/images/Original_Image.png) ![Cropped Image](/images/Cropped_Image.png)  

One other thing I did was add a dropout layer to help prevent model overfitting.  The dropout did not have much of an impact but I left it in anyway as it didn't hurt.  I used Adam optimization and a Mean Square Error loss function as well as RELU activation on all layers.

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

To start the training process I loaded the samples and generated the training and validation data for the model in batches of 32.  The samples were split 80% for training and 20% for validation.
    
    samples=load_samples()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

Finally, I trained my model over 6 epochs.  I ran my model over many more epochs but found the validation stopped decreasing after about 5-6 epochs so I felt I only needed 6 epochs for a quality model.

    model = nvidia_model()
    history_object = model.fit_generator(train_generator, 
                                         steps_per_epoch=len(train_samples)/32, 
                                         validation_data=validation_generator, 
                                         validation_steps=len(validation_samples)/32, 
                                         epochs=6, 
                                         verbose=2)
    plot_history(history_object, './images/', 'model')
    model.save('model.h5')

Here is a plot of training and validation over 20 epochs and you can see the validation starting to get worse after about 5 epochs.

![Training/Validation](/images/samplemodel.png)

---

### Running the Model:

To train the model I used the command:

    python model.py
    
The model trained in about 58 seconds on my local machine using Windows 10 and an Nvidia 1080TI GPU.

The output of the model is saved to a file called: model.h5.

To automously drive the car in the simulator I opened the simulator in automous mode and ran on the command line:

    python drive.py model.h5

This connected to the simulator and passed steering angles as images were captured.  Note: I changed the throttle value in drive.py to 30 to speed up the recorded video.

Once I was satisfied the trained model was able to safely drive the car around the track twice I generated a video of the run using the command:

    python drive.py model.h5 model

This created a 'model' folder of images of my automoous run from the simulator.

To make a video of the automous run I ran the command:

    python video.py model

This created an mp4 video called model.mp4.

---

### Submission:

For my submission, I included the following files in a zipped file:

    model.py
    drive.py
    model.h5
    model.mp4
    mywriteup.pdf
