import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
import cv2
import math

rows, cols, ch = 64, 64, 3
TARGET_SIZE = (64, 64)

#function to pre-process the image to numpy array
def process_image(image):

    # Apply brightness augmentation
    image_bright = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    image_bright[:, :, 2] = image_bright[:, :, 2] * random_bright
    image = cv2.cvtColor(image_bright, cv2.COLOR_HSV2RGB)

    #Crop the image
    cropped_image = image[55:135, :, :]
    processed_image = cv2.resize(cropped_image, TARGET_SIZE)
    image = processed_image.astype(np.float32)

    #Normalize the image
    image = image / 255.0 - 0.5
    return image


def get_augmented_row(row):
    steering = row['steering']
    throttle = row['throttle']
    camera_center=row['center']
    camera_right=row['right']
    camera_left=row['left']
    #print(str(camera_center),str(camera_left),str(camera_right))

    if str(camera_left)!='nan': camera='left'
    elif str(camera_right)!='nan': camera='right'
    else: camera='center'
    #camera = np.random.choice(['center', 'left', 'right'])
    #camera = 'center'
    #print(camera)


    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    image = load_img("data/" + row[camera].strip())
    image = img_to_array(image)

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    # Crop, resize and normalize the image
    image = process_image(image)

    return image, steering



#function to extract the data from driving log
def get_data_generator(data, batch_size=256):
    N = data.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


def get_model():
    model = Sequential()

    # layer 1 output shape is 32x32x24
    model.add(Convolution2D(24, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # layer 2 output shape is 16x16x36
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Dropout(.3))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))


    # layer 3 output shape is 6x6x48
    model.add(Convolution2D(48, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.3))

    # layer 4 output shape is 4x4x64
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.2))

    # Flatten the output 1024
    model.add(Flatten())

    # layer 5
    model.add(Dense(1164))
    model.add(ELU())

    # layer 6
    model.add(Dense(100))
    model.add(ELU())

    # layer 7
    model.add(Dense(50))
    model.add(ELU())

    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    model.summary()

    model.compile(optimizer="adam", loss="mse")

    return model


if __name__ == "__main__":
    #define batch size
    BATCH_SIZE = 256

    #read data from driving log file
    data = pd.read_csv('data/driving_log.csv', usecols=[0, 1, 2, 3, 4])

    # shuffle the data
    data= data.sample(frac=1).reset_index(drop=True)

    # define training validation split
    training_split = 0.75
    n_samples=77530

    num_rows_training = int(data.shape[0] * training_split)

    training_data = data.loc[0:num_rows_training - 1]
    validation_data = data.loc[num_rows_training:]


    data = None

    training_generator = get_data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = get_data_generator(validation_data, batch_size=BATCH_SIZE)

    model = get_model()

    samples_per_epoch = (n_samples // BATCH_SIZE) * BATCH_SIZE

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=5, nb_val_samples=int(samples_per_epoch*(1-training_split)))



    print("Saving model weights and configuration file.")

    model.save_weights('model.h5')  # always save your weights after training or during training
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())