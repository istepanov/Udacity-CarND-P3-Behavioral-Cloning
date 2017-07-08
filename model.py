# CPU:
# docker run -it --rm -v `pwd`:/src udacity/carnd-term1-starter-kit python model.py train/**/driving_log.csv model.h5
#
# GPU:
# nvidia-docker run -it --rm -v `pwd`:/src istepanov/carnd-gpu python model.py train/**/driving_log.csv model.h5
#

import argparse
import os
import csv
import random
import numpy as np
import cv2
import sklearn.utils
import sklearn.model_selection
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


LEARNING_RATE = 0.00005
BATCH_SIZE = 64
EPOCHS = 20


def random_brightness(image):
    brightness = np.random.uniform() + 0.5
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    image[:,:,2] = image[:,:,2] * brightness
    image[:,:,2][image[:,:,2] > 255] = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image


def generator(samples, batch_size, training=False):
    num_samples = len(samples)

    images_and_angles = [
        (0, 0.0),   # center
        (1, 0.15),   # left
        (2, -0.15),  # right
    ] if training else [(0, 0.0)] # just center if not training

    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for batch_sample_index, angle_adjust in images_and_angles:
                    image_file = batch_sample[batch_sample_index]
                    image = cv2.imread(image_file)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3]) + angle_adjust
                    if training:
                        augment_mode = random.choice([None, 'flip', 'brightness'])
                        if augment_mode == 'flip':
                            image = cv2.flip(image, 1)
                            angle *= -1
                        elif augment_mode == 'brightness':
                            image = random_brightness(image)
                    images.append(image)
                    angles.append(angle)
            x = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(x, y)


def main():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument(
        'driving_logs',
        type=str,
        nargs='+',
        default='',
        help='Paths to driving log csv file (hint: you can put wildcard mask here)'
    )
    parser.add_argument(
        'model_file',
        type=str,
        default='',
        help='Path to file where model will be saved.'
    )
    args = parser.parse_args()

    samples = []

    for csv_file_name in args.driving_logs:
        assert(os.path.isfile(csv_file_name))
        print(csv_file_name)
        dirname = os.path.dirname(csv_file_name)
        with open(csv_file_name) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                for i in range(0, 3):
                    line[i] = os.path.join(dirname, line[i])
                    assert(os.path.isfile(line[i]))
                samples.append(line)

    train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=0.2)

    input_shape = (160, 320, 3)

    # NVIDIA
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(48, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, 3, border_mode='valid', activation='elu', subsample=(1, 1), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu', init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='elu', init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='elu', init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='elu', init='glorot_uniform'))
    model.add(Dense(1, init='glorot_uniform'))

    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    training_generator = generator(samples=train_samples, batch_size=BATCH_SIZE, training=True)
    validation_generator = generator(samples=validation_samples, batch_size=BATCH_SIZE)

    model.fit_generator(
        generator=training_generator,
        samples_per_epoch=len(train_samples) * 3,   # 3 images per sample
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=EPOCHS,
        callbacks=[early_stopping]
    )
    model.save(args.model_file)

    # workaround for https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()


if __name__ == '__main__':
    main()
