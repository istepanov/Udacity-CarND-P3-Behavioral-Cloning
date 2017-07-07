# CPU:
# docker run -it --rm -v `pwd`:/src udacity/carnd-term1-starter-kit python model.py train/driving_log.csv model.h5
#
# GPU:
# nvidia-docker run -it --rm -v `pwd`:/src istepanov/carnd-gpu python model.py train/driving_log.csv model.h5
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


LEARNING_RATE = 0.00005
BATCH_SIZE = 64
EPOCHS = 20

IMAGES_AND_ANGLES = [
    (0, 0.0),
    (1, 0.1),
    (2, -0.1)
]


def generator(samples, batch_size, base_train_folder, training=False):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for batch_sample_index, angle_adjust in IMAGES_AND_ANGLES:
                    image_file = os.path.join(base_train_folder, batch_sample[batch_sample_index])
                    image = cv2.imread(image_file)
                    angle = float(batch_sample[3]) + angle_adjust
                    if training and random.random() > 0.5:
                        image = cv2.flip(image, 1)
                        angle *= -1
                    images.append(image)
                    angles.append(angle)
            x = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(x, y)


def main():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument(
        'driving_log',
        type=str,
        default='',
        help='Path to driving log csv file.'
    )
    parser.add_argument(
        'model_file',
        type=str,
        default='',
        help='Path to file where model will be saved.'
    )
    args = parser.parse_args()

    assert(os.path.isfile(args.driving_log))

    base_train_folder = os.path.dirname(args.driving_log)

    samples = []
    with open(args.driving_log) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=0.2)

    input_shape = (160, 320, 3)

    # NVIDIA
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, 5, 5, border_mode='valid', activation='elu', subsample=(2, 2), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(36, 5, 5,border_mode='valid', activation='elu', subsample=(2, 2), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(48, 5, 5,border_mode='valid', activation='elu', subsample=(2, 2), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, 3,border_mode='valid', activation='elu', subsample=(1, 1), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, 3,border_mode='valid', activation='elu', subsample=(1, 1), init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu', init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='elu', init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='elu', init='glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='elu', init='glorot_uniform'))
    model.add(Dense(1, activation='tanh', init='glorot_uniform'))

    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    model.fit_generator(
        generator=generator(samples=train_samples, batch_size=BATCH_SIZE, base_train_folder=base_train_folder, training=True),
        samples_per_epoch=len(train_samples),
        validation_data=generator(samples=validation_samples, batch_size=BATCH_SIZE, base_train_folder=base_train_folder),
        nb_val_samples=len(validation_samples),
        nb_epoch=EPOCHS,
    )
    model.save(args.model_file)

    # workaround for https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()


if __name__ == '__main__':
    main()
