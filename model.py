# CPU:
# docker run -it --rm -v `pwd`:/src udacity/carnd-term1-starter-kit python model.py train/driving_log.csv model.h5
#
# GPU:
# nvidia-docker run -it --rm -v `pwd`:/src istepanov/carnd-gpu python model.py train/driving_log.csv model.h5
#

import argparse
import os
import csv
import numpy as np
import cv2
from keras import backend as K
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Cropping2D
from keras.optimizers import Adam


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

    images = []
    angles = []
    with open(args.driving_log) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            center_image_file = os.path.join(base_train_folder, line[0])
            center_image = cv2.imread(center_image_file)
            angle = float(line[3])
            images.append(center_image)
            angles.append(angle)

            left_image_file = os.path.join(base_train_folder, line[1])
            left_image = cv2.imread(left_image_file)
            images.append(left_image)
            angles.append(angle + 0.1)

            right_image_file = os.path.join(base_train_folder, line[2])
            right_image = cv2.imread(right_image_file)
            images.append(right_image)
            angles.append(angle - 0.1)

            mirrored_image = cv2.flip(center_image, 1)
            images.append(mirrored_image)
            angles.append(-1.0 * angle)

    x_train = np.array(images)
    y_train = np.array(angles)

    input_shape = (160, 320, 3)

    # NVIDIA
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(24, 5, 5,border_mode='valid', activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5,border_mode='valid', activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5,border_mode='valid', activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3,border_mode='valid', activation='elu', subsample=(1, 1)))
    model.add(Conv2D(64, 3, 3,border_mode='valid', activation='elu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1164, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.8))
    model.add(Dense(50, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mse', optimizer=Adam(lr=0.0001))

    model.fit(x_train, y_train, validation_split=0.2, shuffle=True, batch_size=128, nb_epoch=15)
    model.save(args.model_file)

    # workaround for https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()


if __name__ == '__main__':
    main()
