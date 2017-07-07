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
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, BatchNormalization


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
            image_file = os.path.join(base_train_folder, line[0])
            image = cv2.imread(image_file)
            images.append(image)
            angles.append(float(line[3]))

    x_train = np.array(images)
    y_train = np.array(angles)

    input_shape = (160, 320, 3)

    # nVidia
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    # model.add(BatchNormalization(epsilon=0.001, mode=2, axis=1, input_shape=input_shape))
    model.add(Conv2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Conv2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Conv2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, validation_split=0.2, shuffle=True, batch_size=128, nb_epoch=15)

    model.save(args.model_file)


if __name__ == '__main__':
    main()
