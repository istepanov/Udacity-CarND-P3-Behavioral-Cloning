# How to run:
#
# docker run -it --rm -v `pwd`:/src udacity/carnd-term1-starter-kit python model.py train/driving_log.csv model.h5
#

import argparse
import os
import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense


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

    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, validation_split=0.2, shuffle=True)

    model.save(args.model_file)


if __name__ == '__main__':
    main()
