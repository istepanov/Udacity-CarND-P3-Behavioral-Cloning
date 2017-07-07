# This is a helper script that converts absolute paths in the driving log file
# into paths relative to the driving log parent folder,
# so the training data can be easily moved between different machines.

import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'driving_log',
        type=str,
        default='',
        help='Path to input driving log csv file.'
    )
    args = parser.parse_args()

    data = []
    with open(args.driving_log) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            data.append(line)

    with open(args.driving_log, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in data:
            for i in range(0, 3):
                if os.path.isabs(line[i]):
                    common_path = os.path.commonpath([args.driving_log, line[i]])
                    line[i] = os.path.relpath(line[i], common_path)
            writer.writerow(line)

    print('Done.')


if __name__ == '__main__':
    main()
