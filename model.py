import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense, Dropout
from random import shuffle
from pathlib import Path
from os import path
import matplotlib.pyplot as plt


class TrainingDataRecord:
    def __init__(self, image, steering_angle, flip):
        self.image = image
        self.steering_angle = steering_angle
        self.flip = flip


def parse_log(log_file, center_correction_factor, left_correction_factor=None, right_correction_factor=None):
    samples = []
    log_file_path = Path(log_file)
    dir_name = path.dirname(log_file)
    with open(log_file_path) as csv_file:
        reader = csv.DictReader(csv_file)
        for record in reader:
            samples.append(
                TrainingDataRecord(
                    path.join(dir_name, record['center']),
                    float(record['steering']) + center_correction_factor,
                    False))
            samples.append(
                TrainingDataRecord(
                    path.join(dir_name, record['center']),
                    - (float(record['steering']) + center_correction_factor),
                    True))
            if left_correction_factor:
                steering_angle = min(1, float(record['steering']) + left_correction_factor)
                samples.append(
                    TrainingDataRecord(
                        path.join(dir_name, record['left']),
                        steering_angle,
                        False))
                samples.append(
                    TrainingDataRecord(
                        path.join(dir_name, record['left']),
                        -steering_angle,
                        True))
            if right_correction_factor:
                steering_angle = max(-1, float(record['steering']) + right_correction_factor)
                samples.append(
                    TrainingDataRecord(
                        path.join(dir_name, record['right']),
                        steering_angle,
                        False))
                samples.append(
                    TrainingDataRecord(
                        path.join(dir_name, record['right']),
                        -steering_angle,
                        True))
    return samples


def prune_samples(samples):
    bins = 20
    steering_samples = [x.steering_angle for x in samples]
    hist, bin_edges = np.histogram(steering_samples, bins=bins)
    size = len(samples)
    avg_samples_per_bin = size/bins
    new_samples = []
    for sample in samples:
        for idx in range(len(bin_edges) - 1):
            if bin_edges[idx] <= sample.steering_angle  < bin_edges[idx + 1]:
                if hist[idx] <= avg_samples_per_bin:
                    new_samples = np.append(new_samples, sample)
                else:
                    chance_to_keep = avg_samples_per_bin / hist[idx]
                    if np.random.random_sample() < chance_to_keep:
                        new_samples = np.append(new_samples, sample)
                continue
    return new_samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:    # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample.image
                image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                if batch_sample.flip:
                    # Horizontal flip
                    image = cv2.flip(image, 1)
                steering_angle = float(batch_sample.steering_angle)
                images.append(image)
                angles.append(steering_angle)

            x_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(x_train, y_train)


def build_model(drop_off_keep_rate=None, l2_reg_rate=0.001):
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Cropping2D(cropping=((80, 25), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(55, 320, 3)))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_rate)))
    if drop_off_keep_rate:
        model.add(Dropout(drop_off_keep_rate))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_rate)))
    if drop_off_keep_rate:
        model.add(Dropout(drop_off_keep_rate))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_rate)))
    if drop_off_keep_rate:
        model.add(Dropout(drop_off_keep_rate))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg_rate)))
    model.add(Flatten())
    if drop_off_keep_rate:
        model.add(Dropout(drop_off_keep_rate))
    model.add(Dense(100, kernel_regularizer=regularizers.l2(l2_reg_rate)))
    if drop_off_keep_rate:
        model.add(Dropout(drop_off_keep_rate))
    model.add(Dense(50, kernel_regularizer=regularizers.l2(l2_reg_rate)))
    if drop_off_keep_rate:
        model.add(Dropout(drop_off_keep_rate))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def parse_all_data():
    middle_samples = parse_log(
        "./data/middle/driving_log.csv",
        center_correction_factor=0,
        left_correction_factor=0.10,
        right_correction_factor=-0.10)
    left_samples = parse_log("./data/left2/driving_log.csv", center_correction_factor=0.25, right_correction_factor=None)
    right_samples = parse_log("./data/right2/driving_log.csv", center_correction_factor=-0.25, left_correction_factor=None)
    turns_samples = parse_log(
        "./data/turns/driving_log.csv",
        center_correction_factor=0,
        left_correction_factor=0.10,
        right_correction_factor=-0.10)
    mountains_samples = parse_log(
        "./data/mountains/driving_log.csv",
        center_correction_factor=0,
        left_correction_factor=0.10,
        right_correction_factor=-0.10)
    mountains_reverse_samples = parse_log(
        "./data/mountains-reverse/driving_log.csv",
        center_correction_factor=0,
        left_correction_factor=0.10,
        right_correction_factor=-0.10)
    mountains_turns_samples = parse_log(
        "./data/mountains-turns/driving_log.csv",
        center_correction_factor=0,
        left_correction_factor=0.10,
        right_correction_factor=-0.10)
    mountains_turns_samples2 = parse_log(
        "./data/mountains-turns2/driving_log.csv",
        center_correction_factor=0,
        left_correction_factor=0.10,
        right_correction_factor=-0.10)
    all_samples = middle_samples + turns_samples + mountains_samples + mountains_reverse_samples + left_samples + \
                  right_samples + mountains_turns_samples + mountains_turns_samples2
    return all_samples


BATCH_SIZE = 256
all_samples = parse_all_data()
steering_samples = [x.steering_angle for x in all_samples]
plt.interactive(False)
plt.figure()
plt.hist(steering_samples, bins=20)
#plt.show(block=True)
plt.title("Histogram of original data")
plt.savefig("OrigDataHistogram.jpg")

pruned_samples = prune_samples(all_samples)
steering_samples = [x.steering_angle for x in pruned_samples]
plt.figure()
plt.hist(steering_samples, bins=20)
#plt.show(block=True)
plt.title("Histogram of re-balanced data")
plt.savefig("RebalancedDataHistogram.jpg")

train_samples, validation_samples = train_test_split(pruned_samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

model = build_model()

checkpoint_callback = ModelCheckpoint('checkpoints/weights.{epoch}-{val_loss:.3f}.h5', save_weights_only=True)

adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_samples) / BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=len(validation_samples) / BATCH_SIZE,
    nb_epoch=7,
    callbacks=[checkpoint_callback],
    verbose=1)

model.save('model-v17.h5')
print(history.history.keys())
print('Loss on the training set:')
print(history.history['loss'])
print('Loss on the validation set:')
print(history.history['val_loss'])

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("Loss.jpg")