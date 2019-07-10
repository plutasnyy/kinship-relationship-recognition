import pandas as pd
import cv2
import glob
import numpy as np

from keras.models import Sequential
from keras.layers import Lambda, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Input, concatenate

from random import sample
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DATA_PATH = '/content/kinship-relationship-recognition/data/'

RELATIONSHIPS_PATH = DATA_PATH + 'train_relationships.csv'
IMAGE_EXPRESSION = DATA_PATH + 'train/{}/*'
PEOPLE_NAMES_EXPRESSION = DATA_PATH + 'train/*/*'
TRAIN_PATH = DATA_PATH + 'train/'

validate = 0.2
HALF_DATA_SET_SIZE = 1000
RESIZE_TO = 32


def preprocess_image(person_directory_path, i):
    paths = glob.glob(person_directory_path)
    image_path = sample(paths, 1)[0]

    # TODO we should reading and preprocessing every image every time, instead we should remeber preprecossed images
    gray_image = cv2.imread(image_path, 0)
    gray_image = cv2.resize(gray_image, dsize=(RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_CUBIC)

    # TODO do some preprocess, contrast, exposure etc
    norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.expand_dims(norm_image, axis=2)


def is_pair_in_relation(relations, pair):
    p1, p2 = pair
    if p1 == p2:
        return True
    return relations.query('p1=="{}" and p2=="{}" or p1=="{}" and p2=="{}"'.format(p1, p2, p2, p1))['p1'].count() != 0


def clean_relations_batch(relations_batch, people_names):
    relations = relations_batch.values
    result = dict()
    result['p1'], result['p2'] = list(), list()
    for p1, p2 in relations:
        # TODO instead of pd -> numpy -> pd filter on data frame
        if p1 in people_names and p2 in people_names:
            result['p1'].append(p1)
            result['p2'].append(p2)
    return pd.DataFrame.from_dict(result)


def construct_batch(relations_list):
    pairs = list()
    Y = list()
    for i, relations in enumerate(relations_list):
        for left_image_name, right_image_name in relations:
            left = preprocess_image(IMAGE_EXPRESSION.format(left_image_name), i)
            right = preprocess_image(IMAGE_EXPRESSION.format(right_image_name), i)
            pairs += [[left, right]]
            Y += [i]

    return pairs, Y


relations_df = pd.read_csv(RELATIONSHIPS_PATH)

names = glob.glob(PEOPLE_NAMES_EXPRESSION)
people_names = [name.replace(TRAIN_PATH, '') for name in names]

relations_df = clean_relations_batch(relations_df, people_names)
relations_batch = relations_df.sample(HALF_DATA_SET_SIZE, replace=True).values.tolist()

not_relations_batch = list()

while len(not_relations_batch) < HALF_DATA_SET_SIZE:
    random_pair = sample(people_names, 2)
    if not is_pair_in_relation(relations_df, random_pair):
        not_relations_batch.append(random_pair)

pairs, Y = construct_batch([not_relations_batch, relations_batch])

X = np.array(pairs)
y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validate)

from keras.optimizers import Adam


def initialize_weights(shape, name=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, name=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(input_shape):
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                     kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (2, 2), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                    kernel_regularizer=l2(1e-3),
                    kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net


import tensorflow as tf

# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allocator_type = 'BFC'
# # config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.95

model = get_siamese_model(X_train[0][0].shape)
optimizer = Adam(lr=0.0001)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

# model.fit([X_train[:,0],X_train[:,1]],y_train, epochs=10)
model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, validation_data=([X_test[:, 0], X_test[:, 1]], y_test))