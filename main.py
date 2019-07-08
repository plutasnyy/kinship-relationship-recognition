import pandas as pd
import cv2
import glob
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Input, concatenate

from random import sample
from keras.models import Model

RELATIONSHIPS_PATH = 'data/train_relationships.csv'
IMAGE_EXPRESSION = 'data/train/{}/*'
PEOPLE_NAMES_EXPRESSION = 'data/train/*/*'


def preprocess_image(person_directory_path, i):
    paths = glob.glob(person_directory_path)
    image_path = sample(paths, 1)[0]

    # TODO we should reading and preprocessing every image every time, instead we should remeber preprecossed images
    gray_image = cv2.imread(image_path, 0)

    # TODO do some preprocess, contrast, exposure etc
    norm_image = cv2.normalize(gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.expand_dims(norm_image, axis=2)


def is_pair_in_relation(relations, pair):
    p1, p2 = pair
    if p1 == p2:
        return True
    return relations.query('p1=="{}" and p2=="{}" or p1=="{}" and p2=="{}"'.format(p1, p2, p2, p1))['p1'].count() != 0


def clean_relations_batch(relations_batch, people_names):
    relations = relations_batch.as_matrix()
    result = dict()
    result['p1'], result['p2'] = list(), list()
    for p1, p2 in relations:
        # TODO instead of pd -> numpy -> pd filter on data frame
        if p1 in people_names and p2 in people_names:
            result['p1'].append(p1)
            result['p2'].append(p2)
    return pd.DataFrame.from_dict(result)


relations_df = pd.read_csv(RELATIONSHIPS_PATH)

people_names = [name.replace('data/train/', '') for name in glob.glob(PEOPLE_NAMES_EXPRESSION)]

relations_df = clean_relations_batch(relations_df, people_names)
relations_batch = relations_df.sample(16).values.tolist()

not_relations_batch = list()

while len(not_relations_batch) < 16:
    random_pair = sample(people_names, 2)
    if not is_pair_in_relation(relations_df, random_pair):
        not_relations_batch.append(random_pair)


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


pairs, Y = construct_batch([not_relations_batch, relations_batch])

inp1 = Input((224, 224, 1))
inp2 = Input((224, 224, 1))

model = concatenate([inp1, inp2], axis=-1)
model = Conv2D(64, (3, 3))(model)
model = Activation('relu')(model)
model = Flatten()(model)

outputs = Dense(1, activation='softmax')(model)
model = Model([inp1, inp2], outputs)

model.compile(optimizer='sgd', loss='mse')

pairs = np.array(pairs)
Y = np.array(Y)

model.fit([pairs[:, 0], pairs[:, 1]], Y)
