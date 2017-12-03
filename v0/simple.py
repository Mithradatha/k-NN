from collections import Counter
from math import sqrt
import random
import csv
import json
import cProfile


def distance(vector0, vector1):
    """returns the Euclidean distance between two vectors"""

    square_difference = [(final - initial)**2 for initial,
                         final in zip(vector0, vector1)]

    return sqrt(sum(square_difference))


def majority_vote(k_nearest_neighbors):
    """most frequent neighbor wins"""

    counter = Counter([category for (dist, category) in k_nearest_neighbors])
    category = counter.most_common(1)[0][0]

    return category

# [ ([ features ], class) ]


def knn(train_set, query, k):
    """k nearest neighbors classification"""

    neighbors = [(distance(features, query), category)
                 for (features, category) in train_set]

    k_nearest_neighbors = sorted(neighbors)[:k]

    return majority_vote(k_nearest_neighbors)


if __name__ == '__main__':

    with open('./simple.json') as configuration:
        config = json.load(configuration)

        k = config['neighbors']

        train_len = config['train']
        test_len = config['test']

        sample_size = train_len + test_len

    with open(config['input']) as data:
        reader = csv.reader(data)

        train_set = []
        test_set = []

        for i, row in enumerate(reader):

            if i == sample_size:
                break

            # filter out unwanted features
            features = [float(col) for col in row[:-1]]

            # distribute data
            if i < train_len:
                train_set.append((features, row[-1]))
            else:
                test_set.append((features, row[-1]))

    total = len(test_set)
    passed = 0

    print('\nSample Size: {}\n'.format(sample_size))

   # print(timeit.timeit('knn(train_set, q, k) for q in test_set'))
    profile = cProfile.Profile()
    profile.enable()

    for query in test_set:
        category = knn(train_set, query[0], k)

        if category == query[1]:
            passed += 1
        else:
            print('Expected: {}, Actual: {}, Features: {}'
                  .format(query[1], category, query[0]))
    profile.disable()

    print('\nCorrectly Classified Entries: {}'.format(passed))
    print('Accurancy: {0:.2f}%\n'.format(100 * (passed / total)))

    profile.print_stats(sort='time')
