
from collections import Counter, defaultdict
from math import sqrt
import random
import csv
import json


def memoize(distance_fnc):
    """caches previously calculated distances"""

    distances = {}

    def fnc(vector0, vector1):

        displacement = []
        for (point0, point1) in zip(vector0, vector1):

            # hashable pair
            points = frozenset((point0, point1))

            if points not in distances:
                distances[points] = (point1 - point0)**2

            displacement.append(distances[points])

        # euclidean summation
        return sqrt(sum(displacement))

    return fnc


@memoize
def distance(vector0, vector1):
    '''returns the Euclidean distance between two vectors'''

    square_difference = [(final - initial)**2 for initial,
                         final in zip(vector0, vector1)]
    return sqrt(sum(square_difference))


def majority_vote(k_nearest_neighbors):
    '''most frequent neighbor wins'''

    counter = Counter([category for (dist, category) in k_nearest_neighbors])
    most_common = counter.most_common(1)[0]

    category = most_common[0]
    confidence = most_common[1] / k
    return (k_nearest_neighbors, category, confidence)


def weighted_vote(k_nearest_neighbors):
    """average of closest neighbors win"""

    neighbor = defaultdict(list)
    for (dist, category) in k_nearest_neighbors:
        # give closer neighbors larger weight
        neighbor[category].append(1 / dist)

    sums = [(sum(neighbor[category]), category) for category in neighbor]
    result = max(sums)

    category = result[1]
    confidence = result[0] / sum([total for total, _ in sums])

    return (k_nearest_neighbors, category, confidence)


def classify(sample, query, k, weighted):
    '''weighting the classification by distance as well as frequency
    will help normalize a skewed class distribution in the sample data'''

    neighbors = []

    for category in sample:
        for vector in sample[category]:

            dist = (distance(query, vector))
            if dist == 0:  # query is in training set
                return ([(0.0, category)], category, 1.0)

            neighbors.append((dist, category))

    k_nearest_neighbors = sorted(neighbors)[:k]

    if weighted:
        return weighted_vote(k_nearest_neighbors)
    else:
        return majority_vote(k_nearest_neighbors)


if __name__ == '__main__':

    with open('./config.json') as configuration:
        config = json.load(configuration)

        k = config['neighbors']
        skewed = config['skewed']

        group = config['class']
        exclude = config['exclude']

        exclude.append(group)

        sample_size = config['sample']
        distribution = config['test']

    with open(config['input']) as data:
        reader = csv.reader(data)

        train_set = defaultdict(list)
        test_set = defaultdict(list)

        train_len = 0
        test_len = 0

        for i, row in enumerate(reader):

            if i == sample_size:
                break

            # filter out unwanted features
            features = [float(col) for index, col
                        in enumerate(row) if index not in exclude]

            # distribute data
            r = random.random()
            if r < distribution:
                test_set[row[group]].append(features)
                test_len += 1
            else:
                train_set[row[group]].append(features)
                train_len += 1

        print('{} data points in training set..\n'.format(train_len))
        print('training distribution:')

        for key in train_set:
            print('{} data points in class {}..'
                  .format(len(train_set[key]), key))

        results = []
        for category in test_set:
            for feature in test_set[category]:

                (neighbors, result, confidence) = classify(
                    train_set, feature, k, skewed)

                results.append((category, feature, neighbors,
                                result, confidence, category == result))

        print('\n{} data points in testing set..'.format(test_len))

        passed = sum([1 for (_, _, _, _, _, grade) in results if grade])
        print('{} data points classified correctly..'.format(passed))
        print('\naccurancy: {0:.2f}%'.format(100 * (passed / test_len)))

        print('\nwriting results to output.csv..')

    with open('output.csv', 'w') as output:

        headers = ['Expected Class', 'Features',
                   'K Nearest Neighbors',
                   'Actual Class', 'Confidence', 'Pass']

        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow({headers[i]: row[i]
                             for i in range(0, len(headers))})
