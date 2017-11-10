from math import sqrt
from collections import Counter, defaultdict
import csv


def distance(vector0, vector1):
    '''returns the Euclidean distance between two vectors'''
    square_difference = [(final - initial)**2 for initial,
                         final in zip(vector0, vector1)]
    return sqrt(sum(square_difference))


def most_frequent(neighbors, summation=True):

    if summation:
        frequency = {category: [] for dist, category in neighbors}
        for dist, category in neighbors:
            frequency[category].append(dist)

        sums = [(sum(frequency[category]), category) for category in frequency]
        return max(sums)[1]

    else:
        counter = Counter([category for dist, category in neighbors])
        return counter.most_common(1)[0][0]

# { 'class1': [[data_point1], [data_point2]], 'class2': [[data_point1], ... }


def nearest_neighbors(sample, query, k=5, weighted=True):
    '''weighting the classification by distance as well as frequency will help normalize a skewed class distribution in the sample data'''

    neighbors = []

    for category in sample:
        for data_point in sample[category]:
            dist = distance(query, data_point)
            if weighted:
                if dist == 0:  # query feature exists in training sample
                    return category
                dist = 1.0 / dist
            neighbors.append((dist, category))

    k_nearest_neighbors = sorted(neighbors)[:k]
    return most_frequent(k_nearest_neighbors, weighted)


if __name__ == '__main__':
    with open('test.csv') as data:
        reader = csv.reader(data)
        train = defaultdict(list)
        test = defaultdict(list)
        i = 1
        for row in reader:
            if i % 3 == 0:  # 33% of sample goes to training
                train[row[-1]].append([int(val) for val in row[:-1]])
            else:
                test[row[-1]].append([int(val) for val in row[:-1]])
            i = i + 1
        train = dict(train)
        print(len(train['1']), len(train['2']))
        results = []
        for category in test:
            for feature in test[category]:
                result = nearest_neighbors(train, feature, 5, False)
                results.append((category, feature, result, result == category))
        print(results)
