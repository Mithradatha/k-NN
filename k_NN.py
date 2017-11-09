from math import sqrt
from collections import Counter


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

        print(frequency)

        print([(key, sum(frequency[key])) for key in frequency])
        sums = [(sum(frequency[category]), category) for category in frequency]
        return max(sums)[1]

    else:
        counter = Counter([category for dist, category in neighbors])
        print(counter)
        return counter.most_common(1)[0][0]

# { 'class1': [[data_point1], [data_point2]], 'class2': [[data_point1], ... }


def nearest_neighbors(sample, query, k=5, weighted=True):
    '''weighting the classification by distance as well as frequency will help normalize a skewed class distribution in the sample data'''

    neighbors = []

    for category in sample:
        for data_point in sample[category]:
            dist = distance(query, data_point)
            print((category, dist))
            if weighted:
                dist = 1.0 / dist
            neighbors.append((dist, category))

    k_nearest_neighbors = sorted(neighbors)[:k]
    print(k_nearest_neighbors)
    return most_frequent(k_nearest_neighbors, weighted)


if __name__ == '__main__':
    data_set = {2: [[3, 5, 9, 4], [5, 4, 3, 9]], 4: [[2, 5, 9, 5]]}
    unknown = [3, 5, 9, 6]

    # if len(data_set.items) != len(unknown):
    #     raise ValueError('Feature vectors should be the same length')

    print(nearest_neighbors(data_set, unknown, 3, True))
