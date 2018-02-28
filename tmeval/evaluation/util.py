import scipy.optimize as so
import numpy as np


def align_topics(labels_a, labels_b, num_topics):
    """
    Align topic labels from two different topic models

    :param labels_a: list[(topic_number, weight)]
    :param labels_b: list[(topic_number, weight)]
    :param num_topics: int
    :return:
        aligned_a: list[(topic_number, weight)]
        aligned_b: list[(topic_number, weight)]
        topic_map: dict(label_a -> label_b)

    """

    # convert topics to binary array
    A = np.zeros((len(labels_a), num_topics))
    B = np.zeros((len(labels_b), num_topics))

    for idoc, row in enumerate(labels_a):
        for topic, weight in row:
            A[idoc, topic] = weight

    for idoc, row in enumerate(labels_b):
        for topic, weight in row:
            B[idoc, topic] = weight

    # find best match between labels
    # http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html
    # note: flip to a cost
    # note: A'B will give mapping of A->B
    cost = -A.T.dot(B)

    res = so.linear_sum_assignment(cost)
    labels_from_a, labels_from_b = res
    topic_map = dict(zip(labels_from_b, labels_from_a))

    aligned_a = []
    aligned_b = []

    for doc in labels_a:
        row = []
        for topic, weight in doc:
            row.append((topic, weight))
        aligned_a.append(row)

    for doc in labels_b:
        row = []
        for topic, weight in doc:
            row.append((topic_map[topic], weight))
        aligned_b.append(row)

    return aligned_a, aligned_b, topic_map


def topic_list_to_array(topic_list, num_topics):
    """
    Convert list of (topic, weight) tuples to "full" array

    :param num_topics:
    :param topic_list:
    :return:
    """

    A = np.zeros((len(topic_list), num_topics))

    for idoc, row in enumerate(topic_list):
        for topic, weight in row:
            A[idoc, topic] = weight

    return A

