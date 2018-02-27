"""
Create simulated LDA documents

"""

# M: number of documents
# K: number of topics
# V: number of words in vocab
# N: number of words in all documents

# theta: topic distribution over documents (M by K)
# phi: word distribution over topics (V by K) (lambda)

import numpy as np
import scipy.stats as ss
from collections import namedtuple, Counter

ModelParameters = namedtuple("ModelParameters", ["num_topics", "num_documents",
                                                 "num_vocab", "alpha", "beta",
                                                 "seed", "theta", "phi"])


def generate_model_parameters(num_topics, num_documents, num_vocab,
                              alpha=.1, beta=.0001, seed=None):
    """
    Generate parameters for LDA model

    :param num_topics:
    :param num_documents:
    :param num_vocab:
    :param alpha:
    :param beta:
    :param seed:
    :return:
    """
    rs = np.random.RandomState(seed)

    theta = rs.dirichlet(np.ones(num_topics) * alpha, num_documents)
    phi = rs.dirichlet(np.ones(num_vocab) * beta, num_topics)
    parameters = ModelParameters(alpha=alpha,
                                 beta=beta,
                                 num_topics=num_topics,
                                 num_documents=num_documents,
                                 num_vocab=num_vocab,
                                 theta=theta,
                                 phi=phi,
                                 seed=seed)

    return parameters


def generate_documents():
    pass


def generate_document_term_counts(model_parameters, seed=None):
    """
    Generate count of terms per document

    :param model_parameters:
    :param num_documents:
    :param outfile:
    :param seed:
    :return:
    """
    rs = np.random.RandomState(seed)
    num_documents = model_parameters.num_documents
    num_topics = model_parameters.num_topics
    num_vocab = model_parameters.num_vocab

    # make document lenghts follow a "reasonable" distribution
    # based on a gamma function fit from nytimes dataset
    gamma_parameters = (5.4096036273853478, -52.666540545843134, 71.072370010304383)
    min_document_length = 10
    document_lengths = ss.gamma.rvs(*gamma_parameters, size=num_documents)
    document_lengths[document_lengths < min_document_length] = min_document_length

    topic_array = np.arange(num_topics)
    vocab_array = np.arange(num_vocab)
    document_term_counts = []

    for idocument in range(num_documents):
        document_length = int(document_lengths[idocument])
        document_topic_distribution = model_parameters.theta[idocument]

        # topic for each word in document
        document_word_topics = rs.multinomial(1,
                                              document_topic_distribution,
                                              document_length
                                              )
        # document_word_topics is 0/1 matrix that looks like:
        # [[0, 1, 0, 0, 0, 0, 0, 0],  # topic 1
        #  [1, 0, 0, 0, 0, 0, 0, 0],  # topic 0
        #  [0, 0, 0, 0, 1, 0, 0, 0],  # topic 4
        # ...
        # we'll dot this with [ 0, 1, ... num_topics ] to convert
        # to document_word_topics to [ 1, 0, 4 ... ]
        document_word_topics = document_word_topics.dot(topic_array)

        counts = Counter()
        for iword, word_topic in enumerate(document_word_topics):
            topic_word_distribution = model_parameters.phi[word_topic]

            word_topic = rs.multinomial(1, topic_word_distribution)

            # as before (with document_word_topics), word_topic is a 0/1
            # array, so dot with [0, 1, 2, ... num_vocab] to convert to
            # an index into num_vocab
            word_index = word_topic.dot(vocab_array)
            counts[word_index] += 1

        yield list((k, v) for (k, v) in counts.items())
