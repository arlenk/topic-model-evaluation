"""
Create simulated LDA documents

"""

# M: number of documents
# K: number of topics
# V: number of words in vocab
# N: number of words in all documents

# theta: topic distribution over documents (M by K)
# phi: word distribution over topics (V by K) (lambda)
import typing

import numpy as np
import scipy.stats as ss
from collections import namedtuple, Counter
from datetime import datetime

ModelParameters = typing.NamedTuple("ModelParameters",
                                    [("num_topics", int),
                                     ("num_documents", int),
                                     ("num_vocab", int),
                                     ("alpha", float),
                                     ("beta", float),
                                     ("seed", typing.Optional[int]),
                                     ("theta", np.ndarray),
                                     ("phi", np.ndarray)])


def generate_model_parameters(num_topics: int, num_documents: int, num_vocab: int,
                              alpha: float = .1, beta: float = .0001,
                              seed: typing.Optional[int] = None) -> ModelParameters:
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


def generate_document_term_counts(model_parameters: ModelParameters,
                                  seed: typing.Optional[int] = None):
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

    topic_array = np.arange(num_topics)
    vocab_array = np.arange(num_vocab)

    for idocument in range(num_documents):
        document_length = max(int(ss.gamma.rvs(*gamma_parameters)),
                              min_document_length)

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


def generate_mmcorpus_files(model_parameters: ModelParameters,
                            document_term_counts,
                            output_prefix: str,
                            training_pct: float = .8):
    """
    Output training and validate mm files

    :param model_parameters:
    :param document_term_counts:
    :param output_prefix:
    :param training_pct:
    :return:
    """

    num_documents = model_parameters.num_documents
    num_documents_training = int(training_pct * num_documents)
    num_documents_validation = num_documents - num_documents_training
    num_vocab = model_parameters.num_vocab

    print("outputting")
    print("  num documents: {:,.0f}".format(num_documents))
    print("  num training: {:,.0f}".format(num_documents_training))
    print("  num validaiton: {:,.0f}".format(num_documents_validation))

    def _write_headers(f, _num_documents=-1, _num_vocab=-1, _num_non_zero=-1):
        f.seek(0)
        f.write("%%MatrixMarket matrix coordinate real general\n")
        header = "{} {} {}".format(_num_documents, _num_vocab, _num_non_zero)
        header = header.ljust(50) + '\n'
        f.write(header)

    # training
    outfile = output_prefix + ".training.mm"
    with open(outfile, 'w') as f:
        _write_headers(f)
        num_non_zero = 0

        for idocument in range(num_documents_training):
            if idocument % 100 == 0:
                print("{}: training document {}".format(datetime.now(), idocument + 1))

            term_counts = next(document_term_counts)
            for term, count in term_counts:
                f.write("{} {} {}\n".format(idocument + 1, term, count))
                num_non_zero += count
        _write_headers(f, num_documents_training, num_vocab, num_non_zero)

    # validation
    outfile = output_prefix + ".validation.mm"
    with open(outfile, 'w') as f:
        _write_headers(f)
        num_non_zero = 0

        for idocument in range(num_documents_validation):
            if idocument % 100 == 0:
                print("{}: validation document {}".format(datetime.now(), idocument + 1))

            term_counts = next(document_term_counts)
            for term, count in term_counts:
                f.write("{} {} {}\n".format(idocument + 1, term, count))
                num_non_zero += count
        _write_headers(f, num_documents_validation, num_vocab, num_non_zero)
