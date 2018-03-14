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
                              alpha: float = .1, beta: float = .001,
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

    # make document lengths follow a "reasonable" distribution
    # based on a gamma function fit from nytimes dataset
    gamma_parameters = (5.4096036273853478, -52.666540545843134, 71.072370010304383)
    min_document_length = 10

    for idocument in range(num_documents):
        document_length = max(int(ss.gamma.rvs(*gamma_parameters)),
                              min_document_length)

        document_topic_distribution = model_parameters.theta[idocument]

        # topic for each word in document
        document_words_per_topics = rs.multinomial(document_length,
                                                   document_topic_distribution,)

        # document_words_per_topics looks like
        # [5, 1, 0, 0, 0, 8, 0, 0],
        # ie, 5 words in topic 0, 1 word in topic 1, and 8 words in topic 5

        document_word_counts = Counter()
        for topic, topic_word_count in enumerate(document_words_per_topics):
            if topic_word_count:
                topic_word_distribution = model_parameters.phi[topic]
                word_counts = rs.multinomial(topic_word_count, topic_word_distribution)

                for word in np.flatnonzero(word_counts):
                    document_word_counts[word] += word_counts[word]

        yield list((k, v) for (k, v) in document_word_counts.items())


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
            if idocument % 5000 == 0:
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
            if idocument % 5000 == 0:
                print("{}: validation document {}".format(datetime.now(), idocument + 1))

            term_counts = next(document_term_counts)
            for term, count in term_counts:
                f.write("{} {} {}\n".format(idocument + 1, term, count))
                num_non_zero += count
        _write_headers(f, num_documents_validation, num_vocab, num_non_zero)
