"""
Corpus related metrics

"""
import numpy as np
from gensim.interfaces import CorpusABC


def words_per_document(corpus: CorpusABC) -> np.ndarray:
    """
    Measure distribution of number of words per document

    :param corpus: corpus to measure metric on
    :return: number of words per document

    """
    number_of_words = []

    for doc in corpus:
        number_of_words.append(len(doc))

    number_of_words = np.array(number_of_words)
    return number_of_words
