"""
Corpus related metrics

"""
from collections import Counter

import numpy as np
from gensim.interfaces import CorpusABC


def term_count_per_document(corpus: CorpusABC) -> np.ndarray:
    """
    Measure distribution of number of terms (unique words) per document

    :param corpus: corpus to measure metric on
    :return: number of terms per document

    """
    number_of_terms = []

    for doc in corpus:
        number_of_terms.append(len(doc))

    number_of_terms = np.array(number_of_terms)
    return number_of_terms


def word_count_per_document(corpus: CorpusABC) -> np.ndarray:
    """
    Measure distribution of number of words per document

    :param corpus: corpus to measure metric on
    :return: number of words per document

    """
    number_of_words = []

    for doc in corpus:
        document_word_count = 0
        for term, count in doc:
            document_word_count += count
        number_of_words.append(document_word_count)

    number_of_words = np.array(number_of_words)
    return number_of_words


def document_count_per_term(corpus: CorpusABC) -> np.ndarray:
    """
    Measure distribution of number of documents that each term appears in

    :param corpus: corpus to measure metric on
    :return: number of documents per term

    """
    number_of_documents_per_term = Counter()

    for doc in corpus:
        for term, count in doc:
            number_of_documents_per_term[term] += count

    number_of_documents_per_term = np.array(list(number_of_documents_per_term.values()))

    return number_of_documents_per_term
