"""
Datasets from UCI

https://archive.ics.uci.edu/ml/datasets.html

"""
import urllib
import os
from itertools import islice

import copy
from gensim.models import VocabTransform
from gensim.corpora.dictionary import Dictionary
from gensim.corpora import MmCorpus, UciCorpus
from gensim.interfaces import CorpusABC


def generate_mmcorpus_files(corpus_name: str,
                            target_path: str,
                            training_pct: float = .8):
    """
    Output training and validation MM corpus files
    :param corpus_name:
    :param target_path:
    :param training_pct:
    :return:

    """
    corpus = download_corpus(corpus_name, target_path)
    print("downloaded {} corpus [num_docs={}, num_terms={}]".format(corpus_name,
                                                                    corpus.num_docs,
                                                                    corpus.num_terms))

    print("dropping top/bottom words in dictionary")
    corpus, dictionary = filter_corpus(corpus)

    # output mm files
    num_documents = len(corpus)
    num_documents_training = int(training_pct * num_documents)
    num_documents_validation = num_documents - num_documents_training
    num_vocab = len(dictionary)

    print("outputting")
    print("  num documents: {:,.0f}".format(num_documents))
    print("  num training: {:,.0f}".format(num_documents_training))
    print("  num validation: {:,.0f}".format(num_documents_validation))
    print("  vocab size: {:,.0f}".format(num_vocab))

    # output same dictionary for training and validation
    output_prefix = os.path.join(target_path, "{}.filtered".format(corpus_name))
    dictionary_file = output_prefix + ".dictionary"
    dictionary.save(dictionary_file)

    # training data
    output_file = output_prefix + ".training.mm"
    training_corpus = islice(corpus, num_documents_training)
    MmCorpus.serialize(output_file, training_corpus, dictionary)

    # validation
    output_file = output_prefix + ".validation.mm"
    validation_corpus = islice(corpus, num_documents_training, num_documents)
    MmCorpus.serialize(output_file, validation_corpus, dictionary)


def download_corpus(corpus_name: str,
                    target_path: str) -> UciCorpus:
    """
    Download corpus from UCI website

    :param corpus_name:
    :param target_path:
    :return:
    """

    url_root = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"
    target_path = os.path.join(target_path, "uci", "raw")
    if not os.path.exists(target_path):
        print("creating target path: {}".format(target_path))
        os.makedirs(target_path)

    vocab_file = os.path.join(target_path, "vocab.{}.txt".format(corpus_name))
    print("downloading {} vocab file to: {}".format(corpus_name, vocab_file))
    urllib.request.urlretrieve(url_root + "vocab.{}.txt".format(corpus_name),
                               filename=vocab_file)

    docword_file = os.path.join(target_path, "docword.{}.txt.gz".format(corpus_name))
    print("downloading {} bag of words to: {}".format(corpus_name, docword_file))
    urllib.request.urlretrieve(url_root + "docword.{}.txt.gz".format(corpus_name),
                               filename=docword_file)

    corpus = UciCorpus(docword_file, vocab_file)
    return corpus


def filter_corpus(corpus: UciCorpus) -> (CorpusABC, Dictionary):
    """
    Filter extreme (frequent and infrequent) words from dictionary

    :param corpus:
    :return: (filtered corpus, filtered dictionary)
    """

    # filter dictionary first
    original_dict = corpus.create_dictionary()
    filtered_dict = copy.deepcopy(original_dict)
    filtered_dict.filter_extremes(no_below=20, no_above=.1)

    # now transform the corpus
    old2new = {original_dict.token2id[token]: new_id for new_id, token in filtered_dict.iteritems()}
    vt = VocabTransform(old2new)

    return vt[corpus], filtered_dict


