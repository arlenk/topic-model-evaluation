"""
Datasets from UCI

https://archive.ics.uci.edu/ml/datasets.html

"""
import urllib
import os
import gensim.corpora as gc
import copy
from gensim.models import VocabTransform


def download_corpus(corpus_name, target_path):
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

    corpus = gc.UciCorpus(docword_file, vocab_file)
    return corpus


def create_corpus(corpus_name, target_path):
    corpus = download_corpus(corpus_name, target_path)

    print("dropping top/bottom words in dictionary")

    # filter the dictionary
    old_dict = corpus.create_dictionary()
    new_dict = copy.deepcopy(old_dict)
    new_dict.filter_extremes(no_below=20, no_above=.1)
    filtered_dictionary_file = os.path.join(target_path, "{}.filtered.dictionary".format(corpus_name))
    new_dict.save(filtered_dictionary_file)

    # now transform the corpus
    old2new = {old_dict.token2id[token]: new_id for new_id, token in new_dict.iteritems()}
    vt = VocabTransform(old2new)
    filtered_corpus_file = os.path.join(target_path, "{}.filtered.mm".format(corpus_name))
    gc.MmCorpus.serialize(filtered_corpus_file, vt[corpus], id2word=new_dict)


