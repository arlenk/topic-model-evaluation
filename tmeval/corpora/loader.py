"""
Load previously generated/downloaded corpora

"""
from pathlib import Path
from gensim.corpora import MmCorpus, Dictionary


def load(corpus_name: str,
         data_path: str) -> (MmCorpus, MmCorpus, Dictionary):
    """
    Load corpus by name

    :param corpus_name: str
    :param data_path: str
    :return: (MmCorpus, MmCorpus, Dictionary)
        training corpus, validation corpus, and dictionary

    """
    data_path = Path(data_path)
    training_corpus_file = data_path / "{}.training.mm".format(corpus_name)
    validation_corpus_file = data_path / "{}.validation.mm".format(corpus_name)
    dictionary_file = data_path / "{}.dictionary".format(corpus_name)

    if not training_corpus_file.exists():
        raise ValueError("cannot find training file for: {}, "
                         "looking for file: {}".format(corpus_name, training_corpus_file))

    if not validation_corpus_file.exists():
        raise ValueError("cannot find validation file for: {}, "
                         "looking for file: {}".format(corpus_name, validation_corpus_file))

    if not dictionary_file.exists():
        raise ValueError("unknown dictionary: {}, "
                         "looking for file: {}".format(dictionary_name, dictionary_file))

    training_corpus = MmCorpus(str(training_corpus_file))
    validation_corpus = MmCorpus(str(validation_corpus_file))
    dictionary = Dictionary.load(str(dictionary_file))

    return training_corpus, validation_corpus, dictionary

