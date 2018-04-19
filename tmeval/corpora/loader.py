"""
Load previously generated/downloaded corpora

"""
from pathlib import Path
from gensim.corpora import MmCorpus, Dictionary


def load(corpus_name: str,
         data_path: Path) -> (MmCorpus, Dictionary):
    """
    Load corpus by name

    :param corpus_name: str
    :param data_path: Path
    :return: (MmCorpus, Dictionary)

    """
    corpus_file = data_path / "{}.mm".format(corpus_name)
    dictionary_file = data_path / "{}.dictionary".format(corpus_name)

    if not corpus_file.exists():
        raise ValueError("unknown corpus: {}, "
                         "looking for file: {}".format(corpus_name, corpus_file))

    if not dictionary_file.exists():
        raise ValueError("unknown dictionary: {}, "
                         "looking for file: {}".format(dictionary_name, dictionary_file))

    corpus = MmCorpus(str(corpus_file))
    dictionary = Dictionary.load(str(dictionary_file))

    return corpus, dictionary

