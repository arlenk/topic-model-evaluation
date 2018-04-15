"""
Load previously generated/downloaded corpora

"""
from pathlib import Path
from gensim.corpora import MmCorpus


def load(corpus_name: str,
         data_path : Path) -> MmCorpus:
    """
    Load corpus by name

    :param corpus_name: str
    :param data_path: Path
    :return: Corpus
    """
    corpus_file = data_path / "{}.mm".format(corpus_name)
    if not corpus_file.exists():
        raise ValueError("unknown corpus: {}, "
                         "looking for file: {}".format(corpus_name, corpus_file))

    corpus = MmCorpus(str(corpus_file))
    return corpus

