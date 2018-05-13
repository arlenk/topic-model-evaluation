"""
Load previously generated/downloaded corpora

"""
import pickle
from pathlib import Path
from gensim.corpora import MmCorpus, Dictionary
from .generate.simulated import ModelParameters


def load_corpus(corpus_name: str,
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
        raise ValueError("cannot find dictionary file for: {}, "
                         "looking for file: {}".format(corpus_name, dictionary_file))

    training_corpus = MmCorpus(str(training_corpus_file))
    validation_corpus = MmCorpus(str(validation_corpus_file))
    dictionary = Dictionary.load(str(dictionary_file))

    return training_corpus, validation_corpus, dictionary


def load_model_parameters(corpus_name: str,
                          data_path: str) -> (ModelParameters):
    """
    Load model parameters for a simulated corpus

    :param corpus_name: str
    :param data_path: str
    :return: ModelParameters
        true model parameters used for simulated corpus

    """

    data_path = Path(data_path)
    model_parameters_file = data_path / "{}.model_parameters.dat".format(corpus_name)

    if not model_parameters_file.exists():
        raise ValueError("cannot find model parameter file for: {}, "
                         "looking for file: {}".format(corpus_name, model_parameters_file))

    model_parameters = pickle.load(open(str(model_parameters_file), 'rb'))

    return model_parameters

