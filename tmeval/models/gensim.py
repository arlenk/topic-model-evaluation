import gensim.models.ldamodel as gml
from gensim.corpora import MmCorpus, Dictionary
import numpy as np
from typing import Optional


class GensimBatchLDA():
    def __init__(self,
                 dictionary: Dictionary,
                 num_topics: int,
                 passes: int = 20,
                 dtype: type = np.float32):
        """
        Gensim's implementation of a batch (not-online) LDA model

        :param dictionary: gensim Dictionary for both training and validation corpora
        :param num_topics: number of topics
        :param passes: number of passes through training data when fitting model
        :param dtype: fit model using *dtype* for data/calculations

        """

        self.model = gml.LdaModel(num_topics=num_topics, id2word=dictionary, passes=passes, dtype=dtype)
        self.dictionary = dictionary
        self.num_topics = num_topics
        self.passes = passes
        self.dtype = dtype

    def fit(self,
            corpus: MmCorpus,
            passes: Optional[int] = None):
        """
        Train model with corpus of documents

        :param corpus: training corpus
        :return:
        """
        passes = passes or self.passes
        self.model.update(corpus=corpus, passes=passes)

    def score(self,
              corpus: MmCorpus) -> float:
        """
        Evaluate model fit on corpus

        :param corpus: training corpus
        :return: float
        """

        score = self.model.log_perplexity(corpus)
        return score
    

class GensimOnlineLDA():
    def __init__(self, corpus: MmCorpus,
                 dictionary: Dictionary,
                 num_topics: int):
        self.model = gml.LdaModel(num_topics=num_topics, id2word=dictionary)
        self.dictionary = dictionary
        self.num_topics = num_topics
        self.corpus = corpus
