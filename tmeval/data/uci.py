"""
Datasets from UCI

https://archive.ics.uci.edu/ml/datasets.html

"""
import urllib
import os


def download_data(target_path, dataset="enron"):
    url_root = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/"

    outfile = os.path.join(target_path, "uci", "raw", "vocab.{}.txt".format(dataset))
    print("downloading {} vocab file to: {}".format(dataset, outfile))
    urllib.request.urlretrieve(url_root + "vocab.{}.txt".format(dataset),
                               filename=outfile)

    outfile = os.path.join(target_path, "uci", "raw", "vocab.{}.txt".format(dataset))
    print("downloading {} bag of words to: {}".format(dataset, outfile))
    urllib.request.urlretrieve(url_root + "docword.{}.txt.gz".format(dataset),
                               filename=outfile)
