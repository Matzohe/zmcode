from src.utils.utils import INIconfig
from src.NLP.NaiveBayes import NaiveBayes
from src.utils.DataFilePreprocess.HotelPreprocess import emotional_data_preprocess


def naiveBayesTest():
    config = INIconfig("config.cfg")
    nb = NaiveBayes(config)
    nb.predict()