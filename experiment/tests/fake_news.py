from src.utils.utils import INIconfig
from src.utils.DataLoader.fake_news import FakeNewsDataset
from src.NLP.BertClassification import FakeNewsBertClassification

def fakeNewsTest():
    model = FakeNewsBertClassification(INIconfig())
    dataset = FakeNewsDataset(INIconfig())