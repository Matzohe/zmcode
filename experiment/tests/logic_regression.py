import matplotlib.pyplot as plt
from src.NLP.LogicRegression import LogisticRegression
from src.utils.DataFilePreprocess.HotelToPickle import generate_and_save_data
from src.utils.utils import INIconfig
import pickle
import os
from torch.utils.data import DataLoader


def draw_loss(train_losses, test_losses):
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, 'r', label='Train Loss')
    plt.plot(epochs, test_losses, 'b', label='Test Loss')
    plt.legend()

    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def logic_regression_test():

    config = INIconfig('config.cfg')
    generate_and_save_data(config)
    train_set_path = os.path.join(config.DATASET["hotel_emotion"], "train_set.pkl")
    test_set_path = os.path.join(config.DATASET["hotel_emotion"], "test_set.pkl")
    BATCH_SIZE = int(config.HOTEL['batch_size'])
    EPOCHS = int(config.HOTEL['epoch'])

    with open(train_set_path, 'rb') as f:
        train_set = pickle.load(f)

    with open(test_set_path, 'rb') as f:
        test_set = pickle.load(f)
    train_iter = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_iter = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model = LogisticRegression(config)
    train_losses, test_losses = model.train(train_iter, test_iter)
    accuracy = model.test_accuracy(test_iter)
    print("accuracy:", accuracy, "%")
    draw_loss(train_losses, test_losses)