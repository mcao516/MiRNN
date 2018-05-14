from model.data_utils import getDataSet
from model.miRNA_model import Model
from model.config import Config

def main():
    # create instance of config
    config = Config()

    # build model
    model = Model(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    train, dev = getDataSet('data/train/pos_sample.txt', 
        'data/train/neg_sample.txt')

    # train model
    model.train(train, dev)
    model.evaluate(dev)


if __name__ == "__main__":
    main()
