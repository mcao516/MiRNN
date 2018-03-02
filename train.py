from model.data_utils import getDataSet
from model.prem_model import Model
from model.prem_model_old import Model as Simple_Model
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
    train, dev = getDataSet('data/train/hairpin.txt', 'data/train/mrna.txt')

    # train model
    model.train(train[:40], dev[:40])
    model.evaluate(train[:40] + dev[:40])

if __name__ == "__main__":
    main()
