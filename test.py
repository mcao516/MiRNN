from model.data_utils import minibatches, getDataSet
import numpy as np

def main():

    # create datasets
    train, dev = getDataSet('data/train/hairpin.txt', 'data/train/mrna.txt')

    # train model
    batch_size = 32

    (inputs, labels) = minibatches(train, batch_size)
    
    # inputs: (32, None, 4)
    print(labels)

if __name__ == "__main__":
    main()
