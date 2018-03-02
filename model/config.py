import os

from .utils import get_logger

class Config():
    def __init__(self):
        """Initialize hyperparameters of the model

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

    # general config
    dir_output = "results/train/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # training
    nepochs          = 20
    dropout          = 0.5
    batch_size       = 256
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.95
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 30

    # model hyperparameters
    feature_size = 4
    class_num = 2
    layer_num = 2
    hidden_size = 200
    max_length = 200
