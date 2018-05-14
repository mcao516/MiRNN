import os
from datetime import datetime
from .utils import get_logger

class Config():
    def __init__(self):
        """Initialize hyperparameters of the model

        """
        self.dir_output = "results/{}/{:%Y%m%d_%H%M%S}/".format('train', datetime.now())
        self.dir_model  = self.dir_output + "model.weights/"
        self.path_log   = self.dir_output + "log.txt"

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        
        # create instance of logger
        self.logger = get_logger(self.path_log)

    # general config
    # dir_output = "results/test/"

    # training
    nepochs          = 20
    dropout          = 0.5
    batch_size       = 32
    lr_method        = "adam"
    lr               = 0.003
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 5

    # model hyperparameters
    class_num = 2
    layer_num = 2
    hidden_size = 30
    max_length  = 100
    n_window_size = 1
    s_window_size = 1

    nucle_type_num  = 4
    pair_status_num = 2
    nucle_embedding_size = 4
    struc_embedding_size = 4

