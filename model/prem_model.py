import numpy as np
import os
import tensorflow as tf
import collections
import math

from .utils import Progbar
from .model import BaseModel
from .data_utils import minibatches, pad_sequence


class Model(BaseModel):
    """Specialized class of Model for micro RNA prediction"""

    def __init__(self, config):
        super(Model, self).__init__(config)


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # inputs: (batch_size, max_length=200, second_demension=4) (int)
        # labels: (batch_size, class_num=2) (int)
        
        print("adding placeholders into graph...")

        # inputs: [batch_size, max_length=200, second_demension=4]
        self.inputs_placeholder = tf.placeholder(tf.float32, 
            shape=[None, self.config.max_length, self.config.feature_size], 
            name="inputs_placeholder")
        
        # labels: [batch_size]
        self.labels_placeholder = tf.placeholder(tf.int32, 
            shape=[None], name="labels_placeholder")

        # shape = [batch size]
        # self.sequence_lengths = tf.placeholder(tf.int32, 
        #     shape=[None], name="sequence_lengths")
        
        # hyper parameters
        self.dropout_placeholder = tf.placeholder(tf.float32, 
            shape=[], name="dropout_placeholder")

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, 
            shape=[], name="lr_placeholder")

    def get_feed_dict(self, inputs_batch, labels_batch=None, lr=1e-3, dropout=1.0):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # padding input sequences
        inputs_batch = pad_sequence(inputs_batch, self.config.max_length)
        # print('shape test: '+str(np.asarray(inputs_batch).shape))
        
        # build feed dictionary
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.lr_placeholder : lr,
            self.dropout_placeholder: dropout
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_logits_op(self):
        """Defines self.logits, the core part of the model

        """
        print("adding logits operation into the graph...")

        def get_lstm_cell():
            lstmCell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
            lstmCell = tf.contrib.rnn.DropoutWrapper(lstmCell, 
                    output_keep_prob=self.dropout_placeholder)           
            return lstmCell

        with tf.variable_scope('rnn-cells'):
            if self.config.layer_num > 1:
                fw_cells = tf.contrib.rnn.MultiRNNCell(
                    [get_lstm_cell() for _ in range(self.config.layer_num)])
                bw_cells = tf.contrib.rnn.MultiRNNCell(
                    [get_lstm_cell() for _ in range(self.config.layer_num)])                    
            else:
                fw_cells = get_lstm_cell()
                bw_cells = get_lstm_cell()
            
            # initial_state = cells.zero_state(self.config.batch_size, 
            #         dtype=tf.float32)

            # 2. compute using tf.nn.dynamic_rnn
            # cell_outpus: (output_fw, output_bw)
            # output_fw, output_bw: [None, max_length, hidden_size]
            cell_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cells,             bw_cells, self.inputs_placeholder, dtype=tf.float32)

            # concated_output: [batch_size, max_length, hidden_size*2]
            concated_output = tf.concat(cell_outputs, axis=-1)

            # # fw_last, bw_last = [batch_size, hidden_size]
            # fw_last = cell_outputs[0][:, -1, :]
            # bw_last = cell_outputs[1][:, 0, :]

            # # output: [batch_size, 2*hidden_size]
            # output = tf.concat([fw_last, bw_last], axis=-1)

            output = concated_output[:, 0, :] + concated_output[:, -1, :]
            # output = concated_output[:, self.config.max_length//2, :]

        with tf.variable_scope('layer1'):
            W1 = tf.get_variable("W1", 
                shape=[2*self.config.hidden_size, self.config.hidden_size], 
                initializer=tf.contrib.layers.xavier_initializer())

            b1 = tf.get_variable("b1", 
                shape=[1, self.config.hidden_size], 
                initializer=tf.constant_initializer(0))

            # z1: [batch_size, hidden_size]
            z1 = tf.matmul(output, W1) + b1
            a1 = tf.nn.relu(z1)

        with tf.variable_scope('layer2'):
            W2 = tf.get_variable("W2", 
                shape=[self.config.hidden_size, self.config.class_num], 
                initializer=tf.contrib.layers.xavier_initializer())

            b2 = tf.get_variable("b2", 
                shape=[1, self.config.class_num], 
                initializer=tf.constant_initializer(0))            

            # logits: [batch_size, class_numner]
            logits = tf.matmul(a1, W2) + b2

        return logits

    def add_pred_op(self, logits):
        """Defines self.labels_pred

        """
        print("adding prediction operation into the graph...")
        labels_pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        
        return labels_pred


    def add_loss_op(self, logits):
        """Defines the loss"""
        print("adding loss operation into the graph...")
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels_placeholder)
        # mask = tf.sequence_mask(self.sequence_lengths)
        # cross_entropy = tf.boolean_mask(cross_entropy, mask)
        loss_op = tf.reduce_mean(cross_entropy)

        return loss_op
    
    # def add_train_op(self, lr_method, lr, loss, clip=-1):
    #     """Train op is implemented in the basic model"""
    #     pass

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.logits = self.add_logits_op()
        self.pred = self.add_pred_op(self.logits)
        self.loss = self.add_loss_op(self.logits)

        # Generic functions that add training op and initialize session
        # add_train_op(self, lr_method, lr, loss, clip=-1)
        self.train_op = self.add_train_op(self.config.lr_method, 
            self.lr_placeholder, self.loss, self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init

    def predict_batch(self, inputs):
        """
        Args:
            inputs: input data batch without labels

        Returns:
            labels_pred: list of predicted labels: [batch_size]
        """

        fd = self.get_feed_dict(inputs)
        labels_pred = self.sess.run(self.pred, feed_dict=fd)

        return labels_pred

    def run_epoch(self, train, dev):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (inputs, labels) in enumerate(minibatches(train, batch_size)):

            fd = self.get_feed_dict(inputs, labels_batch=labels, 
                 lr=self.config.lr, dropout=self.config.dropout)

            _, train_loss = self.sess.run([self.train_op, self.loss], 
                feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

        # for i in range(nbatches):
        #     (inputs, labels) = minibatches(train, batch_size)

        #     assert len(inputs) == len(labels)

        #     fd = self.get_feed_dict(inputs, labels_batch=labels, 
        #          lr=self.config.lr, dropout=self.config.dropout)

        #     _, train_loss = self.sess.run([self.train_op, self.loss], 
        #         feed_dict=fd)

        #     prog.update(i + 1, [("train loss", train_loss)])

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        batch_size = self.config.batch_size

        tp = 0 # pred: positive, label: positive
        tn = 0 # pred: negative, label: negative
        fp = 0 # pred: negative, label: positive
        fn = 0 # pred: positive, label: negative

        correct_preds = 0 # total number of correct predictions
        total_preds = 0 # total number of predictions

        for inputs, labels in minibatches(test, batch_size):
            # labels_pred: [batch_size]
            # labels: [batch_size]
            labels_pred = self.predict_batch(inputs)

            for lab, lab_pred in zip(labels, labels_pred):
                total_preds += 1

                if lab == lab_pred:
                    correct_preds += 1
                    if lab == 1 and lab_pred == 1:
                        tp += 1
                    elif lab == 0 and lab_pred == 0:
                        tn += 1
                else:
                    if lab == 0 and lab_pred == 1:
                        fn += 1
                    elif lab == 1 and lab_pred == 0:
                        fp += 1

        # P = TP/(TP+FP)  R = TP/(TP+FN)
        # F1 = 2*P*R/(P+R)  ACCURACY = (TP+TN)/TOTAL
        assert correct_preds == (tp + tn)
        assert (total_preds - correct_preds) == (fp + fn)

        p   = tp / (tp + fp) if tp > 0 else 0 # precision
        r   = tp / (tp + fn) if tp > 0 else 0 # recall
        se = tp / (tp + fn) if tp > 0 else 0 # sensitivity
        sp = tn / (fp + tn) if tn > 0 else 0 # specificity

        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0
        acc = correct_preds/total_preds
        mcc = (tp*tn - fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        dic = collections.OrderedDict()
        dic['p'] = 100*p
        dic['r'] = 100*r
        dic['se'] = 100*se
        dic['sp'] = 100*sp
        dic['acc'] = 100*acc
        dic['f1'] = 100*f1
        dic['mcc'] = 100*mcc

        return dic
