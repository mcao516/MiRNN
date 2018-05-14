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

        # inputs: [batch_size, sequence_length, n_features]
        self.inputs_placeholder = tf.placeholder(tf.int32, 
            shape=[None, None, 2*self.config.n_window_size + 1], 
            name="inputs_placeholder")

        # secondary structure: [batch_size, sequence_length, n_features]
        self.structure_placeholder = tf.placeholder(tf.int32, 
            shape=[None, None, 2*self.config.s_window_size + 1], 
            name="structure_placeholder")
        
        # labels: [batch_size]
        self.labels_placeholder = tf.placeholder(tf.int32, 
            shape=[None], name="labels_placeholder")

        # shape = [batch size]
        self.sequence_lengths = tf.placeholder(tf.int32, 
            shape=[None], name="sequence_lengths")
        
        # hyper parameters
        self.dropout_placeholder = tf.placeholder(tf.float32, 
            shape=[], name="dropout_placeholder")

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, 
            shape=[], name="lr_placeholder")


    def get_feed_dict(self, inputs, labels=None, lr=1e-3, dropout=1.0):
        """Given some data, pad it and build a feed dictionary

        Args:
            inputs: list of rna sequence, [[2, 1, 0, ...], ...]
            labels: list of lables, [1, 0, 1, ...]
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # padding input sequences
        sequence_ids, structure_ids = zip(*inputs)
        inputs_padded, sequences_lengths = pad_sequence(sequence_ids, 
            pad_tok=0, max_length=self.config.max_length, 
            window_size=self.config.n_window_size)
        structure_padded, _ = pad_sequence(structure_ids, 
            pad_tok=0, max_length=self.config.max_length, 
            window_size=self.config.s_window_size)
        
        # build feed dictionary
        feed_dict = {
            self.inputs_placeholder: inputs_padded,
            self.structure_placeholder: structure_padded, 
            self.sequence_lengths: sequences_lengths,
            self.lr_placeholder: lr,
            self.dropout_placeholder: dropout
        }

        if labels is not None:
            feed_dict[self.labels_placeholder] = labels

        return feed_dict


    def add_look_up(self):
        """Create a look up table
        """
        print("adding embedding layer into graph...")

        with tf.variable_scope('nucleotides'):
            _nucle_embeddings = tf.get_variable(
                name="_nucleotide_embeddngs", 
                dtype=tf.float32, 
                shape=[self.config.nucle_type_num, 
                       self.config.nucle_embedding_size])
            
            # nucle_embeddings: [batch_size, max_length - 2, n_features, nucle_size]
            nucle_embeddings = tf.nn.embedding_lookup(_nucle_embeddings, 
                self.inputs_placeholder)
            nucle_embeddings = tf.reshape(nucle_embeddings, 
            [-1, self.config.max_length - 2*self.config.n_window_size, 
            (2*self.config.n_window_size + 1)*self.config.nucle_embedding_size])

        with tf.variable_scope('structure'):
            _struc_embeddings = tf.get_variable(
                name="_structure_embeddngs", 
                dtype=tf.float32, 
                shape=[self.config.pair_status_num, 
                       self.config.struc_embedding_size])
            
            # struc_embeddings: [batch_size, max_length - 2, n_features, struc_size]
            struc_embeddings = tf.nn.embedding_lookup(_struc_embeddings, 
                self.structure_placeholder)
            struc_embeddings = tf.reshape(struc_embeddings, 
            [-1, self.config.max_length - 2*self.config.s_window_size, 
            (2*self.config.s_window_size + 1)*self.config.struc_embedding_size])

            # concat
            # look_up: [batch_size, max_length, struc_size + nucle_size]
            sequence_embeddings = tf.concat([nucle_embeddings, 
                struc_embeddings], axis=-1)

        self.sequence_embeddings =  tf.nn.dropout(sequence_embeddings, 
            self.dropout_placeholder)


    def add_logits_op(self):
        """Defines self.logits, the core part of the model

        """
        print("adding logits operation into the graph...")

        def get_rnn_cell():
            rnnCell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
            rnnCell = tf.contrib.rnn.DropoutWrapper(rnnCell, 
                    output_keep_prob=self.dropout_placeholder)           
            return rnnCell


        with tf.variable_scope('rnn-cells'):
            if self.config.layer_num > 1:
                fw_cells = tf.contrib.rnn.MultiRNNCell(
                    [get_rnn_cell() for _ in range(self.config.layer_num)])
                bw_cells = tf.contrib.rnn.MultiRNNCell(
                    [get_rnn_cell() for _ in range(self.config.layer_num)])                    
            else:
                fw_cells = get_rnn_cell()
                bw_cells = get_rnn_cell()

            # 2. compute using tf.nn.dynamic_rnn
            # cell_outpus: (output_fw, output_bw)
            # output_fw, output_bw: [None, max_length, hidden_size]
            cell_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.sequence_embeddings, 
                sequence_length=self.sequence_lengths, 
                dtype=tf.float32)

            # concated_output: [batch_size, max_length, hidden_size*2]
            concated_output = tf.concat(cell_outputs, axis=-1)

            # output: [batch_size, hidden_size*2]
            output = concated_output[:, 0, :] + concated_output[:, -1, :]
            # output = concated_output[:, self.config.max_length//2, :]

        with tf.variable_scope('layer1'):
            W1 = tf.get_variable("W1", 
                shape=[2*self.config.hidden_size, self.config.class_num], 
                initializer=tf.contrib.layers.xavier_initializer())

            b1 = tf.get_variable("b1", 
                shape=[1, self.config.class_num], 
                initializer=tf.constant_initializer(0))

            # z1: [batch_size, class_num]
            logits = tf.matmul(output, W1) + b1
            
        logits = tf.nn.dropout(logits, self.dropout_placeholder)

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
        self.add_look_up()
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

    def probability_on_batch(self, inputs):
        """
        Args:
            inputs: input data batch without labels

        Returns:
            probability: list of predicted probabilities
        """
        fd = self.get_feed_dict(inputs)
        # probability: [batch_size, class_numner]
        probability = tf.nn.softmax(logits=self.logits, dim=-1)

        return self.sess.run(probability, feed_dict=fd)    

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

            fd = self.get_feed_dict(inputs, labels=labels, 
                 lr=self.config.lr, dropout=self.config.dropout)

            _, train_loss = self.sess.run([self.train_op, self.loss], 
                feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

        # evaluate performan on development set
        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_prob(self, test):
        """Compute probability on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            probs: list of prediction probabilities
            labels: list of labels
        """
        batch_size = self.config.batch_size
        probs = np.array([], dtype=np.float)
        labels = np.array([], dtype=np.int32)

        for inputs, batch_labels in minibatches(test, batch_size):
            # [batch_size, class_numner]
            batch_probs = self.probability_on_batch(inputs)[:, -1]

            probs = np.concatenate([probs, batch_probs])
            labels = np.concatenate([labels, batch_labels])

        assert len(probs) == len(labels)

        return probs, labels


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
        if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0:
            mcc = (tp*tn - fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        else:
            mcc = 0

        dic = collections.OrderedDict()
        dic['tp'] = tp
        dic['tn'] = tn
        dic['p'] = 100*p
        dic['r'] = 100*r
        dic['se'] = 100*se
        dic['sp'] = 100*sp
        dic['acc'] = 100*acc
        dic['f1'] = 100*f1
        dic['mcc'] = mcc
        
        return dic
