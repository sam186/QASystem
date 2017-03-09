from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from utils.data_reader import minibatches

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Attention(object):
    def __init__(self):
        pass

    def calculate(self, context_state, question_state):
        # compare the question representation with all the context hidden states.
        attention = tf.nn.softmax(tf.matmul(context_state, tf.expand_dims(question_state, -1)))
        context_attention_state = context_state * attention
        return context_attention_state

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, sequence_length, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        sequence_length, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input (padded all to the same length)
        :param sequence_length: Length of the sequence
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.size, state_is_tuple=True)

        initial_state_fw = None
        initial_state_bw = None
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input


        # Get lstm cell output
        outputs, final_output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
                                                      cell_bw=lstm_bw_cell,\
                                                      inputs=inputs,\
                                                      sequence_length=sequence_length,
                                                      initial_state_fw=initial_state_fw,\
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float64)

        # Concatinate forward and backword hidden output vectors.
        # each vector is of size [batch_size, sequence_length, cell_state_size]
        hidden_state = tf.concat(2, outputs)
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        (final_state_fw, final_state_bw) = final_output_states
        concat_final_state = tf.concat(1, [final_state_fw[1], final_state_bw[1]])
        return hidden_state, concat_final_state, final_output_states


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        lstm_cell = tf.rnn.BasicLSTMCell(self.output_size, state_is_tuple=True)
        (output, _) = tf.nn.dynamic_rnn(lstm_cell, lstm_cell, inputs=knowledge_rep, dtype=tf.float32)

        return output

class QASystem(object):
    def __init__(self, encoder, decoder, pretrained_embeddings, config):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.pretrained_embeddings = pretrained_embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.config = config

        # ==== set up placeholder tokens ========
        self.question_placeholder = tf.placeholder(tf.int32, shape=(None, config.question_maxlen, config.n_features))
        self.question_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.context_placeholder = tf.placeholder(tf.int32, shape=(None, config.context_maxlen, config.n_features))
        self.context_length_placeholder = tf.placeholder(tf.int32, shape=(None,))
        self.answer_placeholders = tf.placeholder(tf.int32, shape=(None, config.answer_size))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            # get embeddings for input
            self.q, self.x = self.setup_embeddings()
            # pred from x and q
            self.pred = self.setup_system(self.x, self.q)
            self.loss = self.setup_loss(self.pred)

        # ==== set up training/updating procedure ====
        get_op = get_optimizer("adam")
        train_op = get_op(self.config.lr).minimize(loss)

    def create_feed_dict(self, question_batch, question_length_batch, context_batch, context_length_batch, labels_batch=None):
        feed_dict = {}
        feed_dict[self.question_placeholder] = question_batch
        feed_dict[self.question_length_placeholder] = question_length_batch
        feed_dict[self.context_placeholder] = context_batch
        feed_dict[self.context_length_placeholder] = context_length_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

    def logistic_regression(self, X):
        n_classes = config.context_maxlen
        """
        With any kind of representation, do 2 independent classifications
        Args:
            X: [N, JX, d_com]
        Returns:
            pred: [N, 2, JX]
        """
        JX = self.config.context_maxlen
        d = tf.shape(X)[-1]
        assert self.x.get_shape().as_list() == [None, JX, d] 

        USE_CONTEXT_MASKS = False

        xavier_initializer = xavier_weight_init()
        W1 = tf.Variable(xavier_initializer((d, )), name='W1')
        b1 = tf.Variable(tf.zeros((1,)), name='b1')
        pred1 = tf.matmul(X, W1)+b1 # [N, JX, d]*[d,] +[1,] -> [N, JX]


        W2 = tf.Variable(xavier_initializer((d, )), name='W2')
        b2 = tf.Variable(tf.zeros((1,)), name='b2')
        pred2 = tf.matmul(X, W2)+b2 # [N, JX, d]*[d,] +[1,] -> [N, JX]

        preds =  tf.stack([pred1, pred2], axis = -2) # -> [N, 2, JX]
        assert preds.get_shape().as_list() == [None, 2, JX]
        return preds


    def setup_system(self, x, q):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!

        :return:
        """
        JX, JQ = self.config.context_maxlen, self.config.question_maxlen
        d = tf.shape(self.x)[-1] # self.config.embedding_size * self.config.n_features
        # Args:
            #   self.x: [None, JX, d]
            #   self.q: [None, JQ, d]
        assert self.x.get_shape().as_list() == [None, JX, d] 
        assert self.q.get_shape().as_list() == [None, JQ, d] 


        # Step 1: encode x and q, respectively, with independent weights
        #         e.g. H = encode_context(x)   # get H (2d*T) as representation of x
        #         e.g. U = encode_question(q)  # get U (2d*J) as representation of q
        with tf.variable_scope('q'):
            question_sentence_repr, question_repr, question_state = \
                 self.encoder.encode(inputs=q, sequence_length=self.question_length_placeholder, encoder_state_input=None)

        with tf.variable_scope('c'):
            context_sentence_repr, context_repr, context_state =\
                 self.encoder.encode(inputs=x, sequence_length=self.context_length_placeholder, encoder_state_input=question_state)

        d_en = d
        assert H.get_shape().as_list() == [None, JX, d_en] 
        assert U.get_shape().as_list() == [None, JQ, d_en] 

        # Step 2: combine H and U using "Attention"
        #         e.g. S = H.T * U
        #              a_x = softmax(S)
        #              a_q = softmax(S.T)
        #              U_hat = sum(a_x*U)
        #              H_hat = sum(a_q*H)

        attention_model = Attention()
        context_attention_state = attention_model.calculate(context_sentence_repr, question_repr)

        # Step 3: further encode
        #         e.g. G = f(H, U, H_hat, U_hat)

        # G = tf.concat(1, [H, U])
        
        d_com = d
        assert G.get_shape().as_list() == [None, JX, d_com] 

        # Step 4: decode
        #         e.g. pred_start = decode_start(G)
        #         e.g. pred_end = decode_end(G)
        answer_start, answer_end = self.decoder.decode(context_attention_state)

        # raise NotImplementedError("Connect all parts of your system here!")
        return preds


    def setup_loss(self, preds):
        """
        Set up your loss computation here
        Args:
            preds: A tensor of shape (batch_size, 2, n_classes) containing the output of the neural
                  network before the softmax layer.
        :return:
        """
        with vs.variable_scope("loss"):
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.answer_placeholders),)  
        return loss

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            if self.config.RE_TRAIN_EMBED:
                pretrained_embeddings = tf.Variable(self.pretrained_embeddings, name="Emb")
            else:
                pretrained_embeddings = self.pretrained_embeddings
            question_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.question_placeholder)
            question_embeddings = tf.reshape(question_embeddings, shape=[-1, self.config.question_maxlen, self.config.embedding_size * self.config.n_features])
            context_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.context_placeholder)
            context_embeddings = tf.reshape(question_embeddings, shape=[-1, self.config.context_maxlen, self.config.embedding_size * self.config.n_features])

        return question_embeddings, context_embeddings

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [] #[self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def create_feed_dict(self, question_batch, question_length_batch, context_batch, context_length_batch, labels_batch=None):
        feed_dict = {}
        feed_dict[self.question_placeholder] = question_batch
        feed_dict[self.question_length_placeholder] = question_length_batch
        feed_dict[self.context_placeholder] = context_batch
        feed_dict[self.context_length_placeholder] = context_length_batch
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

    def train_on_batch(self, sess, question_batch, question_length_batch, context_batch, context_length_batch, labels_batch):
        feed_dict = self.create_feed_dict(question_batch, question_length_batch, context_batch, context_length_batch, labels_batch=labels_batch)
        loss = 0.00
        # TODO: set up loss
        # _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def run_epoch(self, session, training_set, validation_set):
        for i in batch in enumerate(minibatches(training_set, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)

        # TODO: Evaluate on training set
        f1, em = evaluate_answer(session, training_set)
        # TODO: Evaluate on validation set
        f1, em = evaluate_answer(session, validation_set)
        return 0


    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        training_set = dataset['training']
        validation_set = dataset['validation']

        best_score = 0
        for epoch in range(self.config.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.config.epochs)
            score = self.run_epoch(session, training_set, validation_set)

            # Saving the model
            # saver = tf.train.Saver()
            # saver.save(session, train_dir)

