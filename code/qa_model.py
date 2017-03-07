from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

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


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input=None): # where to use encoder_state_input(optional), where to set weights
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # Forward direction cell
        lstm_fw_cell = tf.rnn.BasicLSTMCell(self.size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.rnn.BasicLSTMCell(self.size, state_is_tuple=True)
        # Get lstm cell output
        (fw_h, bw_h), _ = tf.nn.bidirectional_dynamic_rnn( \
            lstm_fw_cell, lstm_bw_cell, inputs=inputs, dtype=tf.float32)
        # Concatinate 2 vectors as context representation.
        h = tf.concat(2, [fw_h, bw_h])
        return


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
        self.config = config

        # ==== set up placeholder tokens ========
        self.question_inputs = tf.placeholder(tf.int32, shape=(None, config.question_maxlen))
        self.question_masks = tf.placeholder(tf.int32, shape=(None, config.question_maxlen))
        self.context_inputs = tf.placeholder(tf.int32, shape=(None, config.context_maxlen))
        self.context_masks = tf.placeholder(tf.int32, shape=(None, config.context_maxlen))


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.x, self.q = self.setup_embeddings()
            self.pred = self.setup_system(self.x, self.q)
            self.loss = self.setup_loss(self.pred)

        # ==== set up training/updating procedure ====
        # ?train_op = e.g. tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        pass


    def setup_system(self, x, q):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # Step 1: encode x and q, respectively, with independent weights
        #         e.g. H = encode_context(x)   # get H (2d*T) as representation of x
        #         e.g. U = encode_question(q)  # get U (2d*J) as representation of q

        # Step 2: combine H and U using "Attention"
        #         e.g. S = H.T * U
        #              a_x = softmax(S)
        #              a_q = softmax(S.T)
        #              U_hat = sum(a_x*U)
        #              H_hat = sum(a_q*H)


        # Step 3: further encode
        #         e.g. G = f(H, U, H_hat, U_hat)

        # Step 4: decode
        #         e.g. pred_start = decode_start(G)
        #         e.g. pred_end = decode_end(G)

        raise NotImplementedError("Connect all parts of your system here!")

        # retunr (pred_start, pred_end)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pretrained_embeddings = tf.Variable(self.pretrained_embeddings, name="Emb")
            question_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.question_inputs)
            question_embeddings = tf.reshape(question_embeddings, shape=[-1, self.config.question_maxlen, self.config.embedding_size])
            context_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.context_inputs)
            context_embeddings = tf.reshape(question_embeddings, shape=[-1, self.config.context_maxlen, self.config.embedding_size])

        return （context_embeddings, question_embeddings）

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
