from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from operator import mul
from tensorflow.python.ops import variable_scope as vs
from utils.util import ConfusionMatrix, Progbar, minibatches, one_hot, minibatch

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
    def __init__(self, config):
        self.config = config

    def calculate(self, h, u):
        # compare the question representation with all the context hidden states.
        #         e.g. S = h.T * u
        #              a_x = softmax(S)
        #              a_q = softmax(S.T)
        #              u_a = sum(a_x*U)
        #              h_a = sum(a_q*H)
        """
        :param h: [N, JX, d_en]
        :param u: [N, JQ, d_en]
        :param h_mask:  [N, JX]
        :param u_mask:  [N, JQ]
        :param scope:

        :return: [N, JX, d_com]
        """
        logging.debug('-'*5 + 'attention' + '-'*5)
        logging.debug('Context representation: %s' % str(h))
        logging.debug('Question representation: %s' % str(u))
        JX, JQ = self.config.context_maxlen, self.config.question_maxlen
        d_en = h.get_shape().as_list()[-1]
        assert h.get_shape().as_list() == [None, JX, d_en]
        assert u.get_shape().as_list() == [None, JQ, d_en]

        h = tf.reshape(h, shape = [-1, JX, 1, d_en])
        u = tf.reshape(u, shape = [-1, 1, JQ, d_en])
        s = tf.reduce_sum(tf.multiply(h, u), axis = -1) # h * u: [N, JX, d_en] * [N, JQ, d_en] -> [N, JX, JQ]
        a_x = tf.nn.softmax(s, dim=-1) # softmax -> [N, JX, softmax(JQ)]
        assert a_x.get_shape().as_list() == [None, JX, JQ]

        a_x = tf.reshape(a_x, shape = [-1, JX, JQ, 1])
        u = tf.reshape(u, shape = [-1, 1, JQ, d_en])
        u_a = tf.reduce_sum(tf.multiply(a_x, u), axis = -2)# a_x * u: [N, JX, JQ](weight) * [N, JQ, d_en] -> [N, JX, d_en]
        assert u_a.get_shape().as_list() == [None, JX, d_en]
        logging.debug('Context with attention: %s' % str(u_a))
        return u_a

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim



    def encode(self, inputs, mask, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        sequence_length, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input (padded all to the same length)
        :param mask: mask of the sequence
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        logging.debug('-'*5 + 'encode' + '-'*5)
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.size, state_is_tuple=True)

        initial_state_fw = None
        initial_state_bw = None
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input

        logging.debug('Inputs: %s' % str(inputs))
        sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), 1)
        flat_len = tf.cast(flatten(sequence_length, 0), 'int32')
        logging.debug('flat_len: %s' % str(flat_len))
        # Get lstm cell output
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
                                                      cell_bw=lstm_bw_cell,\
                                                      inputs=inputs,\
                                                      sequence_length=flat_len,
                                                      initial_state_fw=initial_state_fw,\
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float64)

        # Concatinate forward and backword hidden output vectors.
        # each vector is of size [batch_size, sequence_length, cell_state_size]
        # outputs_fw = tf.Print(outputs_fw, [outputs_fw])
        # final_state_fw = tf.Print(final_state_fw, [final_state_fw])

        logging.debug('fw hidden state: %s' % str(outputs_fw))
        hidden_state = tf.concat(2, [outputs_fw, outputs_bw])
        logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        concat_final_state = tf.concat(1, [final_state_fw[1], final_state_bw[1]])
        logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
        return hidden_state, concat_final_state, (final_state_fw, final_state_bw)


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
        logging.debug('-'*5 + 'decode' + '-'*5)
        logging.debug('Input knowledge_rep: %s' % str(knowledge_rep))
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=1, state_is_tuple=True)
        hidden_states, _ = tf.nn.dynamic_rnn(lstm_cell, inputs=knowledge_rep, dtype=tf.float64)
        logging.debug('Hidden state: %s' % str(hidden_states))
        xavier_initializer=tf.contrib.layers.xavier_initializer()
        b = tf.get_variable("b", shape=(1,), initializer=xavier_initializer,dtype=tf.float64)
        preds = tf.reduce_mean(tf.sigmoid(hidden_states + b), 2)
        start_idx = 0
        end_idx = 0
        return start_idx, end_idx

    def logistic_regression(self, X):
        """
        With any kind of representation, do 2 independent classifications
        Args:
            X: [N, JX, d_en2]
        Returns:
            pred: [N, 2, JX]
        """
        JX = X.get_shape().as_list()[-2]
        d = X.get_shape().as_list()[-1]
        assert X.get_shape().as_list() == [None, JX, d] 

        X = tf.reshape(X, shape = [-1, d])

        xavier_initializer = tf.contrib.layers.xavier_initializer
        W1 = tf.get_variable('W1', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float64)
        b1 = tf.get_variable('b1', initializer=tf.contrib.layers.xavier_initializer(), shape=(1,), dtype=tf.float64)
        W2 = tf.get_variable('W2', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float64)
        b2 = tf.get_variable('b2', initializer=tf.contrib.layers.xavier_initializer(), shape=(1,), dtype=tf.float64)
        
        pred1 = tf.matmul(X, W1)+b1 # [N*JX, d]*[d, 1] +[1,] -> [N*JX, 1]
        pred2 = tf.matmul(X, W2)+b2 # [N*JX, d]*[d, 1] +[1,] -> [N*JX, 1]
        pred1 = tf.reshape(pred1, shape = [-1, JX]) # -> [N, JX]
        pred2 = tf.reshape(pred2, shape = [-1, JX]) # -> [N, JX]

        # preds =  tf.stack([pred1, pred2], axis = -2) # -> [N, 2, JX]
        # assert preds.get_shape().as_list() == [None, 2, JX]
        return pred1, pred2

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
        self.attention = Attention(config)

        # ==== set up placeholder tokens ========
        self.question_placeholder = tf.placeholder(dtype=tf.int32, name="q", shape=(None, config.question_maxlen, config.n_features))
        self.question_mask_placeholder = tf.placeholder(dtype=tf.int32, name="q_mask", shape=(None, config.question_maxlen, config.n_features))
        self.context_placeholder = tf.placeholder(dtype=tf.int32, name="c", shape=(None, config.context_maxlen, config.n_features))
        self.context_mask_placeholder = tf.placeholder(dtype=tf.int32, name="c_mask", shape=(None, config.context_maxlen, config.n_features))
        # self.answer_placeholders = tf.placeholder(dtype=tf.int32, name="a", shape=(None, config.answer_size))
        self.answer_start_placeholders = tf.placeholder(dtype=tf.int32, name="a_s", shape=(None,))
        self.answer_end_placeholders = tf.placeholder(dtype=tf.int32, name="a_e", shape=(None,))
        

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.q, self.x = self.setup_embeddings()
            self.preds = self.setup_system(self.x, self.q)
            self.loss = self.setup_loss(self.preds)

        # ==== set up training/updating procedure ====
        get_op = get_optimizer(self.config.optimizer)
        self.train_op = get_op(self.config.learning_rate).minimize(self.loss)

    def setup_system(self, x, q):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!

        :return:
        """
        JX, JQ = self.config.context_maxlen, self.config.question_maxlen
        d = x.get_shape().as_list()[-1] # self.config.embedding_size * self.config.n_features
        d_ans = self.config.answer_size
        # Args:
            #   x: [None, JX, d]
            #   q: [None, JQ, d]
        assert x.get_shape().as_list() == [None, JX, d], "Expected {}, got {}".format([None, JX, d], x.get_shape().as_list())
        assert q.get_shape().as_list() == [None, JQ, d], "Expected {}, got {}".format([None, JX, d], q.get_shape().as_list())

        # Step 1: encode x and q, respectively, with independent weights
        #         e.g. u = encode_question(q)  # get U (2d*J) as representation of q
        #         e.g. h = encode_context(x, u_state)   # get H (2d*T) as representation of x
        with tf.variable_scope('q'):
            u, question_repr, u_state = \
                 self.encoder.encode(inputs=q, mask=self.question_mask_placeholder, encoder_state_input=None)
        with tf.variable_scope('c'):
            h, context_repr, context_state =\
                 self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=u_state)
                 # self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=None)
        
        d_en = 4*d
        # ---------- opt2 ------------
        # d_en = d
        # h = x
        # u = q
        # -------- opt2 end ---------- 
        assert h.get_shape().as_list() == [None, JX, d_en], "Expected {}, got {}".format([None, JX, d_en], h.get_shape().as_list())
        assert u.get_shape().as_list() == [None, JQ, d_en], "Expected {}, got {}".format([None, JQ, d_en], u.get_shape().as_list())


        # Step 2: combine H and U using "Attention"
        #         e.g. s = h.T * u
        #              a_x = softmax(s)
        #              a_q = softmax(s.T)
        #              u_hat = sum(a_x*u)
        #              h_hat = sum(a_q*h)
        #              g = combine(u, h, u_hat, h_hat)
        g = self.attention.calculate(h, u)

        d_com = d_en
        assert g.get_shape().as_list() == [None, JX, d_com], "Expected {}, got {}".format([None, JX, d_com], g.get_shape().as_list())

        # Step 3: farther encode
        #              m = encode(g), !later bi_LSTM*2
        with tf.variable_scope('g'):
            m, m_repr, m_state = \
                 self.encoder.encode(inputs=g, mask=self.context_mask_placeholder, encoder_state_input=None)
        d_en2 = d_com
        assert m.get_shape().as_list() == [None, JX, d_en2], "Expected {}, got {}".format([None, JX, d_en2], m.get_shape().as_list())

        # Step 4: decode
        #         e.g. pred_start = decode_start(G)
        #         e.g. pred_end = decode_end(G)
        pred1, pred2 = self.decoder.logistic_regression(m)
        assert pred1.get_shape().as_list() == [None, JX], "Expected {}, got {}".format([None, JX], pred1.get_shape().as_list())
        assert pred2.get_shape().as_list() == [None, JX], "Expected {}, got {}".format([None, JX], pred2.get_shape().as_list())
        # raise NotImplementedError("Connect all parts of your system here!")
        return pred1, pred2


    def setup_loss(self, preds):
        """
        Set up your loss computation here
        Args:
            preds: A tensor of shape (batch_size, 2, n_classes) containing the output of the neural
                  network before the softmax layer.
        :return:
        """
        JX, JQ = self.config.context_maxlen, self.config.question_maxlen
        with vs.variable_scope("loss"):
            s, e = preds
            assert s.get_shape().as_list() == [None, JX], "Expected {}, got {}".format([None, JX], s.get_shape().as_list())
            assert e.get_shape().as_list() == [None, JX], "Expected {}, got {}".format([None, JX], e.get_shape().as_list())
            # assert self.answer_start_placeholders.get_shape().as_list() == [None, JX], "Expected {}, got {}".format([None, JX], self.answer_start_placeholders.get_shape().as_list())
            # assert self.answer_end_placeholders.get_shape().as_list() == [None, JX], "Expected {}, got {}".format([None, JX], self.answer_end_placeholders.get_shape().as_list())
            # loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.answer_placeholders),)  
            loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)

            # loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
            
             
        return loss1 + loss2

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            if self.config.RE_TRAIN_EMBED:
                pretrained_embeddings = tf.Variable(self.pretrained_embeddings, name="Emb")
            else:
                pretrained_embeddings = tf.cast(self.pretrained_embeddings, tf.float64)
            question_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.question_placeholder)
            question_embeddings = tf.reshape(question_embeddings, shape=[-1, self.config.question_maxlen, self.config.embedding_size * self.config.n_features])
            context_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.context_placeholder)
            context_embeddings = tf.reshape(context_embeddings, shape=[-1, self.config.context_maxlen, self.config.embedding_size * self.config.n_features])

        return question_embeddings, context_embeddings

    def optimize(self, session, training_set):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        question_batch, question_mask_batch, context_batch, context_mask_batch, answer_batch = training_set
        input_feed = self.create_feed_dict(question_batch, question_mask_batch, context_batch, context_mask_batch, answer_batch=answer_batch)

        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, validation_set):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        question_batch, question_mask_batch, context_batch, context_mask_batch, answer_batch = validation_set
        input_feed = self.create_feed_dict(question_batch, question_mask_batch, context_batch, context_mask_batch, answer_batch=answer_batch)

        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def decode(self, session, test_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        question_batch, question_mask_batch, context_batch, context_mask_batch, answer_batch = test_batch
        input_feed =  self.create_feed_dict(question_batch, question_mask_batch, context_batch, context_mask_batch, answer_batch=None)
        
        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.preds[0], self.preds[1]]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_sample):
        yp, yp2 = self.decode(session, test_sample)

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
        valid_cost = self.test(sess, valid_dataset)

        # TODO: set up validation cost

        return valid_cost

    def evaluate_answer(self, session, dataset, vocab, sample=100, log=False):
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

        N = len(dataset)
        sampleIndices = np.random.choice(N, sample)
        evaluate_set = [dataset[i] for i in sampleIndices]
        predict_s, predict_e = [], []
        for i, batch in enumerate(minibatches(evaluate_set, self.config.batch_size)):
            s, e = self.answer(session, batch)
            predict_s.extend(s)
            predict_e.extend(e)

        for i, s, e in zip(sampleIndices, predict_s, predict_e):
            true_s = int(dataset[i][-1][0])
            true_e = int(dataset[i][-1][1])
            context_words = [vocab[w[0]] for w in dataset[i][2]]
            true_answer = ' '.join(context_words[true_s : true_e + 1])
            # print(s)
            # print(e)
            if s <= e:
                predict_answer = ' '.join(context_words[s : e + 1])
            else:
                predict_answer = ''
            f1 += f1_score(predict_answer, true_answer)
            em += exact_match_score(predict_answer, true_answer)

        f1 /= N
        em /= N

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def create_feed_dict(self, question_batch, question_mask_batch, context_batch, context_mask_batch, answer_batch=None):
        feed_dict = {}
        feed_dict[self.question_placeholder] = question_batch
        feed_dict[self.question_mask_placeholder] = question_mask_batch
        feed_dict[self.context_placeholder] = context_batch
        feed_dict[self.context_mask_placeholder] = context_mask_batch
        if answer_batch is not None:
            start = answer_batch[:,0]
            end = answer_batch[:,1]
            # start_one_hot = np.array([one_hot(self.config.context_maxlen, s) for s in start])
            # end_one_hot = np.array([one_hot(self.config.context_maxlen, e) for e in end])
            feed_dict[self.answer_start_placeholders] = start
            feed_dict[self.answer_end_placeholders] = end
        return feed_dict

    def run_epoch(self, session, training_set):
        prog = Progbar(target=1 + int(len(training_set) / self.config.batch_size))
        for i, batch in enumerate(minibatches(training_set, self.config.batch_size)):
            _, loss = self.optimize(session, batch)
            prog.update(i + 1, [("training loss", loss)])
        print("")
        return 0


    def train(self, session, dataset, train_dir, vocab):
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

        for epoch in range(self.config.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, self.config.epochs)
            score = self.run_epoch(session, training_set)
            self.evaluate_answer(session, training_set, vocab, sample=100, log=True)
            self.evaluate_answer(session, validation_set, vocab, sample=100, log=True)
            # self.validate(session, validation_set)
            # Saving the model
            saver = tf.train.Saver()
            saver.save(session, train_dir)
