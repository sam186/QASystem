from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, datetime
import logging
from tqdm import tqdm
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from operator import mul
from tensorflow.python.ops import variable_scope as vs
from utils.util import ConfusionMatrix, Progbar, minibatches, one_hot, minibatch, get_best_span

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# No gradient clipping:
def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

# With gradient clipping:
def get_optimizer(opt, loss, max_grad_norm, learning_rate):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        assert (False)

    grads_and_vars = optfn.compute_gradients(loss)
    variables = [output[1] for output in grads_and_vars]
    gradients = [output[0] for output in grads_and_vars]

    gradients = tf.clip_by_global_norm(gradients, clip_norm=max_grad_norm)[0]
    grads_and_vars = [(gradients[i], variables[i]) for i in range(len(gradients))]
    train_op = optfn.apply_gradients(grads_and_vars)

    return train_op

def softmax_mask_prepro(tensor, mask): # set huge neg number(-1e10) in padding area
    assert tensor.get_shape().ndims == mask.get_shape().ndims
    m0 = tf.subtract(tf.constant(1.0), tf.cast(mask, 'float32'))
    paddings = tf.multiply(m0,tf.constant(-1e10))
    tensor = tf.select(mask, tensor, paddings)
    return tensor

class Attention(object):
    def __init__(self):
        pass

    def calculate(self, h, u, h_mask, u_mask, JX, JQ):
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

        :return: [N, JX, d_com]
        """
        logging.debug('-'*5 + 'attention' + '-'*5)
        logging.debug('Context representation: %s' % str(h))
        logging.debug('Question representation: %s' % str(u))
        d_en = h.get_shape().as_list()[-1]
        # h [None, JX, d_en]
        # u [None, JQ, d_en]

        # get similarity
        h_aug = tf.reshape(h, shape = [-1, JX, 1, d_en])
        u_aug = tf.reshape(u, shape = [-1, 1, JQ, d_en])
        h_mask_aug = tf.tile(tf.expand_dims(h_mask, -1), [1, 1, JQ]) # [N, JX] -(expend)-> [N, JX, 1] -(tile)-> [N, JX, JQ]
        u_mask_aug = tf.tile(tf.expand_dims(u_mask, -2), [1, JX, 1]) # [N, JQ] -(expend)-> [N, 1, JQ] -(tile)-> [N, JX, JQ]
        s = tf.reduce_sum(tf.multiply(h_aug, u_aug), axis = -1) # h * u: [N, JX, d_en] * [N, JQ, d_en] -> [N, JX, JQ]
        hu_mask_aug = h_mask_aug & u_mask_aug

        s = softmax_mask_prepro(s, hu_mask_aug)

        # get a_x
        a_x = tf.nn.softmax(s, dim=-1) # softmax -> [N, JX, softmax(JQ)]

        #     use a_x to get u_a
        a_x = tf.reshape(a_x, shape = [-1, JX, JQ, 1])
        u_aug = tf.reshape(u_aug, shape = [-1, 1, JQ, d_en])
        u_a = tf.reduce_sum(tf.multiply(a_x, u_aug), axis = -2)# a_x * u: [N, JX, JQ](weight) * [N, JQ, d_en] -> [N, JX, d_en]
        logging.debug('Context with attention: %s' % str(u_a))

        # get a_q
        a_q = tf.reduce_max(s, axis=-1) # max -> [N, JX]
        a_q = tf.nn.softmax(a_q, dim=-1) # softmax -> [N, softmax(JX)]
        #     use a_q to get h_a
        a_q = tf.reshape(a_q, shape = [-1, JX, 1])
        h_aug = tf.reshape(h, shape = [-1, JX, d_en])

        h_a = tf.reduce_sum(tf.multiply(a_q, h_aug), axis = -2)# a_q * h: [N, JX](weight) * [N, JX, d_en] -> [N, d_en]
        assert h_a.get_shape().as_list() == [None, d_en]
        h_a = tf.tile(tf.expand_dims(h_a, -2), [1, JX, 1]) # [None, JX, d_en]

        h_0_u_a = h*u_a #[None, JX, d_en]
        h_0_h_a = h*h_a #[None, JX, d_en]
        return tf.concat(2,[h, u_a, h_0_u_a, h_0_h_a])

class Encoder(object):
    def __init__(self, vocab_dim, state_size, dropout = 0):
        self.vocab_dim = vocab_dim
        self.state_size = state_size
        #self.dropout = dropout
        #logging.info("Dropout rate for encoder: {}".format(self.dropout))

    def encode(self, inputs, mask, encoder_state_input, J, dropout = 1.0):
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
        sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        sequence_length = tf.reshape(sequence_length, [-1,])

        logging.debug('-'*5 + 'encode' + '-'*5)
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        d_in = inputs.get_shape().as_list()[-1]
        W_input = tf.get_variable('Winput', initializer=tf.contrib.layers.xavier_initializer(), shape=(d_in, self.state_size), dtype=tf.float32)
        inputs_transformed = tf.matmul(tf.reshape(inputs, shape = [-1, d_in]), W_input)
        inputs_transformed = tf.reshape(inputs_transformed, shape = [-1, J, self.state_size])

        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout)

        initial_state_fw = None
        initial_state_bw = None
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input

        logging.debug('Inputs: %s' % str(inputs))
        # Get lstm cell output
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
                                                      cell_bw=lstm_bw_cell,\
                                                      inputs=inputs_transformed,\
                                                      sequence_length=sequence_length,
                                                      initial_state_fw=initial_state_fw,\
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float32)

        # Concatinate forward and backword hidden output vectors.
        # each vector is of size [batch_size, sequence_length, cell_state_size]

        logging.debug('fw hidden state: %s' % str(outputs_fw))
        hidden_state = tf.concat(2, [outputs_fw, outputs_bw])
        logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        concat_final_state = tf.concat(1, [final_state_fw[1], final_state_bw[1]])
        logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
        return hidden_state, concat_final_state, (final_state_fw, final_state_bw)


class Decoder(object):
    def __init__(self, output_size, state_size):
        self.output_size = output_size
        self.state_size = state_size

    def decode(self, g, context_mask, JX, dropout = 1.0):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.
        m_2 = bi_LSTM*2(g)
        """
        d_de = self.state_size*2
        with tf.variable_scope('g'):
            m, m_repr, m_state = \
                 self.decode_LSTM(inputs=g, mask=context_mask, encoder_state_input=None, dropout = dropout)
        with tf.variable_scope('m'):
            m_2, m_2_repr, m_2_state = \
                 self.decode_LSTM(inputs=m, mask=context_mask, encoder_state_input=m_state, dropout = dropout)
        # assert m_2.get_shape().as_list() == [None, JX, d_en2]

        s, e = self.get_logit(m_2, JX) #[N, JX]*2
        # or s, e = self.get_logit_start_end(m_2) #[N, JX]*2
        s = softmax_mask_prepro(s, context_mask)
        e = softmax_mask_prepro(e, context_mask)
        return (s, e)

    def decode_LSTM(self, inputs, mask, encoder_state_input, dropout = 1.0):
        logging.debug('-'*5 + 'decode_LSTM' + '-'*5)
        # Forward direction cell
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        # Backward direction cell
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)

        # add dropout

        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, input_keep_prob = dropout)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, input_keep_prob = dropout)

        initial_state_fw = None
        initial_state_bw = None
        if encoder_state_input is not None:
            initial_state_fw, initial_state_bw = encoder_state_input

        logging.debug('Inputs: %s' % str(inputs))
        sequence_length = tf.reduce_sum(tf.cast(mask, 'int32'), axis=1)
        sequence_length = tf.reshape(sequence_length, [-1,])
        # Get lstm cell output
        (outputs_fw, outputs_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,\
                                                      cell_bw=lstm_bw_cell,\
                                                      inputs=inputs,\
                                                      sequence_length=sequence_length,
                                                      initial_state_fw=initial_state_fw,\
                                                      initial_state_bw=initial_state_bw,
                                                      dtype=tf.float32)

        logging.debug('fw hidden state: %s' % str(outputs_fw))
        hidden_state = tf.concat(2, [outputs_fw, outputs_bw])
        logging.debug('Concatenated bi-LSTM hidden state: %s' % str(hidden_state))
        # final_state_fw and final_state_bw are the final states of the forwards/backwards LSTM
        concat_final_state = tf.concat(1, [final_state_fw[1], final_state_bw[1]])
        logging.debug('Concatenated bi-LSTM final hidden state: %s' % str(concat_final_state))
        return hidden_state, concat_final_state, (final_state_fw, final_state_bw)

    def get_logit(self, X, JX):
        d = X.get_shape().as_list()[-1]
        assert X.get_shape().ndims == 3
        X = tf.reshape(X, shape = [-1, d])
        W1 = tf.get_variable('W1', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
        W2 = tf.get_variable('W2', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
        pred1 = tf.matmul(X, W1)
        pred2= tf.matmul(X, W2)
        pred1 = tf.reshape(pred1, shape = [-1, JX])
        pred2 = tf.reshape(pred2, shape = [-1, JX])
        tf.summary.histogram('logit_start', pred1)
        tf.summary.histogram('logit_end', pred2)
        return pred1, pred2

    def get_logit_start_end(self, X, JX):
        d = X.get_shape().as_list()[-1]
        X = tf.reshape(X, shape = [-1, d])
        X = tf.reshape(X, shape = [-1, d])
        W1 = tf.get_variable('W1', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
        W2 = tf.get_variable('W2', initializer=tf.contrib.layers.xavier_initializer(), shape=(d, 1), dtype=tf.float32)
        pred1 = tf.matmul(X, W1)
        pred2_0 = tf.matmul(X, W2)
        pred1 = tf.reshape(pred1, shape = [-1, JX])
        pred2_0 = tf.reshape(pred2_0, shape = [-1, JX])

        pred2_1 = tf.concat(1,[pred1, pred2_0])
        W_se = tf.get_variable('W_se', initializer=tf.contrib.layers.xavier_initializer(), shape=(2*JX, JX), dtype=tf.float32)
        b_se = tf.get_variable('b_se', initializer=tf.contrib.layers.xavier_initializer(), shape=(JX,), dtype=tf.float32)
        pred2 = tf.matmul(pred2_1, W_se)+b_se
        return pred1, pred2

class QASystem(object):
    def __init__(self, pretrained_embeddings, config):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.pretrained_embeddings = pretrained_embeddings
        self.encoder = Encoder(vocab_dim=config.embedding_size, state_size = config.encoder_state_size)
        self.decoder = Decoder(output_size=config.output_size, state_size = config.decoder_state_size)
        self.attention = Attention()
        self.config = config

        # ==== set up placeholder tokens ====
        self.question_placeholder = tf.placeholder(dtype=tf.int32, name="q", shape=(None, None))
        self.question_mask_placeholder = tf.placeholder(dtype=tf.bool, name="q_mask", shape=(None, None))
        self.context_placeholder = tf.placeholder(dtype=tf.int32, name="c", shape=(None, None))
        self.context_mask_placeholder = tf.placeholder(dtype=tf.bool, name="c_mask", shape=(None, None))
        # self.answer_placeholders = tf.placeholder(dtype=tf.int32, name="a", shape=(None, config.answer_size))
        self.answer_start_placeholders = tf.placeholder(dtype=tf.int32, name="a_s", shape=(None,))
        self.answer_end_placeholders = tf.placeholder(dtype=tf.int32, name="a_e", shape=(None,))
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout", shape=())
        self.JX = tf.placeholder(dtype=tf.int32, name='JX', shape=())
        self.JQ = tf.placeholder(dtype=tf.int32, name='JQ', shape=())


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.q, self.x = self.setup_embeddings()
            self.preds = self.setup_system(self.x, self.q)
            self.loss = self.setup_loss(self.preds)

        # ==== set up training/updating procedure ====
        # No gradient clipping:
        # get_op = get_optimizer(self.config.optimizer)
        # self.train_op = get_op(self.config.learning_rate).minimize(self.loss)

        # With gradient clipping:
        opt_op = get_optimizer("adam", self.loss, config.max_gradient_norm, config.learning_rate)

        if config.ema_weight_decay is not None:
            self.train_op = self.build_ema(opt_op)
        else:
            self.train_op = opt_op
        self.merged = tf.summary.merge_all()

    def build_ema(self, opt_op):
        self.ema = tf.train.ExponentialMovingAverage(self.config.ema_weight_decay)
        ema_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([opt_op]):
            train_op = tf.group(ema_op)
        return train_op

    def setup_system(self, x, q):
        d = x.get_shape().as_list()[-1] # self.config.embedding_size
            #   x: [None, JX, d]
            #   q: [None, JQ, d]
        assert x.get_shape().ndims == 3
        assert q.get_shape().ndims == 3

        # Step 1: encode x and q, respectively, with independent weights
        #         e.g. u = encode_question(q)  # get U (2d*J) as representation of q
        #         e.g. h = encode_context(x, u_state)   # get H (2d*T) as representation of x
        with tf.variable_scope('q'):
            u, question_repr, u_state = \
                 self.encoder.encode(inputs=q, mask=self.question_mask_placeholder, encoder_state_input=None, J = self.JQ, dropout = self.dropout_placeholder)
            if self.config.QA_ENCODER_SHARE:
                tf.get_variable_scope().reuse_variables()
                h, context_repr, context_state =\
                     self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=u_state, J = self.JX, dropout = self.dropout_placeholder)
        if not self.config.QA_ENCODER_SHARE:
            with tf.variable_scope('c'):
                h, context_repr, context_state =\
                     self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=u_state, J = self.JX, dropout = self.dropout_placeholder)
                 # self.encoder.encode(inputs=x, mask=self.context_mask_placeholder, encoder_state_input=None)
        d_en = self.config.encoder_state_size*2
        assert h.get_shape().as_list() == [None, None, d_en], "Expected {}, got {}".format([None, JX, d_en], h.get_shape().as_list())
        assert u.get_shape().as_list() == [None, None, d_en], "Expected {}, got {}".format([None, JQ, d_en], u.get_shape().as_list())


        # Step 2: combine H and U using "Attention"
        #         e.g. s = h.T * u
        #              a_x = softmax(s)
        #              a_q = softmax(s.T)
        #              u_hat = sum(a_x*u)
        #              h_hat = sum(a_q*h)
        #              g = combine(u, h, u_hat, h_hat)
        # --------op1--------------
        g = self.attention.calculate(h, u, self.context_mask_placeholder, self.question_mask_placeholder, JX = self.JX, JQ = self.JQ) # concat[h, u_a, h*u_a, h*h_a]
        d_com = d_en*4
        assert g.get_shape().as_list() == [None, None, d_com], "Expected {}, got {}".format([None, JX, d_com], g.get_shape().as_list())

        # Step 3:
        # 2 LSTM layers
        # logistic regressions
        pred1, pred2 = self.decoder.decode(g, self.context_mask_placeholder, dropout = self.dropout_placeholder, JX = self.JX)
        return pred1, pred2

    def setup_loss(self, preds):
        with vs.variable_scope("loss"):
            s, e = preds # [None, JX]*2
            assert s.get_shape().ndims == 2
            assert e.get_shape().ndims == 2
            loss1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            loss2 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
            # loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s, labels=self.answer_start_placeholders),)
            # loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=e, labels=self.answer_end_placeholders),)
        loss = loss1 + loss2
        tf.summary.scalar('loss', loss)
        return loss

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            if self.config.RE_TRAIN_EMBED:
                pretrained_embeddings = tf.Variable(self.pretrained_embeddings, name="Emb", dtype=tf.float32)
            else:
                pretrained_embeddings = tf.cast(self.pretrained_embeddings, tf.float32)

            question_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.question_placeholder)
            question_embeddings = tf.reshape(question_embeddings, shape = [-1, self.JQ, self.config.embedding_size])

            context_embeddings = tf.nn.embedding_lookup(pretrained_embeddings, self.context_placeholder)
            context_embeddings = tf.reshape(context_embeddings, shape = [-1, self.JX, self.config.embedding_size])

        return question_embeddings, context_embeddings

    def create_feed_dict(self, question_batch, question_len_batch, context_batch, context_len_batch, JX=10, JQ=10, answer_batch=None, is_train = True):
        feed_dict = {}
        JQ = np.max(question_len_batch)
        JX = np.max(context_len_batch)
        # print('This batch len: JX = %d, JQ = %d', JX, JQ)
        def add_paddings(sentence, max_length):
            mask = [True] * len(sentence)
            pad_len = max_length - len(sentence)
            if pad_len > 0:
                padded_sentence = sentence + [0] * pad_len
                mask += [False] * pad_len
            else:
                padded_sentence = sentence[:max_length]
                mask = mask[:max_length]
            return padded_sentence, mask

        def padding_batch(data, max_len):
            padded_data = []
            padded_mask = []
            for sentence in data:
                d, m = add_paddings(sentence, max_len)
                padded_data.append(d)
                padded_mask.append(m)
            return (padded_data, padded_mask)

        question, question_mask = padding_batch(question_batch, JQ)
        context, context_mask = padding_batch(context_batch, JX)

        feed_dict[self.question_placeholder] = question
        feed_dict[self.question_mask_placeholder] = question_mask
        feed_dict[self.context_placeholder] = context
        feed_dict[self.context_mask_placeholder] = context_mask
        feed_dict[self.JQ] = JQ
        feed_dict[self.JX] = JX

        if answer_batch is not None:
            start = answer_batch[:,0]
            end = answer_batch[:,1]
            feed_dict[self.answer_start_placeholders] = start
            feed_dict[self.answer_end_placeholders] = end
        if is_train:
            feed_dict[self.dropout_placeholder] = 0.8
        else:
            feed_dict[self.dropout_placeholder] = 1.0

        return feed_dict

    def optimize(self, session, training_set):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = training_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch, context_batch, context_len_batch, answer_batch=answer_batch, is_train = True)

        output_feed = [self.train_op, self.merged, self.loss]

        outputs = session.run(output_feed, input_feed)
        return outputs

    def test(self, session, validation_set):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = validation_set
        input_feed = self.create_feed_dict(question_batch, question_len_batch, context_batch, context_len_batch, answer_batch=answer_batch, is_train = False)

        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)
        return outputs

    def answer(self, session, test_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        question_batch, question_len_batch, context_batch, context_len_batch, answer_batch = test_batch
        input_feed =  self.create_feed_dict(question_batch, question_len_batch, context_batch, context_len_batch, answer_batch=None, is_train = False)
        output_feed = [self.preds[0], self.preds[1]]
        outputs = session.run(output_feed, input_feed)

        s, e = outputs

        best_spans, scores = zip(*[get_best_span(si, ei) for si, ei in zip(s, e)])
        return best_spans

    def predict_on_batch(self, session, dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        # prog = Progbar(target=batch_num)
        predicts = []
        for i, batch in tqdm(enumerate(minibatches(dataset, self.config.batch_size, shuffle=False))):
            pred = self.answer(session, batch)
            # prog.update(i + 1)
            predicts.extend(pred)
        return predicts

    def predict_on_batch(self, session, dataset):
        batch_num = int(np.ceil(len(dataset) * 1.0 / self.config.batch_size))
        # prog = Progbar(target=batch_num)
        predicts = []
        for i, batch in tqdm(enumerate(minibatches(dataset, self.config.batch_size, shuffle=False))):
            pred = self.answer(session, batch)
            # prog.update(i + 1)
            predicts.extend(pred)
        return predicts

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        batch_num = int(np.ceil(len(valid_dataset) * 1.0 / self.config.batch_size))
        prog = Progbar(target=batch_num)
        avg_loss = 0
        for i, batch in enumerate(minibatches(valid_dataset, self.config.batch_size)):
            loss = self.test(sess, batch)[0]
            prog.update(i + 1, [("validation loss", loss)])
            avg_loss += loss
        avg_loss /= batch_num
        logging.info("Average validation loss: {}".format(avg_loss))
        return avg_loss

    def evaluate_answer(self, session, dataset, vocab, sample=400, log=False):
        f1 = 0.
        em = 0.

        N = len(dataset)
        sampleIndices = np.random.choice(N, sample, replace=False)
        evaluate_set = [dataset[i] for i in sampleIndices]
        predicts = self.predict_on_batch(session, evaluate_set)

        for example, (start, end) in zip(evaluate_set, predicts):
            q, _, c, _, (true_s, true_e) = example
            # print (start, end, true_s, true_e)
            context_words = [vocab[w] for w in c]

            true_answer = ' '.join(context_words[true_s : true_e + 1])
            if start <= end:
                predict_answer = ' '.join(context_words[start : end + 1])
            else:
                predict_answer = ''
            f1 += f1_score(predict_answer, true_answer)
            em += exact_match_score(predict_answer, true_answer)


        f1 = 100 * f1 / sample
        em = 100 * em / sample

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def run_epoch(self, session, epoch_num, training_set, vocab, validation_set, sample_size=400):
        set_num = len(training_set)
        batch_size = self.config.batch_size
        batch_num = int(np.ceil(set_num * 1.0 / batch_size))
        sample_size = 400

        prog = Progbar(target=batch_num)
        avg_loss = 0
        for i, batch in enumerate(minibatches(training_set, self.config.batch_size, window_batch = self.config.window_batch)):
            global_batch_num = batch_num * epoch_num + i
            _, summary, loss = self.optimize(session, batch)
            prog.update(i + 1, [("training loss", loss)])
            if self.config.tensorboard and global_batch_num % self.config.log_batch_num == 0:
                self.train_writer.add_summary(summary, global_batch_num)
            if (i+1) % self.config.log_batch_num == 0:
                logging.info('')
                self.evaluate_answer(session, training_set, vocab, sample=sample_size, log=True)
                self.evaluate_answer(session, validation_set, vocab, sample=sample_size, log=True)
            avg_loss += loss
        avg_loss /= batch_num
        logging.info("Average training loss: {}".format(avg_loss))
        return avg_loss


    def train(self, session, dataset, train_dir, vocab):
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        training_set = dataset['training'] # [question, len(question), context, len(context), answer]
        validation_set = dataset['validation']
        f1_best = 0
        if self.config.tensorboard:
            train_writer_dir = self.config.log_dir + '/train/' # + datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
            self.train_writer = tf.summary.FileWriter(train_writer_dir, session.graph)
        for epoch in range(self.config.epochs):
            logging.info("="* 10 + " Epoch %d out of %d " + "="* 10, epoch + 1, self.config.epochs)

            score = self.run_epoch(session, epoch, training_set, vocab, validation_set, sample_size=self.config.evaluate_sample_size)
            logging.info("-- validation --")
            self.validate(session, validation_set)

            f1, em = self.evaluate_answer(session, validation_set, vocab, sample=self.config.model_selection_sample_size, log=True)
            # Saving the model
            if f1>f1_best:
                f1_best = f1
                saver = tf.train.Saver()
                saver.save(session, train_dir+'/fancier_model')
                logging.info('New best f1 in val set')
            logging.info('')
