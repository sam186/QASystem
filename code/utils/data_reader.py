import logging
import numpy as np
from os.path import join as pjoin
from tensorflow.python.platform import gfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config(object):
    def __init__(self, data_dir):
        # self.train_answer_file = pjoin(data_dir, 'train.answer')
        self.train_answer_span_file = pjoin(data_dir, 'train.span')
        self.train_question_file = pjoin(data_dir, 'train.ids.question')
        self.train_context_file = pjoin(data_dir, 'train.ids.context')
        # self.val_answer_file = pjoin(data_dir, 'val.answer')
        self.val_answer_span_file = pjoin(data_dir, 'val.span')
        self.val_question_file = pjoin(data_dir, 'val.ids.question')
        self.val_context_file = pjoin(data_dir, 'val.ids.context')

def load_glove_embeddings(embed_path):
    logger.info("Loading glove embedding...")
    glove = np.load(embed_path)['glove']
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove

def add_paddings(sentence, max_length, n_features=1):
    zero_vector = [0] * n_features
    mask = [[True]] * len(sentence)
    pad_len = max_length - len(sentence)
    if pad_len > 0:
        padded_sentence = sentence + [zero_vector] * pad_len
        mask += [[False]] * pad_len
    else:
        padded_sentence = sentence[:max_length]
        mask = mask[:max_length]
    return padded_sentence, mask

def featurize_paragraph(paragraph, paragraph_length):
    # TODO: Split by sentence instead of word
    sentences = [[word] for word in paragraph]
    return sentences

def preprocess_dataset(dataset, question_maxlen, context_maxlen):
    processed = []
    for q, q_len, c, c_len, ans in dataset:
        q_sentences = featurize_paragraph(q, q_len)
        c_sentences = featurize_paragraph(c, c_len)

        # add padding:
        q_sentences, q_mask = add_paddings(q_sentences, question_maxlen)
        c_sentences, c_mask = add_paddings(c_sentences, context_maxlen)
        processed.append([q_sentences, q_mask, c_sentences, c_mask, ans])
    return processed

def strip(x):
    return map(int, x.strip().split(" "))

def read_data(data_dir, question_maxlen=None, context_maxlen=None, debug=True):
    config = Config(data_dir)

    if debug:
        debug_train_samples = 40 # (1 / 5)
        debug_val_samples = 4284 

    train = []
    max_q_len = 0
    max_c_len = 0
    logger.info("Loading training data...")
    with gfile.GFile(config.train_question_file, mode="rb") as q_file, \
         gfile.GFile(config.train_context_file, mode="rb") as c_file, \
         gfile.GFile(config.train_answer_span_file, mode="rb") as a_file:
            for (q, c, a) in zip(q_file, c_file, a_file):
                question = strip(q)
                context = strip(c)
                answer = strip(a)
                sample = [question, len(question), context, len(context), answer]
                train.append(sample)
                max_q_len = max(max_q_len, len(question))
                max_c_len = max(max_c_len, len(context))
                if debug and len(train) == debug_train_samples:
                    break
    logger.info("Finish loading %d train data." % len(train))

    val = []
    logger.info("Loading validation data...")
    with gfile.GFile(config.val_question_file, mode="rb") as q_file, \
         gfile.GFile(config.val_context_file, mode="rb") as c_file, \
         gfile.GFile(config.val_answer_span_file, mode="rb") as a_file:
            for (q, c, a) in zip(q_file, c_file, a_file):
                question = strip(q)
                context = strip(c)
                answer = strip(a)
                sample = [question, len(question), context, len(context), answer]
                val.append(sample)
                max_q_len = max(max_q_len, len(question))
                max_c_len = max(max_c_len, len(context))
                if debug and len(val) == debug_val_samples:
                    break
    logger.info("Finish loading %d validation data." % len(val))
    logger.info("Max question length %d" % max_q_len)
    logger.info("Max context length %d" % max_c_len)

    if question_maxlen is None:
        question_maxlen = max_q_len
    if context_maxlen is None:
        context_maxlen = max_c_len

    train = preprocess_dataset(train, question_maxlen, context_maxlen)
    val = preprocess_dataset(val, question_maxlen, context_maxlen)

    return {"training": train, "validation": val}


if __name__ == '__main__':
    read_data('../../data/squad', 100)