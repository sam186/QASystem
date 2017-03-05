import numpy as np
from os.path import join as pjoin
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences

class Config(object):
    def __init__(self, data_dir, vocab_dim):
        self.train_answer_file = pjoin(data_dir, 'train.answer')
        self.train_answer_span_file = pjoin(data_dir, 'train.span')
        self.train_question_file = pjoin(data_dir, 'train.ids.question')
        self.train_context_file = pjoin(data_dir, 'train.ids.context')
        self.embed_path = pjoin(data_dir, "glove.trimmed.{}.npz".format(vocab_dim))
        self.question_maxlen = 30
        self.context_maxlen = 400

def read_data(data_dir, vocab_dim):
    config = Config(data_dir, vocab_dim)
    train_questions = []
    with open(config.train_question_file) as f:
        for line in f.readlines():
            sp = line.strip().split(' ')
            train_questions.append([int(n) for n in sp])
    train_questions = pad_sequences(train_questions, maxlen=config.question_maxlen, value=0)
    print "Finish loading %d train questions." % (len(train_questions))
    train_context = []
    with open(config.train_context_file) as f:
        for line in f.readlines():
            sp = line.strip().split(' ')
            train_context.append([int(n) for n in sp])
    assert len(train_questions) == len(train_context), r"ERROR: number of questions != number of contexts"
    train_context = pad_sequences(train_context, maxlen=config.context_maxlen, value=0)
    print "Finish loading %d train contexts." % (len(train_context)) 

    train_answer_span = []
    with open(config.train_answer_span_file) as f:
        for line in f.readlines():
            sp = line.strip().split(' ')
            train_answer_span.append([int(n) for n in sp])
    assert len(train_questions) == len(train_answer_span), r"ERROR: number of questions != number of answers"
    print "Finish loading %d train answers." % (len(train_answer_span)) 

    glove_embedding = []
    with np.load(config.embed_path) as embeddings:
        glove_embedding = embeddings['glove']
    print "Finish loading Embeddings."

    return glove_embedding, train_questions, train_context, train_answer_span


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size):
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1
    return get_minibatches([x, one_hot], batch_size)


if __name__ == '__main__':
    read_data('../../data/squad', 100)

