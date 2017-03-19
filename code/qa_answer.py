from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
from utils.data_reader import preprocess_dataset, load_glove_embeddings
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.005, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("encoder_state_size", 100, "Size of each encoder model layer.")
tf.app.flags.DEFINE_integer("decoder_state_size", 100, "Size of each decoder model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")

tf.app.flags.DEFINE_string("question_maxlen", 60, "Max length of question (default: 30")
tf.app.flags.DEFINE_string("context_maxlen", 766, "Max length of the context (default: 400)")
tf.app.flags.DEFINE_string("n_features", 1, "Number of features for each position in the sentence.")
tf.app.flags.DEFINE_string("answer_size", 2, "Number of features to represent the answer.")
tf.app.flags.DEFINE_string("RE_TRAIN_EMBED", False, "Max length of the context (default: 400)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("decoder_hidden_size", 100, "Number of decoder_hidden_size.")
tf.app.flags.DEFINE_string("QA_ENCODER_SHARE", False, "QA_ENCODER_SHARE weights.")
tf.app.flags.DEFINE_string("ema_weight_decay", 0.9999, "exponential decay for moving averages ")

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data

def expand_vocab(prefix, dev_filename, vocab, embd, raw_glove, raw_glove_vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)
    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    #context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)
    dataset = dev_data
    context_data = []
    query_data = []
    question_uuid_data = []
    tier = 'dev'
    new_vocab = {}
    found = 0
    notfound = 0

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                #context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                #qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]
                #print(context_ids)
                for w in context_tokens:
                    if not w in vocab:
                        if not w in new_vocab:
                            new_vocab[w] = 1
                        else:
                            new_vocab[w] += 1
                        notfound +=1
                    else:
                        found += 1

                for w in question_tokens:
                    if not w in vocab:
                        if not w in new_vocab:
                            new_vocab[w] = 1
                        else:
                            new_vocab[w] += 1
                        notfound +=1
                    else:
                        found +=1


    print('found/not found: {}/{}, {}% not found'.format(found, notfound, 100 * notfound/float(found + notfound)))
    print('New vocabulary:',len(new_vocab))

    vocab_list = list(vocab.items())
    vn = len(vocab_list)
    for i in range((len(new_vocab))):
        vocab_list.append((new_vocab.keys()[i], vn+i))

    vocab = dict(vocab_list)
    rev_vocab = dict([(x, y) for (y,x) in vocab_list])
                #context_data.append(' '.join(context_ids))
                #query_data.append(' '.join(qustion_ids))
                #question_uuid_data.append(question_uuid)
    #return context_data, question_data, question_uuid_data
    _, dim = embd.shape
    new_glove = np.random.randn(len(vocab), dim)
    new_glove[:vn, :] = embd
    
    found = 0
    for i in range(vn, vn+(len(new_vocab))):
        word = vocab_list[i][0]
        if word in raw_glove_vocab:
            found += 1
            idx = raw_glove_vocab[word]
            new_glove[i, :] = raw_glove[idx, :]
        if word.capitalize() in raw_glove_vocab:
            found += 1
            idx = raw_glove_vocab[word.capitalize()]
            new_glove[i, :] = raw_glove[idx, :]
        if word.upper() in raw_glove_vocab:
            found += 1
            idx = raw_glove_vocab[word.upper()]
            new_glove[i, :] = raw_glove[idx, :]
    #from IPython import embed; embed()
    print("{} unseen words found embeddings".format(found))

    return vocab, rev_vocab, new_glove

def strip(x):
    return map(int, x.strip().split(" "))

def preprocessing(context_data, question_data, context_maxlen, question_maxlen):
    logging.debug("Preprocessing evaluation data...")
    dataset = []
    max_q_len = 0
    max_c_len = 0
    for c_data, q_data in zip(context_data, question_data):
        question = strip(q_data)
        context = strip(c_data)
        sample = [question, len(question), context, len(context), None]
        dataset.append(sample)
        max_q_len = max(max_q_len, len(question))
        max_c_len = max(max_c_len, len(context))
    logging.debug("Max question length %d" % max_q_len)
    logging.debug("Max context length %d" % max_c_len)
    return dataset

def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}

    mydata, context_data, context_len_data, question_uuid_data = dataset
    predicts = model.predict_on_batch(sess, mydata)

    for i, uuid in enumerate(question_uuid_data):
        start, end = predicts[i]
        context_length = context_len_data[i]
        context = strip(context_data[i])
        end = min(end, context_length - 1)
        if start <= end:
            predict_answer = ' '.join(rev_vocab[vocab_index] for vocab_index in context[start : end + 1])
        else:
            predict_answer = ''
        answers[uuid] = predict_answer

    return answers


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = load_glove_embeddings(embed_path)
    
    
    raw_embed_path = pjoin("data", "squad", "glove.untrimmed.{}.npz".format(FLAGS.embedding_size))
    raw_glove_data = np.load(raw_embed_path)
    raw_glove = raw_glove_data['glove']
    raw_glove_vocab = raw_glove_data['glove_vocab_dict'][()]
    
    
    # expand vocab
    vocab, rev_vocab, embeddings = expand_vocab(dev_dirname, dev_filename, vocab, embeddings, raw_glove, raw_glove_vocab)

    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    context_len_data = [len(context.split()) for context in context_data]
    mydata = preprocessing(context_data, question_data, FLAGS.context_maxlen, FLAGS.question_maxlen)
    dataset = (mydata, context_data, context_len_data, question_uuid_data)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    #encoder = Encoder(vocab_dim=FLAGS.embedding_size, state_size = FLAGS.encoder_state_size)
    #decoder = Decoder(output_size=FLAGS.output_size, hidden_size = FLAGS.decoder_hidden_size, state_size = FLAGS.decoder_state_size)


    qa = QASystem(embeddings, FLAGS)

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
