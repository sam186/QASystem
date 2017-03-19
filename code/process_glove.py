from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


def setup_args():
    parser = argparse.ArgumentParser()
    #home = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    home = os.getcwd()
    vocab_dir = os.path.join(home, "data", "squad")
    glove_dir = os.path.join(home, "download", "dwr")
    source_dir = os.path.join(home, "data", "squad")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=100, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()



def process_glove(args, save_path, size=4e5):
    """
    :param vocab_list: [vocab]
    :return:
    """
    size = 1917495
    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.42B.{}d.txt".format(args.glove_dim))
        glove_vocab = []
        glove = np.random.randn(int(size), args.glove_dim)
        idx = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                glove_vocab.append(word)
                glove[idx, :] = vector
                idx += 1
        
        glove_vocab_dict = dict(zip(glove_vocab, range(len(glove_vocab))))
        np.savez_compressed(save_path, glove=glove, glove_vocab_dict= glove_vocab_dict)
        print("saved untrimmed glove matrix at: {}".format(save_path))

if __name__ == '__main__':
    args = setup_args()
    process_glove(args, args.source_dir + "/glove.untrimmed.{}".format(args.glove_dim))