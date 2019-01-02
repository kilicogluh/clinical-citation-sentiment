# this is used for predicting citation sentiments for new dataset
#
# Usage: python predict.py [input_file_path] [output_file_path]
#
# Input file requirements:
#   One sentence per line
#   Citations must be tagged with <cit> and </cit> where they are occurred.
#   The number of words in the sentence should be less or equal to 166(based on our training set), any longer string will be trimmed
#
# Python package requirements:
#   python 2.7
#   tensorflow
#   sklearn
#
import sys
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from subprocess import *
import cPickle

def main():
    if len(sys.argv) != 3:
        print "Usage: python predict.py [input_file_path] [output_file_path]"
        return
    input_file_path = sys.argv[1]
    result_file_path = sys.argv[2]

    internal_file_path = 'data/preprocessing.txt'
    word_vocab_path = 'data/word_vocab.p'
    pos_vocab_path = 'data/pos_vocab.p'
    seqlen = 166

    # Preprocess the input text to generate necessary inputs for neural network
    process = Popen(['java', '-jar', 'data/genfeas.jar', input_file_path, internal_file_path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    process.wait()

    results = open("results.txt", 'wb')
    cit_tags, word_idx, pos_idx, add_features = parse_results(internal_file_path, word_vocab_path, pos_vocab_path, seqlen)
    mean = cPickle.load(open('data/mean.p', 'rb'))
    transform_matrix = cPickle.load(open('data/transform_matrix.p', 'rb'))
    add_features = [map(lambda x: float(x),feature) for feature in add_features]
    add_features = np.subtract(add_features, mean)
    reduced_features = np.dot(add_features, transform_matrix)

    sess = tf.Session()
    saver = tf.train.import_meta_graph('best_model/best_model.ckpt.meta')
    saver.restore(sess, 'best_model/best_model.ckpt')
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    dropout_keep = graph.get_tensor_by_name('dropout_keep:0')
    x_input = graph.get_tensor_by_name('x_input:0')
    x_pos_input = graph.get_tensor_by_name('x_pos_input:0')
    mlf_input = graph.get_tensor_by_name('mlf_input:0')
    feed_dict = {dropout_keep: 1, x_input: word_idx, x_pos_input: pos_idx,
                    mlf_input: reduced_features}
    y_pred = sess.run(y_pred, feed_dict)
    labels = ['NEGATIVE', 'NEUTRAL','POSITIVE']
    y_out = [ labels[z] for z in np.argmax(y_pred,axis=-1) ]
    write_result(result_file_path, y_out, cit_tags)

def parse_results(path, word, pos, seqlen):
    word_vocab = cPickle.load(open(word, 'rb'))
    pos_vocab = cPickle.load(open(pos, 'rb'))
    f = open(path, 'rb')
    cit_tags = []
    word_idx = []
    pos_idx = []
    feas = []
    for line in f:
        if line and line.strip():
            tokens = line.split("|")
            cit_tag = tokens[0]
            words = tokens[1].split(" ")
            if len(words) > 166:
                words = words[0:166]
            pos = tokens[2].split(" ")
            if len(pos) > 166:
                pos = pos[0:166]
            features = tokens[3].split(",")

            words, pos = apply_unks(words, pos, word_vocab, pos_vocab)
            cit_tags.append(cit_tag)
            widx = ([ word_vocab.index(z) for z in words ]
                            + [0] * (seqlen - len(words)))
            word_idx.append(widx)
            pidx = ([ pos_vocab.index(z) for z in pos ]
                            + [0] * (seqlen - len(pos)))
            pos_idx.append(pidx)
            feas.append(features)

    return cit_tags, word_idx, pos_idx, feas

def write_result(path, y_out, tags):
    f = open(path, 'wb')
    for i in range(len(y_out)):
        f.write(tags[i] + " " + y_out[i] + "\n")
    f.close()

def apply_unks(words, pos, word_vocab, pos_vocab):
    new_words = [ w if w in word_vocab else 'UNK' for w in words ]
    new_pos = [ p if p in pos_vocab else 'UNK' for p in pos ]
    return new_words, new_pos

if __name__ == '__main__':
    main()
