import os, sys, time, random, re, json, collections
import sklearn.metrics
import pandas as pd
import numpy as np
import tensorflow as tf
import cPickle
from gensim.models import KeyedVectors, Word2Vec
from sklearn.model_selection import train_test_split

NeuralModel = collections.namedtuple('NeuralModel',  ['word_vocab', 'pos_vocab',
        'labels', 'seqlen', 'sess', 'saver', 'dropout_keep', 'x_input', 'x_pos_input', 'mlf_input',
        'x_length', 'y_true','y_out','y_pred', 'y_loss','train_step'])

def main():

    dep_word_embeddings_path = 'data/dbwe_dict.p'
    pos_embeddings_path = 'data/pos_embedding_model_on_additional_biotext_corpus.model'
    add_features_path = 'data/addition_features.p'
    pos_path = 'data/285_pos.txt'
    text_path = 'data/285.txt'
    label_path = 'data/labels.p'

    num_epoch = 30
    batch_size = 16

    trainset = load_data(text_path, label_path, pos_path, add_features_path)

    label_dist, seqlen_mean, seqlen_max = data_stats(trainset)
    print("Label distribution> {}".format(label_dist))
    print("Token length> Mean={} Max={}".format(seqlen_mean,seqlen_max))

    #construct vocabulary for words and pos
    word_vocab, pos_vocab = build_vocab(trainset)
    cPickle.dump(word_vocab, open('data/word_vocab.p', 'wb'))
    cPickle.dump(pos_vocab, open('data/pos_vocab.p', 'wb'))
    #load embeddings for words and pos
    word_embeddings, pos_embeddings = load_embeddings(dep_word_embeddings_path, pos_embeddings_path, word_vocab, pos_vocab)

    print("Training word vocab> Length={} : {}..".format(len(word_vocab),', '.join(word_vocab[:10])))
    #uniform unknown words in the text
    trainset = apply_unks(trainset, word_vocab, pos_vocab)

    print('@ Training set',len(trainset))
    train_model(trainset, word_vocab, pos_vocab, word_embeddings, pos_embeddings, seqlen_max, num_epoch, batch_size)
    print('Model trained...')


def train_model(trainset, word_vocab, pos_vocab, word_embeddings, pos_embeddings, seqlen_max, num_epoch, batch_size):
    # Build the model
    labels = ['NEGATIVE', 'NEUTRAL','POSITIVE']
    model = new_model(word_vocab, pos_vocab, labels, seqlen_max,
                        word_embeddings=word_embeddings, pos_embeddings=pos_embeddings)

    # Train the model
    dropout_keep = 0.5
    tmproot = 'tmp'

    # Stratified split
    labels = get_labels(trainset)
    trainsplit, devsplit, train_label, dev_label = train_test_split(trainset, labels, test_size=0.2, random_state=0, stratify=labels)

    trainsample = random.sample(trainsplit,len(devsplit))
    print('@ Training split',len(trainsplit), ' @ Development split', len(devsplit))

    sess_id = int(time.time())
    best_df1 = -1
    best_model = None
    print('Num. Epochs: {}, Batch Size: {}'.format(num_epoch,batch_size))
    for ep in range(1,num_epoch+1):

        aloss = 0
        for i in range(0,len(trainsplit),batch_size):
            minibatch = trainsplit[i:i+batch_size]
            batch_feed = compile_examples(model, minibatch, keep_prob=dropout_keep)
            _, loss = model.sess.run([model.train_step,model.y_loss], batch_feed)
            aloss = aloss * 0.8 + loss * 0.2
            print("epoch {}> iter {}> {}/{} loss {}          "
                .format(ep, int(i/batch_size), i, len(trainsplit), aloss))

        bf1, _, _, _, _, _, _ = eval_model(model,trainsample,num_samples=1,sample_size=1)
        df1, dp, dr, _, _, _, _ = eval_model(model,devsplit,num_samples=1,sample_size=1)
        if df1 > best_df1:
            best_df1 = df1
            best_model = "{}/model-{}-ep{}.ckpt".format(tmproot,sess_id,ep)

            if not os.path.exists(tmproot):
                os.mkdir(tmproot)

            model.saver.save(model.sess, best_model)
            marker = '*'
        else:
            marker = ' '

        print("epoch {}> loss {} fit> f={:.2%}  dev> f={:.2%} p={:.2%} r={:.2%}  {}"
                    .format(ep, aloss, bf1, df1, dp, dr, marker))

    print("Restoring best model: {}".format(best_model))
    model.saver.restore(model.sess, best_model)
    print("Best model df1: " + str(best_df1))
    model.saver.save(model.sess, "best_model/best_model.ckpt")
    return model

def get_labels(dataset):
    labels = ['NEGATIVE', 'NEUTRAL','POSITIVE']
    idx = []
    for (tokens, mlfs, pos), y in dataset:
        idx.append(labels.index(y))
    return idx

def data_stats(dataset):
    label_freq = {}
    token_lengths = []

    for (tokens, mlfs, pos), y in dataset:
        if y in label_freq:
            label_freq[y] += 1
        else:
            label_freq[y] = 1

        token_lengths.append(len(tokens))

    return label_freq, np.mean(token_lengths), np.max(token_lengths)

def eval_model(models, examples, avg='macro', labels=None, num_samples=20, sample_size=10):
    assert(sample_size > 0 and sample_size <= len(models))
    if not isinstance(models,list):
        models = [models]

    if labels is None:
        target_labels = models[0].labels
    else:
        target_labels = labels

    y_out_store = []
    for model in models:
        feed_dict = compile_examples(model,examples)
        y_true, y_out_probs = model.sess.run([model.y_true, model.y_pred], feed_dict)
        y_out_store.append([y_true, y_out_probs])

    y_true = [ model.labels[z] for z in np.argmax(y_true,axis=-1) ]

    f1s = []
    precisions = []
    recalls = []
    accs = []

    for i in range(0, num_samples):
        y_out = np.zeros((len(examples),len(models[0].labels)))
        y_outs = random.sample(y_out_store, sample_size)
        for y_t, y_ in y_outs:
            y_out += y_

        y_pred = [ model.labels[z] for z in np.argmax(y_out,axis=-1) ]

        f1 = sklearn.metrics.f1_score(y_true, y_pred,
                average=avg, labels=target_labels)
        precision = sklearn.metrics.precision_score(y_true, y_pred,
                average=avg,  labels=target_labels)
        recall = sklearn.metrics.recall_score(y_true, y_pred,
                average=avg,  labels=target_labels)
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)

        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        accs.append(acc)

    return np.mean(f1s), np.mean(precisions), np.mean(recalls), np.mean(accs), y_out_store, max(f1s), min(f1s)

def apply_unks(dataset, word_vocab, pos_vocab):
    new_dataset = []
    for (tokens, mlfs, positions), y in dataset:
        tokens = [ w if w in word_vocab else 'UNK' for w in tokens ]
        positions = [ p if p in pos_vocab else 'UNK' for p in positions ]
        new_dataset.append(((tokens, mlfs, positions), y))

    return new_dataset

def build_vocab(dataset):
    word_freq = {}
    pos_vocab = []
    for (tokens, mlfs, pos), y in dataset:
        for w in tokens:
            if w not in word_freq:
                word_freq[w] = 1
            else:
                word_freq[w] += 1

        for p in pos:
            if p not in pos_vocab:
                pos_vocab.append(p)

    word_vocab = ['ZERO','UNK','THISCITATION','OTHERCITATION']
    word_vocab +=  sorted([ w for w, freq in word_freq.items()
                        if freq >= 1 and w not in word_vocab ])
    pos_vocab += ['UNK']

    return word_vocab, pos_vocab

def compile_examples(model, examples, keep_prob=1):
    bx_length = []
    bx_input = []
    bx_pos_input = []
    by_true = []
    ml_features = []

    for (tokens, mlfs, pos), y in examples:
        bx_length.append(len(tokens))

        tidx = ([ model.word_vocab.index(z) for z in tokens ]
                        + [0] * (model.seqlen - len(tokens)))
        bx_input.append(tidx)

        pidx = ([ model.pos_vocab.index(z) for z in pos ]
                        + [0] * (model.seqlen - len(pos)))
        bx_pos_input.append(pidx)

        onehot = np.zeros(len(model.labels))
        onehot[model.labels.index(y)] = 1
        by_true.append(onehot)
        ml_features.append(mlfs)

    feed_dict = {   model.x_input : np.array(bx_input),
                    model.x_pos_input : np.array(bx_pos_input),
                    model.mlf_input : np.array(ml_features),
                    model.x_length : np.array(bx_length),
                    model.y_true : np.array(by_true),
                    model.dropout_keep : keep_prob }

    return feed_dict

def new_model(word_vocab, pos_vocab, labels, seqlen, bsize=None,
            word_embeddings=None, pos_embeddings=None, embedding_size=300,
            filter_sizes=[3,4,5], num_filters=200):
    tf.reset_default_graph()

    y_true = tf.placeholder(tf.float32, [bsize,len(labels)], name='y_true')
    x_length = tf.placeholder(tf.int32, [bsize], name='x_length')
    x_input = tf.placeholder(tf.int32, [bsize, seqlen], name='x_input')
    x_pos_input = tf.placeholder(tf.int32, [bsize, seqlen], name='x_pos_input')
    dropout_keep = tf.placeholder(tf.float32, None, name='dropout_keep')

    mlf_input = tf.placeholder(tf.float32, [bsize, 200], name='mlf_input')
    pos_embedding_size = 30

    w_input = tf.placeholder(tf.float32, [len(word_vocab), embedding_size], name='w_input')
    pw_input = tf.placeholder(tf.float32, [len(pos_vocab), pos_embedding_size], name='pw_input')

    W_em = tf.Variable(tf.truncated_normal([len(word_vocab), embedding_size], stddev=0.15))
    embedding_init = W_em.assign(w_input)


    W_em_pos = tf.Variable(tf.truncated_normal([len(pos_vocab),
                                    pos_embedding_size], stddev=0.15))
    pos_em_init = W_em_pos.assign(pw_input)

    xw_input = tf.nn.embedding_lookup(W_em, x_input)
    xp_input = tf.nn.embedding_lookup(W_em_pos, x_pos_input)

    xwp_input = tf.expand_dims(tf.concat([xw_input,xp_input],axis=-1),axis=-1)
    input_width = embedding_size + pos_embedding_size

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, input_width, 1, num_filters]
        W_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.15))
        b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]))
        conv = tf.nn.conv2d(xwp_input, W_conv, strides=[1, 1, 1, 1], padding="VALID")

        h = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
        pooled_outputs.append(tf.reduce_max(h,axis=1))
    pooled_outputs.append(tf.expand_dims(mlf_input,axis=1))

    num_filters_total = num_filters * len(filter_sizes)
    cnn_pool = tf.reshape(tf.concat(pooled_outputs,axis=-1), [-1, num_filters_total + 200])


    y_out = tf.layers.dense(tf.nn.dropout(cnn_pool, dropout_keep),len(labels),name='y_out')
    y_pred = tf.nn.softmax(y_out, name='y_pred')

    #reg_loss = sum([ tf.nn.l2_loss(x) for x in tf.trainable_variables() if x != W_em ])
    y_loss = tf.losses.softmax_cross_entropy(y_true,y_out) #+ 0.005 * reg_loss
    train_step = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(y_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if word_embeddings is not None:
        sess.run(embedding_init, { w_input: word_embeddings })

    if pos_embeddings is not None:
        sess.run(pos_em_init, {pw_input: pos_embeddings})

    param_count = 0
    for v in tf.trainable_variables():
        param_count += np.prod([ int(dimsize) for dimsize in v.get_shape() ])

    print("Compiled model with {} variables and {} parameters".format(
            len(tf.trainable_variables()),param_count))

    saver = tf.train.Saver(max_to_keep=100)

    return NeuralModel(word_vocab, pos_vocab, labels, seqlen, sess, saver, dropout_keep, x_input, x_pos_input, mlf_input,
                x_length, y_true, y_out, y_pred, y_loss, train_step)

def load_data(text_path, label_path, pos_path, add_features_path):
    examples = []
    #load labels
    labels = cPickle.load(open(label_path, 'rb'))

    #load pos data
    pos_data = []
    f = open(pos_path, 'rb')
    for line in f:
        pos_data.append(line)
    pos_data = np.array(pos_data)
    f.close()

    #load addtional features
    ml_features = cPickle.load(open(add_features_path, 'rb'))

    #load text
    index = 0
    with open(text_path,'r') as f:
        for line in f:
            label = labels[index]
            tokens = [ x.lower() if x not in ['THISCITATION', 'OTHERCITATION'] else x for x in line.split() ]
            pos = pos_data[index].split()
            examples.append(((tokens, ml_features[index,:], pos),label))
            index = index + 1

    return examples

def load_embeddings(dep_word_embeddings_path, pos_embeddings_path, word_vocab, pos_vocab, dim=300):
    shape = (len(word_vocab), dim)
    weight_matrix = np.random.uniform(-0.15, 0.15, shape).astype(np.float32)
    pos_shape = (len(pos_vocab), 30)
    pos_matrix = np.random.uniform(-0.15, 0.15, pos_shape).astype(np.float32)
    vecs = cPickle.load(open(dep_word_embeddings_path, 'rb'))
    pos_em = Word2Vec.load(pos_embeddings_path)
    pos_vecs = pos_em.wv

    for i in range(len(word_vocab)):
        if word_vocab[i] in vecs:
            weight_matrix[i,:] = vecs[word_vocab[i]]

    for i in range(len(pos_vocab)):
        if pos_vocab[i] in pos_vecs:
            pos_matrix[i,:] = pos_vecs[pos_vocab[i]]

    return weight_matrix, pos_matrix

if __name__ == '__main__':
    main()
