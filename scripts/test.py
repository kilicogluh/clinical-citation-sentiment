import tensorflow as tf
import collections
import random
import numpy as np

# feed_dict = {   model.x_input : np.array(bx_input),
#                 model.x_pos_input : np.array(bx_pos_input),
#                 model.mlf_input : np.array(ml_features),
#                 model.x_length : np.array(bx_length),
#                 model.y_true : np.array(by_true),
#                 model.dropout_keep : keep_prob }
#
# y_true, y_out_probs = model.sess.run([model.y_true, model.y_pred], feed_dict)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('best_model/best_model.ckpt.meta')
    saver.restore(sess, 'best_model/best_model.ckpt')
    print("haha")
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name('y_pred:0')
    dropout_keep = graph.get_tensor_by_name('dropout_keep:0')
    x_input = graph.get_tensor_by_name('x_input:0')
    x_pos_input = graph.get_tensor_by_name('x_pos_input:0')
    mlf_input = graph.get_tensor_by_name('mlf_input:0')
    feed_dict = {dropout_keep: 1, x_input: np.array([np.random.choice(100, 166)]), x_pos_input: np.array([np.random.choice(30, 166)]),
                mlf_input: np.array([np.random.choice(2,200)])}
    print sess.run(y_pred, feed_dict)
