import tensorflow as tf
import numpy as np

def create_gates(input_module_output, question, m, index, tensor, sentence_length, num_sentences=None):
    if index == sentence_length:
        return tf.nn.softmax(tensor)
    a = np.array([index])
    b = np.repeat(False, num_sentences)
    b[a] = True
    s = tf.boolean_mask(input_module_output, b)
    z = tf.concat([s * question, s * m, tf.abs(tf.add(s, -1 * question)), tf.abs(tf.add(s, -1 * m))])
    new_tensor = tf.concat(tensor, z)
    return create_gates(input_module_output, question, m, index + 1, new_tensor)


def episodic_memory_module(input_module_output, question, m, time_Step):

    # TODO
    g = create_gates(input_module_output, question, m, 0, )