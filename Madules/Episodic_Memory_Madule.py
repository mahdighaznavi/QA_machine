import tensorflow as tf
import numpy as np
import hyper_parameters_and_constants as HP


def create_gates(inputs, question, previous_time_step_output, index, tensor):
    """

    :param inputs: concat of embedding context sentences with shape [batch size, number of sentences, embedding size]
    :param question: embedding of the question with shape [batch size, embedding size]
    :param previous_time_step_output: output of previous time step with shape [batch size, embedding size]
    :param index: index of cell
    :param tensor: concat of Z's of previous cells
    :return: a tensor with shape [batch size, sentence length]. g's.
    """
    if index == HP.num_sentence:
        return tf.nn.softmax(tensor)
    b = np.repeat(False, HP.num_sentence)
    b[index] = True
    s = tf.boolean_mask(inputs, b, axis=0)
    assert (tf.shape(s) != tf.shape(question))
    z = tf.concat([s * question, s * previous_time_step_output, tf.abs(tf.add(s, -1 * question)),
                   tf.abs(tf.add(s, -1 * previous_time_step_output))], 1)
    dense_layer1 = tf.layers.Dense(HP.episodic_memory_dim, activation=tf.tanh)
    dense_layer2 = tf.layers.Dense(2)
    Z = dense_layer2(dense_layer1(z))
    g = tf.nn.softmax(Z, axis=1)
    new_tensor = tf.concat([tensor, g], 1)
    return create_gates(inputs, question, previous_time_step_output, index + 1, new_tensor)


def episodic_memory_module(input_module_output, question):
    previous_time_step_output = tf.zeros(shape=[HP.batch_size, HP.embedding_size], dtype=tf.float32)
    gru_cell = tf.nn.rnn_cell.GRUCell(HP.embedding_size)
    for i in range(HP.episodic_memory_time_step):
        gates = create_gates(input_module_output, question, previous_time_step_output, 0,
                             tf.zeros(shape=[HP.batch_size, 0], dtype=tf.float32))
        state = tf.zeros(shape=[HP.batch_size, HP.embedding_size], dtype=tf.float32)
        for j in range(HP.num_sentence):
            b = np.reshape(False, HP.num_sentence)
            b[j] = True
            g = tf.boolean_mask(gates, b)
            input = tf.boolean_mask(input_module_output, b)
            state = tf.add(tf.multiply(g, gru_cell(input, state)), tf.multiply((1 - g), state))

        previous_time_step_output = state
    return previous_time_step_output
