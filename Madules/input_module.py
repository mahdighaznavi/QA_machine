import tensorflow as tf
import hyper_parameters_and_constants as HP


def input_module(embeddings, embedding_size):
    gru_forward = tf.nn.rnn_cell.GRUCell(embedding_size)
    gru_backward = tf.nn.rnn_cell.GRUCell(embedding_size)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_forward, gru_backward, embeddings)
    input_module_return = tf.boolean_mask(tf.concat(outputs, 2), HP.mask)
    return input_module_return
