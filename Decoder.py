import tensorflow as tf
import tensorflow.contrib.seq2seq as tss


def answer_module(inputs, question, max_sentence_length, EOS_token):
    def initializer():
        return [False], inputs

    def sample(time, output, state):
        return tf.zeros(shape=[1])

    def next_inputs(time, output, state, sample_ids):
        #TODO: units?
        y = tf.layers.dense(inputs=outputs, units= ,reuse=True, name='answer_next_input_dense_layer')
        next_input = tf.concat([y, question], axis=1)
        next_state = state
        finished = tf.math.equal(EOS_token, state)
        return finished, next_input, next_state

    helper = tss.CustomHelper(initialize_fn=initializer, sample_fn=sample, next_inputs_fn=next_inputs)
    # TODO: what is num_units?
    gru_cell = tf.nn.rnn_cell.GRUCell()

    decoder = tss.BasicDecoder(gru_cell, helper, inputs)
    outputs, _, _ = tss.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_sentence_length)
    return outputs
