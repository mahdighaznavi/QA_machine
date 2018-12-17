import tensorflow as tf
import tensorflow.contrib.seq2seq as tss
import hyper_parameters_and_constants as HP


def answer_module(inputs, question, dictionary):
    def initializer():
        return False, tf.concat([inputs, question], axis=1)

    def output_word(output):
        temp = tf.layers.dense(inputs=output, units=len(dictionary), reuse=True, name='answer_next_input_dense_layer')
        y = dictionary[tf.argmax(temp)]
        return y

    def sample(time, output, state):
        return output_word(output)

    def next_inputs(time, output, state, sample_ids):
        y = output_word(output)
        next_input = tf.concat([y, question])
        next_state = state
        finished = tf.math.equal(HP.EOS_token, state)
        return finished, next_input, next_state

    helper = tss.CustomHelper(initialize_fn=initializer, sample_fn=sample, next_inputs_fn=next_inputs)
    gru_cell = tf.nn.rnn_cell.GRUCell(HP.answer_module_hidden_size)

    decoder = tss.BasicDecoder(gru_cell, helper, tf.zeros([HP.answer_module_hidden_size]))
    outputs, _, sequence_length = tss.dynamic_decode(decoder, impute_finished=True,
                                                     maximum_iterations=HP.max_sentence_length)
    return outputs.samole_ids, sequence_length
