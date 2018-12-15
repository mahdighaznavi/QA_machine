import tensorflow as tf
import numpy as np

embedding_size = 256
sentence_length = 12
num_sentences = 512
episodic_memory_time_step = 2
answer_module_time_step = 2
mask = np.concatenate(np.repeat(False, sentence_length - 1), True)


class AnswerModuleGRU(tf.nn.rnn_cell.GRUCell):
    def __init__(self, attention, num_units, activation=None):
        super(AnswerModuleGRU, self).__init__(activation=activation, num_units=num_units)

    def call(self, inputs, state):
        z = tf.sigmoid(tf.add(state, inputs))
        r = tf.sigmoid(tf.add(state, inputs))
        h1 = tf.tanh(tf.add(inputs, r * state))
        output = tf.add(h1 * (1 - z), state * z)
        return output, tf.concat(output, self.attention)

def embedd(corpus):
    pass


def input_module(embeddings):  # done
    gru_forward = tf.nn.rnn_cell.GRUCell(embedding_size)
    gru_backward = tf.nn.rnn_cell.GRUCell(embedding_size)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_forward, gru_backward, embeddings)
    input_module_return = tf.boolean_mask(tf.concat(outputs, 2), mask)
    return input_module_return


def create_gates(input_module_output, question, m, index, tensor):
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



def answer_module(hidden_state, question, last_output, time_step, tensor, W):  # done
    gru_cell = AnswerModuleGRU(question, sentence_leng
    tf.nn.bidirectional_dynamic_rnn()
    if time_step == 0:
        return tensor
    new_hidden_state = GRU(tf.concat(last_output, question), hidden_state)
    output = tf.sparse_softmax(tf.matmul(W, hidden_state))
    return answer_module(new_hidden_state, question, output, time_step - 1, tf.concat(tensor, output), W)


corpus = 'salam.'
question = 'aleik'
embeddings = embedd(corpus)
input_module_output = input_module(embeddings)
question_rnn = input_module(question)
