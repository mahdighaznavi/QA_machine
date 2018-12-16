import tensorflow as tf
import numpy as np
import Madules.input_module as Input_Madule

embedding_size = 256
sentence_length = 12
num_sentences = 512
episodic_memory_time_step = 2
answer_module_time_step = 2
mask = np.concatenate(np.repeat(False, sentence_length - 1), True)


def embedd(corpus):
    pass


corpus = 'salam.'
question = 'aleik'
embeddings = embedd(corpus)
input_module_output = Input_Madule.input_module(embeddings, embedding_size)
question_rnn = Input_Madule.input_module(question, embedding_size)
