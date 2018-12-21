import numpy as np

# hyper parameters
answer_module_hidden_size = 128
max_sentence_length = 12
embedding_size = 256
sentence_length = 12
num_sentences = 512
episodic_memory_time_step = 2
answer_module_time_step = 2
batch_size = 64
epsiodic_memory_dim = 128
word_numbers = 10000

# constants
EOS_token = 1
mask = np.concatenate(np.repeat(False, sentence_length - 1), True)
