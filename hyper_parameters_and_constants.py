import numpy as np

# hyper parameters
answer_module_hidden_size = 128
max_sentence_length = 12
embedding_size = 256
num_sentence = 12
episodic_memory_time_step = 2
answer_module_time_step = 2
batch_size = 64
episodic_memory_dim = embedding_size
word_numbers = 10000
max_data_size = 10

# constants
EOS_token = 1
mask = np.concatenate([np.repeat(False, num_sentence - 1), [True]])
