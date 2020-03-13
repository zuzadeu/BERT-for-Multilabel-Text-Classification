from bert_multiclass_torch_extract import ex
import itertools

max_seq_len_values = [256]
batch_size_values = [128]
gamma_values = [2]

for max_seq_len, batch_size, gamma in itertools.product(max_seq_len_values, batch_size_values, gamma_values):
    ex.run(config_updates={'max_seq_len': max_seq_len, 'batch_size': batch_size, 'gamma': gamma})
