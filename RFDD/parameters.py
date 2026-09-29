# Parameters
num_classes = 11
sample_size = 20000
batch_size = 64
n_epochs = 100
total_step = 1000
start_index = 0
end_index = 0+128
signal_len = end_index - start_index
lr = 3e-4
cond_drop_prob = 0
objective_target = 'pred_noise'
auto_normalize = False
