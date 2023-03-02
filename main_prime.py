"""
** deeplean-ai.com **
created by :: GauravBh1010tt
contact :: gauravbhatt.deeplearn@gmail.com
"""

from model import model_congen, mdn_loss, sample_prime
from eval_hand import *

import gc
n = gc.collect()
print("Number of unreachable objects collected by GC:", n)
torch.cuda.empty_cache()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

hidden_size = 400
n_layers = 1
num_gaussian = 20   # number of gaussian for Mixture Density Network
num_attn_gaussian = 10  # number of gaussians for attention window
dropout_p = 0.2
# batch_size = 100    # cuda memory issue
batch_size = 32
max_seq = 700
min_seq = 400       # omit samples below min sample length
max_text_seq = 40       # max length of text sequence 
print_every = batch_size*20
# plot_every = 3
plot_every = 20
    
learning_rate = 0.0005
print_loss = 0
total_loss = torch.Tensor([0]).cuda()
print_loss_total = 0
teacher_forcing_ratio = 1      # do not change this right now
clip = 10.0
np.random.seed(9987)
# epochs = 60
epochs =21
rnn_type = 2 # 1 for gru, 2 for lstm

lr_model = model_congen(input_size = 3, hidden_size=hidden_size, num_gaussian=num_gaussian,\
                 dropout_p = dropout_p, n_layers= n_layers, batch_size=batch_size,\
                 num_attn_gaussian = num_attn_gaussian, rnn_type = rnn_type).to(device)

model_optimizer = optim.Adam(lr_model.parameters(), lr=learning_rate)

num_mini_batch = 6000 - batch_size   # to ensure last batch is of batch length

# which writer we're copying
chosen_writing_idx = 10
strokes = np.load('data/strokes.npy', encoding='latin1', allow_pickle=True)
with open('data/sentences.txt') as f:
    texts = f.readlines()
texts = [a.split('\n')[0] for a in texts]
chosen_stroke = strokes[chosen_writing_idx]
chosen_sentence = texts[chosen_writing_idx]
# print(chosen_stroke.shape, chosen_stroke)
print(chosen_sentence)
plot_stroke(chosen_stroke)

lr_model, char_to_vec, h_size = load_pretrained_congen()
a,b,c,d = sample_prime(lr_model,'welcome to lyrebird',chosen_sentence,chosen_stroke,char_to_vec,hidden_size,time_step=2000,
                       bias1=0.5, bias2=0.5)
# a,b,c,d = sample_congen(lr_model,'welcome to lyrebird',char_to_vec, hidden_size, time_step=800)
print(a)
plot_stroke(a)
