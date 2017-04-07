import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import math
import time
import os

import sconce

n_epochs = 200
n_iters = 100
hidden_size = 50
n_layers = 2
dropout_p = 0.1
learning_rate = 0.05

sconce.start(os.environ['SCONCE_PROGRAM_ID'])
sconce.log_every = n_iters

from data import *
from model import *
from evaluate import *

# # Training

def train(input_variable, target_variable):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    encoder_outputs = Variable(torch.zeros(MAX_LENGTH, encoder.hidden_size))
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_hidden = encoder_hidden
    
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        loss += criterion(decoder_output[0], target_variable[di])
        decoder_input = target_variable[di] # Teacher forcing

    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length

def save_model(model, filename):
    torch.save(model, filename)
    print('Saved %s as %s' % (model.__class__.__name__, filename))

def save():
    save_model(encoder, 'seq2seq-encoder.pt')
    save_model(decoder, 'seq2seq-decoder.pt')

encoder = EncoderRNN(input_lang.size, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.size, n_layers, dropout_p=dropout_p)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

try:
    print("Training for %d epochs..." % n_epochs)

    for epoch in range(n_epochs):
        training_pairs = generate_training_pairs(n_iters)

        for i in range(n_iters):
            input_variable = training_pairs[i][0]
            target_variable = training_pairs[i][1]
            loss = train(input_variable, target_variable)

            sconce.record((n_iters * epoch) + i, loss)

        evaluate_tests()

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

