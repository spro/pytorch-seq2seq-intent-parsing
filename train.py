import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import math
import time
import os

import sconce

n_epochs = 50
n_iters = 100
hidden_size = 50
learning_rate = 0.05

sconce.start(os.environ['SCONCE_PROGRAM_ID'])
sconce.log_every = n_iters

from data import *
from model import *

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

# # Evaluating the trained model

def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = input_lang.variable_from_sentence(sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()
    
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    total_prob = 0
    
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            break
        else:
            total_prob += topv[0][0]
            decoded_words.append(output_lang.index2word[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))
    
    return decoded_words, total_prob, decoder_attentions[:di+1]

test_sentences = [
    'um can you turn on the office light',
    'how are you today',
    'please make the music loud',
    'whats the weather in minnesota',
    'whats the weather in sf',
    'are you on',
    'is my light on'
]

def evaluate_tests():
    for test_sentence in test_sentences:
        command, prob, attn = evaluate(test_sentence)
        if prob > -0.05:
            print(prob, command)
        else:
            print(prob, "UNKNOWN")

def save_model(model, filename):
    torch.save(model, filename)
    print('Saved %s as %s' % (model.__class__.__name__, filename))

def save():
    save_model(encoder, 'seq2seq-encoder.pt')
    save_model(decoder, 'seq2seq-decoder.pt')

encoder = EncoderRNN(input_lang.size, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.size, 2, dropout_p=0.1)

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

