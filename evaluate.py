from data import *
from model import *

encoder = torch.load('seq2seq-encoder.pt')
decoder = torch.load('seq2seq-decoder.pt')

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

if __name__ == '__main__':
    import sys
    input = sys.argv[1]
    print('input', input)
    command, prob, attn = evaluate(input)
    if prob > -0.05:
        print(prob, command)
    else:
        print(prob, "UNKNOWN")

