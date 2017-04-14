import somata
from evaluate import *

def parse(input, cb):
    parsed, prob, attn = evaluate(input)
    print(parsed, prob)
    cb({'parsed': parsed, 'prob': prob})

service = somata.Service('maia:parser', {'parse': parse}, {'bind_port': 7181})
