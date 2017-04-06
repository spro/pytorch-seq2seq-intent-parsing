import somata
from evaluate import *

def parse(input, cb):
    command, prob, attn = evaluate(input)
    if prob > -0.05:
        print(prob, command)
        cb(command)
    else:
        print(prob, "UNKNOWN")
        cb(None)

commands = {'parse': parse}
service = somata.Service('maia:parser', commands, {'bind_port': 7181})
