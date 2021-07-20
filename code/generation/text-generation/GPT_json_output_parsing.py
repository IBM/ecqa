import numpy as np
import json
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-input", type=str, default=None, required=True)
parser.add_argument("-output", type=str, default=None, required=True)
args = parser.parse_args()

with open(args.input) as f:
	inp_data = json.load(f)

inp = []
inp_raw = inp_data['input']
golds = []
golds_raw = inp_data['gold']
outps = []
outps_raw = inp_data['output']
correct = inp_data['correct']
for i, input in enumerate(inp_raw):
	input = input.split()
	input.remove('<BOS>')
	if '<BOP>' in input:
		input.remove('<BOP>')
	input.remove('question:')
	inp.append(' '.join(input))

	reference = golds_raw[i]
	reference = reference.split()
	while '<EOS>' in reference:
		reference.remove('<EOS>')
	while '<EOP>' in reference:
		reference.remove('<EOP>')
	reference = ' '.join(reference)
	golds.append([gold_prop.strip() for gold_prop in reference.split('<BOP>')])
	candidate = outps_raw[i]
	candidate = candidate.split()
	while '<EOP>' in candidate:
		candidate.remove('<EOP>')
	candidate = ' '.join(candidate)
	outps.append([prop.strip() for prop in candidate.split('<BOP>')])


df = {'Input':inp, 'Gold':golds, 'Output':outps, 'Correctness':correct}

with open(args.output, 'w') as f:
	json.dump(df, f)
