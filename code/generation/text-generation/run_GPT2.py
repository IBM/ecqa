from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
# import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import argparse
import json
from nltk.translate.bleu_score import sentence_bleu
# import bcolz
# import pickle
import os
import pathlib

# from dataloader import GPT2Dataset

MODEL_CLASSES = {
	"gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
directory = ''


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-model_type",
		default='gpt2',
		type=str,
		help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=1.0,
		help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
	)
	parser.add_argument("-k", type=int, default=5)
	parser.add_argument("-p", type=float, default=0.9)
	parser.add_argument("-stop_token", type=str, default='<EOS>', help="Token at which text generation is stopped")
	parser.add_argument("-test_file", type=str, default=None, required=True, help="Test file")
	parser.add_argument("-prefix", type=str, default="", help="Text added prior to input.")
	parser.add_argument("-padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
	parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

	parser.add_argument("-batch_size", type=int, default=1)
	# parser.add_argument("-max_eps", type=int, default=10)
	parser.add_argument("-max_length", type=int, default=30)
	parser.add_argument("-output_file", type=str, default='gpt2_output.json')
	parser.add_argument("-pretrained_model", type=str, default='')
	parser.add_argument("-output_dir", type=str, default='./output/')

	args = parser.parse_args()

	print('Parsed Args - ')
	print(args)

	gpu_no = 0 if args.no_cuda else torch.cuda.device_count()



	args.model_type = args.model_type.lower()
	model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

	filename_test = args.test_file
	# test_set = GPT2Dataset(filename = 'data/E2_GTP_test.json', maxlen1 = 1.5*args.max_length, maxlen2 = args.max_length, tokenizer=tokenizer, device=device)
	# test_loader = DataLoader(val_set, batch_size = args.batch_size)
	directory = 'GPT2_output_k_' + str(args.k)

	with open(filename_test) as f:
		data = json.load(f)

	input_data = data['input']
	output_data = data['output']
	q_no = data['q_no']
	q_text = data['q_text']
	option = data['option']
	correct = data['correct']
	free_flow = data['free_flow']

	if args.pretrained_model!='':
		tokenizer = tokenizer_class.from_pretrained(args.pretrained_model)
		model = model_class.from_pretrained(args.pretrained_model)
	else:
		tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='[PAD]')
		model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

	model.to(device)

	avg_blue_score = 0.0
	count = 0

	output_save_dir = os.path.join(args.output_dir, directory)
	pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)
	# output_filename = os.path.join(output_save_dir, args.output_file)
	output_filename = args.output_file

	# file = open(output_filename,'w', encoding='utf-8')
	inps = []
	outps = []
	golds = []
	props = []
	for idx in range(0,len(input_data)):
		input_text = input_data[idx]
		output_text = output_data[idx]

		prompt_text = input_text
		prefix = args.prefix if args.prefix else args.padding_text
		encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=True, return_tensors="pt")
		encoded_prompt = encoded_prompt.to(device)

		if encoded_prompt.size()[-1] == 0:
			input_ids = None
		else:
			input_ids = encoded_prompt

		output_sequences = model.generate(
			input_ids=input_ids,
			max_length=args.max_length + len(encoded_prompt[0]),
			temperature=args.temperature,
			top_k=args.k,
			top_p=args.p,
			repetition_penalty=1.0,
			do_sample=True,
			num_return_sequences=3,
		)

		# Remove the batch dimension when returning multiple sequences
		if len(output_sequences.shape) > 2:
			output_sequences.squeeze_()

		generated_sequences = []

		for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
			# print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
			generated_sequence = generated_sequence.tolist()

			# Decode text
			text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

			# Remove all text after the stop token
			text = text[: text.find(args.stop_token) if args.stop_token else None]

			# Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
			# total_sequence = (
			# 	prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
			# )
			total_sequence = (
				text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
			)

			generated_sequences.append(total_sequence)
			# print(total_sequence)

		model_output = generated_sequences[0]
		bleu_reference = [tokenizer.tokenize(output_text)]
		bleu_candidate = tokenizer.tokenize(model_output)
		score = sentence_bleu(bleu_reference, bleu_candidate)
		# print(input[0] + output_sequence, flush=True)
		inps.append(input_text)
		golds.append(output_text)
		outps.append(model_output)

		candidate = model_output.split()
		while '<EOP>' in candidate:
			candidate.remove('<EOP>')
		candidate = ' '.join(candidate)
		props.append([prop.strip() for prop in candidate.split('<BOP>')])

		# lines = ['inp: '+input_text+'\n', 'gold: '+output_text+'\n', 'outp: '+model_output+'\n', '\n']
		# file.writelines(lines)
		avg_blue_score += score
		count += 1

	avg_blue_score /= count
	print("Average Blue score ", avg_blue_score)
	# file.close()
	final_data = {'q_no': q_no, 'q_text': q_text, 'option': option, 'correct': correct, 'input':  inps, 'output': outps, 'gold': golds, 'property': props, 'free_flow': free_flow}
	with open(output_filename, 'w') as fp:
	    json.dump(final_data, fp)
