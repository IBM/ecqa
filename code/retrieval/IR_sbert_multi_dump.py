from sentence_transformers import SentenceTransformer, models, SentencesDataset, losses, evaluation, util
from sentence_transformers.readers import InputExample
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import json
import random
import argparse
import pathlib
from nltk.translate.bleu_score import sentence_bleu


def create_corpus(filenames):
	train_corpus = []
	total_corpus = []
	for filename in [filenames[0], filenames[1]]:
		with open(filename) as f:
			data = json.load(f)
		train_corpus.extend(data['text'])
		total_corpus.extend(data['text'])
	with open(filenames[2]) as f:
		data = json.load(f)
	total_corpus.extend(data['text'])
	return train_corpus, total_corpus

def create_train_samples(train_file, neg_pos_ratio):
	train_examples = []
	with open(train_file) as f:
		data = json.load(f)
		num_samples = len(data['q_text'])
		for i in range(num_samples):
			query = data['q_text'][i]
			if data['correct'][i]:
				query += ' ' + data['option'][i]
			else:
				query += ' not ' + data['option'][i]
			properties = data['property'][i]
			for prop in properties:
				train_examples.append(InputExample(texts=[query, prop], label=1.0))

	return train_examples

def create_val_set(val_file, pos_only=False):
	val_sentences1 = []
	val_sentences2 = []
	val_queries = []
	val_sentences_all = []
	q_nums = []
	correct = []
	scores = []
	with open(val_file) as f:
		data = json.load(f)
		num_samples = len(data['q_text'])
		for i in range(num_samples):
			query = data['q_text'][i]
			q_nums.append(data['q_no'][i])
			correct.append(data['correct'][i])
			properties = data['property'][i]
			val_sentences_all.append(properties)
			if correct:
				query += ' ' + data['option'][i]
				val_queries.append(query)
				for prop in properties:
					val_sentences1.append(query)
					val_sentences2.append(prop)
					scores.append(1.0)
			elif not pos_only:
				query += ' not ' + data['option'][i]
				val_queries.append(query)
				for prop in properties:
					val_sentences1.append(query)
					val_sentences2.append(prop)
					scores.append(1.0)
	return val_sentences1, val_sentences2, q_nums, correct, val_queries, val_sentences_all, scores


def get_accuracy_multiple(model, sentences1, sentences2, corpus, corpus_embeddings, output_filename):
	# Query sentences:
	queries = sentences1

	# Find the closest k sentences of the corpus for each query sentence based on cosine similarity
	top_k = 15
	hit_1_score = 0
	hit_3_score = 0
	hit_5_score = 0
	hit_10_score = 0
	file = open(output_filename, 'w', encoding='utf-8')

	for i, query in enumerate(queries):
		query_embedding = model.encode(query, convert_to_tensor=True)
		cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
		cos_scores = cos_scores.cpu()
		#We use np.argpartition, to only partially sort the top_k results
		top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]


		results = []
		for idx in top_results[0:top_k]:
			# print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
			results.append(corpus[idx].strip())
		if (any(item in results[:10] for item in sentences2[i])):
			hit_10_score += 1
		if (any(item in results[:5] for item in sentences2[i])):
			hit_5_score += 1
		if (any(item in results[:3] for item in sentences2[i])):
			hit_3_score += 1
		if (any(item in results[:1] for item in sentences2[i])):
			hit_1_score += 1
		lines = ['Input: '+query+'\n', 'Key: '+sentences2[i][0]+'\n', 'SBERT: '+results[0]+'\n', '\n']
		file.writelines(lines)

	file.close()
	hitset1 = hit_1_score / len(queries)
	hitset3 = hit_3_score / len(queries)
	hitset5 = hit_5_score / len(queries)
	hitset10 = hit_10_score / len(queries)
	return hitset1, hitset3, hitset5, hitset10


def get_accuracy_ranks(model, sentences1, sentences2, corpus, corpus_embeddings, k=25):
	# Query sentences:
	queries = sentences1

	# Find the closest k sentences of the corpus for each query sentence based on cosine similarity
	top_k = k
	recall_rate = 0
	binary_recall_rate = 0
	min_k_value = 0
	count_k = 0
	average_golds = 0
	# file = open(output_filename, 'w', encoding='utf-8')

	for i, query in enumerate(queries):
		query_embedding = model.encode(query, convert_to_tensor=True)
		cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
		cos_scores = cos_scores.cpu()
		#We use np.argpartition, to only partially sort the top_k results
		top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
		average_golds += len(sentences2[i])
		results = []
		found_all = False
		for index in range(0,top_k):
			idx = top_results[index]
			sent = corpus[idx].strip()
			if (sent in sentences2[i] and sent not in results):
				results.append(sent)
			if len(results)==len(sentences2[i]):
				found_all = True
				min_k_value += (index+1)
				count_k += 1
				break
		if found_all:
			binary_recall_rate += 1
			recall_rate += 1
		else:
			binary_recall_rate += 0
			recall_rate += (len(results)/len(sentences2[i]))

		# lines = ['Input: '+query+'\n', 'Key: '+sentences2[i][0]+'\n', 'SBERT: '+results[0]+'\n', '\n']
		# file.writelines(lines)

	# file.close()
	recall_rate /= len(queries)
	binary_recall_rate /= len(queries)
	average_golds /= len(queries)
	min_k_value /= count_k
	return recall_rate, binary_recall_rate, average_golds, min_k_value


def dump_file_air(model, input_file, output_file, corpus, corpus_embeddings, k=50):
	air_data = []
	top_k = k

	with open(input_file) as f:
		data = json.load(f)
		num_samples = len(data['q_text'])
		for i in range(num_samples):
			air_instance = {}
			air_instance['id'] = data['q_no'][i]
			air_instance['paragraph'] = {}

			query = data['q_text'][i]
			if data['correct'][i]:
				query += ' ' + data['option'][i]
				answer_text = data['option'][i]
				properties = data['property'][i]
			else:
				query += ' not ' + data['option'][i]
				answer_text = 'not ' + data['option'][i]
				properties = data['property'][i]
			query_embedding = model.encode(query, convert_to_tensor=True)
			cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
			cos_scores = cos_scores.cpu()
			top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
			results = []
			for idx in top_results[0:top_k]:
				results.append(corpus[idx].strip())
			used_sents = []
			air_para_text = ''
			for (idx, sent) in enumerate(results):
				air_para_text += '<b>Sent ' + str(idx+1) + ': </b>' + sent + '<br>'
				if sent in properties:
					used_sents.append(idx)

			air_instance['paragraph']['text'] = air_para_text
			air_para_question = {}
			air_para_question['question'] = data['q_text'][i]
			air_para_question['sentences_used'] = used_sents
			air_para_question['idx'] = 0
			air_para_question['multisent'] = True
			air_answer = {'text': answer_text, 'isAnswer': True, 'Correctness':data['correct'][i], 'scores':{}}
			air_para_question['answers'] = [air_answer]
			air_instance['paragraph']['questions'] = [air_para_question]

			air_data.append(air_instance)

	with open(output_file, 'w') as jf:
		json.dump({'data': air_data}, jf)


def dump_file_hitsets(model, queries, q_nums, correct, sentences, corpus, corpus_embeddings, output_filename):
	inps = []
	golds = []
	outps_3 = []
	outps_5 = []
	outps_10 = []

	top_k = 15

	for i, query in enumerate(queries):
		query_embedding = model.encode(query, convert_to_tensor=True)
		cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
		cos_scores = cos_scores.cpu()
		#We use np.argpartition, to only partially sort the top_k results
		top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]


		results = []
		for idx in top_results[0:top_k]:
			results.append(corpus[idx].strip())


		inps.append(query)
		golds.append(sentences[i])
		outps_3.append(results[:3])
		outps_5.append(results[:5])
		outps_10.append(results[:10])


	df = {'QNo':q_nums, 'Input':inps, 'Gold':golds, 'Output':outps_3, 'Correctness':correct}
	with open('3_'+output_filename, 'w') as f:
		json.dump(df, f)
	# df = {'QNo':q_nums, 'Input':inps, 'Gold':golds, 'Output':outps_5, 'Correctness':correct}
	# with open('5_'+output_filename, 'w') as f:
	# 	json.dump(df, f)
	# df = {'QNo':q_nums, 'Input':inps, 'Gold':golds, 'Output':outps_10, 'Correctness':correct}
	# with open('10_'+output_filename, 'w') as f:
	# 	json.dump(df, f)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-max_length', type = int, default = 30)
	parser.add_argument("-embedding_size", type=int, default = 256)
	parser.add_argument('-batch_size', type = int, default = 90)
	parser.add_argument('-lr', type = float, default = 2e-5)
	parser.add_argument('-neg_pos_ratio', type = int, default = 5)
	parser.add_argument('-warmup_steps', type = int, default = 100)
	parser.add_argument('-evaluation_steps', type = int, default = 500)
	parser.add_argument('-max_eps', type = int, default = 5)
	parser.add_argument('-test', action='store_true')
	parser.add_argument('-pos_only', action='store_true')
	parser.add_argument('-test_omcs', action='store_true')
	parser.add_argument("-output_file",type=str, default='dx_output.txt')
	parser.add_argument("-dump_file",type=str, default='dx_sets.json')
	parser.add_argument("-output_dir", type=str, default='./output/')
	parser.add_argument("-model_save_dir", type=str, default='./checkpoints/')
	parser.add_argument("-pretrained_model", type=str, default='')
	args = parser.parse_args()
	print('Parsed Args - ')


	print(args)

	dir = '/SBERT'
	output_file_dir = args.output_dir + dir
	pathlib.Path(output_file_dir).mkdir(parents=True, exist_ok=True)
	model_save_path = args.model_save_dir + dir

	filenames = ['./data/ED_cqa_train.json', './data/ED_cqa_val.json', './data/ED_cqa_test.json']
	train_file = './data/E2_train.json'
	val_file = './data/E2_val.json'
	test_file = './data/E2_test.json'
	omcs_file = './data/ED_omcs.json'

	train_corpus, total_corpus = create_corpus(filenames)
	corpus = total_corpus

	if args.pretrained_model != '':
		model = SentenceTransformer(args.pretrained_model + '/')
	else:
		word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=args.max_length)
		pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
		dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=args.embedding_size, activation_function=nn.Tanh())
		model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

	if args.test_omcs:
		with open(omcs_file) as f:
			data = json.load(f)
			train_corpus.extend(data['text'])
			corpus = train_corpus


	if args.test:
		# needs to be checked
		# Corpus with example sentences
		corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
		test_sentences1, test_sentences2, q_nums, correct, queries, test_sentences_all, _ = create_val_set(test_file, args.pos_only)
		if args.test_omcs:
			output_filename = 'omcs_' + args.dump_file
			dump_file_hitsets(model, queries, q_nums, correct, test_sentences_all, corpus, corpus_embeddings, output_filename)
			# dump_file_air(model, train_file, './air_omcs_train.json', corpus, corpus_embeddings)
			# dump_file_air(model, val_file, './air_omcs_val.json', corpus, corpus_embeddings)
			dump_file_air(model, test_file, './air_omcs_test.json', corpus, corpus_embeddings)
			exit(0)

		output_filename = args.dump_file
		dump_file_hitsets(model, queries, q_nums, correct, test_sentences_all, corpus, corpus_embeddings, output_filename)

		test_output_filename = output_file_dir + 'test_' + args.output_file
		hitset1, hitset3, hitset5, hitset10 = get_accuracy_multiple(model, queries, test_sentences_all, corpus, corpus_embeddings, test_output_filename)
		print("Test hits@1 score = ", hitset1)
		print("Test hits@3 score = ", hitset3)
		print("Test hits@5 score = ", hitset5)
		print("Test hits@10 score = ", hitset10)

		# dump_file_air(model, train_file, './air_train.json', corpus, corpus_embeddings)
		# dump_file_air(model, val_file, './air_val.json', corpus, corpus_embeddings)
		dump_file_air(model, test_file, './air_test.json', corpus, corpus_embeddings)

		exit(0)

	#Define your train dataset, the dataloader and the train loss
	train_examples = create_train_samples(train_file, args.neg_pos_ratio)
	train_dataset = SentencesDataset(train_examples, model)
	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
	train_loss = losses.MultipleNegativesRankingLoss(model)

	val_sentences1, val_sentences2, q_nums, correct, val_queries, val_sentences_all, scores = create_val_set(val_file)
	evaluator = evaluation.EmbeddingSimilarityEvaluator(val_sentences1+['random sentence'], val_sentences2+['unrelated for -1 score'], scores+[-1.0], args.batch_size)

	model_save_path += '/multi_lr_'+str(args.lr)+'_emb_'+str(args.embedding_size)
	model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=args.max_eps, warmup_steps=args.warmup_steps, evaluator=evaluator, evaluation_steps=args.evaluation_steps, output_path_ignore_not_empty=True, output_path=model_save_path, optimizer_params={'lr':args.lr})

	corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

	train_sentences1, train_sentences2, q_nums, correct, train_queries, train_sentences_all, _ = create_val_set(train_file)
	train_output_filename = output_file_dir + 'train_' + args.output_file
	hitset1, hitset3, hitset5, hitset10 = get_accuracy_multiple(model, train_queries, train_sentences_all, corpus, corpus_embeddings, train_output_filename)
	print("Train hits@1 score = ", hitset1)
	print("Train hits@3 score = ", hitset3)
	print("Train hits@5 score = ", hitset5)
	print("Train hits@10 score = ", hitset10)

	val_output_filename = output_file_dir + 'val_' + args.output_file
	hitset1, hitset3, hitset5, hitset10 = get_accuracy_multiple(model, val_queries, val_sentences_all, corpus, corpus_embeddings, val_output_filename)
	print("Val hits@1 score = ", hitset1)
	print("Val hits@3 score = ", hitset3)
	print("Val hits@5 score = ", hitset5)
	print("Val hits@10 score = ", hitset10)
