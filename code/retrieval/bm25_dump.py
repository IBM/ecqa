from rank_bm25 import BM25Okapi
import argparse
import json

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-dump_file",type=str, default='bm25_sets.json')
	parser.add_argument('-test_omcs', action='store_true')

	args = parser.parse_args()

	corpus = []

	if args.test_omcs:
		filenames = ['./data/ED_cqa_train.json', './data/ED_cqa_val.json', './data/ED_omcs.json']
	else:
		filenames = ['./data/ED_cqa_train.json', './data/ED_cqa_val.json', './data/ED_cqa_test.json']

	for filename in filenames:
		with open(filename) as f:
			data = json.load(f)
			corpus.extend(data['text'])

	test_filenames = ['./data/E2_train.json', './data/E2_val.json', './data/E2_test.json']
	outp_dict = ['train', 'val', 'test']
	# test_filenames = ['./data/E2_test.json']
	tokenized_corpus = [doc.split(" ") for doc in corpus]
	top_k = 50

	bm25 = BM25Okapi(tokenized_corpus)
	for filenum, filename in enumerate(test_filenames):
		air_data = []
		if filenum==2:
			inps = []
			golds = []
			outps_3 = []
			outps_5 = []
			outps_10 = []
			q_nums = []
			correct = []
			with open(filename) as f:
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
				else:
					query += ' not ' + data['option'][i]
					answer_text = 'not ' + data['option'][i]
				properties = data['property'][i]
				tokenized_query = query.split(" ")
				results = bm25.get_top_n(tokenized_query, corpus, n=top_k)
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

				q_nums.append(data['q_no'][i])
				inps.append(query)
				golds.append(properties)
				correct.append(data['correct'][i])
				outps_3.append(results[:3])
				outps_5.append(results[:5])
				outps_10.append(results[:10])

		if filenum==2:
			filename = args.dump_file
			if args.test_omcs:
				filename = 'omcs_' + filename
			df = {'QNo':q_nums, 'Input':inps, 'Gold':golds, 'Output':outps_3, 'Correctness':correct}
			with open('3_'+filename, 'w') as f:
				json.dump(df, f)
			# df = {'QNo':q_nums, 'Input':inps, 'Gold':golds, 'Output':outps_5, 'Correctness':correct}
			# with open('5_'+filename, 'w') as f:
			# 	json.dump(df, f)
			# df = {'QNo':q_nums, 'Input':inps, 'Gold':golds, 'Output':outps_10, 'Correctness':correct}
			# with open('10_'+filename, 'w') as f:
			# 	json.dump(df, f)

			filename = outp_dict[filenum] + '.json'
			if args.test_omcs:
				filename = 'omcs_' + filename
			with open('air_bm25_'+filename, 'w') as jf:
				json.dump({'data': air_data}, jf)
