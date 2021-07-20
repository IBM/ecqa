import json
import re

def make_data_file(filename_in, filename_out):
	with open(filename_in) as f:
		data = json.load(f)
	q_texts = data['q_text']
	options = data['option']
	properties = data['property']
	corrects = data['correct']

	f = open(filename_out, 'w')
	data = ''
	for i in range(0,len(q_texts)):
		q_text = q_texts[i].strip()
		option = options[i].strip()
		q_properties = properties[i]
		correct = corrects[i]

		bos_token = '<BOS>'
		eos_token = '<EOS>'
		text = bos_token + ' question: '+q_text+' '
		if correct:
			text += 'The answer is ' + option + ' because '
		else:
			text += 'The answer is not ' + option + ' because '

		for j in range(0,len(q_properties)):
			property = q_properties[j]
			text += '<BOP> ' + property + ' <EOP> '

		text += eos_token + '\n'
		# text = re.sub(r"\s", " ", text)
		data += text
	f.write(data)

def make_test_file(filename_in, filename_out):
	with open(filename_in) as f:
		data = json.load(f)
	q_texts = data['q_text']
	options = data['option']
	properties = data['property']
	corrects = data['correct']

	data_text = []
	data_target = []
	q_no = []
	free_flow = []
	for i in range(0,len(q_texts)):
		q_text = q_texts[i].strip()
		option = options[i].strip()
		q_num = data['q_no'][i]
		free_flow_exp = data['free_flow'][i]
		q_properties = properties[i]
		correct = corrects[i]
		bos_token = '<BOS>'
		eos_token = '<EOS>'
		text = bos_token + ' question: ' + q_text + ' '
		if correct:
			text += 'The answer is ' + option + ' because <BOP> '
		else:
			text += 'The answer is not ' + option + ' because <BOP> '
		for j in range(0,len(q_properties)):
			property = q_properties[j]
			if j==0:
				target = property + ' <EOP> '
			else:
				target += '<BOP> ' + property + ' <EOP> '
		# target += eos_token
		# text = re.sub(r"\s", " ", text)
		# data += text
		data_text.append(text)
		data_target.append(target)
		q_no.append(q_num)
		free_flow.append(free_flow_exp)
	final_data = {'q_no': q_no, 'q_text': q_texts, 'option': options, 'correct': corrects, 'input':  data_text, 'output': data_target, 'free_flow': free_flow}
	with open(filename_out, 'w') as fp:
	    json.dump(final_data, fp)

def make_noisy_test_file(filename_in, filename_out):
	with open(filename_in) as f:
		data = json.load(f)
	q_texts = data['q_text']
	options = data['option']
	properties = data['property']
	corrects = data['correct']

	data_text = []
	data_target = []
	data_noisy = []
	for i in range(0,len(q_texts)):
		q_text = q_texts[i].strip()
		option = options[i].strip()
		q_properties = properties[i]
		correct = corrects[i]
		bos_token = '<BOS>'
		eos_token = '<EOS>'
		text = bos_token + ' question: ' + q_text + ' '
		if correct:
			text_correct = text + 'The answer is ' + option + ' because <BOP> '
			text_noisy = text + 'The answer is not ' + option + ' because <BOP> '
		else:
			text_correct = text + 'The answer is not ' + option + ' because <BOP> '
			text_noisy = text + 'The answer is ' + option + ' because <BOP> '
		for j in range(0,len(q_properties)):
				property = q_properties[j]
				if j==0:
					target = property + ' <EOP> '
				else:
					target += '<BOP> ' + property + ' <EOP> '
		target += eos_token
		# text = re.sub(r"\s", " ", text)
		# data += text
		data_text.append(text_correct)
		data_target.append(target)
		data_noisy.append(False)
		data_text.append(text_noisy)
		data_target.append(target)
		data_noisy.append(True)
	final_data = {'input':  data_text, 'output': data_target, 'noise': data_noisy}
	with open(filename_out, 'w') as fp:
	    json.dump(final_data, fp)

make_data_file('E2_train.json', 'E2_GPT_train.txt')
make_data_file('E2_val.json', 'E2_GPT_valid.txt')
make_test_file('E2_train.json', 'E2_GPT_train.json')
make_test_file('E2_val.json', 'E2_GPT_valid.json')
make_test_file('E2_test.json', 'E2_GPT_test.json')
# make_noisy_test_file('E2_test.json', 'E2_GPT_test_noisy.json')
