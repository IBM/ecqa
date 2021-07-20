import json
import re


bos_token = '<BOS>'
eos_token = '<EOS>'


def extract_summary_data(data):
	q_no = []
	inp_text = []
	summary = []
	correct_option = None
	wasted = 0
	for i in range(0, len(data['q_no'])):
		first = True if len(q_no)==0 else False
		if first or data['q_no'][i]!=q_no[-1]:
			if not first:
				if correct_option == None:
					# print('Error in parsing')
					wasted += 1
					# for j in range(i-5,i+1):
					# 	print(data['q_no'][j], data['option'][j], data['correct'][j])
					q_no.pop()
					summary.pop()
				else:
					text += '. The best answer is '+correct_option+' because '
					inp_text.append(text)
				correct_option = None
			q_text = data['q_text'][i].strip()
			q_no.append(data['q_no'][i])
			summary.append(data['free_flow'][i] + eos_token)
			option = data['option'][i].strip()
			correct = data['correct'][i]
			if correct:
				correct_option = option
			text = bos_token + ' question: '+q_text+' The options are '+option
		else:
			option = data['option'][i].strip()
			correct = data['correct'][i]
			if correct:
				correct_option = option
			text += ', ' + option
	if correct_option == None:
		wasted += 1
		q_no.pop()
		summary.pop()
	else:
		text += '. The best answer is '+correct_option+' because '
		inp_text.append(text)
	print(wasted, len(q_no))
	return {'q_no': q_no, 'inp_text': inp_text, 'summary': summary}

def make_data_file(filename_in, filename_out):
	with open(filename_in) as f:
		data = json.load(f)
	processed_data = extract_summary_data(data)
	f = open(filename_out, 'w')
	data = ''
	for i in range(0,len(processed_data['q_no'])):
		text = processed_data['inp_text'][i] + processed_data['summary'][i] + eos_token + '\n'
		data += text
	f.write(data)

def make_test_file(filename_in, filename_out):
	with open(filename_in) as f:
		data = json.load(f)
	processed_data = extract_summary_data(data)

	data_text = []
	data_target = []
	for i in range(0,len(processed_data['q_no'])):
		text = processed_data['inp_text'][i]
		target = processed_data['summary'][i]

		data_text.append(text)
		data_target.append(target)
	final_data = {'input':  data_text, 'output': data_target}
	with open(filename_out, 'w') as fp:
	    json.dump(final_data, fp)

make_data_file('E2_train.json', 'E2_GPT_freeflow_train.txt')
make_data_file('E2_val.json', 'E2_GPT_freeflow_valid.txt')
make_test_file('E2_test.json', 'E2_GPT_freeflow_test.json')
