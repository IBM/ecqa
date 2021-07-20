import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import json

def Dump_E2_data(total_data, filename):
	final_data = {'q_no': total_data['q_no'], 'q_text': total_data['q_text'], 'option': total_data['option'],
	 				'free_flow': total_data['free_flow'], 'correct': total_data['correct'], 'property': total_data['properties']}
	with open(filename, 'w') as fp:
	    json.dump(final_data, fp)

def generate_data(df):
	print(df.shape)
	q_no = []
	q_text = []
	option = []
	correct = []
	properties = []
	free_flow = []
	all_properties = []
	wasted = 0

	# need to handle modifiers like plural etc

	for ind in df.index:
		base_index = len(q_no)
		curr_set_options = []
		all_options = []
		all_properties.extend(df['taskA_pos'][ind].splitlines())
		all_properties.extend(df['taskA_neg'][ind].splitlines())
		skipFlag = False
		for i in range(1,6):
			if type(df['q_op'+str(i)][ind]) == type('string'):
				all_options.append(df['q_op'+str(i)][ind].strip().lower())
			else:
				skipFlag = True
		if skipFlag:
			continue
		all_options_set = set(all_options)
		for i in range(1,6):
			q_no.append(df['q_no'][ind])
			q_text.append(df['q_text'][ind].strip().lower())
			free_flow.append(df['taskB'][ind].strip().lower().translate({ord(c): '' for c in "!@#$%^&*()[]{};:,/<>?\|`~-=_+"}))
			curr_option = all_options[i-1]
			option.append(curr_option)
			props_raw = None
			if curr_option==df['q_ans'][ind].strip().lower():
				correct.append(True)
				props_raw = df['taskA_pos'][ind].splitlines()
			else:
				correct.append(False)
				props_raw = df['taskA_neg'][ind].splitlines()
			props_cleaned = [property.lower().translate({ord(c): '' for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"}) for property in props_raw if (property.strip())]
			found = False
			for j in range(0,len(curr_set_options)):
				if curr_option==curr_set_options[j]:
					copy_from_index = base_index + j
					linked_props = properties[copy_from_index]
					properties.append(linked_props)
					found = True
					break
			if found:
				continue
			option_tokens = curr_option.split()
			linked_props = []
			if len(option_tokens)==1:
				search_for = ps.stem(curr_option)
				for property in props_cleaned:
					if search_for in ' '.join([ps.stem(w) for w in property.lower().split()]):
						linked_props.append(property)
			else:
				for token in option_tokens:
					if sum(token in opt for opt in all_options_set)==1:		# found a unique token for this option
						search_for = ps.stem(token)
						for property in props_cleaned:
							if search_for in ' '.join([ps.stem(w) for w in property.lower().split()]):
								linked_props.append(property)
			linked_props = list(set(linked_props))
			if (len(linked_props)==0 and correct[-1] and len(props_cleaned)>0):
				linked_props.append(props_cleaned[0])		# adding one base positive property for correct answer
			if len(linked_props)==0:
				q_no.pop()
				q_text.pop()
				free_flow.pop()
				option.pop()
				correct.pop()
				wasted += 1
			else:
				properties.append(linked_props)
				curr_set_options.append(curr_option)

	all_properties = [prop.lower().translate({ord(c): '' for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"}) for prop in all_properties if (prop.strip() and len(prop.split())<20)]	# removes empty lines
	total = len(set(q_no))*5
	print('Total = ', total, ' Wasted Qs in E2 = ', wasted)
	e2_data = {'q_no':  q_no, 'q_text': q_text, 'option': option, 'free_flow': free_flow,
					'correct': correct, 'properties': properties, 'all_properties': all_properties }
	return e2_data



train_df = pd.read_csv('cqa_data_train.csv', dtype='unicode', header=(0), error_bad_lines=False)
train_data = generate_data(train_df)
val_df = pd.read_csv('cqa_data_val.csv', dtype='unicode', header=(0), error_bad_lines=False)
val_data = generate_data(val_df)
test_df = pd.read_csv('cqa_data_test.csv', dtype='unicode', header=(0), error_bad_lines=False)
test_data = generate_data(test_df)

with open('train_data.json', 'w') as fp:
	json.dump(train_data, fp)

with open('val_data.json', 'w') as fp:
	json.dump(val_data, fp)

with open('test_data.json', 'w') as fp:
	json.dump(test_data, fp)

with open('ED_omcs.json') as f:
    omcs_data = json.load(f)

mixed_props = omcs_data['text']
mixed_props.extend(train_data['all_properties'])

mixed_data = {'text':  mixed_props}

with open('ED_mixed.json', 'w') as fp:
    json.dump(mixed_data, fp)

ED_train_data = {'text': train_data['all_properties']}
ED_val_data = {'text': val_data['all_properties']}
ED_test_data = {'text': test_data['all_properties']}

with open('ED_cqa_train.json', 'w') as fp:
    json.dump(ED_train_data, fp)

with open('ED_cqa_val.json', 'w') as fp:
    json.dump(ED_val_data, fp)

with open('ED_cqa_test.json', 'w') as fp:
    json.dump(ED_test_data, fp)

Dump_E2_data(train_data, 'E2_train.json')
Dump_E2_data(val_data, 'E2_val.json')
Dump_E2_data(test_data, 'E2_test.json')
