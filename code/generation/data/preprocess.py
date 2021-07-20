import random
import json
import re

train_filename = 'ED_mixed.json'
val_filename = 'ED_cqa_val.json'
test_filename = 'ED_cqa_test.json'

def get_text(filename):
	with open(filename, 'r') as fp:
		data = json.load(fp)
	all_text = data['text']
	return all_text


def build_dataset(df, dest_path):
    f = open(dest_path, 'w')
    data = ''
    df_text = df
    for text in df_text:
        text = str(text).strip()
        text = re.sub(r"\s", " ", text)
        bos_token = '<BOP>'
        eos_token = '<EOP>'
        data += bos_token + ' ' + text + ' ' + eos_token + '\n'

    f.write(data)


build_dataset(get_text(train_filename), 'train.txt')
build_dataset(get_text(val_filename), 'valid.txt')
build_dataset(get_text(test_filename), 'test.txt')
