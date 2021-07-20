import pandas as pd
import numpy as np
import random
import json

df = pd.read_csv('omcs-sentences-more.txt', dtype='unicode', sep='\t', header=(0), error_bad_lines=False)
bool_en = df.language_id.str.contains('en')
bool_en.fillna(False, inplace=True)
df_en = df[bool_en]
df_en = df_en[df_en['text'].str.split().str.len().lt(20)]
df_en = df_en[df_en['text'].str.split().str.len().gt(0)]

max_length = 20
corpus = [prop.lower().translate({ord(c): '' for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"}) for prop in df_en['text'].values.astype('str').tolist()]
corpus = [prop for prop in corpus if (prop.strip() and len(prop.split())<20)]
print(len(corpus))

omcs_data = {'text':  corpus}

with open('ED_omcs.json', 'w') as fp:
    json.dump(omcs_data, fp)
