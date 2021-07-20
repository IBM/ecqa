import os
import argparse
import subprocess

import numpy as np
import pdb
import json


parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("-i", "--input", required=True)
#parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

input_file = str(args.input)

def main():
    name = input_file.split('/')
    name = name[-1].replace('.json','')
    print(name)
    # os.system('pwd')
    f = open(input_file)
    data = json.load(f)

    count = 0
    recall = 0.0
    precision = 0.0
    f_score = 0.0
    for i in range(0, len(data["Input"])):
      # if i % 1000 == 0:
      #   print(i)
      true_props = data["Gold"][i]
      # retrieved_props = data["Output"][i]
      if not data["Correctness"][i]:
        retrieved_props = [data["Output"][i][0]]
      else:
        retrieved_props = data["Output"][i]
      num_found = 0
      recall0 = 0
      precision0 = 0
      for prop in retrieved_props:
        if prop in true_props:
          num_found += 1
      precision += float(num_found)/len(retrieved_props)
      precision0 = float(num_found)/len(retrieved_props)

      num_found = 0
      for prop in true_props:
        if prop in retrieved_props:
          num_found += 1
      recall += float(num_found)/len(true_props)
      recall0 = float(num_found)/len(true_props)
      count += 1
      if precision0 + recall0 == 0:
        f_score += 0
      else:
        f_score += 2*precision0*recall0/(precision0 + recall0)

    precision /= count
    recall /= count
    f_score /= count
    print("Exact Score==========")
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1 Score: ', f_score)

if __name__ == '__main__':
    main()
