import argparse
import pandas as pd
import json

parser = argparse.ArgumentParser(description="Evaluation script")
parser.add_argument("-o", "--output", required=True)
parser.add_argument('-t', '--test', required=True)
args = parser.parse_args()

output_file = str(args.output)
test_file = str(args.test)

def main():
    name = output_file.split('/')
    name = name[-1].replace('.tsv','')
    print(name)
    # os.system('pwd')
    f = open(test_file)
    data = json.load(f)

    output_cols = ["start", "index", "zero_zero", "correctness", "ques", "prop0", "prop1", "prop2", "prop3", "prop4", "prop5", "prop6", "prop7","prop8","prop9","prop10","prop11","prop12","prop13","prop14","prop15","prop16","prop17","prop18","prop19","prop20"]
    df = pd.read_csv(output_file, sep='\t', names=output_cols, engine='python', header=None)
    df = df.fillna("")

    output_queries = df['ques']
    output_props = []
    output_correctness = df['correctness']

    for i,row in df.iterrows():
      props = [row['prop0'],row['prop1'],row['prop2'],row['prop3'],row['prop4'],row['prop5'],row['prop6'],row['prop7'],row['prop8'],row['prop9'],row['prop10'],row['prop11'],row['prop12'],row['prop13'],row['prop14'],row['prop15'],row['prop16'],row['prop17'],row['prop18'],row['prop19'],row['prop20']]
      output_props.append(props)
    

    count = 0
    recall = 0.0
    precision = 0.0
    f_score = 0.0
    for i in range(0, len(data['q_text'])):
      # if i % 1000 == 0:
      #   print(i)
      true_props = data['property'][i]
      query = data['q_text'][i].replace('\n','')
      if data['correct'][i]:
        query += ' ' + data['option'][i]
      else:
        query += ' not ' + data['option'][i]

      retrieved_props = []
      found = False
      for cntr in range(i, len(output_queries)):
        if query == output_queries[cntr]:
          retrieved_props = output_props[cntr]
          found = True
          # print(i, cntr)
          break
      if (found == False):
        for cntr in range(0,i):
          if query == output_queries[cntr]:
            retrieved_props = output_props[cntr]
            # print(i, cntr)
            break
      retrieved_props = [x for x in retrieved_props if x]
      
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
        # if 2*precision0*recall0/(precision0 + recall0) == 1.0 and len(retrieved_props) > 1:
        #   print(i)

    precision /= count
    recall /= count
    f_score /= count
    print("Exact score============")
    print('Recall: ', recall)
    print('Precision: ', precision)
    print('F1 Score: ', f_score)

if __name__ == '__main__':
    main()
