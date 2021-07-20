import os
import argparse
import subprocess
from tqdm import tqdm

from semantic_text_similarity.models import WebBertSimilarity
from semantic_text_similarity.models import ClinicalBertSimilarity

web_model = WebBertSimilarity(device='cpu', batch_size=10) #defaults to GPU prediction

# clinical_model = ClinicalBertSimilarity(device='cuda', batch_size=10) #defaults to GPU prediction

import numpy as np
import pdb
import json

class GFG: 
  def __init__(self,graph): 
    self.graph = graph 
    self.ppl = len(graph) 
    self.jobs = len(graph[0]) 

  def bpm(self, u, matchR, seen): 
    for v in range(self.jobs):  
      if self.graph[u][v] and seen[v] == False: 
        seen[v] = True
        if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen): 
          matchR[v] = u 
          return True
    return False
  
  # Returns maximum number of matching 
  def maxBPM(self):
    matchR = [-1] * self.jobs 

    result = 0
    for i in range(self.ppl):
      seen = [False] * self.jobs
      if self.bpm(i, matchR, seen): 
        result += 1
    return result, matchR

def my_lcs(string, sub):
  if(len(string)<= len(sub)):
    sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
      for i in range(1,len(string)+1):
        if (string[i-1] == sub[j-1]):
          lengths[i][j] = lengths[i-1][j-1] + 1
        else:
          lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

class Rouge():
  def __init__(self):
    self.beta = 1.2

  def calc_score(self, candidate, refs):
    assert(len(candidate)==1)   
    assert(len(refs)>0)         
    prec = []
    rec = []

  # split into tokens
    token_c = candidate[0].split(" ")
        
    for reference in refs:
      # split into tokens
      token_r = reference.split(" ")
      # compute the longest common subsequence
      lcs = my_lcs(token_r, token_c)
      if (lcs == None):
        prec.append(0)
        rec.append(0)
      else:
        prec.append(lcs/float(len(token_c)))
        rec.append(lcs/float(len(token_r)))

      prec_max = max(prec)
      rec_max = max(rec)

      if (prec_max!=0 and rec_max !=0):
        score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
      else:
        score = 0.0
      return score

  def compute_score(self, refs, test):
    score = []
    for i in range(len(refs)):
      hypo = test[i]
      ref = refs[i]
      if (hypo == " " or hypo == ""):
        score.append(0)
      else:
        score.append(self.calc_score([hypo], [ref]))
    
    average_score = np.mean(np.array(score))
    return average_score, np.array(score)

  def method(self):
    return "Rouge"
    


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

    

    # Other Metrics
    os.chdir('./cider')
    os.system('pwd')
    spice_thresholder = []
    cider_refs_thresholder = []
    cider_test_thresholder = []
    count = 0
    for k in range(len(data["input"])):
      # if (k % 500 == 0):
      #   print(k)
      l1 = data["gold"][k]
      l2 = data["output"][k]
      # correctness = data["Correctness"][k]
      # if not correctness:
      #   l2 = [data["Output"][k][0]]
      # else:
      #   l2 = data["Output"][k]
      struct = {
          "image_id": count,
          "caption": l2
      }
      cider_test_thresholder.append(struct)
      struct = {
          "image_id": count,
          "caption": l1
      }
      cider_refs_thresholder.append(struct)
      struct = {
          "image_id": count,
          "test": l2,
          "refs": [l1]
      }
      spice_thresholder.append(struct)
      count += 1

    with open('./data/cider_' + name + '_refs.json', 'w') as outfile:
        json.dump(cider_refs_thresholder, outfile)
    with open('./data/cider_' + name + '_test.json', 'w') as outfile:
        json.dump(cider_test_thresholder, outfile)
    with open('../spice/spice_' + name + '.json', 'w') as outfile:
        json.dump(spice_thresholder, outfile)

    params = {
      "pathToData" : "./data/",
        "refName" : 'cider_' + name + '_refs.json',
        "candName" : 'cider_' + name + '_test.json',
        "resultFile" : 'cider_' + name + '_results.json',
        "idf" : "coco-val-df"
    }
    with open('./params.json', 'w') as outfile:
        json.dump(params, outfile)

    os.system('python2 ./cidereval.py')

    file2 = open('./cider_' + name + '_results.json')
    cider_output = json.load(file2)

    # %cd ../spice/
    os.chdir('../spice')
    write_file = open(name + "_meteor_refs", "w")
    for i in range(len(cider_refs_thresholder)):
      new_line = cider_refs_thresholder[i]['caption'].replace("\n", " ") + " \n"
      write_file.write(new_line)
    write_file.close()

    write_file2 = open(name + "_meteor_test", "w")
    for i in range(len(cider_test_thresholder)):
      if (cider_test_thresholder[i]['caption'] == "" or cider_test_thresholder[i]['caption'] == " "):
        new_line = "empty \n"
      else:
        new_line = cider_test_thresholder[i]['caption'].replace("\n", " ") + " \n"
      write_file2.write(new_line)
    write_file2.close()

    # f1 = name + '_meteor_test'
    # f2 = name + '_meteor_refs'
    
    # meteor_scores = out('java -Xmx2G -jar meteor-1.5.jar ' + f1 + ' ' + f2 + ' -l en -norm -a data/paraphrase-en.gz -q')
    # # meteor_scores = os.system('java -Xmx2G -jar meteor-1.5.jar ' + f1 + ' ' + f2 + ' -l en -norm -a data/paraphrase-en.gz -q')
    # meteor_scores = [float(meteor_scores[i]) for i in range(len(meteor_scores))]
    # # meteor_scores[-1]
    # print(len(meteor_scores))

    #in spice directory
    f1 = 'spice_' + name + '.json'
    f2 = 'spice_' + name + '_output.json'  
    cwd = os.path.abspath('')
    cache_dir=os.path.join(cwd, 'cache')
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)
    os.system('java -Xmx8G -jar spice-1.0.jar ' + f1 + ' -cache ./cache -out ' + f2)


    file2 = open('./spice_' + name + '_output.json')
    spice_output = json.load(file2)
    os.chdir('../')
    # len(spice_output)

    rouge_test =  [cider_test_thresholder[i]['caption'] for i in range(len(cider_test_thresholder))]
    rouge_refs =  [cider_refs_thresholder[i]['caption'] for i in range(len(cider_refs_thresholder))]
    r = Rouge()
    rouge_scores = r.compute_score(rouge_refs, rouge_test)
    # rouge_scores[0]


    spice_predictions_array = []
    cider_predictions_array = []
    # meteor_predictions_array = []
    rouge_predictions_array = []
    for k in range(len(spice_output)):
    #   if (k % 500 == 0):
    #     print(k)
      # l1 = data["Gold"][k]
      # l2 = data["GPT2_Output"][k]

      spice_predictions_array.append(spice_output[k]['scores']['All']['f'])
      cider_predictions_array.append(cider_output["CIDEr"][k])
      # meteor_predictions_array.append(meteor_scores[k])
      rouge_predictions_array.append(rouge_scores[1][k])

    print("SPICE==============")
    print('Average: ', np.average(spice_predictions_array))
    print("CIDEr==============")
    print('Average: ', np.average(cider_predictions_array)/10)
    # print("METEOR==============")
    # print(np.average(meteor_predictions_array))
    print("ROUGE==============")
    print('Average: ', np.average(rouge_predictions_array))
    
    
    print("STS-BERT==============")
    sts_bert_output = [web_model.predict([(data['gold'][i], data['output'][i])])[0] for i in tqdm(range(len(data['input'])))]
    print('Average: ', np.average(sts_bert_output)/5)

if __name__ == '__main__':
    main()
