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
    for k in range(len(data["Input"])):
      # if (k % 500 == 0):
      #   print(k)
      l1 = data["Gold"][k]
      l2 = data["Output"][k]
      # correctness = data["Correctness"][k]
      # if not correctness:
      #   l2 = [data["Output"][k][0]]
      # else:
      #   l2 = data["Output"][k]
      for i in range(len(l1)):
        for j in range(len(l2)):
          struct = {
              "image_id": count,
              "caption": l2[j]
          }
          cider_test_thresholder.append(struct)
          struct = {
              "image_id": count,
              "caption": l1[i]
          }
          cider_refs_thresholder.append(struct)
          struct = {
              "image_id": count,
              "test": l2[j],
              "refs": [l1[i]]
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


    spice_recall = 0.0
    spice_precision = 0.0
    spice_fscore = 0.0
    cider_recall = 0.0
    cider_precision = 0.0
    cider_fscore = 0.0
    # meteor_recall = 0.0
    # meteor_precision = 0.0
    # meteor_fscore = 0.0
    rouge_recall = 0.0
    rouge_precision = 0.0
    rouge_fscore = 0.0
    count1 = 0
    count = 0
    spice_threshold = 0.4
    cider_threshold = 3
    meteor_threshold = 0.3
    rouge_threshold = 0.3
    counter = []
    for k in range(len(data["Input"])):
    #   if (k % 500 == 0):
    #     print(k)
      l1 = data["Gold"][k]
      l2 = data["Output"][k]
      
      bipartite_graph = np.zeros((len(l1), len(l2)))
      bipartite_graph_double_spice = np.zeros((len(l1), len(l2)))
      # bipartite_graph_double_meteor = np.zeros((len(l1), len(l2)))
      bipartite_graph_double_rouge = np.zeros((len(l1), len(l2)))
      
      for i in range(len(l1)):
        for j in range(len(l2)):
          cider_score = cider_output['CIDEr'][count1]
          # meteor_score = meteor_scores[count1]
          rouge_score = rouge_scores[1][count1]
          spice_score = spice_output[count1]['scores']['All']['f']
          count1 += 1
          if (spice_score >= spice_threshold):
            bipartite_graph_double_spice[i][j] = 1
          else:
            bipartite_graph_double_spice[i][j] = 0
          if (cider_score >= cider_threshold):
            bipartite_graph[i][j] = 1
          else:
            bipartite_graph[i][j] = 0
          # if (meteor_score >= meteor_threshold):
          #   bipartite_graph_double_meteor[i][j] = 1
          # else:
          #   bipartite_graph_double_meteor[i][j] = 0
          if (rouge_score >= rouge_threshold):
            bipartite_graph_double_rouge[i][j] = 1
          else:
            bipartite_graph_double_rouge[i][j] = 0

      g = GFG(bipartite_graph_double_spice)
      number, division_list = g.maxBPM()
      
      score_recall1 = 0
      score_precision1 = 0
      for i in range(len(l1)):
        j = -1
        for k in range(len(division_list)):
          if (division_list[k] == i):
            j = k
            break
        
        if (j != -1):
          score_recall1 += 1
          score_precision1 += 1
          count += 1
        else:
          count += 1

      spice_recall += score_recall1/len(l1)
      spice_precision += score_precision1/len(l2)
      a1 = score_recall1/len(l1)
      b1 = score_precision1/len(l2)
      if (a1+b1 != 0):
        spice_fscore += 2*a1*b1/(a1+b1)

      g = GFG(bipartite_graph)
      number, division_list = g.maxBPM()
      
      score_recall2 = 0
      score_precision2 = 0
      for i in range(len(l1)):
        j = -1
        for k in range(len(division_list)):
          if (division_list[k] == i):
            j = k
            break
        
        if (j != -1):
          score_recall2 += 1
          score_precision2 += 1
          count += 1 
        else:
          count += 1

      cider_recall += score_recall2/len(l1)
      cider_precision += score_precision2/len(l2)
      a2 = score_recall2/len(l1)
      b2 = score_precision2/len(l2)
      if (a2+b2 != 0):
        cider_fscore += 2*a2*b2/(a2+b2)

      # g = GFG(bipartite_graph_double_meteor) 
      # number, division_list = g.maxBPM()
      
      # score_recall3 = 0
      # score_precision3 = 0
      # for i in range(len(l1)):
      #   j = -1
      #   for k in range(len(division_list)):
      #     if (division_list[k] == i):
      #       j = k
      #       break
      #   if (j != -1):
      #     score_recall3 += 1
      #     score_precision3 += 1
      #     count += 1 
      #   else:
      #     count += 1
      # meteor_recall += score_recall3/len(l1)
      # meteor_precision += score_precision3/len(l2)
      # a2 = score_recall3/len(l1)
      # b2 = score_precision3/len(l2)
      # if (a2+b2 != 0):
      #   meteor_fscore += 2*a2*b2/(a2+b2)

      g = GFG(bipartite_graph_double_rouge) 
      number, division_list = g.maxBPM()
      
      score_recall4 = 0
      score_precision4 = 0
      for i in range(len(l1)):
        j = -1
        for k in range(len(division_list)):
          if (division_list[k] == i):
            j = k
            break
        if (j != -1):
          score_recall4 += 1
          score_precision4 += 1
          count += 1 
        else:
          count += 1
      rouge_recall += score_recall4/len(l1)
      rouge_precision += score_precision4/len(l2)
      a2 = score_recall4/len(l1)
      b2 = score_precision4/len(l2)
      if (a2+b2 != 0):
        rouge_fscore += 2*a2*b2/(a2+b2)

    x = len(data["Input"])
    print("SPICE==============")
    print('Recall: ', spice_recall/x)
    print('Precision: ', spice_precision/x)
    print('F1 Score: ', spice_fscore/x)
    print("CIDEr==============")
    print('Recall: ', cider_recall/x)
    print('Precision: ', cider_precision/x)
    print('F1 Score: ', cider_fscore/x)
    # print("METEOR==============")
    # print(meteor_recall/x)
    # print(meteor_precision/x)
    # print(meteor_fscore/x)
    print("ROUGE==============")
    print('Recall: ', rouge_recall/x)
    print('Precision: ', rouge_precision/x)
    print('F1 Score: ', rouge_fscore/x)
    
    
    # STS-BERT score
    # sts_predictions_array = []
    # sts_predictions_array2 = []
    sts_recall = 0.0
    sts_precision = 0.0
    sts_fscore = 0.0
    count = 0
    sts_threshold = 3

    counter = []
    for k in tqdm(range(len(data["Input"]))):
    #   if (k % 500 == 0):
    #     print(k)
      l1 = data["Gold"][k]
      l2 = data["Output"][k]
      # if data["Correctness"][k]:
      #   l2 = data["Output"][k]
      # else:
      #   l2 = [data["Output"][k][0]]
      bipartite_graph = np.zeros((len(l1), len(l2)))
      bipartite_graph_double = np.zeros((len(l1), len(l2)))
      
      for i in range(len(l1)):
        for j in range(len(l2)):
          sts_score = web_model.predict([(l1[i], l2[j])])[0]
          bipartite_graph_double[i][j] = sts_score
          if (sts_score >= sts_threshold):
            bipartite_graph[i][j] = 1
          else:
            bipartite_graph[i][j] = 0

      # g = GFG(bipartite_graph_double) 
      # number, division_list = g.maxBPM()

      # # property i will be matched with division_list[i]  change this comment
      # score0 = 0
      # for i in range(len(l1)):
      #   j = -1
      #   for k in range(len(division_list)):
      #     if(division_list[k] == i):
      #       j = k
      #       break

      #   if (j != -1):
      #     sts_score = bipartite_graph_double[i][j]
      #     score0 += sts_score
      #     count += 1
      #     # sts_predictions_array2.append(sts_score)
          
      #   else:
      #     count += 1
      #     # sts_predictions_array2.append(0)

      # # sts_predictions_array.append(score0/len(l1))

      g = GFG(bipartite_graph) 
      number, division_list = g.maxBPM()

      # property i will be matched with division_list[i]  change this comment
      score_recall0 = 0
      score_precision0 = 0
      for i in range(len(l1)):
        j = -1
        for k in range(len(division_list)):
          if(division_list[k] == i):
            j = k
            break

        if (j != -1):
          score_recall0 += 1
          score_precision0 += 1
          count += 1
        else:
          count += 1
      sts_recall += score_recall0/len(l1)
      sts_precision += score_precision0/len(l2)
      a0 = score_recall0/len(l1)
      b0 = score_precision0/len(l2)
      if (a0+b0 != 0):
        sts_fscore += 2*a0*b0/(a0+b0)

    x = len(data["Input"])
    print("STS Score==============")
    print('Recall: ', sts_recall/x)
    print('Precision: ', sts_precision/x)
    print('F1 Score: ', sts_fscore/x)

if __name__ == '__main__':
    main()
