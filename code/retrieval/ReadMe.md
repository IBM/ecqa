# Retrieval Models

This section of the repository contains the instructions to preprocess the data in order to be fed into our deep retrieval pipeline which has been developed using [sbert](https://www.sbert.net/) library. We have given the code to train and run retrieval using the same.

## Data Preprocessing

First download the [OMCS Corpus file ](https://s3.amazonaws.com/conceptnet/downloads/2018/omcs-sentences-more.txt), which we refer to as the silver corpus, and paste  it in the data directory of the current root.

Make sure you have the following csv data files, pasted in the data directory of the current root.

```
cqa_data.csv
cqa_data_train.csv
cqa_data_val.csv
cqa_data_test.csv
```

Then run the following commands to generate the processed data for training and running inference on the models.

```bash
cd data
python3 ED_omcs_data_gen.py
```

If you wanna run the experiments with the train, validation and test splits used by us (which excluded certain 32 ambiguous tagged questions), then run the following commands -
```bash
cd data
python3 E2_data_generator_author_split.py
```

If you want to use a random 70-10-20 train-dev-test split from the total dataset of 10,962 annotated samples, then run the following commands -
```bash
cd data
python3 E2_data_generator.py
```

## Reproducibility of Paper's Results

To obtain the exact same results as we obtained from our given trained models, you might need to install older versions of some of the python packages. For the same, run the following command -
```bash
pip3 install -r requirements.txt
```

You can skip the above if you are not concerned if reproducing the exact same numbers as reported in our ACL paper.

## BM-25

We use the python library of [rank-bm25](https://pypi.org/project/rank-bm25/) for this baseline. The following command will generate the naive top-k retrieved files, and the input files for AIR method using BM25 method.

### BM-25 Gold Corpus

```bash
python3 bm25_dump.py
```

### BM-25 Silver Corpus

```bash
python3 bm25_dump.py -test_omcs
```

## Deep SBERT based ranker

The following commands describe the training and retrieval procedure with a Sentence BERT based model, for the gold and silver corpus.

### Training

```bash
python3 IR_sbert_multi_dump.py -embedding_size 512 -model_save_dir <directory to save models>
```
You may change the embedding size as per your needs, the above parameter value was the one used to produce results in our ACL 2021 paper.
The above command would save the model to ```<directory to save models>/SBERT/multi_lr_2e-05_emb_512```, where ```2e-05``` is the learning reate (default) and ```512``` is the mentioned embedding size.

The below commands will generate the naive top-k retrieved files, and the input files for AIR method using a model trained by the above method.

### Retrieval with Gold Corpups

```bash
python3 IR_sbert_multi_dump.py -embedding_size 512 -test -pretrained_model <path to the pretrained model as explained above>
```

### Retrieval with Silver Corpups

```bash
python3 IR_sbert_multi_dump.py -embedding_size 512 -test -test_omcs -pretrained_model <path to the pretrained model as explained above>
```

## Evaluation Setup

Navigate over to the generation directory

```bash
cd ../generation
```

Now follow the evaluation steps mentioned [here](https://github.com/dair-iitd/ECQA/tree/master/generation#evaluation-setup) to set up the environment to run evaluation scripts. You need not repeat this step if you have already set up the environment while running generation experiments.

## Evaluating Model Output
#### Gold corpus
Run the following command to evaluate the input_file for top-k approach where k = 3 for positive properties and k = 1 for negative properties for gold corpus. Results would be the Exact Recall, Precision and F scores.
```bash
python retrieval_eval_gold.py -i input_file
For example:
For Ours + top-k approach, run this: python retrieval_eval_gold.py -i 3_dx_sets.json
For BM25 + top-k approach, run this: python retrieval_eval_gold.py -i 3_bm25_sets.json
```

#### Silver corpus
Setup the all the folders as described in README of generation section for evaluation before running following command.
This would evaluate the input_file for top-k approach where k = 3 for positive properties and k = 1 for negative properties for silver corpus. Results would be the Recall, Precision and F scores for different metrics (STS-BERT score, SPICE, CIDEr and ROUGE) using the bipartite matching as described in the paper.
```bash
python retrieval_eval.py -i input_file
For example:
For Ours + top-k approach, run this: python retrieval_eval.py -i 3_omcs_dx_sets.json
For BM25 + top-k approach, run this: python retrieval_eval.py -i 3_omcs_bm25_sets.json
```

## License
[Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
