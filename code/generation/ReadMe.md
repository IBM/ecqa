# Generation Models

This section of the repository contains the instructions to preprocess the data in order to be fed into fine-tuning the GPT2 model given by the [pytorch-huggingface](huggingface.co/) library. Note that we have modified the [example scripts](https://github.com/huggingface/transformers/tree/master/examples/pytorch) provided by huggingface to fine-tune GPT2 and generate the text from the trained model.

## Data Preprocessing

First run the data pre-processing scripts from the [retrieval section](https://github.com/ShouryaAggarwal/Explanations-for-CommonSenseQA/tree/master/retrieval), and paste all the obtained json files in the data directory of the current root.

```bash
cp ../retrieval/data/*.json ./data/
```
Then run the following commands to generate the processed data for training and running inference on the models.

```bash
cd data
python3 preprocess.py
python3 preprocess_E2.py
python3 preprocess_free_flow.py
python3 preprocess_props_freeflow.py
```

In order to have the data files to train the XGF-II variant, you will first need to complete the training of XGP (see below for that) and run the following commands with the trained XGP model -

```bash
cd text-generation
python3 run_GPT2.py -test_file ../data/E2_GPT_train.json -pretrained_model <path to trained XGP> -max_length 150 -model_type gpt2 -output_file ../data/gpt2_props_train_output.json
python3 run_GPT2.py -test_file ../data/E2_GPT_valid.json -pretrained_model <path to trained XGP> -max_length 150 -model_type gpt2 -output_file ../data/gpt2_props_val_output.json
python3 run_GPT2.py -test_file ../data/E2_GPT_test.json -pretrained_model <path to trained XGP> -max_length 150 -model_type gpt2 -output_file ../data/gpt2_props_test_output.json
cd ../data
python3 preprocess_outp_props_freeflow.py
```

## XGP-W

### Training

```bash
cd language-modeling
python3 run_language_modeling_queries.py --output_dir=<path to save model> --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../data/E2_GPT_train.txt --do_eval --eval_data_file=../data/E2_GPT_valid.txt --per_device_train_batch_size=10 --per_device_eval_batch_size=10 --line_by_line --evaluation_strategy=epoch --learning_rate 5e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 100000 --block_size 250 --prediction_loss_only
```

### Inferencing

```bash
cd text-generation
python3 run_GPT2.py -test_file ../data/E2_GPT_test.json -pretrained_model <path to saved model> -max_length 150 -model_type gpt2 -output_file gpt2_raw_output.json
python3 GPT_json_output_parsing.py -input gpt2_raw_output.json -output gpt2_raw_output.json
```

## XGF-I

### Training

```bash
cd language-modeling
python3 run_language_modeling_queries.py --output_dir=<path to save model> --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../data/E2_GPT_freeflow_train.txt --do_eval --eval_data_file=../data/E2_GPT_freeflow_valid.txt --per_device_train_batch_size=10 --per_device_eval_batch_size=10 --line_by_line --evaluation_strategy=epoch --learning_rate 5e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 100000 --block_size 250 --prediction_loss_only
```

### Inferencing

```bash
cd text-generation
python3 run_GPT2_freeflow.py -test_file ../data/E2_GPT_freeflow_test.json -pretrained_model <path to saved model> -max_length 250 -model_type gpt2 -output_file gpt2_raw_freeflow_output.json
```

## XGP

### Training

```bash
cd language-modeling
python3 run_language_modeling_queries.py --output_dir=<path to save intermediate model> --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../data/train.txt --do_eval --eval_data_file=../data/valid.txt --per_device_train_batch_size=10 --per_device_eval_batch_size=10 --line_by_line --evaluation_strategy=epoch --learning_rate 5e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 100000 --block_size 50 --prediction_loss_only
python3 run_language_modeling_queries.py --output_dir=<path to save model> --model_type=gpt2 --model_name_or_path=<path to saved intermediate model> --do_train --train_data_file=../data/E2_GPT_train.txt --do_eval --eval_data_file=../data/E2_GPT_valid.txt --per_device_train_batch_size=10 --per_device_eval_batch_size=10 --line_by_line --evaluation_strategy=epoch --learning_rate 5e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 100000 --block_size 250 --prediction_loss_only
```

### Inferencing

```bash
cd text-generation
python3 run_GPT2.py -test_file ../data/E2_GPT_test.json -pretrained_model <path to saved model> -max_length 150 -model_type gpt2 -output_file gpt2_props_output.json
python3 GPT_json_output_parsing.py -input gpt2_props_output.json -output gpt2_props_output.json
```

## XGF-II

### Training

```bash
cd language-modeling
python3 run_language_modeling_queries.py --output_dir=<path to save intermediate model> --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=../data/E2_GPT_props_freeflow_train.txt --do_eval --eval_data_file=../data/E2_GPT_props_freeflow_valid.txt --per_device_train_batch_size=10 --per_device_eval_batch_size=10 --line_by_line --evaluation_strategy=epoch --learning_rate 5e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 100000 --block_size 400 --prediction_loss_only
python3 run_language_modeling_queries.py --output_dir=.<path to save model> --model_type=gpt2 --model_name_or_path=<path to saved intermediate model> --do_train --train_data_file=../data/E2_GPT_outp_props_freeflow_train.txt --do_eval --eval_data_file=../data/E2_GPT_outp_props_freeflow_valid.txt --per_device_train_batch_size=10 --per_device_eval_batch_size=10 --line_by_line --evaluation_strategy=epoch --learning_rate 5e-5 --num_train_epochs=5 --overwrite_output_dir --save_steps 100000 --block_size 400 --prediction_loss_only
```

### Inferencing

```bash
cd text-generation
python3 run_GPT2_freeflow.py -test_file ../data/E2_GPT_outp_props_freeflow_test.json -pretrained_model <path to saved model> -max_length 250 -model_type gpt2 -output_file gpt2_outp_props_freeflow_tuned_output.json
```

## Evaluation Setup

#### STS-BERT
We use the [semantic-text-similarity project](https://pypi.org/project/semantic-text-similarity/) to compute the STS-BERT scores. It is an easy-to-use interface to fine-tuned BERT models for computing semantic similarity between 2 sentences.

Install this project using:
```bash
pip install semantic-text-similarity
```
The web-based STS Bert model is downloaded by the 'generation_eval.py' script. Use 'cpu' or 'gpu' according to your machine specs in this script.

#### SPICE
You will first need to download the [Stanford CoreNLP 3.6.0](https://stanfordnlp.github.io/CoreNLP/index.html) code and models for use by SPICE. To do this, run:
```bash
./get_stanford_models.sh
```
Note: SPICE will try to create a cache of parsed sentences in ./spice/cache/. This dramatically speeds up repeated evaluations. Caching can be turned off by removing the '-cache' argument to 'spice_cmd' in all the evaluation scripts. 

#### CIDEr
CIDEr evaluation code is taken from [Consensus-based Image Description Evaluation (CIDEr Code)](https://github.com/vrama91/cider).
First clone their github repo in this folder:
```bash
git clone https://github.com/vrama91/cider
```
During running the evaluaton scripts, if you get a unicode error because of CIDEr, then go to the 'pyciderevalcap/tokenizer/ptbtokenizer.py' file in the cider folder. And in the tokenize function of PTBtokenizer class, update the "prepare data for PTB Tokenizer" block  with this code:
```bash
if self.source == 'gts':
  image_id = [k for k, v in captions_for_image.items() for _ in range(len(v))]
  sentences = '\n'.join([c['caption'].replace('\n', ' ') for k, v in captions_for_image.items() for c in v])
  sentences = sentences.encode('ascii', 'ignore').decode('ascii')
  final_tokenized_captions_for_image = {}

elif self.source == 'res':
  index = [i for i, v in enumerate(captions_for_image)]
  image_id = [v["image_id"] for v in captions_for_image]
  sentences = '\n'.join(v["caption"].replace('\n', ' ') for v in captions_for_image )
  sentences = sentences.encode('ascii', 'ignore').decode('ascii')
  final_tokenized_captions_for_index = []
```
#### METEOR
Follow meteor Readme for downloading one data file before evaluation. Use interactive notebook for calculating METEOR Scores.

## Evaluating Model Output
For evaluation of property generation models (XGP and XGP-W), run the following command:
```bash
python generation_eval_props.py -i input_file
For example: 
For XGP, run this: python generation_eval_props.py -i ./text-generation/gpt2_props_output.json
For XGP-W, run this: python generation_eval_props.py -i ./text-generation/gpt2_raw_output.json
```
For evalution of free-flow generation models (XGF-I and XGF-II), run the following command:
```bash
python generation_eval_free_flow.py -i input_file
For example: 
For XGF-II, run this: python generation_eval_free_flow.py -i ./text-generation/gpt2_outp_props_freeflow_tuned_output.json
For XGF-I, run this: python generation_eval_free_flow.py -i ./text-generation/gpt2_raw_freeflow_output.json
```

## License
[Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
