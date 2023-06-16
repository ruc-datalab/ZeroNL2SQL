# LLM_NL2SQL
## :open_file_folder: Data Preparation
### Train data
- [Spider](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0): Put it under `src/datasets/spider`.
### Test data
- [KaggleDBQA](https://drive.google.com/file/d/1rckVlUcZ1EB7pLJBUzrfeZlEZIc8ruCr/view?usp=share_link): Put it under `src/datasets/kaggledbqa`.
- [Dr.Spider](https://drive.google.com/file/d/1VN0S5Q5NbFe8MGB21T9f6psGw93LRz00/view?usp=share_link): Put it under `src/datasets/drspider`.

```sh
mkdir data/
unzip src/datasets/kaggledbqa/kaggledbqa.zip -d data/
unzip src/datasets/drspider/drspider.zip -d data/
# Don't delete the original .zip file
```
## :computer: Environment Preparation
![PWC](https://img.shields.io/badge/Python-3.8.3-green)

Please refer to `requirements.txt` to download the relevant toolkits.

Prepare the following folders:
```sh
cd LLM_NL2SQL
mkdir logs
mkdir experimental_outputs/train/template_generator
mkdir experimental_outputs/train/aligner
```

## :zap: Quick Start

### Download models
- [Template Generator](https://drive.google.com/drive/folders/1rkHvECBv7zO58q6_kNSFvfEHeRJGZiP1?usp=sharing): Put it under `experimental_outputs/train/template_generator`.
- [Aligner](https://drive.google.com/file/d/1IvvyYo_S2muVr4HyxHyhZifFlIRlPToD/view?usp=share_link): Put it under `experimental_outputs/train/aligner`.

### Text-to-SQL inference

Use the following script to directly infer on the text-to-sql test set. This script will take four steps: 1. generate SQL template; 2. align (SELECT, STRUCTURE) with the user question; 3. prepare data for LLM inference; 4. text2sql using LLM.
```sh
CUDA_VISIBLE_DEVICES={gpu_id} bash scripts/infer_LLM_with_template.sh test_set_name your_openai_key
```

- The first argument is the name of the test set, which can be selected from `kaggledbqa`, `DB_DBcontent_equivalence`, `DB_schema_abbreviation`, `DB_schema_synonym`, `NLQ_keyword_synonym`, `NLQ_keyword_carrier`, `NLQ_column_synonym`, `NLQ_column_carrier`, `NLQ_column_attribute`, `NLQ_column_value`, `NLQ_value_synonym`, `NLQ_multitype`, `NLQ_others`, `SQL_comparison`, `SQL_sort_order`, `SQL_NonDB_number`, `SQL_DB_text`, `SQL_DB_number`. 
- The second argument is your openai key, which you can obtain from the [official website](https://platform.openai.com/account/api-keys).

Note that we evaluate the text-to-SQL results using the [test_suite_evaluation](https://github.com/taoyds/test-suite-sql-eval), and the evaluation results are presented in `eval.output`.

## :open_hands: Train From Scratch

### Train template generator

```sh
CUDA_VISIBLE_DEVICES={gpu_id} bash -c "python src/run.py configs/train_template_generator.json"
```
The best model will be saved at `experimental_outputs/train/template_generator/BEST_MODEL/`.

### Train aligner

```sh
CUDA_VISIBLE_DEVICES={gpu_id} bash -c "python src/run_aligner.py configs/train_aligner.json"
```
The best model will be saved at `experimental_outputs/train/aligner/checkpoint_best.pkl`.

## :speech_balloon:Citation

If our code is helpful to you, please cite our work:
```sh
@misc{gu2023interleaving,
      title={Interleaving Pre-Trained Language Models and Large Language Models for Zero-Shot NL2SQL Generation}, 
      author={Zihui Gu and Ju Fan and Nan Tang and Songyue Zhang and Yuxin Zhang and Zui Chen and Lei Cao and Guoliang Li and Sam Madden and Xiaoyong Du},
      year={2023},
      eprint={2306.08891},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
