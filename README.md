# LLM with Relation Classifier for Document-Level Relation Extraction

Code for paper [LLM with Relation Classifier for Document-Level Relation Extraction](https://www.arxiv.org/abs/2408.13889#)

## Set up

### Prepare Environment for the RC Stage

You can install require packages for the RC stage by running the following command:

```shell
conda create -n lmrc python=3.9
conda activate lmrc
pip install -r requirements.txt
```

### Prepare Environment for the RCP Stage

For the RCP stage, the requirements are:

```text
PyTorch
transformers
numpy
apex=0.1
opt-einsum=3.3.0
wandb
ujson
tqdm
```

Please refer to [ATLOP](https://github.com/wzhouad/ATLOP) for further information. It should be noted that the apex package will report an error when using pip. It is recommended to clone the repository and install it manually.

### Datasets

The DocRED dataset can be downloaded following the instructions at [link](https://github.com/thunlp/DocRED/tree/master/data). The Re-DocRED dataset can be downloaded from [link](https://github.com/tonytan48/Re-DocRED) (You need to replace the original file in DocRED with the file with the suffix 'revised'). The expected structure of files is:

```text
LMRC
 |-- dataset
 |    |-- docred
 |    |    |-- train_annotated.json        
 |    |    |-- train_distant.json
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- rel_info.json
 |    |-- re-docred
 |    |    |-- train_revised.json        
 |    |    |-- dev_revised.json
 |    |    |-- test_revised.json
 |    |    |-- rel_info.json
```

## Run Experiments

### Finetune your model for RC

Run `convert_docred_to_llm.py` first to generate data for finetuning your model.

Finetuning:

```shell
cd RC
bash finetune-lora.sh
```

### For RCP

Train the Roberta model on DocRED/RE-DocRED with the following command:

```shell
cd RCP
bash run_roberta.sh
```

By modifying '--load_dath', it can be converted to testing on dev/test set. Then, run `generate_for_step2.py` to convert structured output into natural language text encapsulated with prompt template. `generate_for_step2.py` will also output an 'idx.json', which is used to determine the document number corresponding to each query.

### For RC

Use the converted RCP results as input for the RC stage:

```shell
cd RC
bash eva-for-lora.sh
```

### Evaluate

Run `convert_llm_result_to_docred` to format the plain text output of the RC stage. 'idx.json' comes from `generate_for_step2.py`.

You can evaluate the structured results with `evaluation.py`, which is modified from [link](https://github.com/thunlp/DocRED/blob/master/code/evaluation.py)

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2024llm,
  title={Llm with relation classifier for document-level relation extraction},
  author={Li, Xingzuo and Chen, Kehai and Long, Yunfei and Zhang, Min},
  journal={arXiv preprint arXiv:2408.13889},
  year={2024}
}
```
