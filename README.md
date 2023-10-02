# Description
**Note that more details will provided soon**

Official repository of the paper [Promoting Generalized Cross-lingual Question Answering in Few-resource Scenarios via Self-knowledge Distillation](https://arxiv.org/abs/2309.17134) containing the implementation to reproduce it. 

Please, refer to the preprint for more details about the fundamental ideas, the method and the evaluation results: URL


# Installation
First, install Python dependencies and extracting training and evaluation data.
```bash
bash ./setup.sh
```
The script creates a Python virtualenv in the `venv` directory and extract all the necessary data in the `corpora` directory.

# How to train
We provide simple script to train the models in the paper. For example, to train the overall best performing model, referred in the paper as _mbert-qa-en, skd, mAP@k_, run the following steps:

1. Standard cross-entropy fine-tuning of the mBERT model for the extractive QA task using the SQuAD v1.1 training dataset in English.
```bash
bash train_qa_mbert.sh
```

2. Cross-lingual fine-tuning of the previous mBERT model, called _mBERT-qa-en_, using **self-knowledge distillation with mAP@k loss coefficients**.
```bash
bash train_skd_map.sh
```

The result will be stored in the `runs` directory along with the tensoboard logs.

All other scripts will fine-tuning the _mBERT-qa-en_ model with different methods.
**Note that each script is use a configuration of hyperparameters correspoding to the best models. Change the configuration inside the scripts to train different models.**

# How to evaluate
We also provide scripts to evaluate the model after training. For example, to evaluate on the _MLQA-test_ dataset, run:

```bash
bash eval_qa.sh <model_path_trained_model> mlqa-test
```

The evaluation result will be stored inside the trained model directory under the name `eval_results_mlqa-test`

Is it possible to choose another test set between `xquad`, `mlqa-dev` and `tydiqa-goldp` datasets.

# How to Cite
To cite our work use the following BibTex:
```
@misc{carrino2023promoting,
      title={Promoting Generalized Cross-lingual Question Answering in Few-resource Scenarios via Self-knowledge Distillation}, 
      author={Casimiro Pio Carrino and Carlos Escolano and José A. R. Fonollosa},
      year={2023},
      eprint={2309.17134},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
