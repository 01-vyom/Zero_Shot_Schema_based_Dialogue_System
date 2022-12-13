# Zero Shot Schema Guided Dialog System for Task Oriented Systems with GPT-2 based responses

### [[Report]](./document/Project%20Final%20Report.pdf) | [[Slides]](./document/Zero_Shot_Bot.pdf)

[Vyom Pathak](https://www.linkedin.com/in/01-vyom/)<sup>1</sup> | [Amogh Mannekote](http://learndialogue.org/person.php?id=amannekote)<sup>1</sup> | [Oluwapemisin Bandy-toyo](https://github.com/pempem12)<sup>1</sup>

[University of Florida](https://www.cise.ufl.edu/)<sup>1</sup>

Designing task-oriented dialogue systems manually has been at the forefront of developing very important systems in the research community. However, when learning NLU and dialogue management from corpora using machine learning algorithms, the tight coupling between the specific task and the general skills precludes the transferability of the learned models to similar, related domains and tasks. With zero-shot learning of dialogue management, the designer of the dialogue agent would not need to spend a lot of time collecting or writing examples for conversations. Instead, he/she can simply exploit the learned behaviors from the related tasks/domains, and would simply have to specify the schema for the new domain/task. Hence, we propose a Schema-based dialogue system for zero-shot task transfer for the NLU unit. For the response generation part, we used zero-shot prompt based Dialogue-GPT2. We performed two user studies to verify the viability of the system.

Please go through the report for more details about the project.

## Setup

### System & Requirements

- Linux OS
- Python-3.6
- TensorFlow-2.2.0

### Setting up repository

  ```shell
  git clone https://github.com/01-vyom/Zero_Shot_Schema_based_Dialogue_System
  python -m venv nn_iop_env
  source $PWD/sgd_gpt2/bin/activate
  ```

### Installing Dependencies

* Change directory to the root of the repository. 
* Follow setup instructions as shown in the [original repository](https://github.com/Shikib/schema_attention_model) for installing dependencies. 
* Download the model checkpoints into the `sam_task_transfer` directory which excludes a task if you are choosing a pre-defined domain, or download all domain trained model for new task interaction.

## Running Code

Change directory to the root of the repository.

### Inferencing

To perform inference on the trained model, run the following command:

```shell
python ./app.py
```

* One can define a new task and add it to the `./STAR/tasks/` directory, and also add a sample example values in `./STAR/apis/apis/`. Change the task name for variable `TASK` in the `app.py` file.
* Following to adding a new task, define new domain and add it to the `./STAR/apis/dbs/` directory. Change the domain name for variable `DOMAIN` in the `app.py` file.

## Acknowledgement
The core part of the project is based on paper titled "Schema-Guided Paradigm for Zero-Shot Dialog"
<details>
<summary>More Details</summary>

# Schema-Guided Paradigm for Zero-Shot Dialog

This is the code for _Schema-Guided Paradigm for Zero-Shot Dialog_ to be published at SIGdial 2021 (paper coming soon).

## Abstract

Developing mechanisms that flexibly adapt dialog systems to unseen tasks and domains is a major challenge in dialog research. Neural models implicitly memorize task-specific dialog policies from the training data. We posit that this implicit memorization has precluded zero-shot transfer learning. To this end, we leverage the *schema-guided paradigm*, wherein the task-specific dialog policy is explicitly provided to the model. We introduce the Schema Attention Model (SAM) and improved schema representations for the STAR corpus. SAM obtains significant improvement in zero-shot settings, with a **+22 F1** score improvement over prior work. These results validate the feasibility of zero-shot generalizability in dialog. Ablation experiments are also presented to demonstrate the efficacy of SAM.

## Instructions for Reproducing Results

### Standard Experiments

The standard experiments assess the performance of a model that is trained and evaluated on the same tasks. Uncomment lines 370 - 372 of `train.py` and run the following command to reproduce the results with SAM on

```
python3.6 train.py --data_path STAR/dialogues/ --schema_path STAR/tasks/ --token_vocab_path bert-base-uncased-vocab.txt --output_dir sam/ --task action --num_epochs 50 --train_batch_size 64 --max_seq_length 100 --schema_max_seq_length 50 --seed 43 --use_schema
```

### Zero-Shot Task Transfer Experiments

The zero-shot task transfer experiments assess the model's ability to train on N-1 tasks and evaluate on the Nth task (N = 23). Uncomment lines 382 - 401 of `train.py` and run the following command to reproduce the results with SAM on

```
python3.6 train.py --data_path STAR/dialogues/ --schema_path STAR/tasks/ --token_vocab_path bert-base-uncased-vocab.txt --output_dir sam_task_transfer/ --task action --num_epochs 10 --train_batch_size 64 --max_seq_length 100 --schema_max_seq_length 50 --seed 42 --use_schema
```

### Zero-Shot Domain Transfer Experiments

The zero-shot domain transfer experiments assess the model's ability to train on N-1 domains and evaluate on the Nth domains (N = 13). Uncomment lines 404 - 423 of `train.py` and run the following command to reproduce the results with SAM on

```
python3.6 train.py --data_path STAR/dialogues/ --schema_path STAR/tasks/ --token_vocab_path bert-base-uncased-vocab.txt --output_dir sam_domain_transfer/ --task action --num_epochs 10 --train_batch_size 64 --max_seq_length 100 --schema_max_seq_length 50 --seed 42 --use_schema
```

## Model Checkpoints

Our model checkpoints can be found at the following links:
- [Standard Experiments with SAM](https://drive.google.com/file/d/1eDbW_xR8jLa-vRzJgY5G2CZm_9l5JbhQ/view?usp=sharing)
- [Zero-Shot Task Transfer Experiments with SAM](https://drive.google.com/file/d/1YTkJ3uXhgc7RI1m499R_nYKo1PsQziJY/view?usp=sharing)
- [Zero-Shot Domain Transfer Experiments with SAM](https://drive.google.com/file/d/19r-QRx0KlzAmPrCyJvjdmtgHnav0xlPN/view?usp=sharing)

## Citations

If you use any of this code or our modified schemas (`STAR/tasks/`) please cite the following paper:

```
TBD
```

The majority of this code is adapted from the work of Mosig et al. Furthermore, the STAR data was produced by Mosig et al. As such, if you use any of the code or the STAR data, please cite the following:

```
@article{mosig2020star,
  title={STAR: A Schema-Guided Dialog Dataset for Transfer Learning},
  author={Mosig, Johannes EM and Mehri, Shikib and Kober, Thomas},
  journal={arXiv preprint arXiv:2010.11853},
  year={2020}
}
```

## Contact

If you have any questions about this code or the paper, please reach out to `amehri@cs.cmu.edu`.
</details>


## Contact

If you have any questions about this code or the paper, please reach out to `v.pathak@ufl.edu`.

Licensed under the [MIT License](LICENSE.md).