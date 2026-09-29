# SADD: SNR-Adaptive Diffusion Defense for AMC

This repository provides the implementation of **SNR-Adaptive Diffusion Defense (SADD)** for automatic modulation classification (AMC).

SADD uses a diffusion-based purification module before AMC classification. The diffusion starting timestep can be dynamically predicted from the received signal, enabling adaptive purification under different SNR conditions.

The general pipeline is:

1. Prepare the AMC dataset.
2. Train baseline AMC classifiers if needed.
3. Pretrain the diffusion purification model and the timestep predictor.
4. Train the SADD-defended AMC classifier.
5. Evaluate the trained SADD model under clean samples or adversarial attacks.

---

## 1. Environment Setup
Install `uv` and create the Python environment:
```bash
pip install uv
uv venv --python 3.10
uv init
uv sync
uv run python <script_name>.py [arguments]
```

## 2. Data Preparation
Place the original raw datasets , for example 'RML2016.10a_dict.pkl,' under data/rawdata/, Please refer to the dataset loading logic in data/RML.py for the exact expected file paths and dataset formats.

## 3. Baseline AMC Model Training
To train the models, run:
uv run python defense.py -data a -gid 0 -defense_method nature -ml all
Arguments
-data a: specifies the dataset. Here, a denotes RML2016.10a.
-gid 0: specifies the GPU ID.
-defense_method nature: specifies the defense/training method. 
nature means the original AMC model without adversarial training.
Other defense methods can be used to train AMC models with different adversarial training strategies.
-ml all: trains all supported model architectures.

## 4. Checkpoint Preparation
Before running attacks or evaluation scripts, make sure the trained model checkpoint files are placed under the checkpoints/ directory.
Example checkpoint paths:
checkpoints/RML2016.10a/psr-10.warm10/pgdat/RML2016.10a_awn.best.pt
checkpoints/RML2016.10a/nature/RML2016.10a_awn.best.pt
......
Alternatively, you may directly copy the pre-generated postdata/ and checkpoints/ directories from another source, and then run the evaluation scripts below.

## 5. Diffusion and Timestep Predictor Pretraining
Before training the SADD classifier, the diffusion purification model and the timestep predictor should be pretrained.
Run:
uv run python exp/df_pipeline/pretrain.py \
  --data data/postdata/RML2016.10a_dict.split.pt \
  --out datasets/models/a_joint_start \
  --device cuda:3 \
  --epochs 100 \
  --batch-size 128 \
  --adv-weight 0.0 \
  --start-mode dynamic \
  --gid 3
This command trains and saves:
datasets/models/a_joint_start/diffusion_best.pt
datasets/models/a_joint_start/mlp_best.pt
Arguments
--data data/postdata/RML2016.10a_dict.split.pt
Path to the preprocessed split dataset.
--out datasets/models/a_joint_start
Output directory for saving the pretrained diffusion model and timestep predictor.
--device cuda:3
Device used for training.
--epochs 100
Number of training epochs.
--batch-size 128
Batch size for diffusion pretraining.
--adv-weight 0.0
Weight of adversarial samples in diffusion pretraining.
In this setting, it is set to 0.0, meaning that adversarial samples are not used during diffusion pretraining.
--start-mode dynamic
Specifies the diffusion starting mode.
Available options:
dynamic: use the MLP timestep predictor to estimate a sample-dependent diffusion starting timestep.
zero: use a fixed starting timestep of 0 for all samples.
--gid 3
GPU ID.

## 6. SADD Classifier Training
After pretraining the diffusion model and the timestep predictor, train the SADD-defended AMC classifier.
Example command:
uv run python exp/defense/sadd_train.py \
  --df-ckpt datasets/models/a_joint_start/diffusion_best.pt \
  --s-ckpt datasets/models/a_joint_start/mlp_best.pt \
  --start-mode dynamic \
  -defense_method sadd \
  -data a \
  -model ctdnn \
  -myt 3 \
  --lambda-nmse 0.0 \
  --lambda-pur 1.0 \
  -gid 7
This command trains an AMC classifier with SADD defense using the pretrained diffusion model and timestep predictor.
Arguments
--df-ckpt datasets/models/a_joint_start/diffusion_best.pt
Path to the pretrained diffusion model checkpoint.
--s-ckpt datasets/models/a_joint_start/mlp_best.pt
Path to the pretrained timestep predictor checkpoint.
--start-mode dynamic
Specifies the timestep starting mode used by SADD.
Available options:
dynamic: use the learned MLP to predict the diffusion starting timestep.
zero: use a fixed starting timestep of 0.
-defense_method sadd
Specifies that the classifier is trained with the SADD defense.
-data a
Specifies the dataset. Here, a denotes RML2016.10a.
-model ctdnn
Specifies the AMC classifier architecture.
Supported models may include:
ctdnn
awn
depending on the models implemented in the project.
-myt 3
Specifies the number of diffusion reconstruction steps used during SADD training.
--lambda-nmse 0.0
Weight of the NMSE-related loss term during adversarial training.
--lambda-pur 1.0
Weight of the purification-related loss term during adversarial training.
-gid 7
GPU ID.
The trained SADD classifier checkpoint will be saved under the corresponding experiment output directory.
For example:
yield_test/.../RML2016.10a/fit/ctdnn/checkpoint/RML2016.10a_ctdnn.best.pt

## 7. SADD Evaluation
To evaluate a trained SADD classifier under different input conditions or adversarial attacks, run:
uv run python exp/df_pipeline/evaluate.py \
  --eval-classifier-ckpt yield_test/sadd.advTraining.sadd.3.0.0.0.0/RML2016.10a/fit/ctdnn/checkpoint/RML2016.10a_ctdnn.best.pt \
  --eval-df-ckpt datasets/models/a_joint_start/diffusion_best.pt \
  --eval-s-ckpt datasets/models/a_joint_start/mlp_best.pt \
  --eval-classifier ctdnn \
  --eval-start-mode dynamic \
  --eval-t 3 \
  --eval-out eval_results \
  -gid 7 \
  --eval-mode fci
This script first generates clean or adversarial samples, then applies diffusion purification, and finally evaluates the classification accuracy of the SADD classifier.
Arguments
--eval-classifier-ckpt
Path to the trained SADD classifier checkpoint.
--eval-df-ckpt
Path to the pretrained diffusion model checkpoint.
--eval-s-ckpt
Path to the pretrained timestep predictor checkpoint.
--eval-classifier ctdnn
Classifier architecture used for evaluation.
This should match the architecture of the checkpoint specified by --eval-classifier-ckpt.
--eval-start-mode dynamic
Starting mode used during purification.
Available options:
dynamic: use the MLP timestep predictor.
zero: use fixed starting timestep 0.
--eval-t 3
Number of diffusion reconstruction steps used during evaluation.
--eval-out eval_results
Output directory for evaluation results.
-gid 7
GPU ID.
--eval-mode fci
Specifies the evaluation mode or attack method.
Available options include:
clean
pgd
mi
fci
sfaa
where:
clean: evaluate clean samples with diffusion purification.
pgd: evaluate under PGD attack.
mi: evaluate under MI attack.
fci: evaluate under FCI attack.
sfaa: evaluate under SFAA attack.