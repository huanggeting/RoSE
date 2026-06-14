# RoSE

## 🚀 How to use our code?

# Project Title

This project provides the implementation of **RoSE**, with support for dataset processing, training, and evaluation.

## 1. Setup Environment

Before starting, ensure you have the required dependencies and environment set up.

### 1.1 Create Conda Environment

First, create a new Conda environment with Python 3.9:

```bash
conda create -n RoSE python=3.9
conda activate RoSE
```

### 1.2 Install Dependencies

After activating the environment, install all necessary packages:

```bash
pip install -r requirements.txt
```

### 1.3 Install SpaCy Language Model

To enable language processing with SpaCy, download the `en_core_web_sm` model using the following command:

```bash
python -m spacy download en_core_web_sm
```

---

## 2. Data Preprocessing

You can run download_data.sh to obtain the datasets.

## 3. Training and Inference

The following scripts can be used to train and evaluate models on different datasets:

### 3.1 Training

You can train models by running the corresponding scripts:

```bash
bash ./scripts/train_ace_roberta.sh
bash ./scripts/train_wikievent_roberta.sh
bash ./scripts/train_rams_roberta.sh
bash ./scripts/train_mlee_roberta.sh
```
### 3.2 Inference
You can inference models by running the corresponding scripts:

```bash
bash ./scripts/infer_ace_roberta.sh
bash ./scripts/infer_wikievent_roberta.sh
bash ./scripts/infer_rams_roberta.sh
bash ./scripts/infer_mlee_roberta.sh
```