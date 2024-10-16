# A new approach for fine-tuning sentence transformers for intent classification and out-of-scope detection tasks
## Overview

This repository contains the code and experiments for the paper titled "[A new approach for fine-tuning sentence transformers for intent classification and out-of-scope detection tasks](https://example.com)"

In this paper, we address the challenge of detecting Out-of-Scope (OOS) queries in virtual assistant (VA) systems. Our approach focuses on improving the ability of a transformer-based sentence encoder to distinguish between in-scope and OOS queries by regularizing its sentence embeddings using an autoencoder. The autoencoder helps to limit the global dispersion of in-scope embeddings in the embedding space, which improves the separation between in-scope and OOS samples.

## Key Features

- Transformer Encoder Fine-tuning: We fine-tune a transformer encoder for intent classification while jointly training an autoencoder head to regularize the sentence embeddings.
- OOS Detection: We improve OOS detection by maintaining compact embeddings for in-scope intents, limiting the overlap with OOS data.
- Improved Precision-Recall: Our approach demonstrates a 1-4% improvement in the Area Under the Precision-Recall Curve (AUPR) for OOS detection across multiple benchmark datasets.
- Maintains Intent Classification Accuracy: Despite improving OOS rejection, the model retains high performance for intent classification tasks.

## Datasets

We evaluated our approach on the following datasets:

- CLINC150: A popular benchmark dataset for intent classification with an out-of-scope (OOS) class.
- StackOverflow: A subset of the StackOverflow dataset categorized into 20 distinct intent classes.
- MTOP-EN: A task-oriented dataset with hierarchical intent labels used for evaluating the performance of the proposed model.
- Car Assistant Dataset: An internal dataset that captures user interactions with a virtual assistant in an automotive setting.

## How to Use
1. Clone the Repository
```bash
git clone https://github.com/SlangLab-NU/autoencoder-oos.git
cd autoencoder-oos
```
2. Install Dependencies
Use the requirements.txt file to install the necessary dependencies. We recommend using a virtual environment to avoid conflicts:
```bash
pip install -r requirements.txt
```
3. Run Experiments in jupyter lab. `crossentropy_fine.ipynb` contains code for cross entropy loss only and `autoencoder_fine` contains our experiment method.
```bash
jupyter-lab 
```

## Results
The results show that our approach consistently improves OOS rejection while maintaining high performance on in-scope intent classification tasks. The improvements in the AUPR for OOS detection across multiple datasets demonstrate the effectiveness of autoencoder regularization.

## Citation

TBD


