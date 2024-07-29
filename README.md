# Graph Neural Network-Based Prediction of Hammett's Constant Parameters

This repository contains the following Python scripts for developing various Torch models trained on a molecular dataset using the Deep Graph Library (DGL) and DGL-LifeSci (DGLLS). These scripts allow you to train, evaluate pre-trained models (AFP, GCN, Weave), and perform cross-validation on a molecular property prediction task:

- model.py – For training different torch models
- eval.py – For evaluating different pre-trained models
- cross_val.py – For cross validating different torch models
- utils.py - Houses all the functions and modules utilized in this study

Other files/folders:

- data.csv – Contains the SMILES strings (SMILES), Hammet sigma (Sigma), IDs (prefix A for para-substituted compounds and prefix B for meta-substituted compounds), and Type (m for meta-substituted compounds and p for para-substituted compounds) values for a total of 985 molecules. 
- requirements.txt – contains all the dependencies
- train_index – contains the index of train set saved as a pickle file.
- final_models - saved torch models for reproducibility

## Torch Model Training

The following Python script model.py was used for training different models. 

Usage

Clone the repository:

```python:
git clone https://github.com/v-saini/hammet-gnn.git
```
Change directory:

```python:
cd hammet-gnn
```
Create a virtual environment using conda:

```python:
conda create --n <env> pyton=3.9.18
```
Activate the environment:

```python:
conda activate <env> 
```
Install dependencies from requirements.txt file.
PyTorch, PyTorch Geometric, DGL, and DGL-LifeSci should be installed using the instructions provided on their respective websites, depending on whether you have CUDA support or not.
After successful installation follow the instructions below:

Use the following command to train a specific model:

```python:
python model.py <model_name>
```

Replace <model_name> with one of afp, afp_final, gcn, nf, gat or weave.

The abbreviations stands for the following models:

- afp – Attentive FP model using default parameters
- afp_final – Attentive FP model using tuned parameters
- gcn – Graph Convolution model using default parameters
- nf – Neural Fingerprint model using default parameters
- gat – Graph Attention Network using default parameters
- weave – Weave model using default parameters

Example:

```python:
python model.py afp
```

This will use the Attentive FP model using default parameters and train it on the dataset.

## Torch Model Evaluation

The following Python script eval.py was used for evaluating different models. 

Similar steps can be followed as before:


Use the following command to evaluate a specific model:

```python:
python eval.py <model_name>
```

Replace <model_name> with one of afp, afp_final, gcn, nf, gat or weave.

The complete model name for the above mentioned abbreviations have been provided in the previous section.

Example:

```python:
python eval.py afp
```

This will load the afp_default.pth model and evaluate it on the dataset.


## Torch Model Cross-Validation

The following Python script cross_val.py was used for cross validating different models. 

Similar steps can be followed as before:

Use the following command to cross validate using a specific model:

```python:
python cross_val.py <model_name>
```
Replace <model_name> with one of afp, gcn, nf, gat or weave.

The complete model name for the above mentioned abbreviations have been provided in the previous section.

Example:
```python:
python cross_val.py afp
```

This will cross validate the dataset using Attentive FP model under default set of parameters.

