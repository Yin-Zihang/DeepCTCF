# Training DeepCTCF

This document describes how to train DeepCTCF using user-provided
quantitative DNA-binding affinity datasets and generate a trained HDF5
model file.

# 1. Requirements

DeepCTCF was developed using:

- Python 3.8.13
- TensorFlow 2.4.1
- Keras 2.4.3
- NumPy 1.24.4
- Pandas 1.4.2
- SciPy 1.7.1
- scikit-learn 0.24.2


The environment can be recreated using:
conda create -n DeepCTCF python=3.8.13
conda activate DeepCTCF
pip install tensorflow==2.4.1
pip install keras==2.4.3
pip install numpy==1.24.4
pip install pandas==1.4.2
pip install scipy==1.7.1
pip install scikit-learn==0.24.2

# 2. Training data preparation

DeepCTCF can be retrained using user-provided quantitative DNA-binding
affinity datasets.

Input format:

    affinity_value    DNA_sequence

Requirements: - Sequence length: 42 bp - DNA alphabet: A/C/G/T -
Quantitative affinity values are required

# 3. Data preprocessing

Sequences are converted using one-hot encoding:

    A = [1,0,0,0]
    C = [0,1,0,0]
    G = [0,0,1,0]
    T = [0,0,0,1]

Each sequence is represented as a 42 × 4 matrix.

# 4. DeepCTCF model architecture

Architecture:

    Input (42,4)
    Conv1D 128 filters, kernel size 5
    MaxPooling1D
    Conv1D 128 filters, kernel size 10
    MaxPooling1D
    Conv1D 128 filters, kernel size 4
    MaxPooling1D
    Flatten
    Dense 32
    Dropout 0.2
    Dense 1

# 5. Model training

Run:

``` bash
python train.py
```

Training parameters:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | Mean squared error |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping patience | 10 |
| L2 regularization | 0.3 |
| Dropout | 0.2 |

# 6. Output model

Training generates:

    example.h5

The pretrained model used in this study is:

    DeepCTCF.h5

# 7. Prediction using trained models

Prediction:

``` bash
python DeepCTCF_predict.py
```

Example files: - New_Sequences.fasta - DeepCTCF_predicted.csv

# 8. Retraining DeepCTCF with new datasets

Prepare a dataset following the required format, modify the input path
in `train.py`, and run:

``` bash
python train.py
```

The resulting HDF5 model can be used for prediction.

# Repository structure


    DeepCTCF/
    │
    ├── README.md
    ├── training_tutorials/
        ├── train.py
        └── TRAINING.md
    ├──model/
        └── DeepCTCF.h5
    ├── DeepCTCF_predict.py
    └── examples/
        ├── DeepCTCF_predict.py
        ├── New_Sequences.fasta
        └── DeepCTCF_predicted.csv

   
