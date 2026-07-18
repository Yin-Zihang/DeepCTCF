# DeepCTCF

DeepCTCF is a convolutional neural network model developed to quantitatively predict CTCF-DNA binding affinity from 42-bp DNA sequences.
The model was trained using MpEMSA-seq data and predicts intrinsic CTCF-DNA binding affinity based on DNA sequence information.


### Predict the Affinity to CTCF for new DNA sequences
To predict the affinity to CTCF protein for new DNA sequences, please run:

```
# Clone this repository

git clone https://github.com/Yin-Zihang/DeepCTCF.git
```

# create 'DeepCTCF' conda environment by running the following:
```
conda create --name DeepCTCF python=3.8.13 tensorflow=2.4.1 keras=2.4.3 numpy=1.24.4 pandas=1.4.2 scipy=1.7.1
source activate DeepCTCF 
conda install scikit-learn
```

# Run prediction script

```
cd DeepCTCF/examples
python DeepCTCF_predict.py \
-s New_Sequences.fasta \
-m ../model/DeepCTCF.h5
```


Where:
* -s FASTA file with input DNA sequences
* The prediction results are saved as: DeepCTCF_predicted.csv.
