# DeepCTCF
Deep learning model built to quantitatively predict the binding affinities of DNA sequences to human CCCTC-binding factor(CTCF)
### Predict the Affinity to CTCF for new DNA sequences
To predict the affinity to CTCF protein for new DNA sequences, please run:
```
# Clone this repository
git clone https://github.com/Yin-Zihang/DeepCTCF.git

# create 'DeepCTCF' conda environment by running the following:
conda create --name DeepCTCF python=3.8.13 tensorflow=2.4.1 keras=2.4.3 numpy=1.19.5 pandas=1.4.2 scipy=1.7.1
source activate DeepCTCF 
conda install scikit-learn

# Run prediction script
python DeepCTCF_predict.py -s New_Sequences.fasta -m DeepCTCF.h5


```
Where:
* -s FASTA file with input DNA sequences
