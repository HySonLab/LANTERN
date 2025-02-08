# Data: https://drive.google.com/drive/folders/1liqU_9XgJYENw9KQW6tLWWYylMVvoZaU?usp=drive_link
## Drug - target interaction (DTI) benchmark
+ DAVIS
+ KIBA
+ BIOSNAP
## Protein - protein interaction (PPI) benchmark
+ Yeast
## Drug- drug interaction (DDI) benchmark
+ Deep DDI


# Pretrained weights
Pretrained embeddings of all entities of all datasets can be found at : https://drive.google.com/drive/folders/1Dph9nUsovudogbIv4v8SuYCspcWdMfOI?usp=drive_link

To run all the training and evaluation commands, please ensure that the corresponding dataset and the pretrained embeddings of all entities in the dataset are placed in the LANTERN/data folder.

Example : BioSNAP :
```
LANTERN
├── data
│   ├──BioSNAP ├── pretrained
               ├── train.csv
               ├── val.csv
               ├── test.csv
            
```


