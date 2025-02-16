# LANTERN: Leveraging Large Language Models And Transformer For Enhanced Molecular Interaction

![LANTERN](assets/lantern.png)

https://doi.org/10.1101/2025.02.10.637522

Contributors:
* Ha Cong Nga
* Phuc Pham
* Truong-Son Hy (PI)

The main functionalities from LANTERN include, but not limited to:
- Featurization of ligand SMILES and protein Amino Acids.
- Training process and prediction scripts.
- A simple but effective methods that generalise across molecular interaction tasks (DTI, DDI, PPI, ...).

The main innovations we made away from LANTERN, but not limited to:
- Integration of pretrained LLM embeddings with Transformer-based interaction modeling.
- Broad applicability and SOTA performance.
- Efficiency and independence from 3D structural data.

## Setup Environment
Clone this repository and install dependencies:
```bash
git clone https://github.com/anonymousreseach99/LANTERN.git
cd LANTERN
conda env create -f environment.yaml
conda activate LANTERN
```

### File structure 
Files should be placed as the following folder structure:
```
LANTERN
├── code
│   ├──file.py ...
├── data
│   ├── README.md
│   ├── embedding
├── log
│   ├── README.md
├── README.md
├── environment.yaml
│ 
```

## Training
### DTI datasets (BioSNAP, DAVIS, KIBA):

First, ensure that pretrained weights and dataset for all entities in the dataset are properly located at LANTERN\data as guided in LANTERN\data\README.md .

Second, cd code .

Finally, run the training script:
```
python main.py \
    --interaction_tyoe "DTI"\
    --dataset_name "BioSNAP"\
    --embed_dim 384 \
    --seed 120 \
    --valid_step 10 \
    --epoch 100 \
    --lr 0.0001 \
    --dropout 0.1 \
    --modality 1 \
    --save_model True \
    --score_fun 'transformer' \
    --save_path path_to_saved_checkpoints \
    --drug_pretrained_dim 768 \
    --protein_sequence_dim 1024 \
   
```

Please modify the dataset_name, path_to_dataset, and save_path according to your experiments.

### DDI datasets (DeepDDI):

First, ensure that pretrained weights and dataset for all entities in the dataset are properly located at LANTERN\data as guided in LANTERN\data\README.md .

Second, cd code .

Finally, run the training script:
```
python main.py \
    --interaction_type "DDI" \
    --dataset_name "DeepDDI" \
    --embed_dim 384 \
    --seed 120 \
    --valid_step 10 \
    --epoch 100 \
    --lr 0.0001 \
    --dropout 0.1 \
    --modality 1 \
    --save_model True \
    --score_fun 'transformer' \
    --save_path path_to_saved_checkpoints \
    --drug_pretrained_dim 768 \
   
```

### PPI datasets (yeast):

First, ensure that pretrained weights for all entities in the dataset are properly located at data\embedding\{dataset_name}.

Second, cd code.

Finally, run the training script:
```
python main.py \
    --interaction_type "PPI" \
    --dataset_name "yeast" \
    --embed_dim 384 \
    --seed 120 \
    --valid_step 10 \
    --epoch 100 \
    --lr 0.0001 \
    --dropout 0.1 \
    --modality 1 \
    --save_model True \
    --score_fun 'transformer' \
    --save_path path_to_saved_checkpoints \
    --protein_sequence_dim 1024 \
   
```

Please modify the dataset_name, path_to_dataset, and save_path according to your experiments.

## Evaluation 
First, cd code.
Second, run the following script :
```
python eval.py \
    --model_save_path path_to_checkpoint \
    --gpu True \
    --interaction_type "DTI" \
    --dataset_name "BioSNAP"
    --test_path path_to_dataset_folder \
```

## Predict interactions between a pair of entities
```
python predict.py \
    --model_save_path path_to_checkpoint \
    --gpu True \
    --type 'dti' \
    --sequence1 amino_acid_sequence_or_smiles_string \
    --sequence2 amino_acid_sequence_or_smiles_string \
```
## Acknowledgements

This work is primarily based on the following repositories:

- https://github.com/samsledje/ConPLex.git (BioSNAP, DAVIS - DTI datasets)
- https://github.com/thinng/GraphDTA.git (KIBA - DTI dataset)
- https://github.com/biomed-AI/MUSE.git (DeepDDI - DDI datasets)
- https://github.com/xzenglab/TAGPPI.git (Yeast - PPI dataset)


## Please cite our work as follows

```bibtex
@article {Ha2025.02.10.637522,
	author = {Ha, Cong Nga and Pham, Phuc and Hy, Truong Son},
	title = {LANTERN: Leveraging Large Language Models and Transformers for Enhanced Molecular Interactions},
	elocation-id = {2025.02.10.637522},
	year = {2025},
	doi = {10.1101/2025.02.10.637522},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Understanding molecular interactions such as Drug-Target Interaction (DTI), Protein-Protein Interaction (PPI), and Drug-Drug Interaction (DDI) is critical for advancing drug discovery and systems biology. However, existing methods often struggle with scalability due to the vast chemical and biological space and suffer from limited accuracy when capturing intricate biochemical relationships. To address these challenges, we introduce LANTERN (Leveraging large LANguage models and Transformers for Enhanced moleculaR interactioNs), a novel deep learning framework that integrates Large Language Models (LLMs) with Transformer-based architectures to model molecular interactions more effectively. LANTERN generates high-quality, context-aware embeddings for drug and protein sequences, enabling richer feature representations and improving predictive accuracy. By leveraging a Transformer-based fusion mechanism, our framework enhances scalability by efficiently integrating diverse interaction data while maintaining computational feasibility. Experimental results demonstrate that LANTERN achieves state-of-the-art performance on multiple DTI and DDI benchmarks, significantly outperforming traditional deep learning approaches. Additionally, LANTERN exhibits competitive performance on challenging PPI tasks, underscoring its versatility across diverse molecular interaction domains. The proposed framework offers a robust and adaptable solution for modeling molecular interactions, efficiently handling a diverse range of molecular entities without the need for 3D structural data and making it a promising framework for foundation models in molecular interaction. Our findings highlight the transformative potential of combining LLM-based embeddings with Transformer architectures, setting a new standard for molecular interaction prediction. The source code and relevant documentation are available at: https://github.com/HySonLab/LANTERNCompeting Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2025/02/15/2025.02.10.637522},
	eprint = {https://www.biorxiv.org/content/early/2025/02/15/2025.02.10.637522.full.pdf},
	journal = {bioRxiv}
}
```
