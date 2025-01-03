import argparse
import torch
import os
import pickle
import sys
import re
from util_representations import load_relation_embed, load_entity_embed, get_bio_bert
from transformers import BertModel, BertTokenizer
import re
import json
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Adjust sys.path to ensure modules are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
code_folder = os.path.dirname(__file__)
kgcnh_folder = os.path.dirname(code_folder)
kgdrp_folder = os.path.dirname(kgcnh_folder)
gene_seq_dict_path = os.path.join(kgcnh_folder, 'data', 'BioSNAP' ,'raw_dict', 'gene', 'gene2sequence.json')
gene_seq_embed_path = os.path.join(kgdrp_folder,'embeddings', 'pretrained', 'gene', 'gene_sequence.json')
gene_func_embed_path = os.path.join(kgdrp_folder,'embeddings', 'pretrained', 'gene', 'gene_function.json')

test_df_path = os.path.join(kgcnh_folder, 'data' , 'BioSNAP', 'test.csv')
test_df = pd.read_csv(test_df_path)

def get_positive_relations_in_test() :
    # Load the CSV file
    df = test_df

    # Filter rows where 'Label' is 1.0
    # 'DrugBank ID', 'Gene', 'Label'
    positive_relations = df[df['Label'] == 1.0]

    # Get the count of positive relations
    #num_positive_relations = len(positive_relations)

    #print(f"Number of positive relations: {num_positive_relations}")
    return positive_relations

def get_biobert_representation(input) :
    tokenizer, model = get_bio_bert()
    input = tokenizer([input], return_tensors="pt", truncation=True, padding=True, max_length=512)
    model.eval()
    with torch.no_grad():
        outputs = model(**input)
        last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_length, hidden_dim)
        embeddings = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        print(27, 'predict bio bert embed shape', embeddings.shape)
    return embeddings.squeeze(0).cpu()

def get_probert_representation(input) :
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model.eval()
    sequence_Example = re.sub(r"[UZOB]", "X", input)
    encoded_input = tokenizer(sequence_Example, return_tensors='pt')
    with torch.no_grad() :
        output = model(**encoded_input)
    last_hidden_states = output.last_hidden_state  # (batch_size, seq_length, hidden_dim)
    embeddings = last_hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
    print(27, 'predict bio bert embed shape', embeddings.shape)
    return embeddings.squeeze(0).cpu()

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "model":
            module = "KGCNH.code.model"
        elif module == "layers":
            module = "KGCNH.code.layers"
        return super().find_class(module, name)
    
def get_top_ranked_drugs(probs, k) :
    """
    return top k drugs ranked by probabilities 
    """
    return torch.topk(probs, k).indices, probs[torch.topk(probs, k).indices]


def get_infered_gene_embedding(model, get_bio_bert_embeddings, get_protbert_embeddings,  
                               n_t_sequence, n_t_function, gene_seq_dict_path, 
                               gene_seq_embed_path, gene_func_embed_path):
    """
    Retrieves or generates the embeddings for a given gene sequence and function.
    
    Parameters:
    - model: The model to use for generating function embeddings if needed.
    - get_bio_bert_embeddings: Function to get BioBERT embeddings for gene functions.
    - get_protbert_embeddings: Function to get ProtBERT embeddings for gene sequences.
    - n_t_sequence: The gene sequence to process.
    - n_t_function: The gene function to process.
    - gene_seq_dict_path: Path to the JSON file containing UniprotID to sequence mapping.
    - gene_seq_embed_path: Path to the JSON file containing UniprotID to sequence embeddings.
    - gene_func_embed_path: Path to the JSON file containing UniprotID to function embeddings.
    
    Returns:
    - A tuple containing sequence embedding and function embedding for the gene.
    """

    # Load the sequence dictionary
    with open(gene_seq_dict_path, 'r') as f:
        gene_seq_dict = json.load(f)

    # Check if the sequence exists in the gene sequence dictionary
    uniprot_id = None
    for uid, sequence in gene_seq_dict.items():
        if sequence == n_t_sequence:
            uniprot_id = uid
            break
    
    if uniprot_id:
        print("IN IN")
        # Load the embeddings if the sequence is found
        with open(gene_seq_embed_path, 'r') as f:
            gene_seq_embeddings = json.load(f)
        with open(gene_func_embed_path, 'r') as f:
            gene_func_embeddings = json.load(f)
        
        # Retrieve embeddings
        sequence_embedding = gene_seq_embeddings.get(uniprot_id, None)
        function_embedding = gene_func_embeddings.get(uniprot_id, None)

        # If any embedding is missing, regenerate using the models
        if sequence_embedding is None:
            sequence_embedding = get_probert_representation(n_t_sequence)
            
        if function_embedding is None:
            function_embedding = get_biobert_representation(n_t_function)
        sequence_embedding = torch.tensor(sequence_embedding)
        function_embedding = torch.tensor(function_embedding)
        #print(109, sequence_embedding.device)
        sequence_embedding = model.project_pretrained_gene_seq(sequence_embedding.to('cuda'))
        function_embedding = model.project_pretrained_gene_func(function_embedding.to('cuda'))
        
    else:
        print("OUT")
        # If the sequence is not found, generate embeddings from scratch
        sequence_embedding = get_probert_representation(n_t_sequence)
        print(117, 'predict', sequence_embedding.device)
        sequence_embedding = model.project_pretrained_gene_seq(sequence_embedding.to('cuda'))
        function_embedding = get_biobert_representation(n_t_sequence + n_t_function)
        function_embedding = model.project_pretrained_gene_func(function_embedding.to('cuda'))
    #print(sequence_embedding)
    return torch.cat([function_embedding,sequence_embedding], dim=-1)

def get_supplementary_drugs(model, embed, n_t_embed, top_drugs, top_probs, drugbank_db_folder, device) :
    """
    top_drugs : indices of top drugs of KG.
    From top drugs obtained from KG, return drugs from drugbank that are similar.
    """
    model.eval()
    model = model.to(device)
    

    drugdes_db_pretrained_embed_path = os.path.join(drugbank_db_folder, 'drugdes_db_pretrained.json')
    with open(drugdes_db_pretrained_embed_path, 'r') as f :
        drugdes_db_pretrained_embed = json.load(f)

    supplementary_drugs_db = list(drugdes_db_pretrained_embed.keys())
    drugdes_db_pretrained_embed = torch.tensor(list(drugdes_db_pretrained_embed.values()), dtype=torch.float32)
    drugdes_db_embed = model.drug_des_project(drugdes_db_pretrained_embed.to(device))
    
    drugsmiles_db_pretrained_embed_path = os.path.join(drugbank_db_folder, 'drugsmiles_db_pretrained.json')
    with open(drugsmiles_db_pretrained_embed_path, 'r') as f :
        drugsmiles_db_pretrained_embed = json.load(f)
    drugsmiles_db_pretrained_embed = torch.tensor(list(drugsmiles_db_pretrained_embed.values()), dtype=torch.float32)
    drugsmiles_db_embed = model.drug_smiles_project(drugsmiles_db_pretrained_embed.to(device))
    drug_db_embed = torch.cat([drugsmiles_db_embed, drugdes_db_embed], dim=1)

    coeff_type = 'normal'
    topk = 10
    representative_kg_embed = get_representative_embed(embed, top_drugs, top_probs, coeff_type, topk)
    # just for test :
    #representative_kg_embed = n_t_embed
    #drug_db_embed = torch.nn.functional.normalize(drug_db_embed, p=2, dim=-1)
    #representative_kg_embed = torch.nn.functional.normalize(representative_kg_embed, p=2, dim=-1)
    dot_scores = (drug_db_embed * representative_kg_embed).sum(dim=-1)
    similarity_scores = torch.sigmoid(dot_scores)

    top_supplementary_drug_indices, top_scores = get_top_ranked_drugs(similarity_scores, len(supplementary_drugs_db))

    print("156, 10th drugs : ", supplementary_drugs_db[top_supplementary_drug_indices[10]])
    supplementary_drugs = [supplementary_drugs_db[id.item()] for id in top_supplementary_drug_indices]
    print("158, 10th drugs : ", supplementary_drugs[10])
    return supplementary_drugs, top_scores

def get_representative_embed(embed, top_drugs, top_probs, coeff_type='exp', topk = 10) :
    top_drugs = top_drugs[:topk]
    top_probs = top_probs[:topk]
    if coeff_type == 'exp' :
        coeff = torch.exp(top_probs)
    else :
        coeff = top_probs
    coeff = coeff.view(-1, 1)
    topk_drugs_embed = embed[top_drugs]
    representative_embed = (topk_drugs_embed * coeff).mean(dim=0)
    return representative_embed

def get_similar_kg_drugs(embed,n_t_embed, entities2id, drug_num, top_drugs, top_probs, top_kg_filtering) :
    coeff_type = 'normal'
    topk = 10
    representative_kg_embed = get_representative_embed(embed, top_drugs, top_probs, coeff_type, topk)
    id2entities = {value : key for key, value in entities2id.items()}

    # get the embeddings of all drugs that are not in the top top_kg_filtering for one more filtering
    unselected_kg_drugs = top_drugs[top_kg_filtering:] # indices of remaining drugs
    remaining_embed = embed[unselected_kg_drugs] # BUGS

    #remaining_embed = torch.nn.functional.normalize(remaining_embed, p=2, dim=-1)
    #representative_kg_embed = torch.nn.functional.normalize(representative_kg_embed, p=2, dim=-1)
    # just for test :
    representative_kg_embed = n_t_embed
    dot_scores = (remaining_embed[:,int(embed.shape[-1]/2):] * representative_kg_embed[int(embed.shape[-1]/2):]).sum(dim=-1)
    similarity_scores = torch.sigmoid(dot_scores)

    # get the indices of good remaining drugs ; indices are starting from 0, 1, ... remaining_embed.shape[0]
    top_supplementary_drug_indices, top_scores = get_top_ranked_drugs(similarity_scores, remaining_embed.shape[0])
    top_supplementary_drug_indices = [unselected_kg_drugs[id.item()] for id in top_supplementary_drug_indices]

    print("156, 10th drugs : ", id2entities[top_supplementary_drug_indices[10].item()])
    supplementary_drugs = [id2entities[id.item()] for id in top_supplementary_drug_indices]
    print("158, 10th drugs : ", supplementary_drugs[10])
    return supplementary_drugs, top_scores

def predict(args) :
    if torch.cuda.is_available() and args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(args.model_save_path, "rb") as f:
      res = CustomUnpickler(f).load()
    print("Type res :", type(res))
    print("Keys of res :" , res.keys())
    model = res['model']
    embed = model.entity_embed
    #print(234, 'predict, embed dim : ', embed.shape)
    drug_num = model.drug_num
    entities2id = model.train_entity2index
    indices2entities = {value : key for key, value in entities2id.items()}

    h = list(entities2id.values())
    n_t_embed = get_infered_gene_embedding(model, get_biobert_representation, get_probert_representation, args.n_t_sequence, args.n_t_function, gene_seq_dict_path, gene_seq_embed_path, gene_func_embed_path)
    h = h[:drug_num]
    #h_embed = embed[h]
    model.eval()
    model = model.to('cuda')
    with torch.no_grad() :
        h_embed = embed[h, :].unsqueeze(0)
        n_t_embed = n_t_embed.unsqueeze(0).unsqueeze(1)
        probs, _ = model.make_link_predictions(h_embed, n_t_embed)
        probs = probs.squeeze(0).squeeze(-1)
        #print(122, 'predict', probs[0])
        #print(123, 'predict', probs[1])

    k = 4400
    #print(255, 'predict', probs.shape)
    top_drugs,top_probs = get_top_ranked_drugs(probs, k)
    drug_names = [key for key, value in entities2id.items() if value in top_drugs]
    #print(top_probs)

    # drugbank id 
    top_entities = [indices2entities[top_drug.item()] for top_drug in top_drugs]
    #print(131, top_entities)

    name = args.name
    with open(f'top_drugs_{name}_ss_2.txt', 'w') as f:
        for entity, prob in zip(top_entities, top_probs):
            f.write(f'{entity} {prob}\n')

    drugbank_db_folder = args.drugbank_db_folder

    # Similar drugs in kg, refiltering; similar_kg_drugs is drugs in similar kg 
    top_kg_filtering = args.top_kg_filtering
    similar_kg_drugs, kg_scores = get_similar_kg_drugs(embed, n_t_embed, entities2id, drug_num, top_drugs, top_probs, top_kg_filtering)

    # Similar drugs in drugbank
    similar_potential_drugs, db_scores = get_supplementary_drugs(model, embed, n_t_embed, top_drugs, top_probs, drugbank_db_folder, device)

    # Append similar potential drugs to the file
    with open(f'top_drugs_{name}_ss_2.txt', 'a') as f:  # 'a' mode is used to append
        f.write("\nSimilar Potential Drugs from KG:\n")
        for drug, score in zip(similar_kg_drugs, kg_scores):
            f.write(f'{drug} {score}\n')

    # Append similar potential drugs to the file
    with open(f'top_drugs_{name}_ss_2.txt', 'a') as f:  # 'a' mode is used to append
        f.write("\nSimilar Potential Drugs from DB:\n")
        for drug, score in zip(similar_potential_drugs, db_scores):
            f.write(f'{drug} {score}\n')

def get_prediction_statistics(args) :
    if torch.cuda.is_available() and args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(args.model_save_path, "rb") as f:
      res = CustomUnpickler(f).load()

    model = res['model']
    embed = model.entity_embed
    drug_num = model.drug_num
    entities2id = model.train_entity2index
    indices2entities = {value : key for key, value in entities2id.items()}

    h = list(entities2id.values())
    h = h[:drug_num]
    #h_embed = embed[h]
    model.eval()
    model = model.to(device)

    positive_relations_df = args.positive_relations_df

    num_item = 0
    with open(f'test_df_predictions.txt', 'w') as f:
        for _, row in positive_relations_df.iterrows():
            target_seq = row['Target Sequence']
            target_func = row['gene_function']
            gene_id = row['Gene']
            pos_drug = row['DrugBank ID']
            assert row['Label'] == 1.0 
            #print(f"Drug: {row['DrugBank ID']}, Gene: {row['Gene']}, Label: {row['Label']}")
            
            #pos_drug = args.data_fr[args.data_fr['GeneID'] == ]
            n_t_embed = get_infered_gene_embedding(model, get_biobert_representation, get_probert_representation, target_seq, target_func, gene_seq_dict_path, gene_seq_embed_path, gene_func_embed_path)
            with torch.no_grad() :
                h_embed = embed[h, :].unsqueeze(0)
                n_t_embed = n_t_embed.unsqueeze(0).unsqueeze(1)
                probs, _ = model.make_link_predictions(h_embed, n_t_embed)
                probs = probs.squeeze(0).squeeze(-1)

            k = 4400
            #print(332, 'predict', probs.shape)
            top_drugs, top_probs = get_top_ranked_drugs(probs, k)  # drugs in ranked order

            # drugbank id 
            top_entities = [indices2entities[top_drug.item()] for top_drug in top_drugs]
            #print(131, top_entities)

            index = top_entities.index(pos_drug) if pos_drug in top_entities else -1

            f.write(f'{gene_id} {pos_drug} {index}\n')
            num_item = num_item + 1
            if num_item % 100 == 0: 
                print(f"{num_item} items written")

"""
if __name__ == '__main__':
    toy_target_sequence =   'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'
    des = "N-terminal domain of the S1 subunit of the Spike (S) protein from Severe acute respiratory syndrome coronavirus and related betacoronaviruses in the B lineage; Region: SARS-CoV-like_Spike_S1_NTD; cd21624"
    onine_seq = 'METSSPRPPRPSSNPGLSLDARLGVDTRLWAKVLFTALYALIWALGAAGNALSAHVVLKARAGRAGRLRHHVLSLALAGLLLLLVGVPVELYSFVWFHYPWVFGDLGCRGYYFVHELCAYATVLSVAGLSAERCLAVCQPLRARSLLTPRRTRWLVALSWAASLGLALPMAVIMGQKHELETADGEPEPASRVCTVLVSRTALQVFIQVNVLVSFVLPLALTAFLNGVTVSHLLALCSQVPSTSTPGSSTPSRLELLSEEGLLSFIVWKKTFIQGGQVSLVRHKDVRRIRSLQRSVQVLRAIVVMYVICWLPYHARRLMYCYVPDDAWTDPLYNFYHYFYMVTNTLFYVSSAVTPLLYNAVSSSFRKLFLEAVSSLCGEHHPMKRLPPKPQSPTLMDTASGFGDPPETRT'
    onine_des = 'Receptor for the tridecapeptide neurotensin. It is associated with G proteins that activate a phosphatidylinositol-calcium second messenger system'
    pzeroone_seq = 'MAWALLLLTLLTQGTGSWAQSALTQPPSASGSPGQSVTISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYEVSKRPSGVPDRFSGSKSGNTASLTVSGLQAEDEADYYCSSYAGSNNF'
    pzeroone_des = 'V region of the variable domain of immunoglobulin light chains that participates in the antigen recognition (PubMed:24600447). Immunoglobulins, also known as antibodies, are membrane-bound or secreted glycoproteins produced by B lymphocytes. In the recognition phase of humoral immunity, the membrane-bound immunoglobulins serve as receptors which, upon binding of a specific antigen, trigger the clonal expansion and differentiation of B lymphocytes into immunoglobulins-secreting plasma cells. Secreted immunoglobulins mediate the effector phase of humoral immunity, which results in the elimination of bound antigens (PubMed:20176268, PubMed:22158414). The antigen binding site is formed by the variable domain of one heavy chain, together with that of its associated light chain. Thus, each immunoglobulin has two antigen binding sites with remarkable affinity for a particular antigen. The variable domains are assembled by a process called V-(D)-J rearrangement and can then be subjected to somatic hypermutations which, after exposure to antigen and selection, allow affinity maturation for a particular antigen (PubMed:17576170, PubMed:20176268)'
    qeightn_des = 'Plays a role in the regulation of ureagenesis by producing the essential cofactor N-acetylglutamate (NAG), thus modulating carbamoylphosphate synthase I (CPS1) activity'
    qeightn_seq = 'MATALMAVVLRAAAVAPRLRGRGGTGGARRLSCGARRRAARGTSPGRRLSTAWSQPQPPPEEYAGADDVSQSPVAEEPSWVPSPRPPVPHESPEPPSGRSLVQRDIQAFLNQCGASPGEARHWLTQFQTCHHSADKPFAVIEVDEEVLKCQQGVSSLAFALAFLQRMDMKPLVVLGLPAPTAPSGCLSFWEAKAQLAKSCKVLVDALRHNAAAAVPFFGGGSVLRAAEPAPHASYGGIVSVETDLLQWCLESGSIPILCPIGETAARRSVLLDSLEVTASLAKALRPTKIIFLNNTGGLRDSSHKVLSNVNLPADLDLVCNAEWVSTKERQQMRLIVDVLSRLPHHSSAVITAASTLLTELFSNKGSGTLFKNAERMLRVRSLDKLDQGRLVDLVNASFGKKLRDDYLASLRPRLHSIYVSEGYNAAAILTMEPVLGGTPYLDKFVVSSSRQGQGSGQMLWECLRRDLQTLFWRSRVTNPINPWYFKHSDGSFSNKQWIFFWFGLADIRDSYELVNHAKGLPDSFHKPASDPGS'
    qonesix_des = 'Receptor tyrosine kinase involved in the development and the maturation of the central and the peripheral nervous systems through regulation of neuron survival, proliferation, migration, differentiation, and synapse formation and plasticity (By similarity). Receptor for BDNF/brain-derived neurotrophic factor and NTF4/neurotrophin-4. Alternatively can also bind NTF3/neurotrophin-3 which is less efficient in activating the receptor but regulates neuron survival through NTRK2 (PubMed:15494731, PubMed:7574684). Upon ligand-binding, undergoes homodimerization, autophosphorylation and activation (PubMed:15494731). Recruits, phosphorylates and/or activates several downstream effectors including SHC1, FRS2, SH2B1, SH2B2 and PLCG1 that regulate distinct overlapping signaling cascades. Through SHC1, FRS2, SH2B1, SH2B2 activates the GRB2-Ras-MAPK cascade that regulates for instance neuronal differentiation including neurite outgrowth. Through the same effectors controls the Ras-PI3 kinase-AKT1 signaling cascade that mainly regulates growth and survival. Through PLCG1 and the downstream protein kinase C-regulated pathways controls synaptic plasticity. Thereby, plays a role in learning and memory by regulating both short term synaptic function and long-term potentiation. PLCG1 also leads to NF-Kappa-B activation and the transcription of genes involved in cell survival. Hence, it is able to suppress anoikis, the apoptosis resulting from loss of cell-matrix interactions. May also play a role in neutrophin-dependent calcium signaling in glial cells and mediate communication between neurons and glia'
    qonesix_seq = 'MSSWIRWHGPAMARLWGFCWLVVGFWRAAFACPTSCKCSASRIWCSDPSPGIVAFPRLEPNSVDPENITEIFIANQKRLEIINEDDVEAYVGLRNLTIVDSGLKFVAHKAFLKNSNLQHINFTRNKLTSLSRKHFRHLDLSELILVGNPFTCSCDIMWIKTLQEAKSSPDTQDLYCLNESSKNIPLANLQIPNCGLPSANLAAPNLTVEEGKSITLSCSVAGDPVPNMYWDVGNLVSKHMNETSHTQGSLRITNISSDDSGKQISCVAENLVGEDQDSVNLTVHFAPTITFLESPTSDHHWCIPFTVKGNPKPALQWFYNGAILNESKYICTKIHVTNHTEYHGCLQLDNPTHMNNGDYTLIAKNEYGKDEKQISAHFMGWPGIDDGANPNYPDVIYEDYGTAANDIGDTTNRSNEIPSTDVTDKTGREHLSVYAVVVIASVVGFCLLVMLFLLKLARHSKFGMKGPASVISNDDDSASPLHHISNGSNTPSSSEGGPDAVIIGMTKIPVIENPQYFGITNSQLKPDTFVQHIKRHNIVLKRELGEGAFGKVFLAECYNLCPEQDKILVAVKTLKDASDNARKDFHREAELLTNLQHEHIVKFYGVCVEGDPLIMVFEYMKHGDLNKFLRAHGPDAVLMAEGNPPTELTQSQMLHIAQQIAAGMVYLASQHFVHRDLATRNCLVGENLLVKIGDFGMSRDVYSTDYYRVGGHTMLPIRWMPPESIMYRKFTTESDVWSLGVVLWEIFTYGKQPWYQLSNNEVIECITQGRVLQRPRTCPQEVYELMLGCWQREPHMRKNIKGIHTLLQNLAKASPVYLDILG'
    
    pzerofive_seq = 'MAGLGPGVGDSEGGPRPLFCRKGALRQKVVHEVKSHKFTARFFKQPTFCSHCTDFIWGIGKQGLQCQVCSFVVHRRCHEFVTFECPGAGKGPQTDDPRNKHKFRLHSYSSPTFCDHCGSLLYGLVHQGMKCSCCEMNVHRRCVRSVPSLCGVDHTERRGRLQLEIRAPTADEIHVTVGEARNLIPMDPNGLSDPYVKLKLIPDPRNLTKQKTRTVKATLNPVWNETFVFNLKPGDVERRLSVEVWDWDRTSRNDFMGAMSFGVSELLKAPVDGWYKLLNQEEGEYYNVPVADADNCSLLQKFEACNYPLELYERVRMGPSSSPIPSPSPSPTDPKRCFFGASPGRLHISDFSFLMVLGKGSFGKVMLAERRGSDELYAIKILKKDVIVQDDDVDCTLVEKRVLALGGRGPGGRPHFLTQLHSTFQTPDRLYFVMEYVTGGDLMYHIQQLGKFKEPHAAFYAAEIAIGLFFLHNQGIIYRDLKLDNVMLDAEGHIKITDFGMCKENVFPGTTTRTFCGTPDYIAPEIIAYQPYGKSVDWWSFGVLLYEMLAGQPPFDGEDEEELFQAIMEQTVTYPKSLSREAVAICKGFLTKHPGKRLGSGPDGEPTIRAHGFFRWIDWERLERLEIPPPFRPRPCGRSGENFDKFFTRAAPALTPPDRLVLASIDQADFQGFTYVNPDFVHPDARSPTSPVPVPVM'
    pzerofive_des =  'Calcium-activated, phospholipid- and diacylglycerol (DAG)-dependent serine/threonine-protein kinase that plays diverse roles in neuronal cells and eye tissues, such as regulation of the neuronal receptors GRIA4/GLUR4 and GRIN1/NMDAR1, modulation of receptors and neuronal functions related to sensitivity to opiates, pain and alcohol, mediation of synaptic function and cell survival after ischemia, and inhibition of gap junction activity after oxidative stress. Binds and phosphorylates GRIA4/GLUR4 glutamate receptor and regulates its function by increasing plasma membrane-associated GRIA4 expression. In primary cerebellar neurons treated with the agonist 3,5-dihyidroxyphenylglycine, functions downstream of the metabotropic glutamate receptor GRM5/MGLUR5 and phosphorylates GRIN1/NMDAR1 receptor which plays a key role in synaptic plasticity, synaptogenesis, excitotoxicity, memory acquisition and learning. May be involved in the regulation of hippocampal long-term potentiation (LTP), but may be not necessary for the process of synaptic plasticity. May be involved in desensitization of mu-type opioid receptor-mediated G-protein activation in the spinal cord, and may be critical for the development and/or maintenance of morphine-induced reinforcing effects in the limbic forebrain. May modulate the functionality of mu-type-opioid receptors by participating in a signaling pathway which leads to the phosphorylation and degradation of opioid receptors. May also contributes to chronic morphine-induced changes in nociceptive processing. Plays a role in neuropathic pain mechanisms and contributes to the maintenance of the allodynia pain produced by peripheral inflammation. Plays an important role in initial sensitivity and tolerance to ethanol, by mediating the behavioral effects of ethanol as well as the effects of this drug on the GABA(A) receptors. During and after cerebral ischemia modulate neurotransmission and cell survival in synaptic membranes, and is involved in insulin-induced inhibition of necrosis, an important mechanism for minimizing ischemic injury. Required for the elimination of multiple climbing fibers during innervation of Purkinje cells in developing cerebellum. Is activated in lens epithelial cells upon hydrogen peroxide treatment, and phosphorylates connexin-43 (GJA1/CX43), resulting in disassembly of GJA1 gap junction plaques and inhibition of gap junction activity which could provide a protective effect against oxidative stress (By similarity). Phosphorylates p53/TP53 and promotes p53/TP53-dependent apoptosis in response to DNA damage. Involved in the phase resetting of the cerebral cortex circadian clock during temporally restricted feeding. Stabilizes the core clock component BMAL1 by interfering with its ubiquitination, thus suppressing its degradation, resulting in phase resetting of the cerebral cortex clock (By similarity)'
    start_time = time.time()
    parser = argparse.ArgumentParser(description="add args for predictions")
    parser.add_argument("--model_save_path", default=os.path.join(kgcnh_folder, 'log','training','2','result_0.9959813699388065_49_auc9966_modal2_lr1e-4_ep100.pkl'), help="path to model saved")
    parser.add_argument("--gpu", default=True, help="enable gpu")
    parser.add_argument("--n_t_sequence", default=pzerofive_seq, help="Protein")
    parser.add_argument("--n_t_function", default=pzerofive_des, help="Protein")
    parser.add_argument("--name", default='P05129', help='to save file')
    parser.add_argument("--drugbank_db_folder", default='drugbank_db')
    parser.add_argument("--top_kg_filtering", type=int, default=1000)
    parser.add_argument("--modal_func", type=bool, default=True)
    parser.add_argument("--modal_seq", type=bool, default=True)
    parser.add_argument("--modal_struct", type=bool, default=False)

    args = parser.parse_args()
    print(args.n_t_sequence)#

    #positive_relations_df = get_positive_relations_in_test()

    predict(args)
    #get_prediction_statistics(args)
    end_time = time.time()
    print(f'The elapsed time for running one pair : {end_time-start_time}')
"""

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description="add args for predictions")
    positive_relations_df = get_positive_relations_in_test()
    parser.add_argument("--gpu", default=True, help="enable gpu")
    parser.add_argument("--positive_relations_df", default=positive_relations_df, help="Positive relations")
    parser.add_argument("--top_kg_filtering", type=int, default=1000)
    parser.add_argument("--model_save_path", default=os.path.join(kgcnh_folder, 'log','training','2','result_0.9959813699388065_49_auc9966_modal2_lr1e-4_ep100.pkl'), help="path to model saved")

    #parser.add_argument("--")
    args = parser.parse_args() 

    #predict(args)
    get_prediction_statistics(args)
    end_time = time.time()
    print(f'The elapsed time for running one pair : {end_time-start_time}')
