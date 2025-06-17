import random
import numpy as np
import torch

from diffab.utils.retrieval.retrieval_utils import tensor_to_pdbid


ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20, '-':21
}




def get_retrieved_sequences_class_test_CDRclass_1(pdb_id_tensor, length, CDR_flag):
    pdb_id = tensor_to_pdbid(pdb_id_tensor)
    if CDR_flag == 1:
        fasta_file = './data/ref_seqs_final/H_CDR1.fasta'
    elif CDR_flag == 2:
        fasta_file = './data/ref_seqs_final/H_CDR2.fasta'
    elif CDR_flag == 3:
        fasta_file = './data/ref_seqs_final/H_CDR3.fasta'

    elif CDR_flag == 4:
        fasta_file = './data/ref_seqs_final/ref_sequences_chothia_CDR4.fasta'
    elif CDR_flag == 5:
        fasta_file = './data/ref_seqs_final/ref_sequences_chothia_CDR5.fasta'
    elif CDR_flag == 6:
        fasta_file = './data/ref_seqs_final/ref_sequences_chothia_CDR6.fasta'

    
    ref_sequences = find_protein_sequences(fasta_file, pdb_id)
 
    num_sequences = 15 

    if ref_sequences !=[]:
        
        if len(ref_sequences) < num_sequences:
            while len(ref_sequences) < num_sequences:
                ref_sequences.append(random.choice(ref_sequences))
        elif len(ref_sequences) > num_sequences:
            while len(ref_sequences) > num_sequences:
                ref_sequences.pop()

        ref_sequences_int = []  
        for seq in ref_sequences:
            sequence_tensor = torch.zeros(len(seq), dtype=torch.long)
            for i, amino_acid in enumerate(seq):
                index = ressymb_to_resindex.get(amino_acid, 20)  
                sequence_tensor[i] = index
        
            ref_sequences_int.append(sequence_tensor)
        
        ref_matrix_list = ref_sequences_int
        
    else:
        ref_matrix_tensor = torch.full((num_sequences, length), 21) 
        ref_matrix_list = ref_matrix_tensor.tolist()

    return ref_matrix_list

def get_retrieved_sequences_class_test_CDRclass_2(pdb_id_tensor, length, CDR_flag): 
    pdb_id = tensor_to_pdbid(pdb_id_tensor)
    if CDR_flag == 1:
        fasta_file = './data/ref_seqs_final/H_CDR1.fasta'
    elif CDR_flag == 2:
        fasta_file = './data/ref_seqs_final/H_CDR2.fasta'
    elif CDR_flag == 3:
        fasta_file = './data/ref_seqs_final/H_CDR3.fasta'

    elif CDR_flag == 4:
        fasta_file = './data/ref_seqs_final/ref_sequences_chothia_CDR4.fasta'
    elif CDR_flag == 5:
        fasta_file = './data/ref_seqs_final/ref_sequences_chothia_CDR5.fasta'
    elif CDR_flag == 6:
        fasta_file = './data/ref_seqs_final/ref_sequences_chothia_CDR6.fasta'

    ref_sequences = find_protein_sequences(fasta_file, pdb_id)
 
    ref_sequences = [x for x in ref_sequences if x != ref_sequences[0]] #avoid leakage
    ref_sequences = ref_sequences[1:]
    num_sequences = 15 

    if ref_sequences !=[]:
        if len(ref_sequences) < num_sequences:
            while len(ref_sequences) < num_sequences:
                ref_sequences.append(random.choice(ref_sequences))
        elif len(ref_sequences) > num_sequences:
            while len(ref_sequences) > num_sequences:
                ref_sequences.pop()

        ref_sequences_int = []  
        for seq in ref_sequences:
            sequence_tensor = torch.zeros(len(seq), dtype=torch.long)
            
            for i, amino_acid in enumerate(seq):
                index = ressymb_to_resindex.get(amino_acid, 20)  
                sequence_tensor[i] = index
        
            ref_sequences_int.append(sequence_tensor)
        
        ref_matrix_list = ref_sequences_int
        
    else:
        ref_matrix_tensor = torch.full((num_sequences, length), 21) 
        ref_matrix_list = ref_matrix_tensor.tolist()

    return ref_matrix_list

def read_numeric_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                continue  
            else:
                sequence = list(map(int, line.strip().split()))
                sequences.append(sequence)
    return sequences

def find_protein_sequences(fasta_file, pdb_id):
    sequences = []
    record = False
    current_sequence = ""

    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if line[1:] == pdb_id:
                    record = True 
                else:
                    if record:
                        break
            elif record:
                if 5 < len(line) < 30:
                    sequences.append(line)
    
    return sequences
