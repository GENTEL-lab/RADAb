from ImmuneBuilder import ABodyBuilder2
import numpy as np
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
from diffab.tools.folding.base import FoldingTask
from diffab.tools.folding.align_folded import align_and_move_chains
import os
predictor = ABodyBuilder2(numbering_scheme = 'chothia')
def extract_reslist(model, residue_first, residue_last):
    assert residue_first[0] == residue_last[0]
    residue_first, residue_last = tuple(residue_first), tuple(residue_last)

    chain_id = residue_first[0]
    pos_first, pos_last = residue_first[1:], residue_last[1:]
    chain = model[chain_id]
    reslist = []
    for res in Selection.unfold_entities(chain, 'R'):
        pos_current = (res.id[1], res.id[2])
        if pos_first <= pos_current <= pos_last:
            reslist.append(res)
    return reslist

def extract_chain(model, chain_name):
    # chain_id = residue_first[0]
    chain = model[chain_name]
    seq = entity_to_seq(chain)[0]
    return seq
def entity_to_seq(entity):
    seq = ''
    mapping = []
    for res in Selection.unfold_entities(entity, 'R'):
        try:
            seq += three_to_one(res.get_resname())
            mapping.append(res.get_id())
        except KeyError:
            pass
    assert len(seq) == len(mapping)
    return seq, mapping


def folding(task:FoldingTask):
   
    Hname = task.Hname
    Lname = task.Lname
    
    out_path =  task.set_current_path_tag('fold')
    ref_out_path =  os.path.join(os.path.dirname(task.ref_path), 'REF1_fold.pdb')
    # in_path = task.in_path
    # ref_path = task.ref_path
    
    print(Hname+"  "+Lname)
    model_gen = task.get_gen_biopython_model()
    model_ref = task.get_ref_biopython_model()
    seq_H = extract_chain(model_gen, Hname)
    seq_L = extract_chain(model_gen, Lname)
    ref_H = extract_chain(model_ref, Hname)
    ref_L = extract_chain(model_ref, Lname)
    
    sequences = {
      'H': seq_H,
      'L': seq_L
      }
    try:
        antibody = predictor.predict(sequences)
        antibody.save(out_path)
    except Exception as e:
        print(e)
  
    
    sequences_REF = {
      'H': ref_H,
      'L': ref_L
      }
    try:
        # os.makedirs(ref_out_path, exist_ok=True)
        antibody1 = predictor.predict(sequences_REF)
        antibody1.save(ref_out_path)
    except Exception as e:
        print(e)
    
    return task

def align(task: FoldingTask):
    ref_path = task.ref_path
    
   
    fpath = task.set_current_path_tag('fold')
    out_path =  task.set_current_path_tag('fold_align')
    ref_fold_path = os.path.join(os.path.dirname(ref_path), 'REF1_fold.pdb')
    ref_out_path =  os.path.join(os.path.dirname(ref_path), 'REF1_fold_align.pdb')
    antigen_chains = set()
    model_ref = task.get_ref_biopython_model()
    for chain in model_ref:
        if chain.id not in task.ab_chains:
            antigen_chains.add(chain.id)
    align_and_move_chains(ref_path, fpath, out_path, antigen_chains)
    
        
    
    
    align_and_move_chains(ref_path, ref_fold_path, ref_out_path,antigen_chains)
    
    return task
