# pyright: reportMissingImports=false
import pyrosetta
import subprocess
from Bio.PDB import PDBParser, Selection
import numpy as np
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
pyrosetta.init(' '.join([
    '-mute', 'all',
    '-use_input_sc',
    '-ignore_unrecognized_res',
    '-ignore_zero_occupancy', 'false',
    '-load_PDB_components', 'false',
    '-relax:default_repeats', '2',
    '-no_fconfig',
]))

from diffab.tools.eval.base import EvalTask


def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return pose.scores['dG_separated']

def reslist_rmsd(res_list1, res_list2):
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = np.array(res_short[i]['CA'].get_coord())
        coord_j = np.array(res_long[j]['CA'].get_coord())
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        SD[i, j] = sum([ d(i+k, j+k) for k in range(N-j) ])
    
    for j in range(N):
        SD[M-1, j] = d(M-1, j)

    for i in range(M-2, -1, -1):
        for j in range((N-(M-i))-1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i+1, j+1],
                SD[i, j+1]
            )

    min_SD = SD[0, :N-M+1].min()
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD
def scRMSD(genCDR, refCDR):
    """
    Calculate scRMSD between predicted and reference chains.
    """
    
    return reslist_rmsd(genCDR, refCDR)

def extract_reslist(model, residue_first, residue_last): 
    assert residue_first[0] == residue_last[0]
    residue_first, residue_last = tuple(residue_first), tuple(residue_last)

    chain_id = residue_first[0]
    pos_first, pos_last = residue_first[1:], residue_last[1:]
    # chain = model[chain_id]
    # print(model)
    chain = model['H']
    reslist = []
    for res in Selection.unfold_entities(chain, 'R'):
        pos_current = (res.id[1], res.id[2])
        if pos_first <= pos_current <= pos_last:
            reslist.append(res)
    return reslist
def extract_reslist_ref(model, residue_first, residue_last):
    assert residue_first[0] == residue_last[0]
    residue_first, residue_last = tuple(residue_first), tuple(residue_last)

    chain_id = residue_first[0]
    pos_first, pos_last = residue_first[1:], residue_last[1:]
    chain = model[chain_id]
    # print(model)
    # chain = model['H']
    reslist = []
    for res in Selection.unfold_entities(chain, 'R'):
        pos_current = (res.id[1], res.id[2])
        if pos_first <= pos_current <= pos_last:
            reslist.append(res)
    return reslist
def eval_interface_energy(task: EvalTask):
    model_gen = task.get_gen_biopython_model()
    model_ref = task.get_ref_biopython_model()
    reslist_gen = extract_reslist(model_gen, task.residue_first, task.residue_last)
    reslist_ref = extract_reslist_ref(model_ref, task.residue_first, task.residue_last)
    
    antigen_chains = set()
    for chain in model_ref:
        if chain.id not in task.ab_chains:
            antigen_chains.add(chain.id)
    # align_and_merge_pymol(file_ref, file_gen, output, antigen_chains)
    antigen_chains = ''.join(list(antigen_chains))
    antibody_chains = 'H'
    ref_antibody_chains = ''.join(task.ab_chains)
    
    interface_gen = f"{antibody_chains}_{antigen_chains}"
    
    interface_ref = f"{ref_antibody_chains}_{antigen_chains}"
    print('gen'+interface_gen)
    print(interface_ref)
    dG_gen = pyrosetta_interface_energy(task.in_path, interface_gen)
    dG_ref = pyrosetta_interface_energy(task.ref_path, interface_ref)
    
    task.scores.update({
        'dG_gen': dG_gen,
        'dG_ref': dG_ref,
        'ddG': dG_gen - dG_ref,
        'relax_rmsd': scRMSD(reslist_gen, reslist_ref),
    })
    return task
