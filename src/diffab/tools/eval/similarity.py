import numpy as np
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
from Bio import pairwise2
from Bio.Align import substitution_matrices
import logging

from diffab.tools.eval.base import EvalTask
from diffab.tools.eval.Chit import caclute_CDRH3,caclute_others
from diffab.tools.eval.scRMSD import scRMSD
from diffab.tools.eval.plausibility import plausibility

import subprocess
import re


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


def reslist_seqid(res_list1, res_list2):
    seq1, _ = entity_to_seq(res_list1)
    seq2, _ = entity_to_seq(res_list2)
    _, seq_id = align_sequences(seq1, seq2)
    return seq_id


def align_sequences(sequence_A, sequence_B, **kwargs):
    """
    Performs a global pairwise alignment between two sequences
    using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
    as implemented in Biopython. Returns the alignment, the sequence
    identity and the residue mapping between both original sequences.
    """

    def _calculate_identity(sequenceA, sequenceB):
        """
        Returns the percentage of identical characters between two sequences.
        Assumes the sequences are aligned.
        """

        sa, sb, sl = sequenceA, sequenceB, len(sequenceA)
        matches = [sa[i] == sb[i] for i in range(sl)]
        seq_id = (100 * sum(matches)) / sl
        return seq_id

        # gapless_sl = sum([1 for i in range(sl) if (sa[i] != '-' and sb[i] != '-')])
        # gap_id = (100 * sum(matches)) / gapless_sl
        # return (seq_id, gap_id)

    #
    matrix = kwargs.get('matrix', substitution_matrices.load("BLOSUM62"))
    gap_open = kwargs.get('gap_open', -10.0)
    gap_extend = kwargs.get('gap_extend', -0.5)

    alns = pairwise2.align.globalds(sequence_A, sequence_B,
                                    matrix, gap_open, gap_extend,
                                    penalize_end_gaps=(False, False) )
    try:
        best_aln = alns[0]
        aligned_A, aligned_B, score, begin, end = best_aln

        # Calculate sequence identity
        seq_id = _calculate_identity(aligned_A, aligned_B)
        return (aligned_A, aligned_B), seq_id
    except Exception as e:
        logging.error(e)
        return None, 0


def extract_reslist(model, residue_first, residue_last): 
    assert residue_first[0] == residue_last[0]
    residue_first, residue_last = tuple(residue_first), tuple(residue_last)

    chain_id = residue_first[0]
    pos_first, pos_last = residue_first[1:], residue_last[1:]
    # chain = model[chain_id]
    # print(model)
    # chain = model['H'] #eval H chain
    chain = model['L'] #eval L chain
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
def extract_chain(model, residue_first):
    chain_id = residue_first[0]
    #chain = model['H'] 
    chain = model['L'] 
    seq = []
    seq.append(entity_to_seq(chain)[0])
    return seq

    


def caclute_Chit(chain, res_first, res_last, CDRtype):
    print(CDRtype)
    if CDRtype == 'H_CDR3':
        return caclute_CDRH3(chain, res_first, res_last)
    else:
        return caclute_others(chain, CDRtype)
        
def eval_similarity(task: EvalTask):
    try:
        model_gen = task.get_gen_biopython_model()
        model_ref = task.get_ref_biopython_model()
        seq_gen = extract_chain(model_gen, task.residue_first)
        print(task.residue_first[0])
        # chain_ref = extract_chain(model_ref, task.residue_first)
        reslist_gen = extract_reslist(model_gen, task.residue_first, task.residue_last)
        reslist_ref = extract_reslist_ref(model_ref, task.residue_first, task.residue_last)
        CDR_gen = entity_to_seq(reslist_gen)[0]
        cdr_seq = []
        cdr_seq.append(CDR_gen)
        # reslist_gen_chain = extract_reslist(model_gen)
        task.scores.update({
            'seqid': reslist_seqid(reslist_gen, reslist_ref),
            'scRMSD': scRMSD(reslist_gen, reslist_ref), 

            'plausiblity': plausibility(seq_gen),#seq
            'pll':plausibility(cdr_seq),
            'seq': entity_to_seq(reslist_gen)[0],
            'seq_ref': entity_to_seq(reslist_ref)[0]
        })
    except Exception as e:
        logging.warning(
            f'{e.__class__.__name__}: {str(e)} '
        )  
        # task.mark_failure()
    return task

