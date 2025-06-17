from antiberty import AntiBERTyRunner
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one

antiberty = AntiBERTyRunner()

def plausibility(sequence):
    # antiberty = AntiBERTyRunner()
    pll = antiberty.pseudo_log_likelihood(sequence, batch_size=16)
    return pll.item()

