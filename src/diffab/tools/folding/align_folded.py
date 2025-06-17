import subprocess

import os
import time
from Bio import PDB

def remove_chain(input_file, output_file, chain_id):
    parser = PDB.PDBParser()
    io = PDB.PDBIO()
    structure = parser.get_structure('structure', input_file)
    
    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain.id != chain_id
    
    io.set_structure(structure)
    io.save(output_file, ChainSelect())

def align_and_move_chains(file_ref, file_gen,file_out, chains):
    timestamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    
    aligned_gen_path = f"./tmp/aligned_gen_{timestamp}.pdb"
    remove_l_chain_path = f"./tmp/gen_no_L_{timestamp}.pdb"
    align_script_path = f"./tmp/align_{timestamp}.pml"
    move_script_path = f"./tmp/move_chains_{timestamp}.pml"
    
    
    # remove_chain(file_gen, remove_l_chain_path, 'L')
    remove_chain(file_gen, remove_l_chain_path, 'H')
    align_script_content = f"""
    load {file_ref}, ref
    load {remove_l_chain_path}, gen
    align gen, ref
    save {aligned_gen_path}, gen
    quit
    """
    with open(align_script_path, "w") as align_script_file:
        align_script_file.write(align_script_content)
    
    subprocess.run(["pymol", "-cq", align_script_path])

    chains_selection = " or ".join([f"chain {chain}" for chain in chains])
    move_script_content = f"""
    load {aligned_gen_path}, gen
    load {file_ref}, ref
    create gen_chains, ref and ({chains_selection})
    save {file_out}, gen or gen_chains
    quit
    """
    with open(move_script_path, "w") as move_script_file:
        move_script_file.write(move_script_content)
    
    
    subprocess.run(["pymol", "-cq", move_script_path])

    os.remove(aligned_gen_path)
    os.remove(remove_l_chain_path)
    os.remove(align_script_path)
    os.remove(move_script_path)
