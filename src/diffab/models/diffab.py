import torch
import torch.nn as nn
import numpy as np
import esm
from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec
from diffab.modules.encoders.residue import ResidueEmbedding
from diffab.modules.encoders.pair import PairEmbedding
from diffab.modules.diffusion.dpm_full import FullDPM
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
from diffab.utils.retrieval.construct_seq_martix import get_retrieved_sequences_class_test_CDRclass, get_retrieved_sequences_class_test_CDRclass_with_noise,get_retrieved_sequences_class_test_CDRclass_1,get_retrieved_sequences_class_test_CDRclass_2,get_retrieved_sequences_class_test_CDRclass_groundtruth, get_retrieved_sequences_class_test_CDRclass_ablation
import esm
from ._base import register_model


resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}


@register_model('diffanti')
class DiffusionAntibodyDesign(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)
        
        self.diffusion = FullDPM(
            cfg.res_feat_dim,
            cfg.pair_feat_dim,
            **cfg.diffusion,
        )


    def encode(self, batch, remove_structure, remove_sequence):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        # This is used throughout embedding and encoding layers
        #   to avoid data leakage.
        context_mask = torch.logical_and(
            batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], 
            ~batch['generate_flag']    
        )
        #print(batch)
        structure_mask = context_mask if remove_structure else None
        sequence_mask = context_mask if remove_sequence else None
        # seqs = batch['aa']
        
        res_feat = self.residue_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            fragment_type = batch['fragment_type'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
        )
        
        pair_feat = self.pair_embed(
            aa = batch['aa'],
            res_nb = batch['res_nb'],
            chain_nb = batch['chain_nb'],
            pos_atoms = batch['pos_heavyatom'],
            mask_atoms = batch['mask_heavyatom'],
            structure_mask = structure_mask,
            sequence_mask = sequence_mask,
            
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA] 
        
        
        
        mask_generate = batch['generate_flag']
        true_counts_per_row = mask_generate.sum(dim=1)
        true_counts_list = true_counts_per_row.tolist()

        cdr_to_mask = batch['cdr_to_mask']  # (N,L)
        ref_martixs_full_list = []
       
       

        non_zero_indices = torch.argmax((cdr_to_mask != 0).int(), dim=1)

        cdr_flag = cdr_to_mask[torch.arange(cdr_to_mask.size(0)), non_zero_indices].tolist()
        
        
        for i, s_0_row in enumerate(batch['aa']):
            if self.training:
                ref_martix_list = get_retrieved_sequences_class_test_CDRclass_1(batch['pdb_id'][i][:4], true_counts_list[i],cdr_flag[i])
            else:
                ref_martix_list  = get_retrieved_sequences_class_test_CDRclass_2(batch['pdb_id'][i][:4], true_counts_list[i],cdr_flag[i])  #  only CDR_ref
            s_0_length = s_0_row.size(0) #L
            ref_depth = len(ref_martix_list)
            
            s_0_expanded_row = s_0_row.unsqueeze(0).expand(ref_depth, -1)#(10,L)
            mask_generate_shrink = mask_generate.clone()
            mask_generate_expanded_row = mask_generate_shrink[i].unsqueeze(0).repeat(ref_depth, 1).clone() #(10,L)
            ref_martix_expanded = torch.full((ref_depth, s_0_row.size(0)), 21, dtype=s_0_row.dtype, device=s_0_row.device) #（10，L）
            for j, seq in enumerate(ref_martix_list): #adapt for shrink
                seq_length = len(seq)
                diff = seq_length - true_counts_list[i] 
                true_positions = mask_generate_expanded_row[j].nonzero(as_tuple=False).squeeze()
                start, end = true_positions.min().item(), true_positions.max().item()
                current_true_length = end - start + 1
                diff = seq_length - current_true_length
                if diff > 0:
                    
                    start =  start - diff
                    if start < 0:
                        start = 0
                        end += diff
                elif diff < 0:
                    potential_starts = list(range(start, min(start - diff, s_0_length - seq_length) + 1))
                    potential_ends = list(range(max(end + diff, seq_length - 1), end + 1))
    
                    while (end - start + 1) != seq_length:
                        start = np.random.choice(potential_starts)
                        end = np.random.choice(potential_ends)
                    
                mask_generate_expanded_row[j, :] = False  
                mask_generate_expanded_row[j, start:end+1] = True  
                true_positions = mask_generate_expanded_row[j].nonzero(as_tuple=False).squeeze()
                seq_tensor = torch.as_tensor(seq, dtype=s_0_row.dtype, device=s_0_row.device )
                ref_martix_expanded[j][true_positions] = seq_tensor
            ref_martix_full_row = torch.where(mask_generate_expanded_row,ref_martix_expanded,s_0_expanded_row)    
            ref_martixs_full_list.append(ref_martix_full_row)
            
        ref_martixs_full = torch.stack(ref_martixs_full_list)  # (N, ref_depth, L)
        fragment_type = batch['fragment_type']
        # Hseqs = torch.
        return res_feat, pair_feat, R, p, ref_martixs_full,cdr_flag, fragment_type
    
    def forward(self, batch):
        mask_generate = batch['generate_flag']
        
        mask_res = batch['mask']
        true_counts_per_row = mask_generate.sum(dim=1)
        true_counts_list = true_counts_per_row.tolist()
        #print(true_counts_list)
        res_feat, pair_feat, R_0, p_0, ref_martixs,cdr_flag, fragment_type = self.encode(
            batch,
            remove_structure = self.cfg.get('train_structure', True),
            remove_sequence = self.cfg.get('train_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        loss_dict = self.diffusion(
            v_0, p_0, s_0, res_feat, pair_feat, fragment_type,mask_generate, mask_res, ref_martixs,cdr_flag,
            denoise_structure = self.cfg.get('train_structure', True),
            denoise_sequence  = self.cfg.get('train_sequence', True),
            )
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        batch, 
        sample_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0, ref_martixs, cdr_flag, fragment_type = self.encode(
            batch,
            remove_structure = sample_opt.get('sample_structure', True),
            remove_sequence = sample_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']

        traj = self.diffusion.sample(v_0, p_0, s_0, ref_martixs,cdr_flag, res_feat, pair_feat,fragment_type, mask_generate, mask_res, **sample_opt)
        return traj

    @torch.no_grad()
    def optimize(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0, ref_martixs = self.encode(
            batch,
            remove_structure = optimize_opt.get('sample_structure', True),
            remove_sequence = optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.optimize(v_0, p_0, s_0, ref_martixs, opt_step, res_feat, pair_feat, mask_generate, mask_res, **optimize_opt)
        return traj
    
    @torch.no_grad()
    def sample_with_retrieval_structure(
        self, 
        batch, 
        opt_step, 
        optimize_opt={
            'sample_structure': True,
            'sample_sequence': True,
        }
    ):
        mask_generate = batch['generate_flag']
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0, ref_martixs = self.encode(
            batch,
            remove_structure = optimize_opt.get('sample_structure', True),
            remove_sequence = optimize_opt.get('sample_sequence', True)
        )
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        traj = self.diffusion.optimize(v_0, p_0, s_0, ref_martixs, opt_step, res_feat, pair_feat, mask_generate, mask_res, **optimize_opt)
        return traj
