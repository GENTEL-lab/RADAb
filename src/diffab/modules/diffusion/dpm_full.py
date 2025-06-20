import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from tqdm.auto import tqdm

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.encoders.ga import GAEncoder
from diffab.modules.common.MSATransformer import MSATransformerTime
from diffab.modules.common.layers import clampped_one_hot
from diffab.utils.protein.constants import ressymb_to_resindex, esm_ressymb_to_resindex
import esm
from .transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


class EpsilonNet(nn.Module):

    def __init__(self, res_feat_dim, pair_feat_dim,timesetps_for_trans_seq, num_layers, encoder_opt={}, trans_seq_opt={}):
        super().__init__()
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)  # 22 is padding
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 3, res_feat_dim*2), nn.ReLU(), 
            nn.Linear(res_feat_dim * 2, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
       
        # self.MSAtransformer = MSATransformerTime(384, 1536, 2, 6,                                                  
        #                                         n_tokens = 22, padding_idx = 21, mask_idx = 20, max_positions=1024, timesteps = 101)
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        self.trans_seq = AminoacidCategoricalTransition(timesetps_for_trans_seq, **trans_seq_opt)
        
        self.eps_crd_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_rot_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 3)
        )

        self.eps_seq_net = nn.Sequential(
            nn.Linear(res_feat_dim+3, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim), nn.ReLU(),
            nn.Linear(res_feat_dim, 20)
            # , nn.Softmax(dim=-1) 
        )
        
    

    def forward(self, v_t, p_t, s_t, res_feat, pair_feat,
                seq_feat, 
                beta, mask_generate, mask_res, ref_martixs, t, cdr_flag ):
        """
        Args:
            v_t:    (N, L, 3).
            p_t:    (N, L, 3).
            s_t:    (N, L).
            res_feat:   (N, L, res_dim).
            pair_feat:  (N, L, L, pair_dim).
            seq_feat:   (N, L, res_dim).
            beta:   (N,).
            mask_generate:    (N, L).
            mask_res:       (N, L).
            ref_martixs:    (N, 10, L)
        Returns:
            v_next: UPDATED (not epsilon) SO3-vector of orietnations, (N, L, 3).
            eps_pos: (N, L, 3).
        """
        N, L = mask_res.size()
        R = so3vec_to_rotation(v_t) # (N, L, 3, 3)

        res_feat = self.res_feat_mixer(torch.cat([res_feat,
                                                  seq_feat,
                                                  self.current_sequence_embedding(s_t)], dim=-1))
        res_feat = self.encoder(R, p_t, res_feat, pair_feat, mask_res)

        
        
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].expand(N, L, 3)
        in_feat = torch.cat([res_feat, t_embed], dim=-1)

        # Position changes
        eps_crd = self.eps_crd_net(in_feat)    # (N, L, 3)
        eps_pos = apply_rotation_to_vector(R, eps_crd)  # (N, L, 3)
        eps_pos = torch.where(mask_generate[:, :, None].expand_as(eps_pos), eps_pos, torch.zeros_like(eps_pos))

        # New orientation
        eps_rot = self.eps_rot_net(in_feat)    # (N, L, 3)
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_next = R @ U
        v_next = rotation_to_so3vec(R_next)     # (N, L, 3)
        v_next = torch.where(mask_generate[:, :, None].expand_as(v_next), v_next, v_t)

        # New sequence categorical distributions
        c_denoised = self.eps_seq_net(in_feat)  
        # c_denoised_0_softmax = torch.nn.functional.softmax(c_denoised_0, dim = -1)# Already softmax-ed, (N, L, 20)
        # _, s_next = self.trans_seq.denoise(s_t, c_denoised_0_softmax, mask_generate, t)
        # s_t_expanded = s_next.unsqueeze(1)  
        # device = s_next.device
        # ref_martixs = ref_martixs.to(device)
        

        # result = torch.cat((s_t_expanded, ref_martixs), dim=1)  
        # result_MSA  = self.MSAtransformer(result, t)
        # p = result_MSA[:, 0, :, :20]
        #t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long)
        #cdr_flag = cdr_flag
        

        # c_denoised = torch.nn.functional.softmax((c_denoised_0 + p), dim = -1)

        return v_next, R_next, eps_pos, c_denoised
class FullDPM(nn.Module):

    def __init__(
        self, 
        res_feat_dim, 
        pair_feat_dim, 
        num_steps, 
        eps_net_opt={}, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
        
    ):
        super().__init__()
        self.eps_net = EpsilonNet(res_feat_dim, pair_feat_dim,num_steps, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)
        self.esm2, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm2.to('cuda')
        for name, param in self.esm2.named_parameters():
            param.requires_grad = False
        self.MLP = nn.Sequential(
            nn.Linear(1280, 540),nn.ReLU(),
            nn.Linear(540, 128),nn.ReLU()
        )
        
       
        self.MSAtransformer = MSATransformerTime(384, 1536, 2, 6,                                                  
                                                 #use_ckpt=True,
                                                n_tokens = 22, padding_idx = 21, mask_idx = 20, max_positions=1024, timesteps = 101)
        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, v_0, p_0, s_0, res_feat, pair_feat, fragment_type, mask_generate, mask_res, ref_martixs, cdr_flag, denoise_structure, denoise_sequence, t=None):
        N, L = res_feat.shape[:2]
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        p_0 = self._normalize_position(p_0)

        if denoise_structure:
            # Add noise to rotation
            R_0 = so3vec_to_rotation(v_0)
            v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
        else:
            R_0 = so3vec_to_rotation(v_0)
            v_noisy = v_0.clone()
            p_noisy = p_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence 
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        else:
            s_noisy = s_0.clone()
        
        # seqs = torch.where(
        #         mask_generate, 
        #         torch.full_like(s_noisy, fill_value=20),    
        #         s_noisy
        #     ) 
        
        seqs = s_noisy
        # N,L = batch['aa'].size()
        antibody_seqs = torch.full((N,L),21,device = 'cuda')
        mask = (fragment_type == 1) | ((fragment_type == 2))
        antibody_seqs[mask] = s_noisy[mask]
        ressymb_to_resindex = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20, '-': 21
        }
        seq_list = []
        for i, seq in enumerate(antibody_seqs, 1):
            str_seq = ''.join([list(ressymb_to_resindex.keys())[list(ressymb_to_resindex.values()).index(res)] for res in seq])
            seq_list.append((str(i), str_seq))
        batch_labels, batch_strs, batch_tokens = self.batch_converter(seq_list)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to('cuda')
        # with torch.no_grad():
        results = self.esm2(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        sequence_representations = token_representations[:,1 : - 1,:]
       
        seq_feat = self.MLP(sequence_representations)
        
        beta = self.trans_pos.var_sched.betas[t]
        
        
        v_pred, R_pred, eps_p_pred, c_denoised_0 = self.eps_net(
            v_noisy, p_noisy, s_noisy, res_feat, pair_feat,
            seq_feat,
            beta, mask_generate, mask_res, ref_martixs, t, cdr_flag
        )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)
        
        # v_pred, R_pred, eps_p_pred, c_denoised_0 = self.eps_net(
        #     v_noisy, p_noisy, s_noisy, res_feat, pair_feat, beta, mask_generate, mask_res, ref_martixs, t, cdr_flag
        # )   # (N, L, 3), (N, L, 3, 3), (N, L, 3), (N, L, 20), (N, L)
        
        c_denoised_0_softmax = torch.nn.functional.softmax(c_denoised_0, dim = -1)
        
        _, s_next = self.trans_seq.denoise(s_noisy, c_denoised_0_softmax, mask_generate, t)
        s_t_expanded = s_next.unsqueeze(1)  
        ref_martixs = ref_martixs.to(s_next.device)
        result = torch.cat((s_t_expanded, ref_martixs), dim=1) 
        
        result_MSA = self.MSAtransformer(result,t) 
        p = result_MSA[:, 0, :, :20] 
        # New sequence categorical distributions
        c_denoised = torch.nn.functional.softmax((c_denoised_0 + p), dim = -1)
        
        loss_dict = {}

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq
        return loss_dict

    @torch.no_grad()
    def sample(
        self, 
        v, p, s, ref_martixs,cdr_flag,
        res_feat, pair_feat, fragment_type,
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False
        
    ):
        """
        Args:
            v:  Orientations of contextual residues, (N, L, 3).
            p:  Positions of contextual residues, (N, L, 3).
            s:  Sequence of contextual residues, (N, L).
            CDR_flag:占位符，后续添加所有CDR有用
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            v_rand = random_uniform_so3([N, L], device=self._dummy.device)
            p_rand = torch.randn_like(p)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_rand, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_rand, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s

        traj = {self.num_steps: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            
            seqs = s_t
            antibody_seqs = torch.full((N,L),21,device = 'cuda')
            mask = (fragment_type == 1) | ((fragment_type == 2))
            antibody_seqs[mask] = seqs[mask]
            ressymb_to_resindex = {
                'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
                'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
                'X': 20, '-': 21
            }
            seq_list = []
            for i, seq in enumerate(antibody_seqs, 1):
                str_seq = ''.join([list(ressymb_to_resindex.keys())[list(ressymb_to_resindex.values()).index(res)] for res in seq])
                seq_list.append((str(i), str_seq))
            batch_labels, batch_strs, batch_tokens = self.batch_converter(seq_list)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to('cuda')
            with torch.no_grad():
                results = self.esm2(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            sequence_representations = []
            for i, tokens_len in enumerate(batch_lens):
                sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0)) #加.mean(0) 为N,1280，不加则为N,L,1280
            sequence_representations = token_representations[:,1 : - 1,:]
            seq_feat = self.MLP(sequence_representations)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, 
                seq_feat, 
                beta, mask_generate, mask_res, ref_martixs, t_tensor, cdr_flag
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)
            #c_denoised_0_softmax = torch.nn.functional.softmax(c_denoised_0, dim = -1)
            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            
            c_denoised_softmax = torch.nn.functional.softmax(c_denoised, dim = -1)
            
            _, s_next_0 = self.trans_seq.denoise(s_t, c_denoised_softmax, mask_generate, t_tensor)
            s_t_expanded = s_next_0.unsqueeze(1)  
            device = s_next_0.device
            ref_martixs = ref_martixs.to(device)
            
            
            result = torch.cat((s_t_expanded, ref_martixs), dim=1)  
            result_MSA  = self.MSAtransformer(result, t_tensor)
            p = result_MSA[:, 0, :, :20]
            
            

            c_denoised_new = torch.nn.functional.softmax((c_denoised + p), dim = -1)
            _,s_next = self.trans_seq.denoise(s_t, c_denoised_new, mask_generate, t_tensor)
            
            
            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def optimize(
        self, 
        v, p, s, ref_martixs,
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=True, sample_sequence=True,
        pbar=False
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t)
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            _, s_noisy = self.trans_seq.add_noise(s, mask_generate, t)
            s_init = torch.where(mask_generate, s_noisy, s)
        else:
            s_init = s

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, ref_martixs
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj

    @torch.no_grad()
    def sample_with_retrieval_structure(
        self, 
        v, p, s, ref_martixs,
        opt_step: int,
        res_feat, pair_feat, 
        mask_generate, mask_res, 
        sample_structure=False, sample_sequence=True,
        pbar=False
    ):
        """
        Description:
            First adds noise to the given structure, then denoises it.
        """
        N, L = v.shape[:2]
        p = self._normalize_position(p)
        t = torch.full([N, ], fill_value=opt_step, dtype=torch.long, device=self._dummy.device)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            # Add noise to rotation
            v_noisy, _ = self.trans_rot.add_noise(v, mask_generate, t) 
            # Add noise to positions
            p_noisy, _ = self.trans_pos.add_noise(p, mask_generate, t)
            v_init = torch.where(mask_generate[:, :, None].expand_as(v), v_noisy, v)
            p_init = torch.where(mask_generate[:, :, None].expand_as(p), p_noisy, p)
        else:
            v_init, p_init = v, p

        if sample_sequence:
            s_rand = torch.randint_like(s, low=0, high=19)
            s_init = torch.where(mask_generate, s_rand, s)
        else:
            s_init = s 

        traj = {opt_step: (v_init, self._unnormalize_position(p_init), s_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=opt_step, desc='Optimizing')
        else:
            pbar = lambda x: x
        for t in pbar(range(opt_step, 0, -1)):
            v_t, p_t, s_t = traj[t]
            p_t = self._normalize_position(p_t)
            
            beta = self.trans_pos.var_sched.betas[t].expand([N, ])
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            v_next, R_next, eps_p, c_denoised = self.eps_net(
                v_t, p_t, s_t, res_feat, pair_feat, beta, mask_generate, mask_res, ref_martixs
            )   # (N, L, 3), (N, L, 3, 3), (N, L, 3)

            v_next = self.trans_rot.denoise(v_t, v_next, mask_generate, t_tensor)
            p_next = self.trans_pos.denoise(p_t, eps_p, mask_generate, t_tensor)
            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            if not sample_structure:
                v_next, p_next = v_t, p_t
            if not sample_sequence:
                s_next = s_t

            traj[t-1] = (v_next, self._unnormalize_position(p_next), s_next)
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj