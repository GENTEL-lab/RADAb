import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.modules.common.layers import clampped_one_hot
from diffab.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, rotation_to_so3vec
from diffab.utils.protein.constants import esm_ressymb_to_resindex


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class PositionTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t):
        """
        Args:
            p_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        """
        alpha_bar = self.var_sched.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(p_0)
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)

        return p_noisy, e_rand

    def denoise(self, p_t, eps_p, mask_generate, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(-1, 1, 1)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(-1, 1, 1)

        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)
        return p_next


class RotationTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        # Forward (perturb)
        c1 = torch.sqrt(1 - self.var_sched.alpha_bars) # (T,).
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        # Inverse (generate)
        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        self.register_buffer('_dummy', torch.empty([0, ]))

    def add_noise(self, v_0, mask_generate, t):
        """
        Args:
            v_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        """
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        # Noise rotation
        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device)    # (N, L, 3)
        e_normal = e_scaled / (c1 + 1e-8)
        E_scaled = so3vec_to_rotation(e_scaled)   # (N, L, 3, 3)

        # Scaled true rotation
        R0_scaled = so3vec_to_rotation(c0 * v_0)  # (N, L, 3, 3)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        return v_noisy, e_scaled

    def denoise(self, v_t, v_next, mask_generate, t):
        N, L = mask_generate.size()
        e = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device) # (N, L, 3)
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e, 
            torch.zeros_like(e) # Simply denoise and don't add noise at the last step
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)

        return v_next


class AminoacidCategoricalTransition(nn.Module):
    
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L).
        """
        N, L, K = c.size()
        c = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t):
        """
        Args:
            x_0:    (N, L)序列
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            c_t:    Probability, (N, L, K).独热
            x_t:    Sample, LongTensor, (N, L).
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, L, K).
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None] # (N, 1, 1)
        c_noisy = (alpha_bar*c_0) + ( (1-alpha_bar)/K )
        c_t = torch.where(mask_generate[..., None].expand(N,L,K), c_noisy, c_0)
        x_t = self._sample(c_t)
        #c_noisy_length = mask_generate.sum().item()
        return c_t, x_t

    def posterior(self, x_t, x_0, t):
        """
        Args:
            x_t:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            x_0:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            t:  (N,).
        Returns:
            theta:  Posterior probability at (t-1)-th step, (N, L, K).
        """
        K = self.num_classes

        if x_t.dim() == 3:
            c_t = x_t   # When x_t is probability distribution.
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float() # (N, L, K)

        if x_0.dim() == 3:
            c_0 = x_0   # When x_0 is probability distribution.
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, L, K)

        alpha = self.var_sched.alpha_bars[t][:, None, None]     # (N, 1, 1)
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None] # (N, 1, 1)

        theta = ((alpha*c_t) + (1-alpha)/K) * ((alpha_bar*c_0) + (1-alpha_bar)/K)   # (N, L, K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def denoise(self, x_t, c_0_pred, mask_generate, t):
        """
        Args:
            x_t:        (N, L).
            c_0_pred:   Normalized probability predicted by networks, (N, L, K).
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            post:   Posterior probability at (t-1)-th step, (N, L, K).
            x_next: Sample at (t-1)-th step, LongTensor, (N, L).
        """
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()  # (N, L, K)
        post = self.posterior(c_t, c_0_pred, t=t)   # (N, L, K)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)
        return post, x_next
 
    
    
    
class DiscreteUniformTransition:
    def __init__(self, x_classes: int):
        self.X_classes = x_classes

        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes


    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx)
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)

        return q_x

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx)
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x

        return q_x

class BlosumTransition:
    def __init__(self, blosum_path='dataset_src/blosum_substitute.pt',x_classes=20,timestep = 100):
        try:
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        except FileNotFoundError:
            blosum_path = '../'+blosum_path
            self.original_score,self.temperature_list,self.Qt_temperature = torch.load(blosum_path)['original_score'], torch.load(blosum_path)['Qtb_temperature'],torch.load(blosum_path)['Qt_temperature'] 
        self.X_classes = x_classes
        self.timestep = timestep
        temperature_list = self.temperature_list.unsqueeze(dim=0)
        temperature_list = temperature_list.unsqueeze(dim=0)
        Qt_temperature = self.Qt_temperature.unsqueeze(dim=0)
        Qt_temperature = Qt_temperature.unsqueeze(dim=0)
        if temperature_list.shape[0] != self.timestep:
            output_tensor = F.interpolate(temperature_list, size=timestep+1, mode='linear', align_corners=True)
            self.temperature_list = output_tensor.squeeze()
            output_tensor = F.interpolate(Qt_temperature, size=timestep+1, mode='linear', align_corners=True)
            self.Qt_temperature = output_tensor.squeeze()
        else:    
            self.temperature_list = self.temperature_list
            self.Qt_temperature = self.Qt_temperature
    
    def get_Qt_bar(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.temperature_list = self.temperature_list.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.temperature_list[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        q_x[q_x < 1e-6] = 1e-6
        return q_x

    def get_Qt(self, t_normal, device):

        self.original_score = self.original_score.to(device)
        self.Qt_temperature = self.Qt_temperature.to(device)
        t_int = torch.round(t_normal * self.timestep).to(device)
        temperatue = self.Qt_temperature[t_int.long()]       
        q_x = self.original_score.unsqueeze(0)/temperatue.unsqueeze(2)
        q_x = torch.softmax(q_x,dim=2)
        return q_x
    
    
