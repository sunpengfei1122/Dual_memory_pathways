import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.init import trunc_normal_
from torch import fft
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete

torch.manual_seed(42)
# ------------------------
# Spiking LMU Cell (with gamma gate & gradient taps)
# ------------------------
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode
# ------------------------
# Utils / Initializers
# ------------------------
def leCunUniform(tensor):
    fan_in = init._calculate_correct_fan(tensor, "fan_in")
    limit = np.sqrt(3. / fan_in)
    init.uniform_(tensor, -limit, limit)

# ------------------------
# Base LMU (FFT form)
# ------------------------
class LMUFFTCell(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):
        super(LMUFFTCell, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u = nn.Linear(in_features=input_size, out_features=1)
        self.f_u = nn.ReLU()

        self.W_x = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.f_h = nn.ReLU()

        A, B = self.stateSpaceMatrices()
        self.register_buffer("A", A)     # [memory_size, memory_size]
        self.register_buffer("B", B)     # [memory_size, 1]

        H, fft_H = self.impulse()
        self.register_buffer("H", H)         # [memory_size, seq_len]
        self.register_buffer("fft_H", fft_H) # [memory_size, seq_len + 1]

    def stateSpaceMatrices(self):
        Q = np.arange(self.memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing="ij")
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))
        A, B, C, D, dt = cont2discrete(system=(A, B, C, D), dt=1.0, method="zoh")
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()
        return A, B

    def impulse(self):
        H = []
        A_i = torch.eye(self.memory_size).to(self.A.device)
        for _ in range(self.seq_len):
            H.append(A_i @ self.B)
            A_i = self.A @ A_i
        H = torch.cat(H, dim=-1)                         # [memory_size, seq_len]
        fft_H = fft.rfft(H, n=2*self.seq_len, dim=-1)    # [memory_size, seq_len+1]
        return H, fft_H



def get_act(act_type='spike', **act_params):
    act_type = act_type.lower()
    if act_type == 'spike':
        return MultiStepLIFNode(**act_params)
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'identity':
        return nn.Identity()

class SpikingLMUFFTCell(LMUFFTCell):
    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):
        super(SpikingLMUFFTCell, self).__init__(input_size, hidden_size, memory_size, seq_len, theta)
        self.f_x = get_act('spike', tau=2.0, detach_reset=True)
        self.weight = nn.Linear(in_features=memory_size, out_features=hidden_size)
        self.act_loss = 0.0
        self.bn_x = nn.BatchNorm1d(hidden_size) 
        
        # --- gradient taps
        self._last_states = None   # [B,T,H]  -> pre activation of current layer
        self._last_x = None        # [B,T,1]  -> LMU input proxy (pre-nonlinearity W_x(x))
        self._states_name = f"v_pre_{input_size}"

        # --- LMU gate gamma for ablation (1.0 => LMU on; 0.0 => LIF-like)
        self.gamma = 1.0

    def set_memory_gate(self, gamma: float):
        self.gamma = float(gamma)

    def cal_act_loss(self, x):
        return torch.sum(torch.abs(x))

    def forward(self, x):
        """
        x: [B,T,C]
        path:
          u = ReLU(W_u(x)) -> FFT -> m
          x_pre = W_x(x)                        # capture cross-layer input proxy
          pre   = x_pre + gamma * weight(m)     # capture same-layer pre for grads
          u_spike = f_x(pre)
        """
        B, T, C = x.shape

        # ---- memory path m[t] via FFT conv of u=W_u(x)
        u = self.f_u(self.W_u(x))                       # [B,T,1]
        fft_input = u.permute(0, 2, 1)                  # [B,1,T]
        fft_u = fft.rfft(fft_input, n=2*T, dim=-1)
        temp = fft_u * self.fft_H.unsqueeze(0)          # [B,memory,T+1]
        m = fft.irfft(temp, n=2*T, dim=-1)              # [B,memory,T+1]
        m = m[:, :, :T].permute(0, 2, 1).contiguous()   # [B,T,memory]

        # ---- cross-layer input proxy (before nonlinearity)
        x_pre = self.W_x(x)                              # [B,T,H]
  
        x_pre_scalar = x_pre.mean(dim=-1, keepdim=True) # [B,T,1]

        # ---- same-layer pre (for v-pre grads)
        pre  = self.W_x(x)+self.weight(m) # [B,N,C]
        u_spike = self.f_x(pre.permute(1, 0, 2))
        u_spike = u_spike.permute(1,0,2).contiguous() # [T,B,C]->[B,T,C]

        # ---- record grad taps
        if torch.is_grad_enabled():
            self._last_states = pre
            self._last_states.retain_grad()
            self._last_x = x_pre_scalar
            self._last_x.retain_grad()
            self._states_name = f"v_pre_{C}"
        else:
            self._last_states = None
            self._last_x = None

        h_n = u_spike[:, -1, :].unsqueeze(-1)           # [B,H,1] for compatibility
        return u_spike, h_n


# ------------------------
# Model wrapper
# ------------------------
class ConvLMU2(nn.Module):
    def __init__(self, dd, TT, t):
        super().__init__()
        self.LMU1 = SpikingLMUFFTCell(input_size=140,   hidden_size=128, memory_size=dd, seq_len=TT, theta=t)
        self.LMU2 = SpikingLMUFFTCell(input_size=128, hidden_size=128, memory_size=dd, seq_len=TT, theta=t)
        self.dense = nn.Linear(in_features=128, out_features=20)
        self.rnn_drop = nn.Dropout(p=0.2)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_layers_for_vis(self):
        return [(self.LMU1, "LMU1"), (self.LMU2, "LMU2")]

    def set_memory_gate(self, gamma: float):
        # broadcast gate to submodules
        if hasattr(self.LMU1, "set_memory_gate"):
            self.LMU1.set_memory_gate(gamma)
        if hasattr(self.LMU2, "set_memory_gate"):
            self.LMU2.set_memory_gate(gamma)

    def forward_features(self, x):
        x, _ = self.LMU1(x)
        x = self.rnn_drop(x)
        x, _ = self.LMU2(x)
        return x

    def forward(self, x):
        self.act_loss = 0.0
        x = self.forward_features(x)                # [B,T,H]
        x = self.dense(x.squeeze())                 # [B,T,10]
        x = x.mean(dim=1)
        return x

