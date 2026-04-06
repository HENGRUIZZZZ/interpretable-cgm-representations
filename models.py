import os
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import NamedTuple

# P1 审计计划：扩大 si/mi 参数范围以增强可辨识度 (审计报告 v1.0)
P1_WIDE_PARAM_RANGE = os.environ.get("P1_WIDE_PARAM_RANGE", "").strip().lower() in ("1", "true", "yes")
# v10：全参数宽边界（缓解 ODE 参数边界饱和）
P1_V10_WIDE_BOUNDS = os.environ.get("P1_V10_WIDE_BOUNDS", "").strip().lower() in ("1", "true", "yes")
# P1 审计计划 2.2：固定 sg、p2 为生理均值，只学习 tau_m, Gb, si, mi 四参数
P1_FIX_SG_P2 = os.environ.get("P1_FIX_SG_P2", "").strip().lower() in ("1", "true", "yes")
# 实验方案 v5.0 终局之战：Prediction Head (z_init 4D + z_nonseq 16D → 2D SSPG/DI)
P1_V5_PREDICTION_HEAD = os.environ.get("P1_V5_PREDICTION_HEAD", "").strip().lower() in ("1", "true", "yes")
# v8: 让 16D non-seq 通过重构修正路径参与 CGM 重构
P1_V8_RECON_CORR = os.environ.get("P1_V8_RECON_CORR", "").strip().lower() in ("1", "true", "yes")
# v8: 让 16D non-seq 作为 ODE 动力学修正项
P1_V8_ODE_CORR = os.environ.get("P1_V8_ODE_CORR", "").strip().lower() in ("1", "true", "yes")


class NanWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        """ Masks the entire last dimension (usually the feature/channel dimension) if any element is NaN. """
        mask = ~torch.any(torch.isnan(x), dim=-1, keepdim=True)   # 0 if nan, 1 otherwise
        masked_x = torch.where(mask, x, torch.zeros_like(x))
        fx = self.module(masked_x)
        masked_fx = fx * mask
        return masked_fx
    
class ConvLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, channel_last=True):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, bias=bias)
        self.channel_last = channel_last
    
    def forward(self, x):
        assert x.ndim == 3, "Expected input to be (batch_size, seq_len, input_size) or (batch_size, input_size, seq_len)"
        if self.channel_last:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
        else:
            x = self.conv(x)
        return x

def constrain(x, min, max, temperature:float=1.):
    return (max - min) * torch.sigmoid(x / temperature) + min

def unconstrain(y, min, max, temperature:float=1, EPS:float=1e-8):
    assert torch.all(y >= min) and torch.all(y <= max)
    # ensure both numerator and denominator are positive
    numerator = y - min 
    denominator = max - min
    return temperature * torch.logit(numerator / denominator, eps=EPS)

def count_params(model: torch.nn.Module):
  """count number trainable parameters in a pytorch model"""
  total_params = sum(torch.numel(x) for x in model.parameters())
  return total_params

def to_seq(non_seq, *, like):
    _, T, _ = like.shape
    return non_seq[:, None, :].repeat(1, T, 1)

def from_seq(seq):
    B, T, D = seq.shape
    return seq.reshape(-1, D)

from typing import NamedTuple
Constraint = NamedTuple("Constraint", [("min", float), ("max", float)])

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

class MechanisticAutoencoder(torch.nn.Module):
    def __init__(self, meal_size, demographics_size, embedding_size, hidden_size, num_layers, encoder_dropout_prob, decoder_dropout_prob):
        super().__init__()
        # v10: 宽边界方案（默认关闭）
        if P1_V10_WIDE_BOUNDS:
            lim_tau_m = (5.0, 200.0)
            lim_gb = (50.0, 300.0)
            lim_sg = (1e-3, 5e-2)
            lim_si = (1e-5, 1e-2)
            lim_p2 = (1e-3, 0.2)
            lim_mi = (0.05, 5.0)
        else:
            # 历史范围：可选仅放宽 si/mi
            _si_lo, _si_hi = (1e-5, 1e-2) if P1_WIDE_PARAM_RANGE else (1e-4, 1e-3)
            _mi_lo, _mi_hi = (0.05, 5.0) if P1_WIDE_PARAM_RANGE else (0.1, 3.0)
            lim_tau_m = (10.0, 120.0)
            lim_gb = (60.0, 250.0)
            lim_sg = (5e-3, 2e-2)
            lim_si = (_si_lo, _si_hi)
            lim_p2 = (1.0 / 60.0, 1.0 / 15.0)
            lim_mi = (_mi_lo, _mi_hi)
        self.register_buffer("param_lims",
            torch.tensor([
                [lim_tau_m[0], lim_tau_m[1]],  # tau_m
                [lim_gb[0], lim_gb[1]],        # Gb
                [lim_sg[0], lim_sg[1]],        # sg
                [lim_si[0], lim_si[1]],        # si
                [lim_p2[0], lim_p2[1]],        # p2
                [lim_mi[0], lim_mi[1]],        # mi
            ], dtype=torch.float,
            )
        )
        self.register_buffer("state_lims",
            torch.tensor([
                [50., 300],  # G
                [0., 1.], # Ieff
                [0., 1.], # G1
                [0., 100.], # G2
            ], dtype=torch.float,
            )
        )
        self.param_size = self.param_lims.shape[0]
        self.state_size = self.state_lims.shape[0]
        self.fix_sg_p2 = P1_FIX_SG_P2
        if self.fix_sg_p2:
            self.learned_param_ix = [0, 1, 3, 5]
            self.encoding_size = 4 + self.state_size
            self.register_buffer("sg_fixed", torch.tensor([0.01], dtype=torch.float))
            self.register_buffer("p2_fixed", torch.tensor([1.0 / 30.0], dtype=torch.float))
            print(" [P1_FIX_SG_P2] ODE 仅学习 tau_m, Gb, si, mi; sg=0.01, p2=1/30 固定")
        else:
            self.learned_param_ix = list(range(6))
            self.encoding_size = self.param_size + self.state_size

        self.meal_size = meal_size
        self.demographics_size = demographics_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        cgm_size = 1
        cgm_diff_size = 1
        timestamp_size = 1

        self.meal_embedding = NanWrapper(ConvLinear(meal_size, embedding_size, channel_last=True))
        self.demographics_embedding = NanWrapper(nn.Linear(demographics_size, embedding_size))

        encoder_input_size = cgm_size + cgm_diff_size + timestamp_size + embedding_size + embedding_size
        decoder_input_size = timestamp_size + embedding_size + embedding_size

        self.encoder_input_size = encoder_input_size
        self.decoder_input_size = decoder_input_size

        self.encoder_lstm = nn.LSTM(input_size=encoder_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=encoder_dropout_prob, bidirectional=True)
        self.seq_proj = ConvLinear(in_features=2 * hidden_size,  # times 2 for bidirectional
                                   out_features=2)  # 2 for mean and std
        _encoder_cells_dim = 2 * hidden_size * num_layers
        self.non_seq_proj = nn.Linear(in_features=_encoder_cells_dim, out_features=2 * self.encoding_size)
        # 实验方案 v4.0：z_nonseq 16D 用于全特征 bakeoff
        self.nonseq_to_16 = nn.Linear(_encoder_cells_dim, 16)
        # v8 模块常驻；按开关决定是否参与 forward，便于 checkpoint 跨配置加载
        self.use_v8_recon_corr = P1_V8_RECON_CORR
        self.use_v8_ode_corr = P1_V8_ODE_CORR
        self.correction_mlp = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.ode_correction = nn.Sequential(
            nn.Linear(16 + 4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        if self.use_v8_recon_corr:
            print(" [V8] 启用 16D->CGM 重构修正路径 (correction_mlp)")
        if self.use_v8_ode_corr:
            print(" [V8] 启用 16D+state -> dstate ODE 修正路径 (ode_correction)")
        # 实验方案 v5.0：Prediction Head 输入 z_init(4D)+z_nonseq(16D)=20D → 2D (SSPG, DI)
        if P1_V5_PREDICTION_HEAD:
            self.prediction_head = nn.Sequential(
                nn.Linear(4 + 16, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 2),
            )
        else:
            self.prediction_head = None
        self.dt = 5.
        self.max_carb_per_min = 1000 # in mg/min

        self.decoder_dropout = nn.Dropout(decoder_dropout_prob)

        self.register_buffer("seq_p_mean",
            unconstrain(torch.tensor([0.], dtype=torch.float), min=0., max=self.max_carb_per_min)
        )
        self.register_buffer("seq_p_std",
            torch.ones_like(self.seq_p_mean) * 10
        )
        if self.fix_sg_p2:
            lims_learned = self.param_lims[self.learned_param_ix]
            self.register_buffer("nonseq_p_mean",
                torch.cat([
                    unconstrain(torch.tensor([30., 120., 5e-4, 1.0], dtype=torch.float), min=lims_learned[:, 0], max=lims_learned[:, 1]),
                    unconstrain(torch.tensor([120., 0.1, 0.1, 20.], dtype=torch.float), min=self.state_lims[:, 0], max=self.state_lims[:, 1]),
                ])
            )
            self.register_buffer("nonseq_p_std",
                torch.cat([
                    torch.tensor([2., 1., 1., 1.], dtype=torch.float),
                    torch.tensor([1., 1., 1., 2.], dtype=torch.float),
                ])
            )
        else:
            self.register_buffer("nonseq_p_mean",
                torch.cat([
                    unconstrain(torch.tensor([30., 120., 1e-2, 5e-4, 1/30., 1.0], dtype=torch.float), min=self.param_lims[:, 0], max=self.param_lims[:, 1]),
                    unconstrain(torch.tensor([120., 0.1, 0.1, 20.], dtype=torch.float), min=self.state_lims[:, 0], max=self.state_lims[:, 1]),
                ])
            )
            self.register_buffer("nonseq_p_std",
                torch.cat([
                    torch.tensor([2., 1., 1., 1., 1., 1.], dtype=torch.float),
                    torch.tensor([1., 1., 1., 2.], dtype=torch.float),
                ])
            )

    @property
    def seq_p(self): 
        return Normal(self.seq_p_mean, self.seq_p_std)

    @property
    def nonseq_p(self):
        return Normal(self.nonseq_p_mean, self.nonseq_p_std)
        
    def encode_dist(self, cgm, timestamps, meals, demographics):
        cgm_diff = torch.diff(cgm, prepend=torch.zeros_like(cgm[:, 0:1, :]), dim=-2)  # N x T x 1
        meal_embeds = self.meal_embedding(meals)
        demo_embeds = to_seq(self.demographics_embedding(demographics), like=cgm)

        encoder_input = torch.cat([cgm, cgm_diff, timestamps, meal_embeds, demo_embeds], dim=-1)
        output, (_, cells) = self.encoder_lstm(encoder_input)
        cells = cells.transpose(0, 1).flatten(-2)  # concatenated cells from each layer, including both forward and reverse 
        seq_encoding = self.seq_proj(output)  # output consists of the hidden states from the last layer; we map it to N x T x (2 * encoding_size)
        nonseq_encoding = self.non_seq_proj(cells)  # N x T x 2
        
        seq_encoding_mean, _seq_encoding_std = torch.chunk(seq_encoding, 2, dim=-1)
        seq_encoding_std = torch.nn.functional.softplus(_seq_encoding_std).clamp(min=1e-6)
        nonseq_encoding_mean, _nonseq_enccoding_std = torch.chunk(nonseq_encoding, 2, dim=-1)
        nonseq_encoding_std = torch.nn.functional.softplus(_nonseq_enccoding_std).clamp(min=1e-6)

        return (seq_encoding_mean, seq_encoding_std), (nonseq_encoding_mean, nonseq_encoding_std), cells

    def get_all_latents(self, cgm, timestamps, meals, demographics):
        """实验方案 v4.0：返回 6D param + 4D z_init + 16D z_nonseq = 26D 全 latent。"""
        (seq_mean, seq_std), (nonseq_mean, nonseq_std), cells = self.encode_dist(cgm, timestamps, meals, demographics)
        with torch.no_grad():
            nonseq_sample = nonseq_mean
            seq_sample = seq_mean
        z_nonseq_16 = self.nonseq_to_16(cells)
        states, param, init_state, carb_rate = self.decode(seq_sample, nonseq_sample, z_nonseq=z_nonseq_16)
        return param, init_state, z_nonseq_16

    def get_all_latents_for_head(self, cgm, timestamps, meals, demographics):
        """V6 路线B：返回 26D 全 latent，保留梯度供 e2e_head 反传。"""
        (seq_mean, seq_std), (nonseq_mean, nonseq_std), cells = self.encode_dist(cgm, timestamps, meals, demographics)
        seq_sample = seq_mean
        nonseq_sample = nonseq_mean
        z_nonseq_16 = self.nonseq_to_16(cells)
        states, param, init_state, carb_rate = self.decode(seq_sample, nonseq_sample, z_nonseq=z_nonseq_16)
        return param, init_state, z_nonseq_16

    def encode(self, cgm, timestamps, meals, demographics):
        raise NotImplementedError

    def t2d_dynamics(
        self, 
        param: torch.Tensor,
        state: torch.Tensor,
        carb_rate: torch.Tensor,
        z_nonseq: torch.Tensor | None = None,
    ):
        carb_rate = carb_rate * 0.75  # assume bioavailability of 0.75
        tau_m, Gb, sg, si, p2, mi = torch.split(param, 1, dim=-1)
    
        vg = 1.0  # assume 1.0L/kg as constant (not identifiable anyway)
        bw = 87.5 # 87.5 kg bodyweight as constant (roughly the mean in our dataset)
    
        # assume external insulin is 0
        state = torch.clamp(state, min=0.0)  # clamp to avoid negative values
        G, X, G1, G2 = torch.split(state, 1, dim=-1)
    
        # Glucose dynamics; note that some papers use mmol/L instead of mg/dL; we use mg/dL
        dG1 = -G1 / tau_m + carb_rate / (vg * bw)  # carb_rate / vg is d(t); note that we divide by (vg * BW) here to make G1 and G2 have units of mg/dL; vg is in L/kg
        dG2 = -G2 / tau_m + G1 / tau_m
        dG = -X * G - sg * (G - Gb) + G2 / tau_m  # Eq 4; G2/tau_m is Ra(t) in units of mg/dL/min
        Iendo = mi * torch.clamp(G - Gb, min=0.)  # NOTE: basal insulin is Ib = mi  * Gb
        #Iendo = mi * (G - Gb)  # NOTE: basal insulin is Ib = mi  * Gb
        dX = -p2 * X + p2 * si * Iendo
    
        dstate = torch.cat([dG, dX, dG1, dG2], dim=-1)
        if self.use_v8_ode_corr and (z_nonseq is not None):
            corr_in = torch.cat([z_nonseq, state], dim=-1)
            dstate = dstate + self.ode_correction(corr_in)
        return dstate

    def decode(self, seq_encoding, nonseq_encoding, z_nonseq=None):
        """
        1. Convert seq into strictly positive variable, constrained to a bounded region
        2. Constrain nonseq_encoding into a mechanistic parameters and init_state 
        """
        if self.fix_sg_p2:
            learned_size = 4
            param_, init_state_ = nonseq_encoding[..., :learned_size], nonseq_encoding[..., learned_size:self.encoding_size]
            lims_learned = self.param_lims[self.learned_param_ix]
            param_learned = constrain(param_, min=lims_learned[:, 0], max=lims_learned[:, 1])
            init_state = constrain(init_state_, min=self.state_lims[:, 0], max=self.state_lims[:, 1])
            # Build full 6D param: [tau_m, Gb, sg_fix, si, p2_fix, mi]
            param = torch.zeros(param_learned.shape[0], self.param_size, device=param_learned.device, dtype=param_learned.dtype)
            for i, ix in enumerate(self.learned_param_ix):
                param[..., ix] = param_learned[..., i]
            param[..., 2] = self.sg_fixed.expand(param.shape[0])
            param[..., 4] = self.p2_fixed.expand(param.shape[0])
        else:
            param_, init_state_ = nonseq_encoding[..., :self.param_size], nonseq_encoding[..., self.param_size:]
            param = constrain(param_, min=self.param_lims[:, 0], max=self.param_lims[:, 1])
            init_state = constrain(init_state_, min=self.state_lims[:, 0], max=self.state_lims[:, 1])
        carb_rate = constrain(seq_encoding, min=0., max=self.max_carb_per_min)  # convert carbs in g to mg, then to average carb rate in mg/min
        carb_rate = self.decoder_dropout(carb_rate)

        state = init_state
        states = [init_state]
        for i in range(seq_encoding.shape[-2] - 1):
            dstate = self.t2d_dynamics(param, state, carb_rate[:, i], z_nonseq=z_nonseq)
            state = state + self.dt * dstate
            states.append(state)
        states = torch.stack(states, dim=-2)
        return states, param, init_state, carb_rate

    def forward(self, cgm, timestamps, meals, demographics):
        (seq_encoding_mean, seq_encoding_std), (nonseq_encoding_mean, nonseq_encoding_std), cells = self.encode_dist(cgm, timestamps, meals, demographics)
        
        if self.training:
            seq_encoding_sample = torch.randn_like(seq_encoding_mean) * seq_encoding_std + seq_encoding_mean
            nonseq_encoding_sample = torch.randn_like(nonseq_encoding_mean) * nonseq_encoding_std + nonseq_encoding_mean
        else:
            seq_encoding_sample = seq_encoding_mean
            nonseq_encoding_sample = nonseq_encoding_mean
        z_nonseq_16 = self.nonseq_to_16(cells)                  # (N,16)
        # now pass to decoder to get the reconstruction
        states, param, init_state, carb_rate = self.decode(seq_encoding_sample, nonseq_encoding_sample, z_nonseq=z_nonseq_16)
        if self.use_v8_recon_corr:
            cgm_corr = self.correction_mlp(z_nonseq_16)         # (N,1)
            states = states.clone()
            states[..., 0:1] = states[..., 0:1] + cgm_corr[:, None, :]
        output = AutoencoderOutput(states, param, init_state, carb_rate)
        seq_q = Normal(seq_encoding_mean, seq_encoding_std)
        nonseq_q = Normal(nonseq_encoding_mean, nonseq_encoding_std)
        sspg_di_pred = None
        if self.prediction_head is not None:
            head_in = torch.cat([init_state, z_nonseq_16], dim=-1)
            sspg_di_pred = self.prediction_head(head_in)
        return output, seq_q, nonseq_q, sspg_di_pred


class BlackboxAutoencoder(torch.nn.Module):
    def __init__(self, meal_size, demographics_size, embedding_size, hidden_size, num_layers, encoder_dropout_prob, decoder_dropout_prob):
        super().__init__()
        self.meal_size = meal_size
        self.demographics_size = demographics_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        encoding_size = hidden_size
        self.encoding_size = hidden_size

        cgm_size = 1
        cgm_diff_size = 1
        timestamp_size = 1

        self.meal_embedding = NanWrapper(ConvLinear(meal_size, embedding_size, channel_last=True))
        self.demographics_embedding = NanWrapper(nn.Linear(demographics_size, embedding_size))

        self.register_buffer("nonseq_mean", torch.zeros((encoding_size,), dtype=torch.float))
        self.register_buffer("nonseq_std", torch.ones((encoding_size,), dtype=torch.float))

        encoder_input_size = cgm_size + cgm_diff_size + timestamp_size + embedding_size + embedding_size
        decoder_input_size = timestamp_size + embedding_size + embedding_size

        self.encoder_input_size = encoder_input_size
        self.decoder_input_size = decoder_input_size

        self.encoder_lstm = nn.LSTM(input_size=encoder_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=encoder_dropout_prob, bidirectional=True)
        self.non_seq_proj = nn.Linear(in_features=2 * hidden_size * num_layers,  # times 2 for bidirectional
                                      out_features=2 * encoding_size)  # times 2 for mean and std
        self.decoder_model = nn.LSTM(input_size=decoder_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=decoder_dropout_prob, bidirectional=False)
        self.decoder_proj = ConvLinear(in_features=hidden_size, out_features=1, channel_last=True)
    
    def encode_dist(self, cgm, timestamps, meals, demographics):
        cgm_diff = torch.diff(cgm, prepend=torch.zeros_like(cgm[:, 0:1, :]), dim=-2)  # N x T x 1
        meal_embeds = self.meal_embedding(meals)
        demo_embeds = to_seq(self.demographics_embedding(demographics), like=cgm)

        encoder_input = torch.cat([cgm, cgm_diff, timestamps, meal_embeds, demo_embeds], dim=-1)
        _, (_, cells) = self.encoder_lstm(encoder_input)
        cells = cells.transpose(0, 1).flatten(-2)  # concatenated cells from each layer, including both forward and reverse 
        nonseq_encoding = self.non_seq_proj(cells)  # N x T x 2
        
        nonseq_encoding_mean, _nonseq_enccoding_std = torch.chunk(nonseq_encoding, 2, dim=-1)
        nonseq_encoding_std = torch.nn.functional.softplus(_nonseq_enccoding_std)

        return nonseq_encoding_mean, nonseq_encoding_std
    
    def decode(self, nonseq_encoding, timestamps, meals, demographics):
        meal_embeds = self.meal_embedding(meals)
        demo_embeds = to_seq(self.demographics_embedding(demographics), like=timestamps)

        decoder_input = torch.cat([timestamps, meal_embeds, demo_embeds], dim=-1)
        c_0 = nonseq_encoding[None, :, :].contiguous()
        h_0 = torch.zeros_like(c_0).contiguous()
        output, (_, _) = self.decoder_model(decoder_input, (h_0, c_0))
        decoding = self.decoder_proj(output)
        return decoding
    
    def forward(self, cgm, timestamps, meals, demographics):
        # TODO: this is not finished
        (nonseq_encoding_mean, nonseq_encoding_std) = self.encode_dist(cgm, timestamps, meals, demographics)
        
        if self.training:
            nonseq_encoding_sample = torch.randn_like(nonseq_encoding_mean) * nonseq_encoding_std + nonseq_encoding_mean
        else:
            nonseq_encoding_sample = nonseq_encoding_mean
        # now pass to decoder to get the reconstruction
        decoding = self.decode(nonseq_encoding_sample, timestamps, meals, demographics)
        nonseq_q = Normal(nonseq_encoding_mean, nonseq_encoding_std)
        return decoding, nonseq_q

    @property
    def nonseq_p(self):
        return Normal(self.nonseq_mean, self.nonseq_std)


class DirectNN(torch.nn.Module):
    """
    M3 消融：仅 Encoder (LSTM) + 6D param，无 VAE、无 Decoder。
    与 MechanisticAutoencoder 共用 encoder 结构，输出 6D param 供 SSPG/DI/IR 预测头使用。
    """
    def __init__(self, meal_size, demographics_size, embedding_size, hidden_size, num_layers, encoder_dropout_prob):
        super().__init__()
        self.param_size = 6
        self.state_size = 4
        encoding_size = self.param_size + self.state_size  # 10
        # 与 MechanisticAutoencoder 一致，放宽 tau_m/Gb 边界；审计计划 1.1 宽范围
        _si_lo, _si_hi = (1e-5, 1e-2) if P1_WIDE_PARAM_RANGE else (1e-4, 1e-3)
        _mi_lo, _mi_hi = (0.05, 5.0) if P1_WIDE_PARAM_RANGE else (0.1, 3.0)
        self.register_buffer("param_lims",
            torch.tensor([
                [10., 120.], [60., 250.], [5e-3, 2e-2], [_si_lo, _si_hi], [1./60, 1./15.], [_mi_lo, _mi_hi],
            ], dtype=torch.float))
        self.meal_size = meal_size
        self.demographics_size = demographics_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        cgm_size = 1
        cgm_diff_size = 1
        timestamp_size = 1
        self.meal_embedding = NanWrapper(ConvLinear(meal_size, embedding_size, channel_last=True))
        self.demographics_embedding = NanWrapper(nn.Linear(demographics_size, embedding_size))
        encoder_input_size = cgm_size + cgm_diff_size + timestamp_size + embedding_size + embedding_size
        self.encoder_lstm = nn.LSTM(
            input_size=encoder_input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=encoder_dropout_prob, bidirectional=True)
        self.non_seq_proj = nn.Linear(
            in_features=2 * hidden_size * num_layers,
            out_features=2 * encoding_size)

    def encode_dist(self, cgm, timestamps, meals, demographics):
        cgm_diff = torch.diff(cgm, prepend=torch.zeros_like(cgm[:, 0:1, :]), dim=-2)
        meal_embeds = self.meal_embedding(meals)
        demo_embeds = to_seq(self.demographics_embedding(demographics), like=cgm)
        encoder_input = torch.cat([cgm, cgm_diff, timestamps, meal_embeds, demo_embeds], dim=-1)
        output, (_, cells) = self.encoder_lstm(encoder_input)
        cells = cells.transpose(0, 1).flatten(-2)
        nonseq_encoding = self.non_seq_proj(cells)
        nonseq_encoding_mean, _ = torch.chunk(nonseq_encoding, 2, dim=-1)
        return nonseq_encoding_mean

    def forward(self, cgm, timestamps, meals, demographics):
        nonseq_mean = self.encode_dist(cgm, timestamps, meals, demographics)  # (N, 10)
        param_ = nonseq_mean[..., :self.param_size]
        param = constrain(param_, min=self.param_lims[:, 0], max=self.param_lims[:, 1])
        return DirectNNOutput(param=param)


class DirectNNOutput(NamedTuple):
    param: torch.Tensor


class AutoencoderOutput(NamedTuple):
    states: torch.Tensor
    param: torch.Tensor
    init_state: torch.Tensor
    carb_rate: torch.Tensor

# Smoke tests
# %%
#N, T = 11, 13
#meal_size = 3
#demo_size = 4
#embed_size = 5
#hidden_size = 6
#encoding_size = 7
#cgm = torch.randn(N, T, 1)
#timestamps = torch.randn(N, T, 1)
#meals = torch.randn(N, T, meal_size)
#demo = torch.randn(N, demo_size)
#model = MechanisticAutoencoder(meal_size, demo_size, embed_size, hidden_size, 2, encoding_size, 0., 0.)
#output, h_n, c_n, nonseq_enc, seq_encoding = model.encode(cgm, timestamps, meals, demo)
# %%
#N, T = 7, 13
#meal_size = 3
#demo_size = 4
#embed_size = 5
#hidden_size = 6
#cgm = torch.randn(N, T, 1)
#timestamps = torch.randn(N, T, 1)
#meals = torch.randn(N, T, meal_size)
#demo = torch.randn(N, demo_size)
#model = BlackboxAutoencoder(meal_size, demo_size, embed_size, hidden_size, 2, 0, 0.)
#model(cgm, timestamps, meals, demo)