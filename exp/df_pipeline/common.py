# df_pipeline/common.py
from pathlib import Path

import torch
from torch import nn

from RFDD.denoising_diffusion_1d import (
    UnetCon1D,
    GaussianDiffusionCon1D,
)
from taskAttack.util import signal_power


class StartMLP(nn.Module):
    """forward 返回 logits；predict_s 返回离散起点。"""

    def __init__(
        self,
        input_dim,
        num_steps,
        hidden_dims=(512, 256, 128, 64),
    ):
        super().__init__()

        layers = []
        previous = input_dim

        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(previous, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            previous = hidden

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous, num_steps)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.classifier(self.features(x))

    @torch.no_grad()
    def predict_s(self, x):
        return self(x).argmax(dim=1)


def build_models(config, device):
    unet = UnetCon1D(
        dim=config["unet_dim"],
        dim_mults=tuple(config["dim_mults"]),
        cond_drop_prob=config["cond_drop_prob"],
        channels=config["channels"],
        sinusoidal_pos_emb_theta=10000,
    )

    diffusion = GaussianDiffusionCon1D(
        unet,
        seq_length=config["seq_length"],
        timesteps=config["num_steps"],
        objective=config["objective"],
        auto_normalize=False,
    ).to(device)

    mlp = StartMLP(
        input_dim=config["channels"] * config["seq_length"],
        num_steps=config["num_steps"],
        hidden_dims=tuple(config["mlp_hidden_dims"]),
    ).to(device)

    return diffusion, mlp


def normalize(x, stats):
    data_min = x.new_tensor(stats["data_min"])
    data_max = x.new_tensor(stats["data_max"])
    span = data_max - data_min

    if span.item() <= 0:
        raise ValueError("归一化范围必须大于 0。")

    return 2.0 * (x - (data_min + data_max) / 2.0) / span


def denormalize(x, stats):
    data_min = x.new_tensor(stats["data_min"])
    data_max = x.new_tensor(stats["data_max"])
    return (
        x * ((data_max - data_min) / 2.0)
        + (data_min + data_max) / 2.0
    )


@torch.no_grad()
def estimate_start_steps(x_normalized, snr_db, alphas_cumprod):
    """
    保留原代码的起点估计规则。

    要求 signal_power 对每个样本返回一个标量。
    """
    batch = x_normalized.shape[0]
    power = signal_power(x_normalized).reshape(-1)

    if power.numel() != batch:
        raise ValueError(
            "signal_power 必须为每个样本返回一个功率值；"
            "请检查 I/Q 功率定义和返回形状。"
        )

    snr_db = torch.as_tensor(
        snr_db,
        device=x_normalized.device,
        dtype=x_normalized.dtype,
    ).reshape(-1)

    if snr_db.numel() == 1:
        snr_db = snr_db.expand(batch)

    if snr_db.numel() != batch:
        raise ValueError("SNR 数量与 batch 大小不一致。")

    snr_linear = torch.pow(10.0, snr_db / 10.0)
    noise_power = power / (snr_linear + 1.0)

    diffusion_noise = (
        1.0 - alphas_cumprod.to(
            device=x_normalized.device,
            dtype=x_normalized.dtype,
        )
    )

    distance = (
        noise_power[:, None] - diffusion_noise[None, :]
    ).abs()

    return distance.argmin(dim=1)


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def save_checkpoint(path, payload):
    """使用临时文件，降低写入中断导致 checkpoint 损坏的风险。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class FrozenPurifier(nn.Module):
    """
    输入、输出均为原始量纲信号。

    start_mode:
        dynamic:
            使用 MLP 预测每个样本的起始扩散位置 s。

        zero:
            所有样本统一使用 s=0。
            用于动态起点定位方法的消融实验。
    """

    def __init__(
        self,
        diffusion,
        mlp,
        normalization,
        boundary_policy="clamp_start",
        start_mode="dynamic",
    ):
        super().__init__()

        self.diffusion = freeze(diffusion)
        self.mlp = freeze(mlp)

        self.normalization = normalization
        self.stats = normalization
        self.boundary_policy = boundary_policy

        if start_mode not in ["dynamic", "zero"]:
            raise ValueError(
                f"Unsupported start_mode={start_mode}. "
                "Expected 'dynamic' or 'zero'."
            )

        self.start_mode = start_mode


    @classmethod
    def from_checkpoints(
        cls,
        diffusion_path,
        mlp_path,
        device,
        boundary_policy="clamp_start",
        start_mode="dynamic",
    ):
        # 仅加载你自己生成、可信的 checkpoint。
        df_ckpt = torch.load(
            diffusion_path, map_location="cpu", weights_only=False
        )
        mlp_ckpt = torch.load(
            mlp_path, map_location="cpu", weights_only=False
        )

        if df_ckpt["config"] != mlp_ckpt["config"]:
            raise ValueError("扩散模型和 MLP 的配置不一致。")

        if df_ckpt["normalization"] != mlp_ckpt["normalization"]:
            raise ValueError("扩散模型和 MLP 的归一化统计不一致。")

        diffusion, mlp = build_models(df_ckpt["config"], device)

        # 保存完整 diffusion state，包括模型权重和调度 buffer。
        diffusion.load_state_dict(df_ckpt["state_dict"], strict=True)
        mlp.load_state_dict(mlp_ckpt["state_dict"], strict=True)

        saved_schedule = mlp_ckpt["alphas_cumprod"].cpu()
        loaded_schedule = diffusion.alphas_cumprod.detach().cpu()

        if (
            saved_schedule.shape != loaded_schedule.shape
            or not torch.allclose(
                saved_schedule, loaded_schedule, atol=1e-7, rtol=1e-6
            )
        ):
            raise ValueError("MLP 标签所用调度与扩散模型调度不一致。")

        return cls(
            diffusion,
            mlp,
            df_ckpt["normalization"],
            boundary_policy=boundary_policy,
            start_mode=start_mode,
        ).to(device).eval()

    @torch.no_grad()
    def forward(self, x_raw, t, return_info=False):
        """
        扩散净化前向过程。

        输入:
            x_raw:
                原始量纲信号，形状通常为 [B, 2, L]

            t:
                净化步数。实际流程为：
                    used_s -> used_s + t 正向加噪
                    used_s + t -> used_s 反向去噪

            return_info:
                是否返回起点信息。

        start_mode:
            dynamic:
                使用 MLP 对每个样本动态预测起点 predicted_s。

            zero:
                所有样本统一使用 predicted_s = 0。
                用于动态起点定位方法的消融实验。
        """
        self.eval()

        t = int(t)
        total_steps = self.diffusion.num_timesteps

        if not 0 <= t < total_steps:
            raise ValueError(
                f"t 必须满足 0 <= t < {total_steps}，当前为 {t}"
            )

        x_normalized = normalize(x_raw, self.stats)

        # ============================================================
        # 1. 起点选择：dynamic 或 zero
        # ============================================================
        start_mode = getattr(self, "start_mode", "dynamic")

        if start_mode == "dynamic":
            predicted_s = self.mlp.predict_s(x_normalized).long()

        elif start_mode == "zero":
            predicted_s = torch.zeros(
                x_normalized.shape[0],
                device=x_normalized.device,
                dtype=torch.long,
            )

        else:
            raise ValueError(
                f"未知 start_mode: {start_mode}。"
                "可选值为 'dynamic' 或 'zero'。"
            )

        # ============================================================
        # 2. 边界处理：保证 used_s + t 不越界
        # ============================================================
        # 采用原代码的 0-based 时间索引。
        max_start = total_steps - 1 - t
        out_of_range = predicted_s > max_start

        if self.boundary_policy == "error":
            if out_of_range.any().item():
                raise ValueError(
                    "出现 s+t 越界。请检查起点分布，"
                    "或显式选择 clamp_start 边界策略。"
                )
            used_s = predicted_s

        elif self.boundary_policy == "clamp_start":
            # 保证每个样本都执行恰好 t 次反向采样。
            # 注意：这会改变越界样本的预测起点。
            used_s = predicted_s.clamp(min=0, max=max_start)

        else:
            raise ValueError(
                f"未知 boundary_policy: {self.boundary_policy}"
            )

        # ============================================================
        # 3. 扩散净化
        # ============================================================
        if t == 0:
            output = x_raw.clone()

        else:
            end_steps = used_s + t

            # 必须是“从 used_s 到 end_steps”的条件正向加噪。
            x = self.diffusion.q_sample(
                x_normalized,
                end_steps,
                used_s,
            )

            # end_steps -> ... -> used_s，共 t 次。
            for offset in range(t, 0, -1):
                current_steps = used_s + offset

                mean, _, log_variance, _ = (
                    self.diffusion.p_mean_variance(
                        x=x,
                        t=current_steps,
                        cond_scale=1.0,
                        rescaled_phi=0.0,
                    )
                )

                # 是否加噪由当前绝对时间步决定，
                # 不是由超参数 t 是否大于 0 决定。
                noise_mask = (current_steps > 0).to(x.dtype)
                noise_mask = noise_mask.reshape(
                    x.shape[0],
                    *([1] * (x.ndim - 1)),
                )

                x = (
                    mean
                    + noise_mask
                    * torch.exp(0.5 * log_variance)
                    * torch.randn_like(x)
                )

            # 必须对净化结果反归一化。
            output = denormalize(x, self.stats)

        # ============================================================
        # 4. 数值检查与调试信息
        # ============================================================
        if not torch.isfinite(output).all().item():
            raise FloatingPointError("净化结果含 NaN 或 Inf。")

        if return_info:
            return output, {
                "start_mode": start_mode,
                "predicted_s": predicted_s,
                "used_s": used_s,
                "clamped_fraction": out_of_range.float().mean(),
            }

        return output

