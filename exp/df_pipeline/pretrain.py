# df_pipeline/pretrain.py

import argparse
import random
from pathlib import Path
import os
import sys

sys.path.append(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir,
))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from RFDD import parameters as rf_params

from taskAttack.attackmethods.gradient.fgsm import PGD
from taskAttack.util import signal_energy, signal_db2pow

from exp.df_pipeline.common import (
    build_models,
    normalize,
    estimate_start_steps,
    save_checkpoint,
)


class SilentLogger:
    """
    给原工程 Task / PGD 使用的简易 logger。

    Wrapper.py 的 conduct_fit 会调用：
        critical / info / exception

    PGD 或其他模块可能调用：
        warning / debug / error
    """

    def info(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()

    return np.asarray(value).item()


def get_split(split_data, split_name, selected_snrs):
    """
    从 split_data 中取出指定 split 的：

        signals: [N, 2, L]
        labels : [N]
        snrs   : [N]

    AMC-PGD purifier 需要 labels，因为 PGD 攻击的是 AMC 分类器。
    """
    split_set = split_data[f"{split_name}_set"]

    signals = torch.as_tensor(
        split_set[0]
    ).detach().cpu().float()

    labels = torch.as_tensor(
        split_set[1]
    ).detach().cpu().long()

    indices = split_data[f"{split_name}_idx"]
    all_snrs = split_data["SNRs"]

    aligned_snrs = torch.tensor([
        float(
            scalar_value(
                all_snrs[int(scalar_value(index))]
            )
        )
        for index in indices
    ], dtype=torch.float32)

    if not (
        len(signals) == len(labels) == len(aligned_snrs)
    ):
        raise ValueError(
            f"{split_name} 数据、标签与 SNR 数量不一致。"
        )

    if selected_snrs is not None:
        keep = torch.zeros(
            len(aligned_snrs),
            dtype=torch.bool,
        )

        for snr in selected_snrs:
            keep |= torch.isclose(
                aligned_snrs,
                torch.tensor(float(snr)),
            )

        signals = signals[keep]
        labels = labels[keep]
        aligned_snrs = aligned_snrs[keep]

    if len(signals) == 0:
        raise ValueError(
            f"{split_name} 在指定 SNR 范围内没有样本。"
        )

    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError(
            f"期望信号形状 [N, 2, L]，实际为 {tuple(signals.shape)}"
        )

    if not torch.isfinite(signals).all().item():
        raise ValueError(f"{split_name} 信号含 NaN 或 Inf。")

    return signals, labels, aligned_snrs


def make_loader(
    signals,
    labels,
    snrs,
    batch_size,
    workers,
    shuffle,
):
    return DataLoader(
        TensorDataset(signals, labels, snrs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def diffusion_loss_for_batch(diffusion, x, s):
    """
    根据每个样本自己的起点 s 训练扩散模型。

    对每个样本从 [s + 1, T - 1] 随机采样目标时间步。
    如果 s 已经是最后一个时间步，则该样本跳过。
    """
    last_step = diffusion.num_timesteps - 1
    valid = s < last_step

    valid_count = int(valid.sum().item())

    if valid_count == 0:
        return None, 0

    x_valid = x[valid]
    s_valid = s[valid]

    available = last_step - s_valid

    offsets = (
        torch.rand(
            s_valid.shape,
            device=s_valid.device,
        )
        * available.float()
    ).floor().long() + 1

    target_steps = s_valid + offsets

    loss = diffusion.p_losses(
        x_valid,
        target_steps,
        s_valid,
    ).mean()

    return loss, valid_count


def get_logits(model, x):
    """
    兼容不同分类器 forward 返回格式。

    常见情况：
        logits = model(x)

    如果模型返回 tuple/list，则默认取第一个元素作为 logits。
    """
    output = model(x)

    if isinstance(output, (tuple, list)):
        output = output[0]

    return output


def load_amc_classifier_with_task(
    classifier_name,
    checkpoint_path,
    device,
    gid=0,
):
    """
    使用原工程 Task 机制加载 AMC 分类器。

    注意：
        当前工程的 Wrapper.py 中 conduct_fit(dlogger) 不能接收 None，
        因为内部会调用 dlogger.critical / info / exception。

        此外，conduct_fit() 里无条件访问了一些 SADD 相关参数：
            df_ckpt, s_ckpt, myt, lambda_nmse, lambda_pur,
            nmse_beta, boundary_policy

        所以这里需要给 project_args 补安全默认值。
    """
    from taskDefense.Wrapper import Task
    from taskDefense.Parser import get_parser
    from taskRecog.util import Opt

    old_argv = sys.argv[:]

    try:
        # 给原工程 parser 一个干净 argv，避免解析 pretrain.py 参数时报错。
        sys.argv = [old_argv[0]]
        project_args, project_parser = get_parser()

    finally:
        sys.argv = old_argv

    # ============================================================
    # 基础模型加载参数
    # ============================================================
    project_args.log_level = "i"

    project_args.model = classifier_name
    project_args.target_model = classifier_name
    project_args.surrogate_model = classifier_name

    project_args.cuda = torch.cuda.is_available()
    project_args.gid = gid

    # 这里只是加载已有分类器，不是真的重新训练 SADD。
    project_args.test = True
    project_args.clean = False

    project_args.defense_method = "nature"
    project_args.algo = "clean"

    # 给原工程中可能被访问的攻击字段安全默认值。
    project_args.bound = "psr"
    project_args.psr = -10.0
    project_args.pnr = 0.0
    project_args.epsilon = 0.06
    project_args.alpha = 0.03
    project_args.epoch = 5
    project_args.norm = "l2"

    project_args.snr = torch.tensor(
        list(range(-20, 19, 2)),
        dtype=torch.float32,
    )

    project_args.exp_name = "tmp.pretrain_amc_pgd_loader"

    # ============================================================
    # 关键：给 conduct_fit() 中无条件访问的 SADD 参数补默认值
    # ============================================================
    project_args.data = getattr(
        project_args,
        "data",
        "rml16a",
    )

    project_args.df_ckpt = getattr(
        project_args,
        "df_ckpt",
        "",
    )

    project_args.s_ckpt = getattr(
        project_args,
        "s_ckpt",
        "",
    )

    project_args.myt = getattr(
        project_args,
        "myt",
        9,
    )

    project_args.lambda_nmse = getattr(
        project_args,
        "lambda_nmse",
        0.0,
    )

    project_args.lambda_pur = getattr(
        project_args,
        "lambda_pur",
        0.0,
    )

    project_args.nmse_beta = getattr(
        project_args,
        "nmse_beta",
        1.0,
    )

    project_args.boundary_policy = getattr(
        project_args,
        "boundary_policy",
        "clamp_start",
    )

    project_args.finetune = getattr(
        project_args,
        "finetune",
        False,
    )

    # ============================================================
    # 指定预训练分类器 checkpoint
    # ============================================================
    project_args.hyper = Opt(init=dict(
        pretraining_file=str(Path(checkpoint_path).resolve()),
    ))

    task = Task(project_args, project_parser)

    # 不能传 None。Wrapper.py 里会调用 dlogger.critical / exception。
    silent_logger = SilentLogger()

    task.conduct_fit(
        dlogger=silent_logger,
        xfit_stats=False,
    )

    classifier = task.model
    classifier.to(device)
    classifier.eval()
    classifier.requires_grad_(False)

    return classifier



def make_pgd_attacker(
    model,
    logger,
    args,
):
    """
    构造项目已有 PGD attacker。

    参数默认值与原框架 attacker_opts 对齐：

        PSR = -10
        epsilon = 0.06
        alpha = 0.03
        epoch = 5
        norm = 'l2'

    注意：
        对 norm != 'linfty' 的情况，epsilon / alpha 会在每个 batch
        通过 update_attacker_epsilon() 按 PSR 动态覆盖。
    """
    attacker = PGD(
        model=model,
        logger=logger,
        PSR=float(args.adv_psr),
        epsilon=float(args.adv_epsilon),
        alpha=float(args.adv_alpha),
        epoch=int(args.adv_steps),
        norm=str(args.adv_norm),
    )

    return attacker


def update_attacker_epsilon(
    attacker,
    signals,
    args,
):
    """
    复刻原框架里的 update_psr_epsilon 逻辑。

    原始逻辑来自：

        class PGD_AT(AnnealingTrainer):
            def update_psr_epsilon(self, X_detached):
                if self.attacker_opts.norm != 'linfty':
                    self.attacker.epsilon, self.attacker.alpha = \
                        self.attacker_opts.update_psr_epsilon(X_detached)

        class attacker_opts(Opt):
            def update_psr_epsilon(self, signals):
                if self.norm != 'linfty':
                    if len(signals.size()) == 2:
                        signals = signals.unsqueeze(0)
                    sig_energy = signal_energy(signals).mean()
                    gain = signal_db2pow(self.PSR)
                    epsilon = torch.sqrt(sig_energy * gain)
                    alpha = epsilon / self.epoch

                    self.epsilon = epsilon
                    self.alpha = alpha

                return self.epsilon, self.alpha

    这里保持一致：
        - norm == 'linfty'：不根据 PSR 动态更新；
        - norm != 'linfty'：按 batch 信号能量和 PSR 更新 epsilon；
        - alpha = epsilon / adv_steps。
    """
    norm = str(args.adv_norm).lower()

    if norm == "linfty":
        attacker.epsilon = float(args.adv_epsilon)
        attacker.alpha = float(args.adv_alpha)

        return attacker.epsilon, attacker.alpha

    if len(signals.size()) == 2:
        signals = signals.unsqueeze(0)

    sig_energy = signal_energy(signals).mean()
    gain = signal_db2pow(float(args.adv_psr))

    epsilon = torch.sqrt(sig_energy * gain)
    alpha = epsilon / int(args.adv_steps)

    attacker.epsilon = epsilon
    attacker.alpha = alpha

    return attacker.epsilon, attacker.alpha


def generate_project_pgd_adv(
    classifier,
    attacker,
    raw_x,
    labels,
    args,
    device,
):
    """
    使用项目已有 PGD 类生成 AMC-PGD 对抗样本。

    逻辑对应 SADD.py：

        self.attacker = PGD(
            model=self.model,
            logger=self.logger,
            **self.attacker_opts.dict,
        )

        attack_x = sig_batch.detach().clone().requires_grad_(True)
        attack_y = lab_batch.detach()

        self.model.eval()

        try:
            self.update_psr_epsilon(attack_x)

            with torch.enable_grad():
                delta = self.attacker(attack_x, attack_y)
        finally:
            self.model.train()

        delta = delta.detach().to(device)

        adv_x = sig_batch.detach() + delta
    """
    attack_x = raw_x.detach().clone().requires_grad_(True)
    attack_y = labels.detach()

    was_training = classifier.training
    classifier.eval()

    try:
        update_attacker_epsilon(
            attacker=attacker,
            signals=attack_x,
            args=args,
        )

        with torch.enable_grad():
            delta = attacker(
                attack_x,
                attack_y,
            )

    finally:
        classifier.train(was_training)

    delta = delta.detach().to(device)

    if not torch.isfinite(delta).all().item():
        raise FloatingPointError(
            "PGD 扰动含 NaN/Inf。"
            "请检查攻击步长、功率约束和攻击实现，"
            "不要静默替换成零后继续实验。"
        )

    adv_raw_x = raw_x.detach() + delta

    if not torch.isfinite(adv_raw_x).all().item():
        raise FloatingPointError(
            "PGD 生成的 adv_raw_x 含 NaN 或 Inf。"
        )

    return adv_raw_x.detach()

def get_start_steps(
    x_normalized,
    snr_db,
    alphas_cumprod,
    start_mode,
):
    """
    获取扩散起点标签。

    start_mode:
        dynamic:
            使用原 estimate_start_steps 规则。

        zero:
            所有样本固定为 s=0。
            用于动态起点定位消融实验。
    """
    if start_mode == "dynamic":
        return estimate_start_steps(
            x_normalized,
            snr_db,
            alphas_cumprod,
        )

    if start_mode == "zero":
        return torch.zeros(
            x_normalized.shape[0],
            device=x_normalized.device,
            dtype=torch.long,
        )

    raise ValueError(
        f"Unsupported start_mode={start_mode}. "
        "Expected 'dynamic' or 'zero'."
    )

def run_epoch(
    diffusion,
    mlp,
    loader,
    normalization,
    device,
    df_optimizer=None,
    mlp_optimizer=None,
    adv_classifier=None,
    adv_attacker=None,
    args=None,
):
    """
    训练/验证一个 epoch。

    训练阶段：
        clean:
            raw_x -> normalize -> clean_x
            clean_x -> clean_s
            diffusion 和 MLP 都使用 clean 样本训练

        AMC-PGD adv:
            raw_x 经项目 PGD 生成 adv_raw_x
            adv_raw_x -> normalize -> adv_x
            adv_x -> adv_s
            diffusion 和 MLP 都使用 adv 样本训练

    验证阶段：
        默认只验证 clean 输入。
    """
    training = df_optimizer is not None


    if training != (mlp_optimizer is not None):
        raise ValueError("训练时必须同时提供两个 optimizer。")

    use_adv = (
        training
        and adv_classifier is not None
        and adv_attacker is not None
        and args is not None
        and float(args.adv_weight) > 0.0
    )

    diffusion.train(training)
    mlp.train(training)

    if adv_classifier is not None:
        adv_classifier.eval()

    sums = {
        "df_loss": 0.0,
        "df_clean_loss": 0.0,
        "df_adv_loss": 0.0,

        "mlp_ce": 0.0,
        "mlp_clean_ce": 0.0,
        "mlp_adv_ce": 0.0,

        "s_abs_error": 0.0,
        "s_correct": 0,
        "last_step_count": 0,

        "adv_s_abs_error": 0.0,
        "adv_s_correct": 0,
        "adv_last_step_count": 0,
        "adv_count": 0,

        "amc_clean_correct": 0,
        "amc_adv_correct": 0,
    }

    count = 0
    df_count = 0
    df_clean_count = 0
    df_adv_count = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for raw_x, labels, snr_db in loader:
            raw_x = raw_x.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            snr_db = snr_db.to(
                device,
                non_blocking=True,
            )

            batch_count = len(raw_x)

            # ============================================================
            # 1. clean 样本
            # ============================================================
            clean_x = normalize(
                raw_x,
                normalization,
            )

            clean_s = get_start_steps(
                clean_x,
                snr_db,
                diffusion.alphas_cumprod,
                args.start_mode,
            )


            # ============================================================
            # 2. 项目 PGD 生成 AMC-PGD 对抗样本
            # ============================================================
            if use_adv:
                adv_raw_x = generate_project_pgd_adv(
                    classifier=adv_classifier,
                    attacker=adv_attacker,
                    raw_x=raw_x,
                    labels=labels,
                    args=args,
                    device=device,
                )

                adv_x = normalize(
                    adv_raw_x,
                    normalization,
                )

                adv_s = get_start_steps(
                    adv_x,
                    snr_db,
                    diffusion.alphas_cumprod,
                    args.start_mode,
                )


                with torch.no_grad():
                    clean_logits_amc = get_logits(
                        adv_classifier,
                        raw_x,
                    )

                    adv_logits_amc = get_logits(
                        adv_classifier,
                        adv_raw_x,
                    )

                    sums["amc_clean_correct"] += (
                        clean_logits_amc.argmax(dim=1) == labels
                    ).sum().item()

                    sums["amc_adv_correct"] += (
                        adv_logits_amc.argmax(dim=1) == labels
                    ).sum().item()

            else:
                adv_x = None
                adv_s = None

            # ============================================================
            # 3. diffusion 更新：clean + AMC-PGD adv
            # ============================================================
            if training:
                df_optimizer.zero_grad(set_to_none=True)

            df_clean_loss, clean_valid_count = diffusion_loss_for_batch(
                diffusion,
                clean_x,
                clean_s,
            )

            df_loss = None

            if df_clean_loss is not None:
                if not torch.isfinite(df_clean_loss).item():
                    raise FloatingPointError(
                        "clean 扩散训练损失非有限值。"
                    )

                df_loss = df_clean_loss

            df_adv_loss = None
            adv_valid_count = 0

            if use_adv:
                df_adv_loss, adv_valid_count = diffusion_loss_for_batch(
                    diffusion,
                    adv_x,
                    adv_s,
                )

                if df_adv_loss is not None:
                    if not torch.isfinite(df_adv_loss).item():
                        raise FloatingPointError(
                            "AMC-PGD adv 扩散训练损失非有限值。"
                        )

                    if df_loss is None:
                        df_loss = float(args.adv_weight) * df_adv_loss
                    else:
                        df_loss = (
                            df_loss
                            + float(args.adv_weight) * df_adv_loss
                        )

            if df_loss is not None:
                if not torch.isfinite(df_loss).item():
                    raise FloatingPointError(
                        "扩散总损失非有限值。"
                    )

                if training:
                    df_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        diffusion.parameters(),
                        max_norm=1.0,
                    )

                    df_optimizer.step()

                if df_clean_loss is not None:
                    sums["df_clean_loss"] += (
                        df_clean_loss.detach().item()
                        * clean_valid_count
                    )
                    df_clean_count += clean_valid_count

                if df_adv_loss is not None:
                    sums["df_adv_loss"] += (
                        df_adv_loss.detach().item()
                        * adv_valid_count
                    )
                    df_adv_count += adv_valid_count

                total_valid_count = clean_valid_count + adv_valid_count

                if total_valid_count > 0:
                    sums["df_loss"] += (
                        df_loss.detach().item()
                        * total_valid_count
                    )
                    df_count += total_valid_count

            # ============================================================
            # 4. MLP 更新：clean + AMC-PGD adv
            # ============================================================
            if training:
                mlp_optimizer.zero_grad(set_to_none=True)

            logits_clean = mlp(clean_x)

            mlp_clean_loss = F.cross_entropy(
                logits_clean,
                clean_s,
            )

            if not torch.isfinite(mlp_clean_loss).item():
                raise FloatingPointError(
                    "MLP clean 损失非有限值。"
                )

            mlp_loss = mlp_clean_loss
            logits_adv = None
            mlp_adv_loss = None

            if use_adv:
                logits_adv = mlp(adv_x)

                mlp_adv_loss = F.cross_entropy(
                    logits_adv,
                    adv_s,
                )

                if not torch.isfinite(mlp_adv_loss).item():
                    raise FloatingPointError(
                        "MLP AMC-PGD adv 损失非有限值。"
                    )

                mlp_loss = (
                    mlp_loss
                    + float(args.adv_weight) * mlp_adv_loss
                )

            if not torch.isfinite(mlp_loss).item():
                raise FloatingPointError(
                    "MLP 总损失非有限值。"
                )

            if training:
                mlp_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    mlp.parameters(),
                    max_norm=1.0,
                )

                mlp_optimizer.step()

            # ============================================================
            # 5. 统计 clean MLP
            # ============================================================
            clean_prediction = logits_clean.detach().argmax(dim=1)

            sums["mlp_ce"] += (
                mlp_loss.detach().item()
                * batch_count
            )

            sums["mlp_clean_ce"] += (
                mlp_clean_loss.detach().item()
                * batch_count
            )

            sums["s_abs_error"] += (
                clean_prediction - clean_s
            ).abs().float().sum().item()

            sums["s_correct"] += (
                clean_prediction == clean_s
            ).sum().item()

            sums["last_step_count"] += (
                clean_s == diffusion.num_timesteps - 1
            ).sum().item()

            # ============================================================
            # 6. 统计 adv MLP
            # ============================================================
            if use_adv:
                adv_prediction = logits_adv.detach().argmax(dim=1)

                sums["mlp_adv_ce"] += (
                    mlp_adv_loss.detach().item()
                    * batch_count
                )

                sums["adv_s_abs_error"] += (
                    adv_prediction - adv_s
                ).abs().float().sum().item()

                sums["adv_s_correct"] += (
                    adv_prediction == adv_s
                ).sum().item()

                sums["adv_last_step_count"] += (
                    adv_s == diffusion.num_timesteps - 1
                ).sum().item()

                sums["adv_count"] += batch_count

            count += batch_count

    if df_count == 0:
        raise RuntimeError(
            "没有可用于扩散训练/验证的样本。"
            "请检查功率定义、归一化和起点分布。"
        )

    result = {
        "df_loss": sums["df_loss"] / df_count,

        "df_clean_loss": (
            sums["df_clean_loss"] / df_clean_count
            if df_clean_count > 0
            else 0.0
        ),

        "df_adv_loss": (
            sums["df_adv_loss"] / df_adv_count
            if df_adv_count > 0
            else 0.0
        ),

        "mlp_ce": sums["mlp_ce"] / count,

        "mlp_clean_ce": sums["mlp_clean_ce"] / count,

        "mlp_adv_ce": (
            sums["mlp_adv_ce"] / sums["adv_count"]
            if sums["adv_count"] > 0
            else 0.0
        ),

        "s_mae": sums["s_abs_error"] / count,

        "s_acc": sums["s_correct"] / count,

        "last_step_fraction": (
            sums["last_step_count"] / count
        ),

        "adv_s_mae": (
            sums["adv_s_abs_error"] / sums["adv_count"]
            if sums["adv_count"] > 0
            else 0.0
        ),

        "adv_s_acc": (
            sums["adv_s_correct"] / sums["adv_count"]
            if sums["adv_count"] > 0
            else 0.0
        ),

        "adv_last_step_fraction": (
            sums["adv_last_step_count"] / sums["adv_count"]
            if sums["adv_count"] > 0
            else 0.0
        ),

        "amc_clean_acc": (
            sums["amc_clean_correct"] / sums["adv_count"]
            if sums["adv_count"] > 0
            else 0.0
        ),

        "amc_adv_acc": (
            sums["amc_adv_correct"] / sums["adv_count"]
            if sums["adv_count"] > 0
            else 0.0
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
    )

    parser.add_argument(
        "--out",
        default="datasets/models/a_joint",
    )

    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--df-lr",
        type=float,
        default=float(rf_params.lr),
    )

    parser.add_argument(
        "--mlp-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(rf_params.total_step),
    )
    parser.add_argument(
        "--start-mode",
        choices=["dynamic", "zero"],
        default="dynamic",
        help=(
            "扩散起点模式。"
            "dynamic 表示使用 estimate_start_steps / MLP 动态定位；"
            "zero 表示所有样本固定 s=0，用于消融实验。"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=list(range(-20, 19, 2)),
    )

    # ============================================================
    # AMC-PGD purifier 参数
    # ============================================================
    parser.add_argument(
        "--adv-weight",
        type=float,
        default=1.0,
        help=(
            "AMC-PGD 对抗样本损失权重。"
            "1.0 表示 clean + adv 等权训练；"
            "0.0 表示 clean-only 预训练。"
        ),
    )

    parser.add_argument(
        "--adv-classifier",
        default=None,
        help=(
            "用于生成 PGD 对抗样本的 AMC 分类器。"
            "例如 awn、ctdnn、cldnn、msmc。"
            "adv-weight > 0 时必须提供。"
        ),
    )

    parser.add_argument(
        "--adv-classifier-ckpt",
        default=None,
        help=(
            "用于生成 PGD 对抗样本的 AMC 分类器 checkpoint。"
            "adv-weight > 0 时必须提供。"
        ),
    )

    parser.add_argument(
        "--adv-psr",
        type=float,
        default=-10.0,
        help=(
            "PGD PSR，默认与原框架 attacker_opts.PSR=-10 一致。"
            "norm != linfty 时用于动态计算 epsilon。"
        ),
    )

    parser.add_argument(
        "--adv-epsilon",
        type=float,
        default=0.06,
        help=(
            "PGD 初始 epsilon，默认与原框架 attacker_opts.epsilon=0.06 一致。"
            "norm != linfty 时会被 PSR 动态更新覆盖。"
        ),
    )

    parser.add_argument(
        "--adv-alpha",
        type=float,
        default=0.03,
        help=(
            "PGD 初始 alpha，默认与原框架 attacker_opts.alpha=0.03 一致。"
            "norm != linfty 时会被 epsilon / epoch 覆盖。"
        ),
    )

    parser.add_argument(
        "--adv-steps",
        type=int,
        default=5,
        help=(
            "PGD 迭代步数，对应原框架 attacker_opts.epoch=5。"
        ),
    )

    parser.add_argument(
        "--adv-norm",
        default="l2",
        help=(
            "PGD norm，默认与原框架 attacker_opts.norm='l2' 一致。"
        ),
    )

    parser.add_argument(
        "--gid",
        type=int,
        default=0,
        help=(
            "加载 AMC 分类器时传给原工程 Task 的 GPU id。"
            "通常和 --device cuda:x 保持一致。"
        ),
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs 至少为 1。")

    if args.num_steps <= 9:
        raise ValueError("为了支持 t=1..9，num_steps 必须大于 9。")

    if args.adv_weight < 0:
        raise ValueError("adv-weight 不能小于 0。")

    if args.adv_weight > 0:
        if args.adv_classifier is None:
            raise ValueError(
                "adv-weight > 0 时必须提供 --adv-classifier。"
            )

        if args.adv_classifier_ckpt is None:
            raise ValueError(
                "adv-weight > 0 时必须提供 --adv-classifier-ckpt。"
            )

        if not Path(args.adv_classifier_ckpt).is_file():
            raise FileNotFoundError(
                f"adv-classifier-ckpt 不存在：{args.adv_classifier_ckpt}"
            )

    if args.adv_steps < 1:
        raise ValueError("adv-steps 至少为 1。")

    if args.adv_epsilon < 0:
        raise ValueError("adv-epsilon 不能小于 0。")

    if args.adv_alpha < 0:
        raise ValueError("adv-alpha 不能小于 0。")

    seed_everything(args.seed)

    device = torch.device(args.device)

    split_data = torch.load(
        args.data,
        map_location="cpu",
        weights_only=False,
    )

    train_x, train_y, train_snr = get_split(
        split_data,
        "train",
        args.snrs,
    )

    val_x, val_y, val_snr = get_split(
        split_data,
        "val",
        args.snrs,
    )

    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError("训练集和验证集信号形状不一致。")

    # 仅从训练数据计算归一化统计，不读取测试集极值。
    normalization = {
        "data_min": train_x.min().item(),
        "data_max": train_x.max().item(),
    }

    if normalization["data_max"] <= normalization["data_min"]:
        raise ValueError("训练集归一化范围无效。")

    config = {
        "unet_dim": 64,
        "dim_mults": [1, 2, 4, 8],
        "cond_drop_prob": float(rf_params.cond_drop_prob),
        "channels": int(train_x.shape[1]),
        "seq_length": int(train_x.shape[2]),
        "num_steps": args.num_steps,
        "objective": rf_params.objective_target,
        "mlp_hidden_dims": [512, 256, 128, 64],
    }

    diffusion, mlp = build_models(
        config,
        device,
    )

    train_loader = make_loader(
        train_x,
        train_y,
        train_snr,
        args.batch_size,
        args.workers,
        shuffle=True,
    )

    val_loader = make_loader(
        val_x,
        val_y,
        val_snr,
        args.batch_size,
        args.workers,
        shuffle=False,
    )

    logger = SilentLogger()

    # ============================================================
    # 加载 AMC 分类器并构造项目 PGD attacker
    # ============================================================
    if args.adv_weight > 0:
        print(
            f"加载 AMC 分类器用于项目 PGD："
            f"model={args.adv_classifier}, "
            f"ckpt={args.adv_classifier_ckpt}"
        )

        adv_classifier = load_amc_classifier_with_task(
            classifier_name=args.adv_classifier,
            checkpoint_path=args.adv_classifier_ckpt,
            device=device,
            gid=args.gid,
        )

        adv_attacker = make_pgd_attacker(
            model=adv_classifier,
            logger=logger,
            args=args,
        )

        print(
            "AMC 分类器和 PGD attacker 加载完成。"
            f" PSR={args.adv_psr}, "
            f"epsilon={args.adv_epsilon}, "
            f"alpha={args.adv_alpha}, "
            f"epoch={args.adv_steps}, "
            f"norm={args.adv_norm}"
        )

    else:
        adv_classifier = None
        adv_attacker = None
        print("adv-weight=0，当前为 clean-only diffusion + MLP 预训练。")

    df_optimizer = torch.optim.Adam(
        diffusion.parameters(),
        lr=args.df_lr,
    )

    mlp_optimizer = torch.optim.Adam(
        mlp.parameters(),
        lr=args.mlp_lr,
        weight_decay=1e-5,
    )

    mlp_scheduler = torch.optim.lr_scheduler.StepLR(
        mlp_optimizer,
        step_size=10,
        gamma=0.1,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_df = float("inf")
    best_mlp = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            diffusion=diffusion,
            mlp=mlp,
            loader=train_loader,
            normalization=normalization,
            device=device,
            df_optimizer=df_optimizer,
            mlp_optimizer=mlp_optimizer,
            adv_classifier=adv_classifier,
            adv_attacker=adv_attacker,
            args=args,
        )

        # 验证阶段默认只验证 clean 输入。
        val_metrics = run_epoch(
            diffusion=diffusion,
            mlp=mlp,
            loader=val_loader,
            normalization=normalization,
            device=device,
            df_optimizer=None,
            mlp_optimizer=None,
            adv_classifier=None,
            adv_attacker=None,
            args=args,
        )

        mlp_scheduler.step()

        common = {
            "format_version": 2,
            "config": config,
            "normalization": normalization,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "seed": args.seed,
            "selected_snrs": args.snrs,
            "start_mode": args.start_mode,
            "adv_training": {
                "enabled": args.adv_weight > 0,
                "adv_weight": args.adv_weight,
                "attack_target": "AMC_classifier",
                "adv_classifier": args.adv_classifier,
                "adv_classifier_ckpt": (
                    None
                    if args.adv_classifier_ckpt is None
                    else str(Path(args.adv_classifier_ckpt).resolve())
                ),
                "attack_implementation": (
                    "taskAttack.attackmethods.gradient.fgsm.PGD"
                ),
                "PSR": args.adv_psr,
                "epsilon_init": args.adv_epsilon,
                "alpha_init": args.adv_alpha,
                "epoch": args.adv_steps,
                "norm": args.adv_norm,
                "epsilon_update": (
                    "if norm != linfty: "
                    "epsilon=sqrt(mean(signal_energy(x))*signal_db2pow(PSR)); "
                    "alpha=epsilon/epoch"
                ),
                "label_for_pgd": "clean_sample_modulation_label",
                "adv_used_for": [
                    "diffusion",
                    "mlp",
                ],
                "start_label_for_adv": (
                    "recomputed_by_estimate_start_steps_on_adv_x"
                ),
            },
        }

        if val_metrics["df_loss"] < best_df:
            best_df = val_metrics["df_loss"]

            save_checkpoint(
                out / "diffusion_best.pt",
                {
                    **common,
                    "state_dict": diffusion.state_dict(),
                },
            )

        if val_metrics["mlp_ce"] < best_mlp:
            best_mlp = val_metrics["mlp_ce"]

            save_checkpoint(
                out / "mlp_best.pt",
                {
                    **common,
                    "state_dict": mlp.state_dict(),
                    "alphas_cumprod": (
                        diffusion.alphas_cumprod.detach().cpu()
                    ),
                },
            )

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_df={train_metrics['df_loss']:.6f} "
            f"train_df_clean={train_metrics['df_clean_loss']:.6f} "
            f"train_df_adv={train_metrics['df_adv_loss']:.6f} "
            f"val_df={val_metrics['df_loss']:.6f} "
            f"train_mlp={train_metrics['mlp_ce']:.6f} "
            f"train_mlp_clean={train_metrics['mlp_clean_ce']:.6f} "
            f"train_mlp_adv={train_metrics['mlp_adv_ce']:.6f} "
            f"val_mlp={val_metrics['mlp_ce']:.6f} "
            f"val_s_MAE={val_metrics['s_mae']:.3f} "
            f"val_s_acc={val_metrics['s_acc']:.3%} "
            f"train_adv_s_MAE={train_metrics['adv_s_mae']:.3f} "
            f"train_adv_s_acc={train_metrics['adv_s_acc']:.3%} "
            f"amc_clean_acc={train_metrics['amc_clean_acc']:.3%} "
            f"amc_adv_acc={train_metrics['amc_adv_acc']:.3%} "
            f"last_s_ratio={val_metrics['last_step_fraction']:.3%}"
        )

    print(f"扩散 checkpoint: {out / 'diffusion_best.pt'}")
    print(f"MLP checkpoint:  {out / 'mlp_best.pt'}")
    print(f"当前扩散起点模式：{args.start_mode}")


    if args.adv_weight > 0:
        print(
            "当前预训练类型：AMC-PGD purifier。"
            f"PGD 来源模型：{args.adv_classifier}。"
            "PGD 实现：taskAttack.attackmethods.gradient.fgsm.PGD。"
        )
    else:
        print("当前预训练类型：clean-only purifier。")


if __name__ == "__main__":
    main()
