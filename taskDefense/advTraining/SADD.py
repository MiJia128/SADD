import torch

from taskDefense.advTraining.PIAT import PIAT, NMSE_loss
from taskAttack.attackmethods.gradient.fgsm import PGD
from taskRecog.util import logit_acc

from exp.df_pipeline.common import FrozenPurifier


class SADD_Trainer(PIAT):
    def __init__(self, hyper, logger):
        super().__init__(hyper, logger)

        self.lambda_nmse = float(
            getattr(hyper, "lambda_nmse", 1.0)
        )
        self.lambda_pur = float(
            getattr(hyper, "lambda_pur", 1.0)
        )
        self.nmse_beta = float(
            getattr(hyper, "nmse_beta", 4.0)
        )
        self.start_mode = getattr(
            hyper,
            "start_mode",
            "zero",
        )
        
        # False 保留原来 NMSE 对自然分支也传播梯度的语义。
        self.detach_nat_target = bool(
            getattr(hyper, "detach_nat_target", False)
        )
        print(self.start_mode)
        self.purifier = FrozenPurifier.from_checkpoints(
            diffusion_path=hyper.df_ckpt,
            mlp_path=hyper.s_ckpt,
            device=hyper.device,
            boundary_policy=getattr(
                hyper, "boundary_policy", "clamp_start"
            ),
            start_mode=self.start_mode,
        )


        # 保留这些成员名，便于兼容已有调用。
        self.diffusion = self.purifier.diffusion
        self.model_s = self.purifier.mlp

        t = int(self.hyper.t)

        if not 0 <= t < self.diffusion.num_timesteps:
            raise ValueError(f"非法 t={t}")

    @torch.no_grad()
    def purify_data(self, x_raw, t, snr_batch=None):
        """
        新接口：输入原始量纲信号，内部完成归一化和反归一化。
        snr_batch 仅为兼容旧调用保留，不使用真实 SNR。
        """
        return self.purifier(x_raw, int(t))

    def _metric_dict(self):
        return {
            "train_loss_adv": self.train_loss.avg,
            "train_acc_adv": self.train_adv_acc.avg,
            "train_acc_nat": self.train_nat_acc.avg,
        }

    def run_train_batch(self, data_batch, warmup=False):
        device = self.hyper.device

        sig_batch = data_batch[0].to(
            device, non_blocking=True
        )
        lab_batch = data_batch[1].to(
            device, non_blocking=True
        )

        # 这里不读取 data_batch[2] 中的 SNR。
        # 对抗训练阶段的净化完全依赖 MLP 预测起点。
        self.purifier.eval()

        if warmup:
            self.model.train()

            nat_logits, nat_loss = self.sig_logits_loss(
                sig_batch, lab_batch
            )

            self.optimizer.zero_grad(set_to_none=True)
            nat_loss.backward()
            self.optimizer.step()

            self.train_nat_acc.update(
                logit_acc(nat_logits, lab_batch)
            )
            self.train_loss.update(nat_loss.item())

            return self._metric_dict()

        # ---------- 1. 在当前裸分类器上生成 PGD 对抗样本 ----------
        # 保留你原来的攻击对象、攻击约束和 epsilon 更新逻辑。
        self.attacker = PGD(
            model=self.model,
            logger=self.logger,
            **self.attacker_opts.dict,
        )

        attack_x = (
            sig_batch.detach().clone().requires_grad_(True)
        )
        attack_y = lab_batch.detach()

        self.model.eval()

        try:
            self.update_psr_epsilon(attack_x)

            with torch.enable_grad():
                delta = self.attacker(attack_x, attack_y)
        finally:
            self.model.train()

        delta = delta.detach().to(device)

        if not torch.isfinite(delta).all().item():
            raise FloatingPointError(
                "PGD 扰动含 NaN/Inf。"
                "请检查攻击步长、功率约束和攻击实现，"
                "不要静默替换成零后继续实验。"
            )

        adv_x = sig_batch.detach() + delta

        # ---------- 2. 使用冻结的净化器 ----------
        t = int(self.hyper.t)

        with torch.no_grad():
            purified_x, purification_info = self.purifier(
                adv_x,
                t=t,
                return_info=True,
            )

        purified_x = purified_x.detach()

        # ---------- 3. 分类器损失 ----------
        # 清除 PGD 过程可能留下的分类器参数梯度。
        self.optimizer.zero_grad(set_to_none=True)

        nat_logits, nat_loss = self.sig_logits_loss(
            sig_batch, lab_batch
        )

        purified_logits, loss2 = self.sig_logits_loss(
            purified_x, lab_batch
        )

        nat_reference = (
            nat_logits.detach()
            if self.detach_nat_target
            else nat_logits
        )

        # 正则项改为使用“净化后”的分类输出。
        loss1 = NMSE_loss(
            purified_logits,
            nat_reference,
            lab_batch,
            beta=self.nmse_beta,
        )

        total_loss = (
            nat_loss
            + self.lambda_nmse * loss1
            + self.lambda_pur * loss2
        )

        if not torch.isfinite(total_loss).item():
            raise FloatingPointError("分类器训练损失含 NaN/Inf。")

        total_loss.backward()
        self.optimizer.step()

        # ---------- 4. 统计 ----------
        nat_acc = logit_acc(
            nat_logits.detach(), lab_batch
        )
        purified_adv_acc = logit_acc(
            purified_logits.detach(), lab_batch
        )

        self.train_nat_acc.update(nat_acc)
        self.train_adv_acc.update(purified_adv_acc)
        self.train_loss.update(total_loss.item())

        metrics = self._metric_dict()
        metrics.update({
            "batch_loss_nat": nat_loss.detach().item(),
            "batch_loss_nmse": loss1.detach().item(),
            "batch_loss_pur": loss2.detach().item(),
            "batch_s_mean": (
                purification_info["predicted_s"]
                .float().mean().item()
            ),
            "batch_s_clamped_fraction": (
                purification_info["clamped_fraction"].item()
            ),
            "purify_t": t,
        })

        return metrics
