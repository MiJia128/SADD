# exp/df_pipeline/evaluate.py

import argparse
import copy
import json
import random
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(
    os.path.dirname(__file__),
    os.path.pardir,
    os.path.pardir,
))

import numpy as np
import torch
from tqdm.auto import tqdm

from taskDefense.Wrapper import Task
from taskDefense.Parser import get_parser
from taskRecog.util import Opt

from exp.df_pipeline.common import FrozenPurifier


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


def parse_eval_args():
    """
    --eval-* 参数由本程序处理。
    其他参数交给项目原来的 get_parser() 处理。

    当前统一流程：

        clean:
            Task.conduct(eval=False)
            -> CLEAN pseudo attack 生成零扰动
            -> 扩散净化
            -> 分类器评估

        pgd / mi / fci / sfaa:
            Task.conduct(eval=False)
            -> 对抗攻击样本
            -> 扩散净化
            -> 分类器评估
    """
    parser = argparse.ArgumentParser(
        description="攻击样本生成 + MLP 定位 + 扩散净化 + 分类器评估",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--eval-start-mode",
        choices=["dynamic", "zero"],
        default="dynamic",
        help=(
            "评估时 purifier 的起点模式。"
            "dynamic 表示使用 MLP 动态预测起点；"
            "zero 表示所有样本固定 s=0。"
        ),
    )

    parser.add_argument(
        "--eval-classifier-ckpt",
        required=True,
        help="当前待评估分类器的 checkpoint。",
    )

    parser.add_argument(
        "--eval-df-ckpt",
        required=True,
        help="共享扩散模型 checkpoint。",
    )

    parser.add_argument(
        "--eval-s-ckpt",
        required=True,
        help="共享 MLP checkpoint。",
    )

    parser.add_argument(
        "--eval-split-file",
        default="data/postdata/RML2016.10a_dict.split.pt",
        help="包含 train/val/test 划分的数据文件。",
    )

    parser.add_argument(
        "--eval-t",
        type=int,
        required=True,
        help="当前分类器对应的净化步数。",
    )

    parser.add_argument(
        "--eval-classifier",
        required=True,
        help="分类器架构，例如 awn、msmc、cldnn、ctdnn。",
    )

    parser.add_argument(
        "--eval-mode",
        choices=[
            "clean",
            "pgd",
            "fci",
            "mi",
            "sfaa",
        ],
        default="clean",
        help=(
            "评估模式。clean 表示 CLEAN pseudo attack，即零扰动；"
            "其他值表示生成对应攻击样本。"
        ),
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="扩散净化使用的 batch size。",
    )

    parser.add_argument(
        "--eval-snrs",
        type=float,
        nargs="+",
        default=list(range(-20, 19, 2)),
        help="需要评估的 SNR 列表。",
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=2022,
    )

    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=1,
        help="相同测试输入重复进行随机净化和评估的次数。",
    )

    parser.add_argument(
        "--eval-out",
        required=True,
        help="本次评估的独立输出目录。",
    )

    parser.add_argument(
        "--eval-boundary-policy",
        choices=["error", "clamp_start"],
        default="clamp_start",
        help="MLP 预测起点 s 导致 s+t 越界时的处理策略。",
    )

    parser.add_argument(
        "--eval-attack-bound",
        choices=["psr", "pnr"],
        default="psr",
        help="攻击强度约束类型，沿用原项目定义。",
    )

    parser.add_argument(
        "--eval-psr",
        type=float,
        default=-10.0,
        help="攻击模式使用的 PSR，沿用原项目定义。",
    )

    parser.add_argument(
        "--eval-pnr",
        type=float,
        default=0.0,
        help="攻击模式使用的 PNR，沿用原项目定义。",
    )

    parser.add_argument(
        "--eval-save-signals",
        action="store_true",
        help="保存送给 task.evaluation 的净化后数据字典。",
    )

    parser.add_argument(
        "--eval-baseline",
        action="store_true",
        help=(
            "额外评估未经扩散净化时的准确率。"
            "clean 模式下是原始信号准确率；攻击模式下是攻击样本准确率。"
        ),
    )

    eval_args, remaining = parser.parse_known_args()

    # 关键：
    # 原工程 get_parser() 会解析 sys.argv。
    # 因此这里必须把 --eval-* 参数从 sys.argv 中永久过滤掉，
    # 否则 Task 或原工程内部再次解析命令行时会报 unrecognized arguments。
    sys.argv = [sys.argv[0], *remaining]

    project_args, project_parser = get_parser()

    return eval_args, project_args, project_parser


def validate_files(eval_args):
    for name in [
        "eval_classifier_ckpt",
        "eval_df_ckpt",
        "eval_s_ckpt",
        "eval_split_file",
    ]:
        path = Path(getattr(eval_args, name))

        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    if eval_args.eval_batch_size < 1:
        raise ValueError("eval-batch-size 必须大于 0。")

    if eval_args.eval_repeats < 1:
        raise ValueError("eval-repeats 必须大于 0。")

    if eval_args.eval_t < 0:
        raise ValueError("eval-t 不能小于 0。")


def build_task(eval_args, args, parser, out_dir):
    """
    沿用原工程 Task 初始化方式。

    当前 clean 也作为一种 pseudo attack：

        args.algo = "clean"

    因此 clean / pgd / mi / fci / sfaa 都统一走 Task.conduct()。
    """
    args.log_level = "i"

    args.model = eval_args.eval_classifier
    args.target_model = eval_args.eval_classifier
    args.surrogate_model = eval_args.eval_classifier

    args.df_ckpt = eval_args.eval_df_ckpt
    args.s_ckpt = eval_args.eval_s_ckpt
    args.myt = eval_args.eval_t

    args.lambda_nmse = 1.0
    args.lambda_pur = 1.0
    args.nmse_beta = 4.0
    args.boundary_policy = eval_args.eval_boundary_policy
    args.start_mode = eval_args.eval_start_mode
    args.test = True

    # 重要：
    # 不能设为 True。
    # Wrapper.exp_config() 里有：
    #
    #     if args.test and args.clean:
    #         os_rmdirs(self.model_fit_dir)
    #
    # 会删除 model_fit_dir，导致已有结果被清理。
    args.clean = False

    args.cuda = torch.cuda.is_available()

    args.snr = torch.tensor(
        eval_args.eval_snrs,
        dtype=torch.float32,
    )

    args.t = eval_args.eval_t

    # 沿用原工程调度方式。
    # 它不表示所加载分类器一定是自然训练模型；
    # 实际分类器权重由 args.hyper.pretraining_file 指定。
    args.defense_method = "nature"

    # 保持你之前想要的命名方式：
    # 例如 --eval-out eval_results, --eval-t 1
    # 则 exp_name 为 eval_results/1.task_logs。
    args.exp_name = str(out_dir / f"{args.t}.task_logs")

    args.hyper = Opt(init=dict(
        pretraining_file=str(
            Path(eval_args.eval_classifier_ckpt).resolve()
        ),
        t=eval_args.eval_t,
        start_mode=eval_args.eval_start_mode,
    ))


    # 核心：
    # clean 现在也是一种攻击算法。
    # clean -> CLEAN -> 返回零扰动。
    args.algo = eval_args.eval_mode

    # clean 模式下这些参数虽然不会真正用于生成扰动，
    # 但原工程 attack_config / BaseAttackAlgo 可能会访问，所以仍然设置。
    args.bound = eval_args.eval_attack_bound
    args.psr = eval_args.eval_psr
    args.pnr = eval_args.eval_pnr

    task = Task(args, parser)

    # 用于 Wrapper.evaluation() 里打印 t。
    task.t = eval_args.eval_t

    return task


def build_clean_test_dict(split_file, selected_snrs):
    """
    从 split_file 中明确构造 test_set 数据字典。

    返回结构与原 task.evaluation 兼容：

        {
            snr: {
                "x": 原始信号,
                "pert": 0,
                "label": 标签
            }
        }

    主要用于校验 Task.conduct() 返回的数据是否确实来自同一份 test_set。
    """
    split_data = torch.load(
        split_file,
        map_location="cpu",
        weights_only=False,
    )

    test_set = split_data["test_set"]
    test_indices = split_data["test_idx"]
    all_snrs = split_data["SNRs"]

    signals = torch.as_tensor(
        test_set[0]
    ).detach().cpu().float()

    labels = torch.as_tensor(
        test_set[1]
    ).detach().cpu()

    aligned_snrs = torch.tensor([
        float(
            scalar_value(
                all_snrs[int(scalar_value(index))]
            )
        )
        for index in test_indices
    ], dtype=torch.float32)

    if not (
        len(signals) == len(labels) == len(aligned_snrs)
    ):
        raise ValueError("test_set、标签、test_idx 数量不一致。")

    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError(
            f"期望信号形状为 [N, 2, L]，实际为 {tuple(signals.shape)}"
        )

    if not torch.isfinite(signals).all().item():
        raise ValueError("测试信号含 NaN 或 Inf。")

    result = {}

    for snr in dict.fromkeys(selected_snrs):
        mask = torch.isclose(
            aligned_snrs,
            torch.tensor(float(snr)),
        )

        if not mask.any().item():
            raise ValueError(
                f"测试集中没有 SNR={snr} 的样本。"
            )

        x = signals[mask].clone()
        y = labels[mask].clone()

        key = int(snr) if float(snr).is_integer() else float(snr)

        result[key] = {
            "x": x,
            "pert": torch.zeros_like(x),
            "label": y,
        }

    return result


def canonicalize_task_dict(source):
    """
    将 Task.conduct() 的返回结果整理到 CPU。

    要求 source 的结构兼容：

        {
            snr: {
                "x": 原始信号,
                "pert": 攻击扰动或零扰动,
                "label": 标签
            }
        }

    evaluation 使用 x + pert。
    """
    result = {}

    for snr, entry in source.items():
        key_value = float(scalar_value(snr))

        key = (
            int(key_value)
            if key_value.is_integer()
            else key_value
        )

        if key in result:
            raise ValueError(f"出现重复 SNR 键：{key}")

        for required in ("x", "pert", "label"):
            if required not in entry:
                raise KeyError(
                    f"SNR={key} 缺少字段 {required}"
                )

        new_entry = dict(entry)

        new_entry["x"] = torch.as_tensor(
            entry["x"]
        ).detach().cpu().float()

        new_entry["pert"] = torch.as_tensor(
            entry["pert"]
        ).detach().cpu().float()

        new_entry["label"] = torch.as_tensor(
            entry["label"]
        ).detach().cpu()

        if new_entry["x"].shape != new_entry["pert"].shape:
            raise ValueError(
                f"SNR={key}: x 与 pert 形状不同。"
            )

        if len(new_entry["x"]) != len(new_entry["label"]):
            raise ValueError(
                f"SNR={key}: 信号与标签数量不同。"
            )

        if not torch.isfinite(new_entry["x"]).all().item():
            raise ValueError(f"SNR={key}: x 含 NaN 或 Inf。")

        if not torch.isfinite(new_entry["pert"]).all().item():
            raise ValueError(f"SNR={key}: pert 含 NaN 或 Inf。")

        result[key] = new_entry

    if not result:
        raise ValueError("没有可评估的数据。")

    return result


def check_attack_uses_test_samples(attack_dict, test_dict):
    """
    防止 Task.conduct() 意外返回训练集或不同测试子集。

    当前严格要求每个 SNR 内的原始样本和顺序
    与 split_file 的 test_set 一致。

    注意：
    这里检查的是 attack_dict[snr]["x"] 是否等于原始 test x。
    攻击扰动或 clean 零扰动应该在 attack_dict[snr]["pert"] 里。
    """
    if set(attack_dict) != set(test_dict):
        raise ValueError(
            "Task.conduct() 返回的 SNR 范围与指定测试集不一致。"
        )

    for snr in test_dict:
        actual = attack_dict[snr]
        expected = test_dict[snr]

        same_x = (
            actual["x"].shape == expected["x"].shape
            and torch.allclose(
                actual["x"],
                expected["x"],
                atol=1e-7,
                rtol=1e-5,
            )
        )

        same_y = torch.equal(
            actual["label"],
            expected["label"],
        )

        if not same_x or not same_y:
            raise ValueError(
                f"SNR={snr}: Task.conduct() 返回数据与 "
                "split_file 的 test_set 不一致。"
                "可能是顺序、子集、预处理或数据划分不同；"
                "请先按样本 ID 对齐，不能直接混用。"
            )


def check_clean_zero_pert(source_dict, eval_mode):
    """
    clean 模式下，CLEAN pseudo attack 应该返回全零扰动。
    这个检查能防止 clean 算法注册错或误调用其他攻击。
    """
    if eval_mode != "clean":
        return

    for snr, entry in source_dict.items():
        pert = entry["pert"]

        if not torch.allclose(
            pert,
            torch.zeros_like(pert),
            atol=0.0,
            rtol=0.0,
        ):
            max_abs = pert.abs().max().item()

            raise ValueError(
                f"clean 模式下 SNR={snr} 的扰动不是全零，"
                f"max_abs_pert={max_abs}。"
                "请检查 CLEAN 攻击算法是否正确返回 torch.zeros_like(data)。"
            )


@torch.no_grad()
def purify_dictionary(
    source,
    purifier,
    t,
    device,
    batch_size,
):
    """
    对 source 中的信号进行扩散净化。

    输入 source 约定：

        source[snr]["x"]     : 原始信号
        source[snr]["pert"]  : 扰动

    实际送入扩散净化的是：

        purifier_input = x + pert

    clean 模式：
        pert = 0
        purifier_input = 原始信号

    攻击模式：
        pert = 攻击扰动
        purifier_input = 对抗样本

    输出 purified_dict 保持原工程 evaluation 习惯：

        new_entry["x"] = 原始信号
        new_entry["pert"] = purified_x - 原始信号

    所以 evaluation 内部使用 x + pert 时，得到 purified_x。
    """
    purified_dict = {}
    statistics = {}

    for snr, entry in source.items():
        original_x = entry["x"]
        source_pert = entry["pert"]

        purifier_input = original_x + source_pert

        outputs = []

        total_s = 0.0
        total_clamped = 0.0
        total_change = 0.0

        count = len(original_x)

        for start in tqdm(
            range(0, count, batch_size),
            desc=f"SNR={snr}, t={t}",
        ):
            batch = purifier_input[
                start:start + batch_size
            ].to(device)

            purified, info = purifier(
                batch,
                t=t,
                return_info=True,
            )

            outputs.append(
                purified.detach().cpu()
            )

            total_s += (
                info["predicted_s"].float().sum().item()
            )

            total_clamped += (
                info["clamped_fraction"].item() * len(batch)
            )

            per_sample_change = (
                purified - batch
            ).abs().reshape(len(batch), -1).mean(dim=1)

            total_change += per_sample_change.sum().item()

        purified_x = torch.cat(outputs, dim=0)

        new_entry = dict(entry)

        new_entry["x"] = original_x.clone()
        new_entry["label"] = entry["label"].clone()

        # evaluation 使用 x + pert 时得到 purified_x。
        new_entry["pert"] = purified_x - original_x

        purified_dict[snr] = new_entry

        statistics[str(snr)] = {
            "num_samples": count,
            "mean_predicted_s": total_s / count,
            "clamped_fraction": total_clamped / count,
            "mean_abs_purification_change": total_change / count,
        }

    return purified_dict, statistics


def call_evaluation(task, signal_dict, title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    # conduct(eval=False) 后，Wrapper.conduct() 会设置：
    #     self.mylogger = alogger
    task.mylogger.info(title)

    # 防止 Task 继续使用配置中的其他 SNR 集合。
    task.exp_snrs = list(signal_dict.keys())

    return task.evaluation(
        signal_dict,
        task.mylogger,
        ave_confMax=False,
        show_variance=False,
    )


def main():
    eval_args, args, parser = parse_eval_args()

    validate_files(eval_args)

    out_dir = Path(eval_args.eval_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(eval_args.eval_seed)

    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{getattr(args, 'gid', 0)}"
        )
    else:
        device = torch.device("cpu")

    # 加载扩散模型和 MLP 起点模型。
    purifier = FrozenPurifier.from_checkpoints(
        diffusion_path=eval_args.eval_df_ckpt,
        mlp_path=eval_args.eval_s_ckpt,
        device=device,
        boundary_policy=eval_args.eval_boundary_policy,
        start_mode=eval_args.eval_start_mode,
    )

    if eval_args.eval_t >= purifier.diffusion.num_timesteps:
        raise ValueError(
            "eval-t 必须小于 diffusion.num_timesteps。"
        )

    # 统一从 split_file 的 test_set 构造原始测试数据。
    # 主要用于校验 Task.conduct() 返回的数据是否来自同一测试集。
    clean_test_dict = build_clean_test_dict(
        eval_args.eval_split_file,
        eval_args.eval_snrs,
    )

    # 构造 Task。
    # 注意：这里只初始化 Task，不再对 clean 做特殊处理。
    task = build_task(
        eval_args,
        args,
        parser,
        out_dir,
    )

    # 关键统一逻辑：
    # clean / pgd / mi / fci / sfaa 都走 Task.conduct(eval=False)。
    #
    # clean:
    #   args.algo = "clean"
    #   CLEAN pseudo attack 返回零扰动。
    #
    # pgd / mi / fci / sfaa:
    #   生成对应攻击扰动。
    #
    # eval=False 表示 conduct() 只生成 source_dict，
    # 不在 conduct() 内部直接调用 evaluation。
    attack_result = task.conduct(
        eval=False,
        ave_confMax=False,
        show_variance=False,
    )

    source_dict = canonicalize_task_dict(
        attack_result
    )

    # 严格确认 Task.conduct() 返回的数据来自 split_file 的 test_set。
    check_attack_uses_test_samples(
        source_dict,
        clean_test_dict,
    )

    # clean 模式下额外确认扰动为零。
    check_clean_zero_pert(
        source_dict,
        eval_args.eval_mode,
    )

    manifest = {
        "classifier_architecture": eval_args.eval_classifier,
        "classifier_checkpoint": str(
            Path(eval_args.eval_classifier_ckpt).resolve()
        ),
        "diffusion_checkpoint": str(
            Path(eval_args.eval_df_ckpt).resolve()
        ),
        "mlp_checkpoint": str(
            Path(eval_args.eval_s_ckpt).resolve()
        ),
        "split_file": str(
            Path(eval_args.eval_split_file).resolve()
        ),
        "t": eval_args.eval_t,
        "mode": eval_args.eval_mode,
        "seed": eval_args.eval_seed,
        "repeats": eval_args.eval_repeats,
        "purification_batch_size": eval_args.eval_batch_size,
        "boundary_policy": eval_args.eval_boundary_policy,
        "snrs": eval_args.eval_snrs,
        "attack_bound": eval_args.eval_attack_bound,
        "psr": eval_args.eval_psr,
        "pnr": eval_args.eval_pnr,
        "clean_is_zero_attack": eval_args.eval_mode == "clean",
        "num_samples": sum(
            len(entry["x"]) for entry in source_dict.values()
        ),
    }

    with open(
        out_dir / "manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # baseline 的含义：
    #
    # clean 模式：
    #   原始信号 -> 分类器
    #
    # 攻击模式：
    #   攻击样本 -> 分类器
    #
    # baseline 不经过扩散净化。
    if eval_args.eval_baseline:
        if eval_args.eval_mode == "clean":
            baseline_title = (
                "BASELINE | mode=clean | "
                "CLEAN pseudo attack，零扰动，未经扩散净化"
            )
        else:
            baseline_title = (
                f"BASELINE | mode={eval_args.eval_mode} | "
                "攻击样本，未经扩散净化"
            )

        baseline_result = call_evaluation(
            task,
            copy.deepcopy(source_dict),
            title=baseline_title,
        )

        if baseline_result is not None:
            torch.save(
                baseline_result,
                out_dir / "baseline_evaluation_return.pt",
            )

    # 无论 clean 还是攻击模式，这里都会执行扩散净化。
    for repeat in range(eval_args.eval_repeats):
        purified_dict, statistics = purify_dictionary(
            source=source_dict,
            purifier=purifier,
            t=eval_args.eval_t,
            device=device,
            batch_size=eval_args.eval_batch_size,
        )

        if eval_args.eval_mode == "clean":
            title = (
                f"PURIFIED | "
                f"model={eval_args.eval_classifier} | "
                f"mode=clean | "
                f"input=original_signal_from_clean_zero_attack | "
                f"t={eval_args.eval_t} | "
                f"repeat={repeat + 1}/{eval_args.eval_repeats}"
            )
        else:
            title = (
                f"PURIFIED | "
                f"model={eval_args.eval_classifier} | "
                f"mode={eval_args.eval_mode} | "
                f"input=adversarial_signal | "
                f"t={eval_args.eval_t} | "
                f"repeat={repeat + 1}/{eval_args.eval_repeats}"
            )

        if eval_args.eval_save_signals:
            torch.save(
                purified_dict,
                out_dir / f"purified_signals_repeat_{repeat + 1}.pt",
            )

        evaluation_result = call_evaluation(
            task,
            purified_dict,
            title=title,
        )

        with open(
            out_dir / f"purification_stats_repeat_{repeat + 1}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                statistics,
                file,
                ensure_ascii=False,
                indent=2,
            )

        if evaluation_result is not None:
            torch.save(
                evaluation_result,
                out_dir / f"evaluation_return_repeat_{repeat + 1}.pt",
            )

    print("\n评估完成。")
    print(f"结果目录：{out_dir.resolve()}")

    if eval_args.eval_mode == "clean":
        print(
            "当前模式：clean。流程为："
            "原始测试信号 -> CLEAN零扰动 -> 扩散净化 -> 分类器。"
        )
    else:
        print(
            f"当前模式：{eval_args.eval_mode}。流程为："
            "原始测试信号 -> 攻击样本 -> 扩散净化 -> 分类器。"
        )

    print(
        "分类准确率由 task.evaluation 输出；"
        "请查看 PURIFIED 段落中 x+pert 对应的分支。"
    )


if __name__ == "__main__":
    main()
