import os
import sys
sys.path.append(os.path.join(os.path.dirname(
    __file__), os.path.pardir, os.path.pardir))
from taskDefense.Wrapper import Task
from taskDefense.Parser import get_parser


        
if __name__ == "__main__":
    args, parser = get_parser(parsing=True)
    parser.add_argument('-myt', type=int, default=0)
    parser.add_argument('-mya', type=float, default=1.0)
    parser.add_argument('-myb', type=float, default=1.0)
    parser.add_argument("--df-ckpt", required=True)
    parser.add_argument("--s-ckpt", required=True)
    parser.add_argument("--lambda-nmse", type=float, default=1.0)
    parser.add_argument("--lambda-pur", type=float, default=1.0)
    parser.add_argument("--nmse-beta", type=float, default=4.0)
    parser.add_argument("--boundary-policy",choices=["clamp_start", "error"],default="clamp_start",)
    parser.add_argument(
        "--start-mode",
        choices=["dynamic", "zero"],
        default="dynamic",
        help=(
            "SADD 训练时 purifier 的起点模式。"
            "dynamic 使用 MLP 动态预测；zero 固定 s=0。"
        ),
    )

    args = parser.parse_args()

    from models.nn._baseNet import Opt
    args.hyper = Opt()
 
    args.using_snr = True
    args.exp_name = 'sadd.advTraining.{}.{}.{}.{}'.format(args.defense_method,args.myt,args.lambda_nmse,args.lambda_pur)
    args.algo = 'mi'
    args.psr = -10
    args.cuda = True
    args.test = True
    args.clean = True

    task = Task(args, parser)
    task.conduct(eval=True, ave_confMax=False, show_variance=False)
    
