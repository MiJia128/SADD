import os
import traceback

from taskDefense.Wrapper import Task
from taskDefense.Parser import get_parser
from data import data_zoo
from models.nn._baseNet import Opt

if __name__ == "__main__":
    args, parser = get_parser(parsing=False)
    parser.add_argument('-ml', default=['ctdnn','res'],  nargs='+', type=str, help='the model list to conduct the defense, default is ctdnn and res')
    args = parser.parse_known_args()[0] 
    # args.defense_method = 'pgdat'
    args.psr = -10
    args.cuda = True
    args.warmup_epoch = 10

    #args.snr=[10]
    args.snr=['all']
    # args.gid = 2
    args.exp_name = f'defense.psr{args.psr}.warm{args.warmup_epoch}.{args.defense_method}'

    models = ['ctdnn','res'] + [ 'mcd','amcnet','awn', 'mcl', 'msmc',]
    models = args.ml if args.ml != ['all'] else models
    
    log_path = f'yield_results/{args.exp_name}/exp.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
   
    data_name = data_zoo[args.data]['data_name']
    for model in models:
        args.model = model
        args.hyper = Opt(
            init=dict(
                patience=12,
                )
            )
              
        dst = f'yield_results/defense.psr-10.warm10.{args.defense_method}/{data_name}/fit/{model}/checkpoint/{data_name}_{model}.best.pt'
        
        if os.path.exists(dst):
            print(f"Checkpoint already exists, skipping: {dst}")
            continue
        
        try:
            task = Task(args, parser)
            task.conduct(eval=True, ave_confMax=False, show_variance=False)
        except Exception as e:
            error_msg = (
                f"Task conduct failed while processing data: {args.data}: model {model}: {e}"
            )
            print(error_msg)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(error_msg + '\n')
                f.write(traceback.format_exc() + '\n')
            continue
    
