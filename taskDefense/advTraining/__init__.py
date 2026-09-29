from taskDefense.advTraining.AMD import Adversarial_Multi_Distillation
from taskDefense.advTraining.PGD_AT import PGD_AT
from taskDefense.advTraining.PIAT import PIAT
from taskDefense.advTraining.TRADES import TRADES
from models.nn._baseTrainer import AnnealingTrainer
from taskDefense.advTraining.SADD import SADD_Trainer


defense_zoo = dict(
    nature = dict(trainer = AnnealingTrainer),
    pgdat = dict(trainer = PGD_AT),
    trades = dict(trainer = TRADES),
    amd = dict(trainer = Adversarial_Multi_Distillation),
    piat = dict(trainer = PIAT),
    sadd = dict(trainer = SADD_Trainer),
)


def load_defense_class(defense_name):
    if defense_name not in defense_zoo:
        raise Exception('Unspported defense algorithm {}'.format(defense_name))
    return defense_zoo[defense_name]['trainer']

__version__ = '1.0.0'