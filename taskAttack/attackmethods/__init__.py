import importlib

attack_zoo = dict(
    fgsm= ('taskAttack.attackmethods.gradient.fgsm', 'FGSM'),
    fgm= ('taskAttack.attackmethods.gradient.fgsm', 'FGSM'),
    bim= ('taskAttack.attackmethods.gradient.fgsm', 'IFGSM'),
    pgd= ('taskAttack.attackmethods.gradient.fgsm', 'PGD'),
    mi= ('taskAttack.attackmethods.gradient.fgsm', 'MIFGSM'),
    ni= ('taskAttack.attackmethods.gradient.fgsm', 'NIFGSM'),
    vmi= ('taskAttack.attackmethods.gradient.fgsm', 'VMIFGSM'),
    vni= ('taskAttack.attackmethods.gradient.fgsm', 'VNIFGSM'),
    sfaa=('taskAttack.attackmethods.gradient.SFAA','SFAA'),
    fci = ('taskAttack.attackmethods.gradient.FCIAA', 'FCIAA'),
)


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise Exception('Unspported attack algorithm {}'.format(attack_name))
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)
    return attack_class

__version__ = '1.0.0'