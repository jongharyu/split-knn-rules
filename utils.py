import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generate_keys(base_k):
    base_keys = ['soft1_selective1', 'soft1_selective0', 'soft0_selective1', 'soft0_selective0']
    keys = []
    for k in base_k:
        keys.extend(['{}_{}NN'.format(key, k) for key in base_keys])
    return keys
