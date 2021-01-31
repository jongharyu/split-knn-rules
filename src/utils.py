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
    base_keys = ['Msplit_select1', # proposed
                 'Msplit_select0', # \approx big k-NN
                ]
    keys = []
    for k in base_k:
        keys.extend(['{}_{}NN'.format(key, k) for key in base_keys])
    return keys
