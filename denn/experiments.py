import torch
import torch.nn as nn
import argparse
import numpy as np

from denn.algos_NTK import train_GAN_2D
from denn.models import MLP, MultiHeadGen
from denn.config.config import get_config
import denn.pde_problems as pde

def get_problem(pkey, params):
    """ helper to parse problem key and return appropriate problem
    """
    pkey = pkey.lower().strip()
    if pkey == 'pos':
        return pde.PoissonEquation(**params['problem'])
    elif pkey == 'pos_ntk' or pkey.startswith('pos_ntk_'):
        return pde.PoissonEquation(**params['problem'])
    elif pkey == 'kle':
        return pde.KleinEquation(**params['problem'])
    elif pkey == 'kle_ntk' or pkey.startswith('kle_ntk_'):
        return pde.KleinEquation(**params['problem'])
    elif pkey == 'burv':
        return pde.BurgersViscous(**params['problem'])
    elif pkey == 'burv_ntk' or pkey.startswith('burv_ntk_'):
        return pde.BurgersViscous(**params['problem'])
    elif pkey == 'rea':
        return pde.ReactionEquation(**params['problem'])
    elif pkey == 'rea_ntk' or pkey.startswith('rea_ntk_'):
        return pde.ReactionEquation(**params['problem'])
    elif pkey == 'lap1':
        return pde.Laplace1Equation(**params['problem'])
    elif pkey == 'lap1_ntk' or pkey.startswith('lap1_ntk_'):
        return pde.Laplace1Equation(**params['problem'])
    elif pkey == 'lap2':
        return pde.Laplace2Equation(**params['problem'])
    elif pkey == 'lap2_ntk' or pkey.startswith('lap2_ntk_'):
        return pde.Laplace2Equation(**params['problem'])
    else:
        raise RuntimeError(f'Did not understand problem key (pkey): {pkey}')


def gan_experiment(pkey, params):
    # model init seed
    torch.manual_seed(params['training']['modelseed'])
    np.random.seed(params['training']['modelseed'])

    # models
    gen = MLP(**params['generator'])
    disc = MLP(**params['discriminator'])

    # experiment seed
    torch.manual_seed(params['training']['seed'])
    np.random.seed(params['training']['seed'])

    # run
    problem = get_problem(pkey, params)
    res = train_GAN_2D(gen, disc, problem, **params['training'], config=params)

    return res

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=True,
        help='whether to use GAN-based training')
    args.add_argument('--pkey', type=str, default='pos_ntk_lsgan',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args = args.parse_args()

    params = get_config(args.pkey)

    if args.gan:
        print(f'Running GAN training for {args.pkey} problem...')
        gan_experiment(args.pkey, params)
