import torch
import torch.nn as nn
import argparse
import numpy as np

from denn.algos_NTK import train_GAN_2D
from denn.models import MLP, MultiHeadGen
from denn.config.config import get_config
import denn.pde_problems as pde


def resolve_device(requested='auto'):
    """Resolve and validate the device used to construct an experiment."""
    if requested in (None, '', 'auto'):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f'Invalid device {requested!r}: {exc}') from exc

    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested, but torch.cuda.is_available() is False'
            )

        index = 0 if device.index is None else device.index
        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise RuntimeError(
                f'Requested cuda:{index}, but only {device_count} CUDA '
                'device(s) are visible'
            )

        device = torch.device(f'cuda:{index}')
        torch.cuda.set_device(device)
    elif device.type != 'cpu':
        raise RuntimeError(
            f'Unsupported device type {device.type!r}; use auto, cpu, cuda, '
            'or cuda:N'
        )

    return device


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


def gan_experiment(pkey, params, device='auto', train_fn=None):
    # Set the default before constructing models, PDE objects, grids, labels, or
    # scalar constants.  This keeps programmatic/direct use on the same device
    # path as run_experiment.py without requiring each research class to be
    # rewritten with explicit ``device=`` arguments.
    device = resolve_device(device)
    torch.set_default_device(device)
    if train_fn is None:
        train_fn = train_GAN_2D

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

    x, y = problem.get_grid()
    generator_device = next(gen.parameters()).device
    discriminator_device = next(disc.parameters()).device

    devices = {
        generator_device,
        discriminator_device,
        x.device,
        y.device,
    }

    print(
        f"Selected device: {device}\n"
        "Actual devices: "
        f"generator={generator_device}, "
        f"discriminator={discriminator_device}, "
        f"grid_x={x.device}, "
        f"grid_y={y.device}"
    )

    if len(devices) != 1:
        raise RuntimeError(
            "Device mismatch: "
            f"generator={generator_device}, "
            f"discriminator={discriminator_device}, "
            f"grid_x={x.device}, "
            f"grid_y={y.device}"
        )

    res = train_fn(
        gen,
        disc,
        problem,
        **params["training"],
        config=params,
    )

    return res


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=True,
        help='whether to use GAN-based training')
    args.add_argument('--pkey', type=str, default='pos_ntk_lsgan',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args.add_argument('--device', type=str, default='auto',
        help='PyTorch device: auto, cpu, cuda, or cuda:N')
    args.add_argument('--niters', type=int, default=None,
        help='optional iteration override for a launchability check')
    args.add_argument('--dirname', type=str, default=None,
        help='optional output-directory override')
    args = args.parse_args()

    params = get_config(args.pkey)
    if args.niters is not None:
        params['training']['niters'] = args.niters
    if args.dirname is not None:
        params['training']['dirname'] = args.dirname

    if args.gan:
        print(f'Running GAN training for {args.pkey} problem on {args.device}...')
        gan_experiment(args.pkey, params, device=args.device)
