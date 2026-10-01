import torch
import torch.nn as nn
import argparse
import numpy as np

from denn.algos_del import train_GAN_2D
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
    elif pkey == 'wav':
        return pde.WaveEquation(**params['problem'])
    elif pkey == 'wav_ntk' or pkey.startswith('wav_ntk_'):
        return pde.WaveEquation(**params['problem'])
    elif pkey == 'burv':
        return pde.BurgersViscous(**params['problem'])
    elif pkey == 'burv_ntk' or pkey.startswith('burv_ntk_'):
        return pde.BurgersViscous(**params['problem'])
    elif pkey == 'hea':
        return pde.HeatEquation(**params['problem'])
    elif pkey == 'hea_ntk' or pkey.startswith('hea_ntk_'):
        return pde.HeatEquation(**params['problem'])
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

def gan_experiment(pkey, params, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # model init seed
    torch.manual_seed(params['training']['modelseed'])
    np.random.seed(params['training']['modelseed'])

    # Construct all tensors directly on the selected device without leaving a
    # global torch default-device context active during training.
    with torch.device(device):
        if params['generator']['pretrained']:
            from denn.pretrained import get_pretrained
            # Importing neurodiffeq changes PyTorch's global default device and
            # dtype. Restore the original float32 behavior and keep only this
            # local construction context active.
            torch.set_default_device(None)
            torch.set_default_dtype(torch.float32)
            gen = get_pretrained(pkey, params)
        elif pkey.lower().strip() in ["rays"]:
            gen = MultiHeadGen(**params['generator'])
        else:
            gen = MLP(**params['generator'])

        # Optionally decouple the discriminator initialization from the
        # generator architecture. Without this reset, changing the generator
        # width consumes a different number of random values before D is
        # constructed, so D would not start from the same weights even when
        # ``modelseed`` is unchanged.
        discriminator_modelseed = params['training'].get(
            'discriminator_modelseed'
        )
        if discriminator_modelseed is not None:
            torch.manual_seed(discriminator_modelseed)
            np.random.seed(discriminator_modelseed)
        disc = MLP(**params['discriminator'])
        problem = get_problem(pkey, params)

    # experiment seed
    torch.manual_seed(params['training']['seed'])
    np.random.seed(params['training']['seed'])

    # run
    #if pkey.lower().strip() in ["pos", "wav", "bur", "burv", "burva", "hea", "aca","lap"]:
    res = train_GAN_2D(gen, disc, problem, **params['training'], config=params)

    return res

if __name__ == '__main__':
    print(torch.cuda.is_available())  # True 才说明可以用GPU
    print(torch.cuda.device_count())  # GPU数量
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))  # 第0块GPU名称

    args = argparse.ArgumentParser()
    args.add_argument('--gan', action='store_true', default=True,
        help='whether to use GAN-based training')
    args.add_argument('--pkey', type=str, default='hea',
        help='problem to run (exp=Exponential, sho=SimpleOscillator, nlo=NonlinearOscillator)')
    args.add_argument('--device', type=str, default=None,
        help='torch device to use (default: cuda when available, otherwise cpu)')
    args.add_argument('--generator-layers', type=int, default=None,
        help='override generator n_hidden_layers from the YAML config')
    args.add_argument('--discriminator-layers', type=int, default=None,
        help='override discriminator n_hidden_layers from the YAML config')
    args.add_argument('--generator-width', type=int, default=None,
        help='override generator n_hidden_units from the YAML config')
    args.add_argument('--discriminator-width', type=int, default=None,
        help='override discriminator n_hidden_units from the YAML config')
    args.add_argument('--modelseed', type=int, default=None,
        help='override the model-initialization seed from the YAML config')
    args.add_argument('--discriminator-modelseed', type=int, default=None,
        help='independently fix the discriminator initialization seed')
    args.add_argument('--niters', type=int, default=None,
        help='override the number of training epochs from the YAML config')
    args.add_argument('--dirname', type=str, default=None,
        help='override the result directory name from the YAML config')
    args = args.parse_args()

    params = get_config(args.pkey)

    architecture_overridden = any(value is not None for value in (
        args.generator_layers,
        args.discriminator_layers,
        args.generator_width,
        args.discriminator_width,
    ))
    if args.generator_layers is not None:
        params['generator']['n_hidden_layers'] = args.generator_layers
    if args.discriminator_layers is not None:
        params['discriminator']['n_hidden_layers'] = args.discriminator_layers
    if args.generator_width is not None:
        params['generator']['n_hidden_units'] = args.generator_width
    if args.discriminator_width is not None:
        params['discriminator']['n_hidden_units'] = args.discriminator_width
    if args.modelseed is not None:
        params['training']['modelseed'] = args.modelseed
    if args.discriminator_modelseed is not None:
        params['training']['discriminator_modelseed'] = (
            args.discriminator_modelseed
        )
    if args.niters is not None:
        params['training']['niters'] = args.niters

    if args.dirname is not None:
        params['training']['dirname'] = args.dirname
    elif (
        architecture_overridden
        or args.modelseed is not None
        or args.discriminator_modelseed is not None
    ):
        discriminator_seed_suffix = (
            f"_dseed{params['training']['discriminator_modelseed']}"
            if 'discriminator_modelseed' in params['training']
            else ""
        )
        params['training']['dirname'] = (
            f"WAV_NTK_kernel_stability_"
            f"g{params['generator']['n_hidden_layers']}"
            f"x{params['generator']['n_hidden_units']}_"
            f"d{params['discriminator']['n_hidden_layers']}"
            f"x{params['discriminator']['n_hidden_units']}_"
            f"seed{params['training']['modelseed']}"
            f"{discriminator_seed_suffix}"
        )

    if args.gan:
        print(f'Running GAN training for {args.pkey} problem...')
        gan_experiment(args.pkey, params, device=args.device)

    # from denn.utils import diff

    # params = get_config('exp')
    # G = MLP(**params['generator'])
    # D = MLP(**params['discriminator'])

    # def _exp_eqn(x, t):
    #     return diff(x, t) + x

    # def adjust(x, t):
    #     x_adj = 1 + (1 - torch.exp(-t)) * x
    #     return x_adj

    # def get_equation(x, t):
    #     adj_x = adjust(x, t)
    #     return _exp_eqn(adj_x, t)

    # grid_samp = torch.linspace(0, 10, 100, requires_grad=True).reshape(-1, 1) # get grid
    # pred = G(grid_samp) # make prediction
    # residuals = get_equation(pred, grid_samp) # get residuals
    # fake = residuals

    # g_loss = D(fake).sum() # calculate some generator loss based on residuals
    # g_loss.backward(retain_graph=True)

    # d_loss = D(fake.detach()).sum() # calculate some descriminator loss
    # d_loss.backward()

    # grads = diff(residuals, grid_samp)
