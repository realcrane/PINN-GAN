#!/usr/bin/env python3
"""Select the DEQGAN or NTK trainer without editing the legacy entry point."""

import argparse

import denn.experiments as experiments
from denn.algos_NTK import train_GAN_2D as train_ntk
from denn.algos_del import train_GAN_2D as train_deqgan
from denn.config.config import get_config


PROBLEMS = ("pos", "lap1", "lap2", "kle", "rea", "burv")
NTK_METHODS = ("gan", "lsgan", "wgan")
DEQGAN_KEYS = set(PROBLEMS)
NTK_KEYS = {
    f"{problem}_ntk_{method}"
    for problem in PROBLEMS
    for method in NTK_METHODS
}
NTK_KEYS.add("lap1_ntk_gan_deqgan")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one of the reproducible six-PDE experiments."
    )
    parser.add_argument(
        "--trainer",
        required=True,
        choices=("deqgan", "ntk"),
        help="training implementation to use",
    )
    parser.add_argument("--pkey", required=True, help="YAML configuration key")
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device, e.g. cuda:0 or cpu (default: CUDA if available)",
    )
    parser.add_argument("--niters", type=int, default=None)
    parser.add_argument("--dirname", default=None)
    parser.add_argument("--modelseed", type=int, default=None)
    parser.add_argument("--discriminator-modelseed", type=int, default=None)
    parser.add_argument("--generator-layers", type=int, default=None)
    parser.add_argument("--generator-width", type=int, default=None)
    parser.add_argument("--discriminator-layers", type=int, default=None)
    parser.add_argument("--discriminator-width", type=int, default=None)
    parser.add_argument(
        "--disable-rollback",
        action="store_true",
        help=(
            "run the README GAN-without-rollback baseline and write it to a "
            "separate result directory"
        ),
    )
    return parser


def validate_key(trainer, pkey):
    allowed = DEQGAN_KEYS if trainer == "deqgan" else NTK_KEYS
    if pkey not in allowed:
        choices = ", ".join(sorted(allowed))
        raise SystemExit(
            f"Configuration {pkey!r} is not valid for {trainer}. "
            f"Choose one of: {choices}"
        )


def apply_overrides(params, args):
    training = params["training"]
    generator = params["generator"]
    discriminator = params["discriminator"]

    if args.niters is not None:
        training["niters"] = args.niters
    if args.dirname is not None:
        training["dirname"] = args.dirname
    if args.modelseed is not None:
        training["modelseed"] = args.modelseed
    if args.discriminator_modelseed is not None:
        training["discriminator_modelseed"] = args.discriminator_modelseed
    if args.generator_layers is not None:
        generator["n_hidden_layers"] = args.generator_layers
    if args.generator_width is not None:
        generator["n_hidden_units"] = args.generator_width
    if args.discriminator_layers is not None:
        discriminator["n_hidden_layers"] = args.discriminator_layers
    if args.discriminator_width is not None:
        discriminator["n_hidden_units"] = args.discriminator_width


def disable_rollback(params, pkey):
    if pkey == "lap1_ntk_gan_deqgan":
        raise SystemExit("--disable-rollback is only valid for the 18 standard NTK keys")

    problem, method = pkey.split("_ntk_", 1)
    params["training"].update(
        {
            "G_iters": 1,
            "D_iters": 1,
            "adaptive_G": False,
            "adaptive_D": False,
            "ntk": False,
            "energy": False,
            "dirname": f"{problem.capitalize()}_NTK_none_1_1_{method}",
        }
    )


def main():
    args = build_parser().parse_args()
    pkey = args.pkey.lower().strip()
    validate_key(args.trainer, pkey)

    params = get_config(pkey)
    if args.disable_rollback:
        if args.trainer != "ntk":
            raise SystemExit("--disable-rollback requires --trainer ntk")
        disable_rollback(params, pkey)
    apply_overrides(params, args)

    experiments.train_GAN_2D = (
        train_deqgan if args.trainer == "deqgan" else train_ntk
    )
    print(
        f"Running trainer={args.trainer}, pkey={pkey}, "
        f"device={args.device or 'auto'}"
    )
    experiments.gan_experiment(pkey, params, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
