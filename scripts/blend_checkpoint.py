"""Blend a fine-tuned checkpoint toward its base: python scripts/blend_checkpoint.py --alpha 0.7 \
--base gs://openpi-assets/checkpoints/pi05_base/params --tuned <ckpt>/<step>/params --out <dir>

Alpha 1 is the tuned checkpoint, alpha 0 the base. Serve the result like any checkpoint, with the tuned run's config.
"""

import dataclasses

import tyro

from openpi.training import blend


@dataclasses.dataclass
class Args:
    base: str
    tuned: str
    out: str
    alpha: float = 0.7


if __name__ == "__main__":
    args = tyro.cli(Args)
    print(blend.write_blended_checkpoint(args.base, args.tuned, args.alpha, args.out))
