"""Model construction entry point.

The `mode` spec decides, per transformer block, whether it uses standard
self-attention ('sa') or differential attention ('da'); see `model.resolve_modes`.

Examples:
    python main.py --arch vit_small --mode da
    python main.py --arch vit_small --mode sa,sa,sa,sa,sa,sa,da,da,da,da,da,da
    python main.py --arch vit_small --mode 1:da,4:da,9:da            # rest default to sa
"""
import argparse

import torch

import model as models

ARCHS = {
    'vit_tiny': models.vit_tiny,
    'vit_small': models.vit_small,
    'vit_base': models.vit_base,
}
ARCH_DEPTH = {'vit_tiny': 12, 'vit_small': 12, 'vit_base': 12}


def parse_mode(spec, depth):
    """Turn a CLI --mode string into something `model.resolve_modes` accepts.

    'da'                    -> every block
    'sa,da,sa,...'          -> one entry per block (len must equal depth)
    '1:da,4:da'             -> only those indices, the rest default to 'sa'
    '0:sa,1:da,...,11:sa'   -> fully enumerated dict
    """
    spec = spec.strip()
    if ',' not in spec and ':' not in spec:
        return spec
    parts = [p.strip() for p in spec.split(',') if p.strip()]
    if any(':' in p for p in parts):
        if not all(':' in p for p in parts):
            raise ValueError("--mode mixes 'i:mode' and bare entries; use one style")
        out = {}
        for p in parts:
            idx, m = p.split(':', 1)
            out[int(idx)] = m.strip()
        # a fully enumerated dict needs no default; a partial one falls back to 'sa'
        if len(out) < depth:
            out['default'] = 'sa'
        return out
    return parts


def build_model(arch='vit_small', patch_size=16, img_size=224, mode='sa', num_classes=0, **kwargs):
    if arch not in ARCHS:
        raise ValueError(f'unknown arch {arch!r}, expected one of {sorted(ARCHS)}')
    
    if isinstance(mode, str):
        mode = parse_mode(mode, ARCH_DEPTH[arch])

    return ARCHS[arch](
        patch_size=patch_size,
        img_size=[img_size],
        num_classes=num_classes,
        mode=mode,
        **kwargs,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arch', default='vit_tiny', choices=sorted(ARCHS))
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--mode', default='sa',
                   help="'sa' | 'da' | per-block list 'sa,da,...' | index form '1:da,4:da'")
    p.add_argument('--num_classes', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=2, help='batch size of the smoke-test forward pass')
    args = p.parse_args()

    net = build_model(
        arch=args.arch,
        patch_size=args.patch_size,
        img_size=args.img_size,
        mode=args.mode,
        num_classes=args.num_classes,
    )


    net = net.to(args.device)

    n_params = sum(p.numel() for p in net.parameters())
    print(f'{args.arch} patch{args.patch_size} img{args.img_size} on {args.device}')
    print(f'blocks: {net.modes}')
    print(f'params: {n_params / 1e6:.2f}M')

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=args.device)
    with torch.no_grad():
        out = net(x)
        attn = net.get_last_selfattention(x)
    shapes = ({k: tuple(v.shape) for k, v in attn.items()} if isinstance(attn, dict) else tuple(attn.shape))
    print(f'forward: {tuple(out.shape)} | last attn ({net.modes[-1]}): {shapes}')
    

if __name__ == '__main__':
    main()
