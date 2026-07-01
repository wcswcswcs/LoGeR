import argparse
from dataset.scannet import ScanNetDataset
from dataset.matterport import MatterportDataset
from dataset.scannetpp import ScanNetPPDataset
import json

try:
    from dataset.demo import DemoDataset
except ImportError:
    DemoDataset = None

def update_args(args):
    config_path = f'configs/{args.config}.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    for key in config:
        setattr(args, key, config[key])
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str)
    parser.add_argument('--seq_name_list', type=str)
    parser.add_argument('--config', type=str, default='scannet')
    parser.add_argument('--backbone', type=str, default='Cropformer')  # Cropformer SAM2 SAM FastSAM EfficientSAM
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--para', type=float)
    parser.add_argument('--frame-stride', dest='frame_stride', type=int, default=None)
    parser.add_argument('--frame-id-allowlist', type=str, default=None)
    parser.add_argument('--export-local-stage', action="store_true")
    parser.add_argument('--local-stage-output-dir', type=str, default=None)
    parser.add_argument('--segmentation-dir-override', type=str, default=None)

    cli_overrides = {}
    args = parser.parse_args()
    for key in ("frame_stride", "frame_id_allowlist", "local_stage_output_dir", "segmentation_dir_override"):
        value = getattr(args, key, None)
        if value is not None:
            cli_overrides[key] = value
    if bool(getattr(args, "export_local_stage", False)):
        cli_overrides["export_local_stage"] = True
    args = update_args(args)
    for key, value in cli_overrides.items():
        setattr(args, key, value)
    if isinstance(getattr(args, "frame_id_allowlist", None), str):
        args.frame_id_allowlist = [
            int(part.strip())
            for part in args.frame_id_allowlist.split(",")
            if part.strip()
        ]
    return args

def get_dataset(args, model):
    if args.dataset == 'scannet':
        dataset = ScanNetDataset(args.seq_name, model)
    elif args.dataset == 'scannetpp':
        dataset = ScanNetPPDataset(args.seq_name, model)
    elif args.dataset == 'matterport3d':
        dataset = MatterportDataset(args.seq_name, model)
    elif args.dataset == 'demo':
        if DemoDataset is None:
            raise ImportError("dataset.demo is not available in this checkout")
        dataset = DemoDataset(args.seq_name)
    else:
        print(args.dataset)
        raise NotImplementedError
    override = getattr(args, "segmentation_dir_override", None)
    if override:
        dataset.segmentation_dir = str(override).format(seq_name=args.seq_name)
    return dataset
