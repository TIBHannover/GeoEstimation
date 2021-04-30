from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import torch
import pytorch_lightning as pl
import pandas as pd

from classification.train_base import MultiPartitioningClassifier
from classification.dataset import FiveCropImageDataset


def parse_args():
    args = ArgumentParser()
    # model
    args.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/base_M/epoch=014-val_loss=18.4833.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams",
        type=Path,
        default=Path("models/base_M/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    # testsets
    args.add_argument(
        "--image_dirs",
        nargs="+",
        default=["resources/images/im2gps", "resources/images/im2gps3k"],
        help="Whitespace separated list of image folders to evaluate",
    )
    args.add_argument(
        "--meta_files",
        nargs="+",
        default=[
            "resources/images/im2gps_places365.csv",
            "resources/images/im2gps3k_places365.csv",
        ],
        help="Whitespace separated list of respective meta data (ground-truth GPS positions). Required columns: 'IMG_ID,LAT,LON' or 'img_id, latitude, longitude'",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference if CUDA is available",
    )
    args.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Full precision (32), half precision (16)",
    )
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for image loading and pre-processing",
    )
    return args.parse_args()


args = parse_args()
print("Load model from checkpoint", args.checkpoint)
model = MultiPartitioningClassifier.load_from_checkpoint(
    checkpoint_path=str(args.checkpoint),
    hparams_file=str(args.hparams),
    map_location=None,
)

if args.gpu and torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = None
trainer = pl.Trainer(gpus=args.gpu, precision=args.precision)

print("Init Testsets")
dataloader = []
for image_dir, meta_csv in zip(args.image_dirs, args.meta_files):
    dataset = FiveCropImageDataset(meta_csv, image_dir)
    dataloader.append(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=ceil(args.batch_size / 5),
            shuffle=False,
            num_workers=args.num_workers,
        )
    )
print("Testing")
r = trainer.test(model, test_dataloaders=dataloader, verbose=False)
# formatting results
dfs = []
if len(args.image_dirs) > 1:
    r = r[0].values()
for results, name in zip(r, [Path(x).stem for x in args.image_dirs]):
    df = pd.DataFrame(results).T
    df["dataset"] = name
    df["partitioning"] = df.index
    df["partitioning"] = df["partitioning"].apply(lambda x: x.split("/")[-1])
    df.set_index(keys=["dataset", "partitioning"], inplace=True)
    dfs.append(df)

df = pd.concat(dfs)

print(df)
fout = Path(args.checkpoint).parent / ("test-" + "_".join(
    [str(Path(x).stem) for x in args.image_dirs]) + ".csv")
print("Write to", fout)
df.to_csv(fout)
