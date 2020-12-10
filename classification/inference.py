from argparse import ArgumentParser
from pathlib import Path
from math import ceil
import pandas as pd
import torch
from tqdm.auto import tqdm

from classification.train_base import MultiPartitioningClassifier
from classification.dataset import FiveCropImageDataset


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/trained_models/baseM/epoch=014-val_loss=18.4833.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument(
        "--hparams",
        type=Path,
        default=Path("data/trained_models/baseM/hparams.yaml"),
        help="Path to hparams file (*.yaml) generated during training",
    )
    args.add_argument(
        "--image_dir",
        type=Path,
        default=Path("data/images/im2gps"),
        help="Folder containing images. Supported file extensions: (*.jpg, *.jpeg, *.png)",
    )
    # environment
    args.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU for inference if CUDA is available",
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

print("Load model from ", args.checkpoint)
model = MultiPartitioningClassifier.load_from_checkpoint(
    checkpoint_path=str(args.checkpoint),
    hparams_file=str(args.hparams),
    map_location=None,
)
model.eval()
if args.gpu:
    model.cuda()

print("Init dataloader")
dataloader = torch.utils.data.DataLoader(
    FiveCropImageDataset(meta_csv=None, image_dir=args.image_dir),
    batch_size=ceil(args.batch_size / 5),
    shuffle=False,
    num_workers=args.num_workers,
)
print("Number of images: ", len(dataloader.dataset))
if len(dataloader.dataset) == 0:
    raise RuntimeError(f"No images found in {args.image_dir}")

rows = []
for X in tqdm(dataloader):
    if args.gpu:
        X[0] = X[0].cuda()
    img_paths, pred_classes, pred_latitudes, pred_longitudes = model.inference(X)
    for p_key in pred_classes.keys():
        for img_path, pred_class, pred_lat, pred_lng in zip(
            img_paths,
            pred_classes[p_key].cpu().numpy(),
            pred_latitudes[p_key].cpu().numpy(),
            pred_longitudes[p_key].cpu().numpy(),
        ):
            rows.append(
                {
                    "img_id": Path(img_path).stem,
                    "p_key": p_key,
                    "pred_class": pred_class,
                    "pred_lat": pred_lat,
                    "pred_lng": pred_lng,
                }
            )
df = pd.DataFrame.from_records(rows)
df.set_index(keys=["img_id", "p_key"], inplace=True)
print(df)
fout = Path(args.checkpoint).parent / f"inference_{args.image_dir.stem}.csv"
print("Write output to", fout)
df.to_csv(fout)
