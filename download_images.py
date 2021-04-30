from argparse import ArgumentParser
import sys
from io import BytesIO
from pathlib import Path
import time
from multiprocessing import Pool
from functools import partial
import re
import logging
import requests

import msgpack
import pandas as pd
import PIL
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class MsgPackWriter:
    def __init__(self, path, chunk_size=4096):
        self.path = Path(path).absolute()
        self.path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        shards_re = r"shard_(\d+).msg"
        self.shards_index = [
            int(re.match(shards_re, x).group(1))
            for x in self.path.iterdir()
            if x.is_dir() and re.match(shards_re, x)
        ]
        self.shard_open = None

    def open_next(self):
        if len(self.shards_index) == 0:
            next_index = 0
        else:
            next_index = sorted(self.shards_index)[-1] + 1
        self.shards_index.append(next_index)

        if self.shard_open is not None and not self.shard_open.closed:
            self.shard_open.close()

        self.count = 0
        self.shard_open = open(self.path / f"shard_{next_index}.msg", "wb")

    def __enter__(self):
        self.open_next()
        return self

    def __exit__(self, type, value, tb):
        self.shard_open.close()

    def write(self, data):
        if self.count >= self.chunk_size:
            self.open_next()

        self.shard_open.write(msgpack.packb(data))
        self.count += 1


def _thumbnail(img: PIL.Image, size: int) -> PIL.Image:
    # resize an image maintaining the aspect ratio
    # the smaller edge of the image will be matched to 'size'
    w, h = img.size
    if (w <= size) or (h <= size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), PIL.Image.BILINEAR)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), PIL.Image.BILINEAR)


def flickr_download(x, size_suffix="z", min_edge_size=None):

    # prevent downloading in full resolution using size_suffix
    # https://www.flickr.com/services/api/misc.urls.html

    image_id = x["image_id"]
    url_original = x["url"]
    if size_suffix != "":
        url = url_original
        # modify url to download image with specific size
        ext = Path(url).suffix
        url = f"{url.split(ext)[0]}_{size_suffix}{ext}"
    else:
        url = url_original

    r = requests.get(url)
    if r:
        try:
            image = PIL.Image.open(BytesIO(r.content))
        except PIL.UnidentifiedImageError as e:
            logger.error(f"{image_id} : {url}: {e}")
            return
    elif r.status_code == 129:
        time.sleep(60)
        logger.warning("To many requests, sleep for 60s...")
        flickr_download(x, min_edge_size=min_edge_size, size_suffix=size_suffix)
    else:
        logger.error(f"{image_id} : {url}: {r.status_code}")
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize if necessary
    image = _thumbnail(image, min_edge_size)
    # convert to jpeg
    fp = BytesIO()
    image.save(fp, "JPEG")

    raw_bytes = fp.getvalue()
    return {"image": raw_bytes, "id": image_id}


class ImageDataloader:
    def __init__(self, url_csv: Path, shuffle=False, nrows=None):

        logger.info("Read dataset")
        self.df = pd.read_csv(
            url_csv, names=["image_id", "url"], header=None, nrows=nrows
        )
        # remove rows without url
        self.df = self.df.dropna()
        if shuffle:
            logger.info("Shuffle images")
            self.df = self.df.sample(frac=1, random_state=10)
        logger.info(f"Number of URLs: {len(self.df.index)}")

    def __len__(self):
        return len(self.df.index)

    def __iter__(self):
        for image_id, url in zip(self.df["image_id"].values, self.df["url"].values):
            yield {"image_id": image_id, "url": url}


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--threads",
        type=int,
        default=24,
        help="Number of threads to download and process images",
    )
    args.add_argument(
        "--output",
        type=Path,
        default=Path("resources/images/mp16"),
        help="Output directory where images are stored",
    )
    args.add_argument(
        "--url_csv",
        type=Path,
        default=Path("resources/mp16_urls.csv"),
        help="CSV with Flickr image id and URL for downloading",
    )
    args.add_argument(
        "--size",
        type=int,
        default=320,
        help="Rescale image to a minimum edge size of SIZE",
    )
    args.add_argument(
        "--size_suffix",
        type=str,
        default="z",
        help="Image size suffix according to the Flickr API; Empty string for original image",
    )
    args.add_argument("--nrows", type=int)
    args.add_argument(
        "--shuffle", action="store_true", help="Shuffle list of URLs before downloading"
    )
    return args.parse_args()


def main():

    image_loader = ImageDataloader(args.url_csv, nrows=args.nrows, shuffle=args.shuffle)

    counter_successful = 0
    with Pool(args.threads) as p:
        with MsgPackWriter(args.output) as f:
            start = time.time()
            for i, x in enumerate(
                p.imap(
                    partial(
                        flickr_download,
                        size_suffix=args.size_suffix,
                        min_edge_size=args.size,
                    ),
                    image_loader,
                )
            ):
                if x is None:
                    continue

                f.write(x)
                counter_successful += 1

                if i % 1000 == 0:
                    end = time.time()
                    logger.info(f"{i}: {1000 / (end - start):.2f} image/s")
                    start = end
    logger.info(
        f"Sucesfully downloaded {counter_successful}/{len(image_loader)} images ({counter_successful / len(image_loader):.3f})"
    )
    return 0


if __name__ == "__main__":
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ImageDownloader")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(args.output / "writer.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    sys.exit(main())
