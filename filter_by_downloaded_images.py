from argparse import ArgumentParser
import logging
import os
import re
import json
from io import BytesIO
from pathlib import Path
from typing import Union

import yaml
import msgpack
import pandas as pd

import torch
import torchvision
from PIL import Image


class MsgPackIterableMetaDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        msgpack_path: Union[str, Path],
        image_ids_path: Union[str, Path],
        meta_path: Union[str, Path],
        image_ids_index_col: Union[str, int] = 0,
        meta_index_col: Union[str, int] = 0,
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        cache_size=4096,
        ignore_image=False,
    ):

        super(MsgPackIterableMetaDataset, self).__init__()
        self.path = msgpack_path
        self.cache_size = cache_size
        self.transformation = transformation

        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.ignore_image = ignore_image
        self.image_ids = self.__init_image_ids(image_ids_path, image_ids_index_col)

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.meta = pd.read_csv(meta_path, index_col=meta_index_col)

        if "LAT" in self.meta.columns:
            self.meta.rename(
                columns={"LAT": "latitude", "LON": "longitude"}, inplace=True
            )

        self.meta = self.meta.astype({"latitude": "float32", "longitude": "float32"})
        logging.debug(self.meta)
        self.shards = self.__init_shards(self.path)

    @staticmethod
    def __init_image_ids(image_ids_path: Union[Path, str], index_col=0) -> set:
        """
        Args:
            image_ids_path: path to CSV
            index_col: column name or index with image ids

        Returns: set of image ids to filter
        """

        df = pd.read_csv(image_ids_path, index_col=index_col)
        logging.debug(df)
        image_ids = set(df.index.tolist())
        return image_ids

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        img = None
        if not self.ignore_image:
            # prepare image and meta_data
            # decode and initial resize if necessary
            img = Image.open(BytesIO(x[self.key_img_encoded]))
            if img.mode != "RGB":
                img = img.convert("RGB")

            if img.width > 320 and img.height > 320:
                img = torchvision.transforms.Resize(320)(img)

            # apply all user specified image transformations
            if self.transformation is not None:
                img = self.transformation(img)

        _id = x[self.key_img_id].decode("utf-8")
        meta = self.meta.loc[_id].to_dict()
        meta["img_id"] = _id
        return img, meta

    def __iter__(self):

        shard_indices = list(range(len(self.shards)))

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    if _id not in self.image_ids:
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:
                        while cache:
                            yield self._process_sample(cache.pop())
        while cache:
            yield self._process_sample(cache.pop())


def main():

    for dataset_type in ["train", "val"]:
        with open(config[f"{dataset_type}_label_mapping"]) as f:
            mapping = json.load(f)

        logging.info(f"Expected dataset size: {len(mapping)}")
        msgpack_path = config[f"msgpack_{dataset_type}_dir"]
        image_ids_path = config[f"{dataset_type}_meta_path"]
        dataset = MsgPackIterableMetaDataset(
            msgpack_path,
            image_ids_path,
            image_ids_path,
            key_img_id=config["key_img_id"],
            key_img_encoded=config["key_img_encoded"],
            ignore_image=True,
        )

        filtered_mapping = {}
        for _, meta in dataset:
            if meta["img_id"] in mapping:
                filtered_mapping[meta["img_id"]] = mapping[meta["img_id"]]
        logging.info(f"True dataset size: {len(filtered_mapping)}")

        with open(config[f"{dataset_type}_label_mapping"], "w") as fw:
            json.dump(filtered_mapping, fw)
    return


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default="config/baseM.yml")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = config["model_params"]

    main()
