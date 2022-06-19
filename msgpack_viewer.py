import os
import argparse
import re
from typing import Union
from io import BytesIO
import random
from pathlib import Path

from PIL import Image
import torchvision
import torch
import msgpack


class MsgPackIterableDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        path: str,
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=False,
        cache_size=6 * 4096,
    ):

        super(MsgPackIterableDataset, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]
        
        self.shards = self.__init_shards(self.path)

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
        return img, _id

    def __iter__(self):

        shard_indices = list(range(len(self.shards)))

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

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

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--data", type=str, default="resources/images/mp16")
    args = args.parse_args()

    tfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    dataset = MsgPackIterableDataset(path=args.data, transformation=tfm)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=6,
            pin_memory=False,
        )

    num_images = 0
    for x, image_id in dataloader:
        if num_images == 0:
            print(x.shape, image_id)
        num_images +=1
    
    print(f"{num_images=}")
