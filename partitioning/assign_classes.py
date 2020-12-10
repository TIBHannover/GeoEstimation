import argparse
import logging
import yaml
from pathlib import Path
from typing import Union

import pandas as pd
import s2sphere as s2
from tqdm import tqdm

tqdm.pandas()


def create_s2_cell(latlng):
    p1 = s2.LatLng.from_degrees(latlng["lat"], latlng["lng"])
    cell = s2.Cell.from_lat_lng(p1)
    return cell


def get_id_s2cell_mapping_from_raw(
    csv_file, col_img_id, col_lat, col_lng
) -> pd.DataFrame:

    usecols = [col_img_id, col_lat, col_lng]
    df = pd.read_csv(csv_file, usecols=usecols)
    df = df.rename(columns={k: v for k, v in zip(usecols, ["img_path", "lat", "lng"])})

    logging.info("Initialize s2 cells...")
    df["s2cell"] = df[["lat", "lng"]].progress_apply(create_s2_cell, axis=1)
    df = df.set_index(df["img_path"])
    return df[["s2cell"]]


def assign_class_index(cell: s2.Cell, mapping: dict) -> Union[int, None]:

    for l in range(2, 30):
        cell_parent = cell.id().parent(l)
        hexid = cell_parent.to_token()
        if hexid in mapping:
            return int(mapping[hexid])  # class index

    return None  # valid return since not all regions are covered


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default="config/baseM.yml")
    parser.add_argument(
        "-ci",
        "--column_img_path",
        type=str,
        default="IMG_ID",
        help="Column name image id / path",
    )
    parser.add_argument(
        "-clat", "--column_lat", type=str, default="LAT", help="Column name latitude"
    )
    parser.add_argument(
        "-clng", "--column_lng", type=str, default="LON", help="Column name longitude"
    )
    parser.add_argument(
        "-pskip",
        "--skiprows",
        type=int,
        default=2,
        help="skip first n rows for each partitioning",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=logging.INFO,
        filename=Path(config["model_params"]["train_label_mapping"]).parent
        / f"{Path(__file__).stem}.log",
    )

    config = config["model_params"]
    for dataset_type in ["val", "train"]:
        label_mapping_file = config[f"{dataset_type}_meta_path"]
        output_file = config[f"{dataset_type}_label_mapping"]
        logging.info(label_mapping_file)
        logging.info(output_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        logging.info("Load CSV and initialize s2 cells")
        logging.info(f"Column image path: {args.column_img_path}")
        logging.info(f"Column latitude: {args.column_lat}")
        logging.info(f"Column longitude: {args.column_lng}")
        df_mapping = get_id_s2cell_mapping_from_raw(
            label_mapping_file,
            col_img_id=args.column_img_path,
            col_lat=args.column_lat,
            col_lng=args.column_lng,
        )

        partitioning_files = [Path(p) for p in config["partitionings"]["files"]]
        for partitioning_file in partitioning_files:
            column_name = partitioning_file.name.split(".")[0]
            logging.info(f"Processing partitioning: {column_name}")
            partitioning = pd.read_csv(
                partitioning_file,
                encoding="utf-8",
                index_col="hex_id",
                skiprows=args.skiprows,
            )

            # create column with class indexes for respective partitioning
            mapping = partitioning["class_label"].to_dict()
            df_mapping[column_name] = df_mapping["s2cell"].progress_apply(
                lambda cell: assign_class_index(cell, mapping)
            )
            nans = df_mapping[column_name].isna().sum()
            logging.info(
                f"Cannot assign a hexid for {nans} of {len(df_mapping.index)} images "
                f"({nans / len(df_mapping.index) * 100:.2f}%)"
            )

        # drop unimportant information
        df_mapping = df_mapping.drop(columns=["s2cell"])
        logging.info("Remove all images that could not be assigned a cell")
        original_dataset_size = len(df_mapping.index)
        df_mapping = df_mapping.dropna()
        column_names = []
        for partitioning_file in partitioning_files:
            column_name = partitioning_file.name.split(".")[0]
            column_names.append(column_name)
            df_mapping[column_name] = df_mapping[column_name].astype("int32")

        df_mapping["targets"] = df_mapping[column_names].agg(list, axis="columns")

        fraction = len(df_mapping.index) / original_dataset_size * 100
        logging.info(
            f"Final dataset size: {len(df_mapping.index)}/{original_dataset_size} ({fraction:.2f})% from original"
        )

        # store final dataset to file
        logging.info(f"Store dataset to {output_file}")

        df_mapping = df_mapping["targets"]
        df_mapping.to_json(output_file, orient="index")

    exit(0)
