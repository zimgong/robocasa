import os
import json
import shutil

# import logging
from pathlib import Path
from typing import Callable
from functools import partial

import h5py
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# from lerobot.common.datasets.compute_stats import compute_stats
# from lerobot.common.datasets.utils import (
#     STATS_PATH,
#     check_timestamps_sync,
#     get_episode_data_index,
#     serialize_dict,
#     write_json,
# )

AGENTVIEW_LEFT = "robot0_agentview_left.mp4"
AGENTVIEW_RIGHT = "robot0_agentview_right.mp4"
EYE_IN_HAND = "robot0_eye_in_hand.mp4"

FEATURES = {
    "observation.images.agentview_left": {
        "dtype": "video",
        "shape": [512, 512, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.agentview_right": {
        "dtype": "video",
        "shape": [512, 512, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.eye_in_hand": {
        "dtype": "video",
        "shape": [512, 512, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 20.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [26],
    },
    "action": {
        "dtype": "float32",
        "shape": [13],
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}


class RoboCasaDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)
            item = (
                frame[key].numpy()
                if isinstance(frame[key], torch.Tensor)
                else frame[key]
            )
            self.episode_buffer[key].append(item)

        self.episode_buffer["size"] += 1

    def save_episode(
        self, task: str, episode_data: dict | None = None, videos: dict | None = None
    ) -> None:
        """
        We rewrite this method to copy mp4 videos to the target position
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        episode_length = episode_buffer.pop("size")
        episode_index = episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `add_episode`."
            )

        task_index = self.meta.get_task_index(task)

        if not set(episode_buffer.keys()) == set(self.features):
            raise ValueError()

        for key, ft in self.features.items():
            if key == "index":
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                episode_buffer[key] = np.full((episode_length,), episode_index)
            elif key == "task_index":
                episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video"]:
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(key)

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        self.meta.save_episode(episode_index, episode_length, task, task_index)
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
        self.consolidated = False

    # def consolidate(self, run_compute_stats: bool = True, keep_image_files: bool = False) -> None:
    #     self.hf_dataset = self.load_hf_dataset()
    #     self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
    #     check_timestamps_sync(self.hf_dataset, self.episode_data_index, self.tolerance_s)

    #     if len(self.meta.video_keys) > 0:
    #         self.meta.write_video_info()

    #     if not keep_image_files:
    #         img_dir = self.root / "images"
    #         if img_dir.is_dir():
    #             shutil.rmtree(self.root / "images")

    #     video_files = list(self.root.rglob("*.mp4"))
    #     assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

    #     parquet_files = list(self.root.rglob("*.parquet"))
    #     assert len(parquet_files) == self.num_episodes

    #     if run_compute_stats:
    #         self.stop_image_writer()
    #         self.meta.stats = compute_stats(self)
    #         serialized_stats = serialize_dict(self.meta.stats)
    #         write_json(serialized_stats, self.root / STATS_PATH)
    #         self.consolidated = True
    #     else:
    #         logging.warning(
    #             "Skipping computation of the dataset statistics, dataset is not fully consolidated."
    #         )


def load_local_dataset(
    f: h5py.File, demo_id: str, src_path: str, task_id: str
) -> list | None:
    """Load local dataset and return a dict with observations and actions"""

    ob_dir = Path(src_path) / f"{task_id}" / f"{demo_id}"

    # Note: the state is downsampled by 5x to comply with the video fps
    demo_dim = f["data/{}/states".format(demo_id)].shape[1]
    vel_start_idx = int((demo_dim - 1) / 2 + 2)
    state_pos = np.array(f["data/{}/states".format(demo_id)][::5, 1:14])
    state_vel = np.array(
        f["data/{}/states".format(demo_id)][::5, vel_start_idx : vel_start_idx + 13]
    )
    demo_len = state_pos.shape[0]
    states_value = np.hstack([state_pos, state_vel]).astype(np.float32)
    action_value = np.zeros((demo_len, 13))
    action_value[: demo_len - 1, :] = np.array(
        f["data/{}/states".format(demo_id)][::5, 1:14][1:]
    ).astype(np.float32)

    frames = [
        {
            "observation.state": states_value[i],
            "action": action_value[i],
        }
        for i in range(len(states_value))
    ]

    v_path = ob_dir / "videos"
    videos = {
        "observation.images.agentview_left": v_path / AGENTVIEW_LEFT,
        "observation.images.agentview_right": v_path / AGENTVIEW_RIGHT,
        "observation.images.eye_in_hand": v_path / EYE_IN_HAND,
    }
    return frames, videos


def get_task_instruction(f: h5py.File, task_demos: list[str]) -> dict:
    """Get task language instruction"""
    ep = task_demos[0]
    ep_meta = json.loads(f["data/{}".format(ep)].attrs["ep_meta"])
    task_instruction = ep_meta["lang"]
    print(f"Get Task Instruction <{task_instruction}>")
    return task_instruction


def main(
    src_path: str,
    tgt_path: str,
    task_id: str,
    repo_id: str,
    task_info_hdf5: str,
    filter_key: str | None = None,
    debug: bool = False,
):
    all_filter_keys = None
    f = h5py.File(task_info_hdf5, "r")
    if filter_key is not None:
        # use the demonstrations from the filter key instead
        print("NOTE: using filter key {}".format(filter_key))
        demos = sorted(
            [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
        )
    else:
        # use all demonstrations
        demos = sorted(list(f["data"].keys()))

        # extract filter key information
        if "mask" in f:
            all_filter_keys = {}
            for fk in f["mask"]:
                fk_demos = sorted(
                    [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])]
                )
                all_filter_keys[fk] = fk_demos

    # put demonstration list in increasing episode order
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    env_meta = json.loads(f["data"].attrs["env_args"])
    robot_type = env_meta["env_kwargs"]["robots"]
    # camera_names = env_meta["env_kwargs"]["camera_names"]
    # features = {}
    # for camera_name in camera_names:
    #     features[f"observation.images.{camera_name}"] = {
    #         "dtype": "video",
    #         "shape": [128, 128, 3],
    #         "names": ["height", "width", "channel"],
    #         "video_info": {
    #             "video.fps": 20.0,
    #             "video.codec": "av1",
    #             "video.pix_fmt": "yuv420p",
    #             "video.is_depth_map": False,
    #             "has_audio": False,
    #         },
    #     }
    # features.update(FEATURES_INDEX)

    task_name = get_task_instruction(f, demos)

    dataset = RoboCasaDataset.create(
        repo_id=repo_id,
        root=f"{tgt_path}/{repo_id}",
        fps=20,
        robot_type=robot_type,
        features=FEATURES,
    )

    if debug:
        raw_datasets = [
            load_local_dataset(f, demo, src_path, task_id) for demo in tqdm(demos)
        ]
    else:
        raw_datasets = process_map(
            partial(load_local_dataset, f=f, src_path=src_path, task_id=task_id),
            demos,
            max_workers=os.cpu_count() // 2,
            desc="Generating local dataset",
        )

    all_sub_dir_demo_desc = [task_name] * len(demos)

    for raw_dataset, episode_desc in zip(
        tqdm(raw_datasets, desc="Generating dataset from raw datasets"),
        all_sub_dir_demo_desc,
    ):
        for raw_dataset_sub in tqdm(
            raw_dataset[0], desc="Generating dataset from raw dataset"
        ):
            dataset.add_frame(raw_dataset_sub)
        dataset.save_episode(task=episode_desc, videos=raw_dataset[1])
    dataset.consolidate()


if __name__ == "__main__":
    # src_path = "/home/zimgong/Documents/datasets/v0.1"
    src_path = "datasets/v0.1"
    task_id = "single_stage/kitchen_coffee/CoffeePressButton"
    filter_key = "valid"
    tgt_path = "datasets/lerobot"
    debug = True

    # hdf5_file = None
    # for root, dirs, files in os.walk(f"{src_path}/{task_id}"):
    #     for file in files:
    #         if file == "demo_gentex_im128_randcams.hdf5":
    #             hdf5_file = os.path.join(root, file)
    #             break
    # hdf5_file = f"{src_path}/{task_id}/2024-04-25/demo_gentex_im128_randcams.hdf5"
    hdf5_file = f"{src_path}/{task_id}/2024-04-25/demo.hdf5"
    dataset_base = f"robocasa/{task_id}"

    assert hdf5_file, f"hdf5 file not found."
    main(src_path, tgt_path, task_id, dataset_base, hdf5_file, filter_key, debug)
