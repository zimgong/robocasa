import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Callable
from functools import partial

import h5py
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


AGENTVIEW_LEFT = "robot0_agentview_left.mp4"
AGENTVIEW_RIGHT = "robot0_agentview_right.mp4"
EYE_IN_HAND = "robot0_eye_in_hand.mp4"

FEATURES = {
    "observation.images.agentview_left": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 25.0,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.agentview_right": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 25.0,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.eye_in_hand": {
        "dtype": "video",
        "shape": [224, 224, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 25.0,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [13],
        "names": [
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_yaw",
            "mobilebase0_joint_torso_height",
            "robot0_joint1",
            "robot0_joint2",
            "robot0_joint3",
            "robot0_joint4",
            "robot0_joint5",
            "robot0_joint6",
            "robot0_joint7",
            "gripper0_right_finger_joint1",
            "gripper0_right_finger_joint2",
        ]
    },
    "action": {
        "dtype": "float32",
        "shape": [13],
        "names": [
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_yaw",
            "mobilebase0_joint_torso_height",
            "robot0_joint1",
            "robot0_joint2",
            "robot0_joint3",
            "robot0_joint4",
            "robot0_joint5",
            "robot0_joint6",
            "robot0_joint7",
            "gripper0_right_finger_joint1",
            "gripper0_right_finger_joint2",
        ]
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


def load_local_dataset(demo_and_file: tuple, src_path: str, task_id: str, video_skip: int, format: str) -> list | None:
    demo_id, hdf5_file = demo_and_file
    f = h5py.File(hdf5_file, "r")
    if format == "robocasa":
        ob_dir = Path("/".join(hdf5_file.split("/")[:-2])) / f"{demo_id}"
    elif format == "lightwheel":
        demo_dir = hdf5_file.split("/")[-2]
        ob_dir = Path(src_path) / f"{task_id}" / f"{demo_dir}"

    # Note: the state is downsampled by video_skip
    state_pos = np.array(f["data/{}/states".format(demo_id)][::video_skip, 1:14])
    # demo_dim = f["data/{}/states".format(demo_id)].shape[1]
    # vel_start_idx = int((demo_dim - 1) / 2 + 2)
    # state_vel = np.array(
    #     f["data/{}/states".format(demo_id)][::video_skip, vel_start_idx : vel_start_idx + 13]
    # )
    states_value = state_pos.astype(np.float32)
    # states_value = np.hstack([state_pos, state_vel]).astype(np.float32)
    demo_len = state_pos.shape[0]
    action_value = np.zeros((demo_len, 13))
    action_value[:demo_len-1, :] = np.array(
        f["data/{}/states".format(demo_id)][::video_skip, 1:14][1:]
    ).astype(np.float32)
    action_value[-1, :] = action_value[-2, :]

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


def convert_robocasa(args):
    repo_id = f"robocasa/{args.src_path.split('/')[-1]}"

    if os.path.exists(f"{args.tgt_path}/{repo_id}"):
        raise ValueError(f"Dataset {repo_id} already exists")

    dataset = RoboCasaDataset.create(
        repo_id=repo_id,
        root=f"{args.tgt_path}/{repo_id}",
        fps=25,
        robot_type="PandaOmron",
        features=FEATURES,
    )

    hdf5_files = []
    for root, dirs, files in os.walk(args.src_path):
        for dir in dirs:
            if not dir.startswith("demo"):
                files = os.listdir(os.path.join(root, dir))
                hdf5_files.extend([os.path.join(root, dir, f) for f in files if f.endswith(".hdf5")])
                break
    hdf5_files.sort()

    for hdf5_file in tqdm(hdf5_files):
        f = h5py.File(hdf5_file, "r")
        if args.filter_key is not None:
            # use the demonstrations from the filter key instead
            print("NOTE: using filter key {}".format(args.filter_key))
            demos = sorted(
                [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
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

        task_name = get_task_instruction(f, demos)
        task_id = hdf5_file.split("/")[-3]

        if args.debug:
            raw_datasets = [
                load_local_dataset((demo, hdf5_file), args.src_path, task_id, args.video_skip, args.format) for demo in tqdm(demos)
            ]
        else:
            raw_datasets = process_map(
                partial(load_local_dataset, src_path=args.src_path, task_id=task_id, video_skip=args.video_skip, format=args.format),
                [(demo, hdf5_file) for demo in demos],
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


def convert_lightwheel(args):
    repo_id = f"robocasa/{args.src_path.split('/')[-1]}"

    if os.path.exists(f"{args.tgt_path}/{repo_id}"):
        raise ValueError(f"Dataset {repo_id} already exists")

    dataset = RoboCasaDataset.create(
        repo_id=repo_id,
        root=f"{args.tgt_path}/{repo_id}",
        fps=25,
        robot_type='PandaOmron',
        features=FEATURES,
    )

    task_ids = sorted(os.listdir(args.src_path))
    demo = "demo_1"

    for task_id in task_ids[:args.n]:
        episode_ids = sorted(os.listdir(os.path.join(args.src_path, task_id)))
        hdf5_files = [os.path.join(args.src_path, task_id, episode_id, "ep_demo.hdf5") for episode_id in episode_ids]

        f = h5py.File(hdf5_files[0], "r")
        task_name = get_task_instruction(f, [demo])

        if args.debug:
            raw_datasets = [
                load_local_dataset((demo, hdf5_file), args.src_path, task_id, args.video_skip, args.format) for hdf5_file in tqdm(hdf5_files)
            ]
        else:
            raw_datasets = process_map(
                partial(load_local_dataset, src_path=args.src_path, task_id=task_id, video_skip=args.video_skip, format=args.format),
                [(demo, hdf5_file) for hdf5_file in hdf5_files],
                max_workers=os.cpu_count() // 2,
                desc="Generating local dataset",
            )

        all_sub_dir_demo_desc = [task_name] * len(hdf5_files)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        default="datasets/v0.1",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tgt_path", 
        type=str,
        default="datasets/lerobot",
    )
    parser.add_argument(
        "--tgt_repo_id",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="robocasa",
    )
    args = parser.parse_args()

    if args.format == "robocasa":
        convert_robocasa(args)
    elif args.format == "lightwheel":
        convert_lightwheel(args)
