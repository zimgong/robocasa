import os
import json
from functools import partial

import h5py
import imageio
import numpy as np
from termcolor import colored


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
            video_count += 1

        if first:
            break


def main(
    dataset: str,
    filter_key: str | None = None,
    use_obs: bool = False,
    debug: bool = False,
):
    render_image_names = [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ]

    f = h5py.File(dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if filter_key is not None:
        print("using filter key: {}".format(filter_key))
        demos = [
            elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    src_path = os.path.dirname("/".join(dataset.split("/")[:-1]))

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        src_ep_path = os.path.join(src_path, ep, "videos")
        if not os.path.exists(src_ep_path):
            os.makedirs(src_ep_path)
        else:
            raise ValueError("Episode {} already exists".format(ep))

        for camera in render_image_names:
            video_path = os.path.join(src_ep_path, "{}.mp4".format(camera))
            video_writer = imageio.get_writer(
                video_path, fps=20, codec="av1", pixelformat="yuv420p"
            )

            if use_obs:
                playback_trajectory_with_obs(
                    traj_grp=f["data/{}".format(ep)],
                    video_writer=video_writer,
                    video_skip=1,
                    image_names=[camera],
                    first=False,
                )
                video_writer.close()
                continue

    f.close()
    print(colored("Saved videos to {}".format(src_path), "green"))


if __name__ == "__main__":
    dataset = "datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams.hdf5"
    filter_key = "valid"
    use_obs = True
    debug = True

    main(dataset, filter_key, use_obs, debug)
