import os
import json
import time
import random

import h5py
import imageio
import numpy as np
from termcolor import colored

import robosuite
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset, reset_to


def playback_trajectory_with_env_multi_cam(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writers=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=512,
    camera_width=512,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writers (list of imageio writers): video writers
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = video_writers is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    if render is False:
        print(colored("Running episode...", "yellow"))

    for i in range(traj_len):
        start = time.time()

        if action_playback:
            env.step(actions[i])
            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = np.array(env.sim.get_state().flatten())
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    if verbose or i == traj_len - 2:
                        print(
                            colored(
                                "warning: playback diverged by {} at step {}".format(
                                    err, i
                                ),
                                "yellow",
                            )
                        )
        else:
            reset_to(env, {"states": states[i]})

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # video render
        if write_video:
            if video_count % video_skip == 0:
                for cam_name, video_writer in zip(camera_names, video_writers):
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]
                    video_writer.append_data(im)

            video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None


def playback_trajectory_with_obs_multi_cam(
    traj_grp,
    video_writers,
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
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            for image_name, video_writer in zip(image_names, video_writers):
                im = traj_grp["obs/{}".format(image_name + "_image")][i]
                video_writer.append_data(im)
        video_count += 1

        if first:
            break


def main(
    dataset: str,
    filter_key: str | None = None,
    n: int | None = None,
    use_obs: bool = False,
    use_actions: bool = False,
    use_abs_actions: bool = False,
    render: bool = False,
    video_skip: int = 5,
    render_image_names: list[str] | None = None,
    first: bool = False,
    extend_states: bool = False,
    debug: bool = False,
    camera_height: int = 512,
    camera_width: int = 512,
):
    # some arg checking
    write_video = render is not True
    src_path = os.path.dirname("/".join(dataset.split("/")[:-1]))

    # Auto-fill camera rendering info if not specified
    if render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=dataset)
        render_image_names = "robot0_agentview_center"

    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    if use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not use_actions and not use_abs_actions
        ), "playback with observations is offline and does not support action playback"

    env = None

    # create environment only if not playing back with observations
    if not use_obs:
        # # need to make sure ObsUtils knows which observations are images, but it doesn't matter
        # # for playback since observations are unused. Pass a dummy spec here.
        # dummy_spec = dict(
        #     obs=dict(
        #             low_dim=["robot0_eef_pos"],
        #             rgb=[],
        #         ),
        # )
        # initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=dataset)
        if use_abs_actions:
            env_meta["env_kwargs"]["controller_configs"][
                "control_delta"
            ] = False  # absolute action space

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = write_video
        env_kwargs["use_camera_obs"] = False

        if debug:
            print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )

        env = robosuite.make(**env_kwargs)

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

    # maybe reduce the number of demonstrations to playback
    if n is not None:
        random.shuffle(demos)
        demos = demos[:n]

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        src_ep_path = os.path.join(src_path, ep, "videos")
        if not os.path.exists(src_ep_path):
            os.makedirs(src_ep_path)
        else:
            raise ValueError("Episode {} already exists".format(ep))

        video_writers = []
        for camera in render_image_names:
            video_path = os.path.join(src_ep_path, "{}.mp4".format(camera))
            video_writer = imageio.get_writer(
                video_path, fps=20, codec="av1", pixelformat="yuv420p"
            )
            video_writers.append(video_writer)

        if use_obs:
            playback_trajectory_with_obs_multi_cam(
                traj_grp=f["data/{}".format(ep)],
                video_writers=video_writers,
                video_skip=video_skip,
                image_names=render_image_names,
                first=first,
            )
            for video_writer in video_writers:
                video_writer.close()
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        if extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            use_actions and use_abs_actions
        )  # cannot use both relative and absolute actions
        if use_actions:
            actions = f["data/{}/actions".format(ep)][()]
        elif use_abs_actions:
            actions = f["data/{}/actions_abs".format(ep)][()]  # absolute actions

        playback_trajectory_with_env_multi_cam(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            render=render,
            video_writers=video_writers,
            video_skip=video_skip,
            camera_names=render_image_names,
            first=first,
            verbose=debug,
            camera_height=camera_height,
            camera_width=camera_width,
        )
        for video_writer in video_writers:
            video_writer.close()

    f.close()
    print(colored("Saved videos to {}".format(src_path), "green"))

    if env is not None:
        env.close()


if __name__ == "__main__":
    # dataset = "/home/zimgong/Documents/datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams.hdf5"
    dataset = "datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo.hdf5"
    filter_key = "valid"
    # filter_key = None
    n = None
    use_obs = False
    use_actions = False
    use_abs_actions = False
    render = False
    video_skip = 5
    render_image_names = [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ]
    first = False
    extend_states = False
    debug = True
    camera_height = 512
    camera_width = 512
    main(
        dataset,
        filter_key,
        n,
        use_obs,
        use_actions,
        use_abs_actions,
        render,
        video_skip,
        render_image_names,
        first,
        extend_states,
        debug,
        camera_height,
        camera_width,
    )
