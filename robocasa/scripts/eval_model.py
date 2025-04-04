import argparse
import json
import os
import random
import time
import pickle
import torch

import h5py
import imageio
import numpy as np
import robosuite
from termcolor import colored

import robocasa

from lerobot.common.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig


def get_task_instruction(f: h5py.File, task_demos: list[str]) -> dict:
    """Get task language instruction"""
    ep = task_demos[0]
    ep_meta = json.loads(f["data/{}".format(ep)].attrs["ep_meta"])
    task_instruction = ep_meta["lang"]
    print(f"Get Task Instruction <{task_instruction}>")
    return task_instruction


def eval_model_with_env(
    env,
    initial_state,
    states,
    policy,
    task_name,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
    camera_height=512,
    camera_width=512,
    mode="replay",
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        policy (instance of Policy): policy to use for evaluation
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    if mode != "replay":
        device = policy.config.device
    else:
        device = None
    write_video = video_writer is not None
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

    if render is False:
        print(colored("Running episode...", "yellow"))

    obs, _, _, _ = env.step(np.zeros(env.action_dim))
    for i in range(5 * traj_len - 1):
        start = time.time()

        if mode != "replay":
            policy_obs = {}
            policy_obs["observation.images.agentview_left"] = (
                torch.from_numpy(obs["robot0_agentview_left_image"].transpose(2, 0, 1))
                .unsqueeze(0)
                .to(device)
                .to(torch.float32)
            )
            policy_obs["observation.images.agentview_right"] = (
                torch.from_numpy(obs["robot0_agentview_right_image"].transpose(2, 0, 1))
                .unsqueeze(0)
                .to(device)
                .to(torch.float32)
            )
            policy_obs["observation.images.robot0_eye_in_hand"] = (
                torch.from_numpy(obs["robot0_eye_in_hand_image"].transpose(2, 0, 1))
                .unsqueeze(0)
                .to(device)
                .to(torch.float32)
            )
            policy_obs["observation.state"] = [0, 0, 0, 0]
            policy_obs["observation.state"].extend(obs["robot0_joint_pos"])
            policy_obs["observation.state"].extend(obs["robot0_gripper_qpos"])
            policy_obs["observation.state"] = (
                torch.tensor(policy_obs["observation.state"], device=device)
                .unsqueeze(0)
                .to(torch.float32)
            )
            policy_obs["task"] = [task_name]
            action_policy = policy.select_action(policy_obs).flatten().cpu().numpy()
            action = []
            action.extend(action_policy[4:])
            action.extend([0, 0, 0, 0])
            action = np.array(action, dtype=np.float32)
        else:
            action = []
            if i + 1 < traj_len:
                action.extend(states[i + 1, 5:14])
            else:
                action.extend(states[-1, 5:14])
            action.extend([0, 0, 0, 0])
            action = np.array(action, dtype=np.float32)
        # action = np.zeros(env.action_dim)
        obs, _, _, _ = env.step(action)
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
                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)

            video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def eval_model(args):
    if args.mode == "replay":
        policy = None
    else:
        # load model
        device = "cuda"
        # ckpt_torch_dir = "lerobot/pi0"
        ckpt_torch_dir = "/data/ceph_hdd/main/dev/shaoze.yang/code_ly/openpi/checkpoints/pi0_0330_30000_pytorch"

        cfg = PreTrainedConfig.from_pretrained(ckpt_torch_dir)
        cfg.pretrained_path = ckpt_torch_dir
        dataset_meta = pickle.load(
            open(
                "/data/ceph_hdd/main/datasets/lerobot/robocasa/v0.1/meta/meta.pkl", "rb"
            )
        )
        policy = make_policy(cfg, ds_meta=dataset_meta)

    # some arg checking
    write_video = args.render is not True
    if args.video_path is None:
        args.video_path = args.dataset.split(".hdf5")[0] + ".mp4"
        if args.use_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_actions.mp4"
        elif args.use_abs_actions:
            args.video_path = args.dataset.split(".hdf5")[0] + "_use_abs_actions.mp4"
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = "robot0_agentview_center"

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    env = None

    env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.use_abs_actions:
        env_meta["env_kwargs"]["controller_configs"][
            "control_delta"
        ] = False  # absolute action space

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = write_video
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_heights"] = 224
    env_kwargs["camera_widths"] = 224

    env_kwargs[
        "controller_configs"
    ] = robosuite.controllers.load_composite_controller_config(
        controller="/data/ceph_hdd/main/dev/zim.gong/robosuite/robosuite/controllers/config/robots/custom_pandaomron.json"
    )

    if args.verbose:
        print(
            colored(
                "Initializing environment for {}...".format(env_kwargs["env_name"]),
                "yellow",
            )
        )

    env = robosuite.make(**env_kwargs)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    elif "data" in f.keys():
        demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[: args.n]

    task_name = get_task_instruction(f, demos)

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        if args.extend_states:
            states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = None
        assert not (
            args.use_actions and args.use_abs_actions
        )  # cannot use both relative and absolute actions
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]
        elif args.use_abs_actions:
            actions = f["data/{}/actions_abs".format(ep)][()]  # absolute actions

        eval_model_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            policy=policy,
            task_name=task_name,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
            verbose=args.verbose,
            camera_height=args.camera_height,
            camera_width=args.camera_width,
            mode=args.mode,
        )

    f.close()
    if write_video:
        print(colored(f"Saved video to {args.video_path}", "green"))
        video_writer.close()

    if env is not None:
        env.close()


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Playback stored dataset absolute actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-abs-actions",
        action="store_true",
        help="use open-loop action playback with absolute position actions instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--extend_states",
        action="store_true",
        help="play last step of episodes for 50 extra frames",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=512,
        help="(optional, for offscreen rendering) width of image observations",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="replay",
        help="mode to use",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_eval_args()
    eval_model(args)
