#!/usr/bin/env python3
"""Visualize 3D SMPL joints from a WHAM ``wham_output.json`` file.

This script loads a WHAM result JSON, reconstructs SMPL joints for a
chosen subject and frame and renders them as a 3D skeleton using
matplotlib. Left body parts are shown in red and right parts in black.
If joint coordinates are already present in the JSON, they are used
directly, otherwise an SMPL model is required to compute them. No CUDA is
required.

Example usage:
    python visualize_wham_pose.py --json wham_output.json \
        --model-folder /path/to/smpl_models --subject 0 --frame 0 \
        --output frame.png
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

# Skeleton definition for the 24 SMPL joints
SMPL_SKELETON = np.array([
    [0, 1], [0, 2], [0, 3],
    [1, 4], [2, 5], [3, 6],
    [4, 7], [5, 8], [6, 9],
    [7, 10], [8, 11], [9, 12],
    [9, 13], [9, 14],
    [12, 15], [13, 16], [14, 17],
    [16, 18], [17, 19], [18, 20],
    [19, 21], [20, 22], [21, 23],
])

LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT_JOINTS = {2, 5, 8, 11, 14, 17, 19, 21, 23}

def main(args: argparse.Namespace) -> None:
    with open(args.json, 'r') as f:
        data = json.load(f)

    subj_id = str(args.subject) if args.subject is not None else list(data.keys())[0]
    record = {k: np.asarray(v) for k, v in data[subj_id].items()}
    frame = args.frame

    if 'joints' in record:
        joints = record['joints'][frame, :24]
    else:
        if args.model_folder is None:
            raise ValueError('model-folder must be specified when joints are absent')
        import torch
        import smplx

        pose = torch.from_numpy(record['pose'][frame]).float().unsqueeze(0)
        betas = torch.from_numpy(record['betas']).float().unsqueeze(0)
        trans = torch.from_numpy(record['trans'][frame]).float().unsqueeze(0)

        global_orient = pose[:, :3]
        body_pose = pose[:, 3:]

        smpl = smplx.create(args.model_folder, model_type='smpl',
                            gender='neutral', ext='pkl', batch_size=1)
        output = smpl(global_orient=global_orient,
                      body_pose=body_pose,
                      betas=betas,
                      transl=trans)

        joints = output.joints[0, :24].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['k'] * len(joints)
    for idx in LEFT_JOINTS:
        colors[idx] = 'r'
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=colors)

    for i, j in SMPL_SKELETON:
        if i in LEFT_JOINTS and j in LEFT_JOINTS:
            c = 'r'
        elif i in RIGHT_JOINTS and j in RIGHT_JOINTS:
            c = 'k'
        else:
            c = 'k'
        ax.plot(joints[[i, j], 0], joints[[i, j], 1], joints[[i, j], 2], c=c)

    # coordinate axes
    axis_len = 0.5
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r')
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g')
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=-90)
    ax.invert_yaxis()
    ax.set_title(f'Subject {subj_id} frame {frame}')
    plt.savefig(args.output)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D pose from WHAM output.')
    parser.add_argument('--json', type=str, default='wham_output.json',
                        help='Path to wham_output.json')
    parser.add_argument('--model-folder', type=str, default=None,
                        help='Directory containing SMPL model files')
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject id to visualize (default: first one)')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to visualize')
    parser.add_argument('--output', type=str, default='pose.png',
                        help='Path to save the rendered image')
    main(parser.parse_args())
