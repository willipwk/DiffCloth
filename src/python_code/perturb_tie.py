import argparse
import os

import diffcloth_py as diffcloth
import numpy as np
import torch
from tqdm import tqdm
from tulip.utils.io_utils import stdout_redirected

import common
from pySim.pySim import pySim

parser = argparse.ArgumentParser("Perturb flat cloth")
parser.add_argument(
    "-mode",
    type=int,
    default=0,
    choices=[0, 1, 2],
    help="0: fixed attachment index, 1: random index along the middle line, 2: keypoints along the middle line",
)
parser.add_argument("-task_name", type=str, default="perturb_tie")
parser.add_argument("-in_fn", type=str, required=True)
parser.add_argument("-out_fn", type=str, required=True)
parser.add_argument("-att_idx", type=int, default=-1)
parser.add_argument("-n_openmp_thread", type=int, default=4)
parser.add_argument("-seed", type=int, default=8824325)
parser.add_argument("-r", dest="render", action="store_true", default=False)
parser.add_argument("-s", dest="save", action="store_true", default=True)
args = parser.parse_args()


def get_state(sim: diffcloth.Simulation, to_tensor: bool = False) -> tuple:
    state_info_init = sim.getStateInfo()
    x, v = state_info_init.x, state_info_init.v
    clip_pos = np.array(sim.getStateInfo().x_fixedpoints)
    if to_tensor:
        x_t = common.toTorchTensor(x, False, False).clone()
        v_t = common.toTorchTensor(v, False, False).clone()
        a_t = common.toTorchTensor(clip_pos, False, False).clone()
        return x_t, v_t, a_t
    else:
        return x, v, clip_pos


def parse_v_middle(start_idx, end_idx, distance):
    v_middle = [start_idx, end_idx]
    while True:
        if v_middle[-2] + distance < v_middle[-1]:
            v_middle.insert(-1, v_middle[-2] + distance)
        else:
            break
    return v_middle


def perturb(args):
    # Parsing attachment index
    v_middle = parse_v_middle(start_idx=2, end_idx=364, distance=3)
    if args.mode == 0:
        assert args.att_idx in v_middle, "Invalid control vertex index."
        att_idx = args.att_idx
    elif args.mode == 1:
        att_idx = np.random.choice(v_middle)
    else:
        att_idx = -1
    print(f"==> Selecting control point to vertex {att_idx}")

    # Initialize the scene
    helper = diffcloth.makeOptimizeHelper(args.task_name)
    sim = diffcloth.makeSim(
        exampleName=args.task_name,
        objFilename=args.in_fn,
        runBackward=False,
        customAttachmentVertexIdx=att_idx,
    )
    sim.forwardConvergenceThreshold = 1e-8
    pysim = pySim(sim, helper, True)

    # Reset the system
    sim.resetSystem()
    x0_t, v0_t, a0_t = get_state(sim, to_tensor=True)

    # Start perturb the tie
    x_t, v_t, a_t = get_state(sim, to_tensor=True)
    steps = sim.sceneConfig.stepNum
    change_thresh = 10
    actions = 4

    a_control_list = []
    kp_num = a_t.shape[0] // 21
    for j in range(actions):
        a_control_ep_list = []
        for k in range(kp_num):
            a_control_unit_k = (torch.rand(3) - 0.5) * 0.2
            a_control_ep_k = a_control_unit_k.repeat((steps // actions, 7))
            a_control_ep_list.append(a_control_ep_k)
        a_control_ep = torch.hstack(a_control_ep_list)
        a_control_list.append(a_control_ep)
    a_control = torch.vstack(a_control_list)
    
    xvPairs = common.forwardSimulation2(sim, x0_t.clone(), v0_t.clone(), a0_t.clone(), a_control, pysim)
    x_init = xvPairs[0][0]
    x_final = xvPairs[steps][0]
    change = torch.sum((x_final - x_init) ** 2)
    with open("log.txt", "a+") as f:
        f.write(args.out_fn + " change: " + str(change) + "\n")
    # Rendering the simulationg
    if args.render:
        diffcloth.render(sim, renderPosPairs=True, autoExit=True)
    # save perturbation results
    if args.save and change < change_thresh:
        out_folder = "output/" + args.out_fn[:args.out_fn.rfind("/")]
        os.makedirs(out_folder, exist_ok=True)
        # Export final configuration into wavefront file
        sim.exportCurrentMeshPos(steps, args.out_fn.replace(".obj", ""))

    del sim, pysim


if __name__ == "__main__":

    diffcloth.enableOpenMP(n_threads=args.n_openmp_thread)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert args.out_fn.endswith(".obj"), "Please provide a valid filename."
    if os.path.isfile(f"output/{args.out_fn}"):
        print(f"{args.out_fn} exists. Skip...")
    else:
        try:
            with stdout_redirected():
                perturb(args)
        except:
            print("Error encountered. Retry....")
