import argparse
import random
from datetime import datetime

import numpy as np
import torch


def parseInputs():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--eval", action="store_true", help="run model evaluation"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Enable rendering of sequences",
    )
    parser.add_argument(
        "--train_resume", action="store_true", help="whether to resume training"
    )
    parser.add_argument(
        "--load_expname",
        type=str,
        default="20210809-trainedBest",
        help="if evaluating or resuming training, the name of the experiment to load",
    )
    parser.add_argument("--randSeed", type=int, default=2)
    parser.add_argument(
        "--epochNum", type=int, default=200, help="Number of epochs to train"
    )
    parser.add_argument(
        "--load_epoch",
        type=str,
        default="testBestEpoch",
        help="if evaluating or resuming training, the epoch of the experiment to load",
    )

    args = parser.parse_args()
    return args


def setRandomSeed(randSeed):
    torch.manual_seed(randSeed)
    random.seed(randSeed)
    np.random.seed(randSeed)


def setRandomLoadEval(isRender, isLoad, isEval):
    global renderOn, loadFromCheckPoint, evalMode
    renderOn = isRender
    loadFromCheckPoint = isLoad
    evalMode = isEval


def getTimeString():
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H%M%S-")
    return dt_string


def toNumpy(x):
    return x.contiguous().detach().cpu().numpy()


def toTorchTensor(x, requriesGrad=False, toDouble=False):
    torchX = torch.Tensor(x)
    if toDouble:
        torchX = torchX.double()
    torchX = torchX.view(-1).clone().detach().requires_grad_(requriesGrad)
    return torchX


def getTorchVectors(x0, v0, fixedPointInitPoses, targetShape):
    toDouble = False
    fixedPointInitDistance = fixedPointInitPoses[3:6] - fixedPointInitPoses[0:3]
    x0_torch = toTorchTensor(x0, False, toDouble)
    v0_torch = toTorchTensor(v0, False, toDouble)
    a_torch = toTorchTensor(fixedPointInitPoses, True, toDouble)
    a0_torch = toTorchTensor(fixedPointInitPoses, False, toDouble)
    fxiedPointInitDist_torch = toTorchTensor(
        fixedPointInitDistance, False, toDouble
    )
    targetshape_torch = toTorchTensor(targetShape, False, toDouble)
    CLIP_REST_DIST = np.linalg.norm(fixedPointInitDistance, 2)
    return (
        x0_torch,
        v0_torch,
        a_torch,
        a0_torch,
        targetshape_torch,
        fxiedPointInitDist_torch,
        CLIP_REST_DIST,
    )


def forwardSimulation(
    sim, x_i, v_i, a_torch, getStateFunc, controller, simModule
):
    records = []
    vMin, vMax = -0.1, 0.1
    for step in range(sim.sceneConfig.stepNum):
        records.append((x_i, v_i))
        state = getStateFunc(x_i, v_i)
        controllerOut = controller(state)[0, :]
        torch.clamp(controllerOut, min=-1.0, max=1.0)
        delta_a_torch = (controllerOut + 1.0) / 2.0 * (vMax - vMin) + vMin

        # delta_a_torch = controllerOut
        if torch.any(torch.isnan(delta_a_torch)):
            print(
                "NaN encountered for action in step {}: {}".format(
                    step, delta_a_torch
                )
            )
            input("wait")
        a_torch = a_torch + delta_a_torch
        x_i, v_i = simModule(x_i, v_i, a_torch)
    records.append((x_i, v_i))
    return records


def forwardSimulation2(sim, x_i, v_i, a_torch, a_control, simModule):
    records = []
    vMin, vMax= -0.1, 0.1
    key_vertex = sim.sceneConfig.customAttachmentVertexIdx[0][1]
    for step in range(sim.sceneConfig.stepNum): # sim.sceneConfig.stepNum
        # print("step", step, "in total step", sim.sceneConfig.stepNum)
        records.append((x_i, v_i))
        controllerOut = a_control[step]
        controllerOut = torch.clamp(controllerOut, min=-0.1, max=0.1)
        delta_a_torch = (controllerOut + 1.) / 2. * (vMax - vMin) + vMin

        if torch.any(torch.isnan(delta_a_torch)):
            print("NaN encountered for action in step {}: {}".format(step, delta_a_torch))
            input("wait")
        a_torch = a_torch + delta_a_torch

        attach_point_pos = torch.cat([x_i[i*3:(i+1)*3] for i in key_vertex], dim=0)
        a_torch = torch.max(torch.min(a_torch, attach_point_pos+0.1), attach_point_pos-0.1)
        x_i, v_i = simModule(x_i, v_i,  a_torch)
    records.append((x_i, v_i))
    return records
