from typing import Mapping, NoReturn, Optional, Sequence, Tuple, Union

import diffcloth_py as diffcloth
import torch
import torch.nn as nn
from torch import Tensor

from .functional import SimFunction


class pySim(nn.Module):
    def __init__(
        self,
        cppSim: diffcloth.Simulation,
        optimizeHelper: diffcloth.OptimizeHelper,
        useFixedPoint: bool,
    ) -> NoReturn:
        super().__init__()
        self.cppSim = cppSim
        self.optimizeHelper = optimizeHelper

        self.cppSim.useCustomRLFixedPoint = useFixedPoint

    def forward(self, x: Tensor, v: Tensor, a: Tensor) -> Tuple[Tensor, Tensor]:

        return SimFunction.apply(x, v, a, self.cppSim, self.optimizeHelper)
