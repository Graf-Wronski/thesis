from dataclasses import dataclass
from typing import Dict, Type

from grecco_sim.controller import local_control

from .second_order import CoordinatorSecondOrder
from .central_opt import CentralOptimizationCoordinator
from .admm import CoordinatorADMM
from .first_order import CoordinatorVujanic
from .first_order import CoordinatorGradientDescent
from .first_order import CoordinatorDailyGridFee
from .uncoordinated import NoCentralControl

#make as sorted tuple
@dataclass
class CoordinationMechanism:
    """Data structure to specify the used coordination mechanism."""

    name: str
    controller_local: Dict[str, Type]
    coordinator: Type


CM_CENTRAL_OPTIMIZATION = CoordinationMechanism(
    "central_optimization",
    {
        "pv_bat": local_control.LocalControllerPassControl,
        "load": local_control.LocalControllerNoBat,
        "heatpump": local_control.LocalControllerNoBat,
    },
    CentralOptimizationCoordinator,
)

CM_VUJANIC = CoordinationMechanism("vujanic", {}, CoordinatorVujanic)
CM_ADMM = CoordinationMechanism(
    "admm",
    {
        "pv_bat": local_control_distr_opt.LocalControllerSecondOrder,
        "load": local_control.LocalControllerNoBat,
        "heatpump": local_control_distr_opt.LocalControllerSecondOrder,
    },
    CoordinatorADMM,
)

CM_GRADIENT_DESCENT = CoordinationMechanism(
    "gradient_descent",
    {
        "pv_bat": local_control_distr_opt.LocalControllerFirstOrder,
        "load": local_control.LocalControllerNoBat,
    },
    CoordinatorGradientDescent,
)

CM_DAILY_GRID_FEE = CoordinationMechanism(
    "plain_grid_fee",
    {
        "pv_bat": local_control_distr_opt.LocalControllerFirstOrder,
        "load": local_control.LocalControllerNoBat,
        "heatpump": local_control_distr_opt.LocalControllerFirstOrder,
        "ev": local_control_distr_opt.LocalControllerFirstOrder,
    },
    CoordinatorDailyGridFee)

# Quick Fix to incorporate MultiUnit Controller in GrECCo logic.
CM_MULTI_UNIT_GRID_FEE = CoordinationMechanism(
    "multi_unit_grid_fee",
    {},
    CoordinatorDailyGridFee)

CM_SECOND_ORDER = CoordinationMechanism(
    "second_order",
    {
        "pv_bat": local_control_distr_opt.LocalControllerSecondOrder,
        "load": local_control.LocalControllerNoBat,
        "heatpump": local_control_distr_opt.LocalControllerSecondOrder,
    },
    CoordinatorSecondOrder,
)

CM_NONE = CoordinationMechanism(
    "none",
    {sys_type: local_control.LocalControllerNoBat for sys_type in ["load", "pv_bat", "ev", "heatpump"]},
    NoCentralControl,
)

CM_LOCAL_SELF_SUFF = CoordinationMechanism(
    "local_self_suff",
    {
        "pv_bat": local_control.LocalControllerSelfSuff,
        "ev": local_control.LocalControllerEVBaseline,
        "load": local_control.LocalControllerNoBat,
        "heatpump": local_control.LocalControllerHeatPumpOnOff,
    },
    NoCentralControl,
)

# ToDo: Choose Either call objects 'coordination_mechanism' or 'coordinator'.
AVAILABLE_COORDINATORS = [
    CM_SECOND_ORDER,
    CM_CENTRAL_OPTIMIZATION,
    CM_VUJANIC,
    CM_ADMM,
    CM_GRADIENT_DESCENT,
    CM_NONE,
    CM_LOCAL_SELF_SUFF,
    CM_DAILY_GRID_FEE,
    CM_MULTI_UNIT_GRID_FEE,
]

# ToDo: Discuss compromise: No loop in grid_node.py, but also no extra method.
ALL_COORDINATORS = {
    "second_order": CoordinatorSecondOrder,
    "central_optimization": CentralOptimizationCoordinator,
    "vujanic": CoordinatorVujanic,
    "admm": CoordinatorADMM,
    "gradient_descent": CoordinatorGradientDescent,
    "none": NoCentralControl,
    "local_self_suff": NoCentralControl,
    "plain_grid_fee": CoordinatorDailyGridFee,
    "multi_unit_grid_fee": CoordinatorDailyGridFee,
}