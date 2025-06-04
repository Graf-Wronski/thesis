from dataclasses import dataclass
from typing import Dict, Type

from grecco_sim.controller import local_control
from grecco_sim.controller import local_control_distr_opt


from .coord_second_order import CoordinatorSecondOrder
from .coord_central_opt import CentralOptimizationCoordinator
from .coord_admm import CoordinatorADMM
from .coord_first_order import CoordinatorVujanic
from .coord_first_order import CoordinatorGradientDescent
from .coord_first_order import CoordinatorDailyGridFee
from .coord_base import NoCentralControl


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
    },
    CoordinatorDailyGridFee,
)

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
        "heatpump": local_control.LocalControllerNoBat,
    },
    NoCentralControl,
)


AVAILABLE_COORDINATORS = [
    CM_SECOND_ORDER,
    CM_CENTRAL_OPTIMIZATION,
    CM_VUJANIC,
    CM_ADMM,
    CM_GRADIENT_DESCENT,
    CM_NONE,
    CM_LOCAL_SELF_SUFF,
    CM_DAILY_GRID_FEE,
]
