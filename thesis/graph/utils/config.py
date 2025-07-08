from dataclasses import dataclass
from typing import Optional
from grecco_sim.util import configs


@dataclass
class SamplerConfiguration:
    p_heatpump: float = 0.2
    p_pv: float = 0.5
    transformer_type: str = "0.25 MVA 20/0.4 kV"
    s_nom: Optional[float] = .05
    seed: Optional[int] = None

@dataclass
class PushRelabelConfiguration:
    sim_config: configs.SimulationConfiguration

    verbose: bool = False
    slack_as_source: bool = True
    conversion_order: int = 2
    dt_h: float = 0.25
    max_runtime: float = 1500 # The maximal runtime in seconds.

@dataclass
class BuilderConfig:
    push_relabel_config: PushRelabelConfiguration
