import abc
from grecco_sim.controller import mycas


class LocalSolverBase(abc.ABC):
    """Base class for the physical optimization model formulation."""
    @abc.abstractmethod
    def get_central_problem_contribution(
        self,
    ) -> tuple[tuple[mycas.casadi.SX, list[mycas.MySX], list[mycas.MyConstr], dict[str, mycas.MyPar]], mycas.MySX]:
        pass


    def get_user_functions(self) -> dict[str, mycas.casadi.SX]:
        """Extend with user functions. (See paper which user functions are necessary.)"""
        return {}


