from typing import cast, Iterable
import numpy as np
from numpy.typing import NDArray


import cobra
from cobra import Model, Reaction
from cobra.util.solver import linear_reaction_coefficients


from .analyze import DiatomAnalyze
from .grid import DiatomGrid
from .plot import DiatomPlot
from scripts.model_io import ModelIO, load_model
from scripts.model_clustering import ModelClustering


Numerical = int | float


class Diatom():
    """
    Class for handling a diatom metabolic model and its analysis pipeline.

    This class centralizes access to the COBRA model and orchestrates grid sampling,
    qualitative and quantitative analyses, clustering, plotting, and I/O utilities.

    Parameters
    ----------
    model_id : str
        Identifier of the metabolic model to load.
    model_name : str, default="diatom"
        Name used for outputs and saved artifacts.
    solver : str, default="gurobi"
        Linear solver backend to use with COBRApy.

    Attributes
    ----------
    model : cobra.Model
        Loaded COBRA metabolic model.
    objectives : dict[str, float]
        Mapping from reaction IDs to objective coefficients.
    non_blocked : set[str]
        Set of reaction IDs that are not blocked.
    grid : DiatomGrid
        Grid sampling and polytope discretization utilities.
    analyze : DiatomAnalyze
        Qualitative and quantitative FVA analysis utilities.
    clustering : ModelClustering
        Grid-point clustering and cluster-level analysis utilities.
    plot : DiatomPlot
        Visualization utilities.
    io : ModelIO
        Input/output utilities for saving and loading results.
    """
    def __init__(self, model_id: str, model_name: str = "diatom", solver: str = "gurobi"):
        self.model_id = model_id
        self.model_name = model_name
                                 
        self.model: Model = load_model(model_id, name = model_name, solver=solver)
        self.objectives: dict[str, float] = {}
        self.non_blocked: set[str] = set()

        self.grid = DiatomGrid(self)
        self.analyze = DiatomAnalyze(self)
        self.plot = DiatomPlot(self)
        self.clustering = ModelClustering(self)
        self.io = ModelIO(self, model_name)
        

    def set_objective_functions(self, objective_reactions_dict: dict[str, float] | None = None) -> None:
        """Set the objective function of the model.

        If no objective dictionary is provided, the current objective coefficients
        defined in the model are read and stored. Otherwise, the objective is replaced
        by the provided reaction–coefficient mapping.

        Parameters
        ----------
        objective_reactions_dict : dict[str, float] or None, default=None
            Mapping from reaction IDs to linear objective coefficients.
            If None or empty, the existing model objective is used.
        """
        model = self.model

        # use predefined objective functions
        if objective_reactions_dict is None or len(objective_reactions_dict) == 0:
            linear_coeffs = linear_reaction_coefficients(model)
            self.objectives = {reaction.id: coeff for reaction, coeff in linear_coeffs.items()} # should be single key,value dict, could hold more   
            return
        
        # set new objective functions
        for reaction_id, coeff in objective_reactions_dict.items():
            self.objectives[reaction_id] = coeff

        model.objective = {model.reactions.get_by_id(r): coeff for r, coeff in self.objectives.items()}  

        print(model.objective) 

    
    def modify_bounds(self, bounds_dict: dict[str, tuple[Numerical, Numerical]]) -> None:
        """Modify reaction bounds in the model.

        Parameters
        ----------
        bounds_dict : dict[str, tuple[float, float]]
            Mapping from reaction IDs to (lower_bound, upper_bound).
        """
        for reaction_id, bounds in bounds_dict.items():
            reaction = cast(Reaction, self.model.reactions.get_by_id(reaction_id))
            reaction.bounds = bounds


    def _set_non_blocked_reactions(self) -> None:
        """Identify and store all non-blocked reactions in the model."""

        blocked = cobra.flux_analysis.find_blocked_reactions(self.model)
        reactions = cast(Iterable[Reaction], self.model.reactions)
        all_ids = [reaction.id for reaction in reactions]
        non_blocked = set(all_ids).difference(set(blocked))
        self.non_blocked = non_blocked


    def fix_growth_rates(self, mirror_model: Model, grid_point: NDArray[np.floating]) -> None:
        """Fix the biomass production rate of each community member.

        This method constrains the biomass reaction of each member to a fixed growth rate by setting its 
        lower and upper bounds to the same value.

        IMPORTANT: This method must be called *only* inside a model context manager, e.g.::

            with community_model:
                community.apply_member_fraction_bounds(community_model, fractions)
                
        Otherwise, reaction bounds will be permanently modified.

        Parameters
        ----------
        mirror_community_model : cobra.Model
            Community model whose biomass reaction bounds will be temporarily constrained.

        member_mu : np.ndarray, shape (n_members,)
            Fixed biomass production rates for each community member. Each value is imposed as an equality 
            constraint: v_biomass_i = member_mu[i]
        """

        # change bounds for each objective reaction
        for index, reaction_id in enumerate(self.analyze.analyzed_reactions): # member_objectives should be single key dictionary
            value = grid_point[index]
            reaction = cast(Reaction, mirror_model.reactions.get_by_id(reaction_id))
            reaction.bounds = (value, value)    


    def _require(
        self, 
        polytope: bool = False, 
        grid_points: bool = False, 
        clusters: bool = False, 
        qual_vector: bool = False, 
        qfca: bool = False,
    ) -> None:
        """Internal consistency check for required analysis stages.

        Raises a RuntimeError if a requested artifact has not been computed yet.
        """
        if polytope and self.analyze.polytope.is_empty:
            raise RuntimeError(f"Projected polytope not yet computed. Run {self.analyze.project_polytope_2d.__name__} first!")

        if grid_points and self.grid.points.size == 0:
            raise RuntimeError(f"Grid points not yet computed. Run {self.grid.sample_polytope.__name__} first!")
        
        if qual_vector and self.analyze.qual_vector.empty:
            raise RuntimeError(f"Qualitative FVA values not yet computed. Run {self.analyze.qualitative_analysis.__name__} first!")

        if clusters and self.clustering.grid_clusters.size == 0:
            raise RuntimeError(f"Clusters not yet computed. Run {self.clustering.set_grid_clusters.__name__} first!")

        if qfca and self.analyze.qFCA.empty:
            raise RuntimeError(f"qFCA not yet computed. Run {self.analyze.quan_FCA.__name__} first!")

