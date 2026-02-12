from typing import TYPE_CHECKING, Literal
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from cobra import Model
import cobra.io

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem
    from diatom.diatom import Diatom


MODEL_DIR = "models"
SAVE_POINTS_DIR = "models/points"


def load_model(model_name: str, model_directory: str = MODEL_DIR, solver: str = 'gurobi', **kwargs) -> Model:
    """Loads a COBRA model from an SBML file using the specified solver.

    Parameters
    ----------
    model_name : str
        SBML filename (relative to `model_directory`).
    model_directory : str, default=MODEL_DIR
        Directory containing SBML models.
    solver : str, default='gurobi'
        Solver backend to attach to the model.
    **kwargs
        Additional arguments passed to `cobra.io.read_sbml_model`.

    Returns
    -------
    cobra.Model
        Loaded COBRA model with configured solver.
    """
    path = Path(model_directory) / model_name
    model = cobra.io.read_sbml_model(path, solver=solver, **kwargs)
    model.solver.configuration.threads = 0

    return model 


def save_models(model_dict: dict[str, Model], model_directory: str = MODEL_DIR) -> None:
    """Saves all COBRA models in `model_dict` to `model_directory`."""    
    output_dir = Path(model_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in model_dict.items():
        filename = output_dir / f"{model_name}.xml"
        cobra.io.write_sbml_model(model, filename)
        print(f'model {model_name} stored')


class ModelIO():
    """Handle filesystem I/O for analysis results associated with a model instance.

    This class is responsible for:
    - constructing directory.
    - saving and loading per-grid-point results and clustering data.
    - exporting dataframes and plots.
    """
    def __init__(self, modelclass: "BaseEcosystem | Diatom", model_name: str):
        self.modelclass = modelclass
        self.model_name: str = model_name


    @property
    def analyzed_tuple_string(self) -> str:
        """String identifier for the current analyzed reaction tuple.
        
        Raises an assertion error if analyzed reactions have not been set."""
        reaction1, reaction2 = self.modelclass.analyze.analyzed_reactions
        assert not (reaction1 == "" or reaction2 == "")
        return f"{reaction1}_{reaction2}"


    @property
    def results_directory(self) -> Path:
        path = Path("result_files")
        return path / self.model_name / self.analyzed_tuple_string
    

    @property
    def analysis_directory(self) -> Path:
        path = self.results_directory / "analysis"
        return path
    

    @property
    def dataframe_directory(self) -> Path:
        path = self.results_directory / "dataframes"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    
    @property
    def plots_directory(self) -> Path:
        path = self.results_directory / "plots"
        path.mkdir(parents=True, exist_ok=True)
        return path


    # =================================================== ANALYSIS DIRECTORY ===================================================

    
    @staticmethod
    def _format_coord(x: float) -> str:
        return f"{round(x, 6):.6f}".replace('.', 'p').replace('-', 'm')
    

    def _coordinates_to_filename(self, grid_point: np.ndarray) -> str:
        """Formats a floating point coordinate into a filesystem-safe string.""" 
        x_str = self._format_coord(grid_point[0])
        y_str = self._format_coord(grid_point[1])
        return f"x_{x_str}_y_{y_str}.pkl"
    

    def _get_directory(self, subdirectory: Literal["feasibility", "qual_fva", "clustering"]) -> Path:
        """Returns the analysis subdirectory requested for the current grid. 
        If it doesn't exists, the method creates it."""
        Lx, Ly = self.modelclass.grid.grid_dimensions
        grid_dim = f"Lx_{Lx:.4f}_Ly_{Ly:.4f}"

        directory = self.analysis_directory / grid_dim / subdirectory
        directory.mkdir(parents=True, exist_ok=True) 

        return directory


    def _get_point_directory(self, grid_point: np.ndarray, subdirectory: Literal["feasibility", "qual_fva"]) -> Path:
        directory = self._get_directory(subdirectory)
        filename = f"{subdirectory}_{self._coordinates_to_filename(grid_point)}"
        return directory / filename
    

    def load_point(self, grid_point: np.ndarray, analysis: Literal["feasibility", "qual_fva"]) -> bool | tuple | None:
        """Load per-point analysis results from disk.

        Returns
        -------
        bool | tuple | None
            - feasibility: bool
            - qualitative FVA: tuple
            - None if the point is not saved.
        """
        path = self._get_point_directory(grid_point, analysis)
        if not path.exists():
            #print(f"directory doesn't exists")
            return

        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
            if analysis == "feasibility":
                return loaded_data["is_feasible"]
            return loaded_data["fva_tuple"]
        

    def save_feasible_point(self, grid_point: np.ndarray, is_feasible: bool, update_bounds: bool = True) -> None:
        point_dict = {
            "is_feasible": is_feasible,
            "update_bounds": update_bounds,
        }
        
        # making the directory to store the point
        path = self._get_point_directory(grid_point, "feasibility")
        
        with open(path, "wb") as f:
            pickle.dump(point_dict, f)


    def save_fva_result(self, grid_point: np.ndarray, fva_tuple: tuple, update_bounds: bool = True) -> None:
        point_dict = {
            "fva_tuple": fva_tuple,
            "update_bounds": update_bounds
        }
        
        # making the directory to store the point
        path = self._get_point_directory(grid_point, "qual_fva")
        
        with open(path, "wb") as f:
            pickle.dump(point_dict, f)


    def get_clusters_directory(self) -> Path:
        reaction1, reaction2 = self.modelclass.analyze.analyzed_reactions
        delta = self.modelclass.grid.delta
        n_clusters = self.modelclass.clustering.initial_n_clusters
        filename = f"{reaction1}_{reaction2}_clusters_Delta{delta}_NC{n_clusters}.pkl"

        directory = self._get_directory("clustering")

        return directory / filename


    def load_clusters(self) -> tuple | None:
        path = self.get_clusters_directory()
        if not path.exists():
            return

        with open(path, 'rb') as f:
            clusters_tuple = pickle.load(f)
            return clusters_tuple

    
    def save_clusters(self, n_clusters: int, clusters: np.ndarray) -> None:
        clusters_tuple = (n_clusters, clusters)
        
        # making the directory to store clusters
        path = self.get_clusters_directory()
        with open(path, "wb") as f:
            pickle.dump(clusters_tuple, f)


    # =================================================== DATAFRAME DIRECTORY ===================================================


    def save_cluster_df(
        self, 
        df: pd.DataFrame, 
        type: Literal["Qualitative_profiles", "Metrics_per_reaction", "Global_metrics"], 
        index: bool = False,
        reaction_len: int = -1,
        metric_list: list[str] | None = None,
        overwrite: bool = False,
    ) -> None: 
        delta = self.modelclass.grid.delta
        file = f"{type}_NR{reaction_len}_Delta{delta}"
        if metric_list is not None:
            file += f"_M{len(metric_list)}"

        path = self.dataframe_directory / f"{file}.csv"
        if path.exists() and not overwrite:
            print(f"Skipping existing file: {path.name}")
            return
        
        df.to_csv(path, index=index, encoding='utf-8')


    import xlsxwriter
    def merge_to_excel(self, df_dict: dict[str, pd.DataFrame]) -> None:
        path = Path("result_files") / "DM" / "supplementary" / f"{self.analyzed_tuple_string}.xlsx"
        path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for df_name, df in df_dict.items():
                df.to_excel(writer, sheet_name=df_name, index=False)


    # =================================================== PLOT DIRECTORY ===================================================


    def save_plot_path(self, overwrite: bool = False) -> Path | None:
        reaction1, reaction2 = self.modelclass.analyze.analyzed_reactions
        delta = self.modelclass.grid.delta
        n_clusters = self.modelclass.clustering.grid_n_clusters
        path = self.plots_directory / f"{reaction1}_{reaction2}_NC{n_clusters}_Delta{delta}.png"
        if path.exists() and not overwrite:
            return None
        return path