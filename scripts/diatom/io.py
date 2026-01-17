from typing import TYPE_CHECKING, cast, Literal
import numpy as np


import pickle
from pathlib import Path


from cobra import Model
import cobra.io


import ecosystem.model_io as mio


if TYPE_CHECKING:
    from diatom.diatom import Diatom


MODEL_DIR = "models"
SAVE_POINTS_DIR = "models/points"


class DiatomIO():
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom
        self.directory: Path | None = None


    def save_model(self):
        model = self.diatom.model
        mio.save_models({self.diatom.model_name: model})

        
    @property
    def grid_dimensions(self) -> np.ndarray:
        return self.diatom.grid.grid_dimensions
    

    @property
    def points_per_axis(self) -> tuple[int, int]:
        return self.diatom.grid.points_per_axis
    

    def coordinates_to_index(self, grid_point: np.ndarray) -> tuple[int, int]:
        Lx, Ly = self.grid_dimensions
        assert Lx > 0 and Ly > 0

        x, y = grid_point
        i = round((self.points_per_axis[0]-1) / Lx * x)
        j = round((self.points_per_axis[1]-1) / Ly * y)

        return i, j
    

    def get_directory(self, grid_point: np.ndarray, analysis: Literal["feasibility", "qual_fva"]) -> Path:
        model_name = self.diatom.model_name
        
        Lx, Ly = self.grid_dimensions
        grid_dim = f"Lx_{Lx:.4f}_Ly_{Ly:.4f}"

        i, j = self.coordinates_to_index(grid_point)
        filename = f"{analysis}_{self.points_per_axis}_i_{i}_j_{j}.pkl"
        points_per_axis = f"N_{self.points_per_axis}"

        directory = Path(SAVE_POINTS_DIR) / model_name / grid_dim / points_per_axis / analysis
        directory.mkdir(parents=True, exist_ok=True) 

        if self.directory is None:
            self.directory = Path(SAVE_POINTS_DIR) / model_name / grid_dim / points_per_axis
        

        return directory / filename
    

    def is_saved(self, grid_point: np.ndarray, analysis: Literal["feasibility", "qual_fva"]) -> bool:
        return self.get_directory(grid_point, analysis).exists()
    

    def load_point(self, grid_point: np.ndarray, analysis: Literal["feasibility", "qual_fva"]) -> tuple | None:
        if not self.is_saved(grid_point, analysis):
            #print(f"directory doesn't exists")
            return None 
        
        path = self.get_directory(grid_point, analysis)

        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
            if analysis == "feasibility":
                return loaded_data["is_feasible"]
            return loaded_data["fva_tuple"]
        

    def save_feasible_point(self, grid_point: np.ndarray, is_feasible: bool, update_bounds: bool) -> None:
        point_dict = {
            "is_feasible": is_feasible,
            "update_bounds": update_bounds,
        }
        
        # making the directory to store the point
        path = self.get_directory(grid_point, "feasibility")
        
        with open(path, "wb") as f:
            pickle.dump(point_dict, f)


    def save_fva_result(self, grid_point: np.ndarray, fva_tuple: tuple) -> None:
        point_dict = {
            "fva_tuple": fva_tuple
        }
        
        # making the directory to store the point
        path = self.get_directory(grid_point, "qual_fva")
        
        with open(path, "wb") as f:
            pickle.dump(point_dict, f)


    def save_qual_df(self) -> None:
        if self.directory is None:
            raise Exception()

        path = self.directory / "qual_fva" / "qual_vector.json"
        self.diatom.analyze.qual_vector_df.to_json(path, orient="records", indent=2)

