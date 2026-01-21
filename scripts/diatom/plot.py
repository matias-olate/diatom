from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from .diatom import Diatom


class DiatomPlot():
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom


    def plot_sampled_polytope(self, show_boundary: bool = True, show_points: bool = True,
        s: float = 10.0, alpha: float = 0.6) -> None:
        clusters = self.diatom.clustering.grid_clusters
        if clusters is None:
            raise RuntimeError("Grid clusters not computed.")
        
        grid = self.diatom.grid
        analyze = self.diatom.analyze

        poly = analyze.polytope
        points = grid.points
        feasible = grid.feasible_points
        
        points = points[feasible]

        fig, ax = plt.subplots(figsize=(7, 7))

        # --- Polígono ---
        if show_boundary:
            assert isinstance(poly, Polygon)
            x, y = poly.exterior.xy
            ax.plot(x, y, linewidth=1, linestyle="dashed", alpha=0.5, color="gray")
  
        # --- Puntos sampleados ---
        if show_points and points.size > 0:
            k = int(np.max(clusters)) + 1
            cmap = plt.get_cmap("tab20", k)
            vmin = 0.5
            vmax = k + 0.5

            sc = ax.scatter(points[:, 0], points[:, 1], c=clusters, s=s, alpha=alpha, 
                            cmap = cmap, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(1, k + 1))
            cbar.ax.set_yticklabels([f"C{i}" for i in range(k)])
            cbar.set_label("Cluster ID")

        reactions = analyze.analyzed_reactions
        ax.set_xlabel(reactions[0])
        ax.set_ylabel(reactions[1])
        ax.set_title("Projected Feasible Polytope")
        ax.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_cluster_distribution(self, clusters_df: pd.DataFrame, cmap: str = 'Accent', figsize: tuple[int, int] = (10,5)) -> pd.DataFrame:
        """Plot the distribution of qualitative reaction categories per cluster.

        Each column in `clusters_df` is interpreted as a cluster and each row as a
        reaction category assignment. NaN values are treated as an additional
        'variable' category. Outputs a dataframe contaning the percentage of reactions per category for each cluster.
        """
        cluster_columns = list(clusters_df)
        n_reactions = clusters_df.shape[0]

        nan_rep = 100.0 # Change nan to additional category 'variable'
        filled_clusters_df = clusters_df.fillna(nan_rep)

        category_percents_dict = dict()

        for category in cluster_columns:
            vc = filled_clusters_df[category].value_counts()
            vc = 100 * vc/n_reactions # to percentages
            category_percents_dict[category] = vc.to_dict()   
            
        category_percents = pd.DataFrame.from_dict(category_percents_dict, orient='index')
        category_percents.fillna(0, inplace=True)
        category_percents.rename(columns = self.diatom.analyze.category_dict, inplace=True)

        # plot
        ax = category_percents.plot.barh(stacked=True, cmap=cmap, figsize=figsize)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='reaction category')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel('clusters')
        ax.set_xlabel('reactions')

        return category_percents 


