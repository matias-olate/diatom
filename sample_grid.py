from scripts import Diatom

diatom = Diatom("iLB1027_lipid.xml", "DM")
diatom.set_objective_functions()

def polytope_pipeline(reactions: tuple[str, str], n_angles: int = 360, delta: float = 0.01):
    diatom.analyze.project_polytope_2d(reactions, n_angles = n_angles)
    diatom.grid.sample_polytope(delta = delta)
    diatom.grid.debug_plot(delta = delta)
    diatom.analyze.qualitative_analysis()
    diatom.clustering.set_grid_clusters('hierarchical', k = 10)
    #df = diatom.clustering.get_grid_cluster_qual_profiles(threshold=0.8, changing= True)
    #diatom.clustering.compare_clusters(df, reactions[0], reactions[1]).head(200)
    diatom.plot.plot_sampled_polytope(show_boundary=True)


def analyze_reactions(reaction_list1: list[str], reaction_list2: list[str], n_angles: int = 360, delta: float = 0.0125) -> None:
    for reaction1 in reaction_list1:
        for reaction2 in reaction_list2:
            polytope_pipeline((reaction1, reaction2), n_angles = n_angles, delta = delta)


reactions1 = ["EX_no3_e", "EX_photon_e"]
reactions2 = ["DM_biomass_c"]

analyze_reactions(reactions1, reactions2)