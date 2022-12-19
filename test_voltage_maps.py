import matplotlib.pyplot as plt
import numpy as np

from main_classes.voltage_maps import VoltageMapCollection
from main_classes.epsilon_cover import BasicCoverTree, run_cover_properties_test, estimate_span
from utilities.generate_toy_data import non_uniform_1d_experiment

if __name__ == '__main__':
    max_lvl = 6
    start_lvl = 3
    data, _, _ = non_uniform_1d_experiment()
    voltage_map_collections = {}
    epsilon_cover = {}

    # Construct epsilon cover
    epsilon_reduce_factor = 2
    init_radius = estimate_span(data)
    epsilon_cover_factory = BasicCoverTree(init_radius)
    for x in data:
        epsilon_cover_factory.insert(x)

    data_for_density_estimation, ref_epsilon_cover, ref_epsilon_cover_densities = non_uniform_1d_experiment()
    epsilon_cover_factory.estimate_densities(data_for_density_estimation)

    epsilon_cover = epsilon_cover_factory.get_epsilon_cover(max_lvl = max_lvl+1)

    # Source indices
    source_indices = np.arange(0, len(epsilon_cover[start_lvl]['centers']), 1)

    # Initialize voltage maps
    for lvl in range(start_lvl, max_lvl):
        voltage_map_collections[lvl] = VoltageMapCollection(epsilon_cover[lvl], source_indices)

    # Propagate voltages
    voltage_map_collections[start_lvl].propagate_voltage_maps()

    voltage_map_collection = voltage_map_collections[start_lvl].get_voltage_map_collection()

    for voltage_map in voltage_map_collection:
        plt.figure()
        plt.plot(voltage_map.get_voltage())
    plt.show()