import matplotlib.pyplot as plt
import numpy as np

from main_classes.voltage_maps import VoltageMapCollection
from main_classes.epsilon_cover import BasicCoverTree, run_cover_properties_test, estimate_span
from utilities.generate_toy_data import non_uniform_1d_experiment, generate_2D_plane

if __name__ == '__main__':
    max_lvl = 4
    start_lvl = 3
    dim = 1
    if dim == 1:
        data, _, _ = non_uniform_1d_experiment()
    elif dim ==2:
        data = generate_2D_plane(datasize=10000)
    voltage_map_collections = {}
    epsilon_cover = {}

    # Construct epsilon cover
    epsilon_reduce_factor = 2
    init_radius = estimate_span(data)
    epsilon_cover_factory = BasicCoverTree(init_radius)
    for x in data:
        epsilon_cover_factory.insert(x)

    #data, ref_epsilon_cover, ref_epsilon_cover_densities = non_uniform_1d_experiment()
    epsilon_cover_factory.estimate_densities(data)
    epsilon_cover = epsilon_cover_factory.get_epsilon_cover(max_lvl = max_lvl+1)

    # Source indices
    #source_indices = np.arange(0, len(epsilon_cover[start_lvl]['centers']), 1)
    source_indices = {}
    source_centers_start_level = epsilon_cover[start_lvl]['centers']
    for lvl in range(start_lvl, max_lvl):
        indices = []
        for source_center in source_centers_start_level:
            indices.append(np.where(np.all(epsilon_cover[lvl]['centers'] == source_center, axis=1))[0])
        source_indices[lvl] = np.array(indices).flatten()

    # Initialize voltage maps
    for lvl in range(start_lvl, max_lvl):
        voltage_map_collections[lvl] = VoltageMapCollection(epsilon_cover[lvl], source_indices[lvl])

    # Propagate voltages
    for lvl in range(start_lvl, max_lvl):
        voltage_map_collections[lvl].propagate_voltage_maps()

    for lvl in range(start_lvl, max_lvl):
        voltage_map_collection = voltage_map_collections[lvl].get_voltage_map_collection()

        for voltage_map in voltage_map_collection:
            if dim == 1:
                plt.figure()
                plt.plot(voltage_map.get_voltage())

            elif dim == 2:
                plt.figure()
                plt.scatter(epsilon_cover[lvl]['centers'][:, 0], epsilon_cover[lvl]['centers'][:, 1], c=voltage_map.get_voltage())


    ###################################
    ############ Reference ############
    ###################################
    epsilon_cover_ref = {}
    epsilon_cover_ref['centers'] = data
    epsilon_cover_ref['densities'] = np.ones(data.shape[0])/data.shape[0]

    source_indices_in_data = {}
    for lvl in range(start_lvl, max_lvl):
        source_centers = epsilon_cover[lvl]['centers'][source_indices[lvl]]
        indices = []
        for source_center in source_centers:
            indices.append(np.where(np.all(data == source_center, axis=0))[0])
        source_indices_in_data[lvl] = np.array(indices).flatten()

    # Initialize voltage maps
    voltage_map_ref_collections = {}
    for lvl in range(start_lvl, max_lvl):
        voltage_map_ref_collections[lvl] = VoltageMapCollection(epsilon_cover_ref, source_indices_in_data[lvl])

    # Propagate voltages
    voltage_map_ref_collections[start_lvl].propagate_voltage_maps()
    voltage_map_ref_collection = voltage_map_ref_collections[start_lvl].get_voltage_map_collection()

    for voltage_map_ref in voltage_map_ref_collection:
        plt.figure()
        plt.plot(voltage_map_ref.get_voltage())
    plt.show()

