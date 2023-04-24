from utilities.kernels import select_kernel
from utilities.util import get_nn_indices, load_pickle_data, timer_func
from pathlib import Path

import json
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm

from definitions import CONFIG_VOLTAGE_MAPS_PATH
from config.yaml_functions import yaml_loader


class VoltageMap():
    """ This class generates a voltage function over
        an epsilon cover with respect to a given source region

    ## Variables
        - epsilon_cover:
            - Location of points in epsilon cover.
            - numpy array shape (n,d).
            - n is the number of points and d the dimension.
        - source_center_indices:
            - Indices of points (in epsilon cover) that constitutes the source set.
            - numpy array shape (n_s).
            - n_s is the size of the source set.
        - kernel_bandwidth: (Optional) (float) Bandwidth of kernel
        - weight_to_ground: (Optional) (float) Weight of ground node
        - source_radius: (Optional) (float) Radius of source region
        - weight_to_ground (Optional) (float) weight to ground
        - is_source_region (Optional) (Bool) If true use source region, if false only is source center
        - kernel_type (Optional) (string) Name of kernel type

    ## Public methods
        # propagate_voltage
        # construct_transition_matrix
        # init_voltage_from_rougher_resolution
        # trimmer (Not implemented)

    ## Private methods
        # _add_source_constraints_to_voltage
        # _find_source_region
        # _initialize_voltage_default

    ## Get functions
        # get_voltage
        # get_trimmed_epsilon_cover
    """

    def __init__(self, config, epsilon_cover, source_center_index,
                 kernel_bandwidth=None, weight_to_ground=None, source_radius=None,
                 is_source_region=None, kernel_type=None):
        self.config = config

        if kernel_type is None:
            self.kernel_type = self.config['grounded_graph']['kernel_type']
        else:
            self.kernel_type = kernel_type
        if is_source_region is None:
            self.is_source_region = self.config['grounded_graph']['is_source_region']
        else:
            self.is_source_region = is_source_region
        if kernel_bandwidth is None:
            self.kernel_bandwidth = self.config['grounded_graph']['default_kernel_bandwidth']
        else:
            self.kernel_bandwidth = kernel_bandwidth
        if weight_to_ground is None:
            self.weight_to_ground = self.config['grounded_graph']['default_weight_to_ground']
        else:
            self.weight_to_ground = weight_to_ground
        if source_radius is None:
            self.source_radius = self.config['grounded_graph']['default_source_radius']
        else:
            self.source_radius = source_radius

        self.disable_tqdm = self.config['propagator']['disable_tqdm']
        self.progress_thr = self.config['propagator']['progress_thr']
        self.max_iter = self.config['propagator']['max_iter']
        self.use_sparse = self.config['propagator']['use_sparse']
        self.voltage_inclusion_thr = self.config['trimming']['thr_voltage_magnitude']

        self.epsilon_cover = epsilon_cover
        self.epsilon_cover['densities'] = np.ones(len(self.epsilon_cover['densities']))/len(self.epsilon_cover['densities'])

        self.trimmed_epsilon_cover = {}
        self.transition_matrix = None

        self.source_center_index = source_center_index
        self.source_region_indices = None

        self.voltage = None
        self._initialize_voltage_default()
        return

    def _voltage_monitor(self, voltage, voltage_prev):
        stop_condition = False
        if not any(np.abs(voltage - voltage_prev) > self.progress_thr):
            stop_condition = True
        return stop_condition

    def _initialize_voltage_default(self):
        """As a default, initialize voltage on all nodes to zero and the voltage on sources to the source constraint"""
        self.voltage = np.zeros(self.epsilon_cover['centers'].shape[0])
        self.source_region_indices = self._find_source_region()
        self._add_source_constraints_to_voltage()

    def init_voltage_from_rougher_resolution(self, rougher_epsilon_cover, rougher_voltages):
        """Initialize voltage over current epsilon cover from a rougher epsilon cover.
        :param rougher_epsilon_cover: Dictionary with the keys: centers, densities and voltages.
        :param rougher_voltages: numpy array with voltage distribution from given source
        """
        distances = cdist(self.epsilon_cover['centers'], rougher_epsilon_cover['centers'])
        self.voltage = distances.dot(np.multiply(rougher_voltages, rougher_epsilon_cover['densities']))
        self.source_region_indices = self._find_source_region()
        self._add_source_constraints_to_voltage()

    def _find_source_region(self):
        idx_x, idx_y = get_nn_indices(self.epsilon_cover['centers'][self.source_center_index].reshape(1, -1),
                                      self.epsilon_cover['centers'],
                                      self.source_radius)
        return idx_y

    def _add_source_constraints_to_voltage(self):
        """Add source constraints to voltage"""
        if self.is_source_region:
            self.voltage[self.source_region_indices] = 1
        else:
            """NB! voltage = source density does not give nice results. This has to be investigated"""
            self.voltage[self.source_center_index] = self.epsilon_cover['densities'][self.source_center_index]

    def trimmer(self):
        """Removes the points in the epsilon cover that have voltage below voltage_inclusion_thr"""
        #indices = (self.voltage >= self.voltage_inclusion_thr)
        indices = (self.voltage >= 0) # Currently threshold set to 0, since with threshold there are certain problems we need to resolve
        self.trimmed_epsilon_cover['centers'] = self.epsilon_cover['centers'][indices]
        self.trimmed_epsilon_cover['densities']= self.epsilon_cover['densities'][indices]
        self.voltage = self.voltage[indices]

    def construct_transition_matrix(self):
        """Constructs the transition matrix"""
        transition_matrix = select_kernel(self.trimmed_epsilon_cover['centers'], self.kernel_bandwidth, self.kernel_type)

        # Apply densities to transition matrix
        transition_matrix = np.multiply(transition_matrix, self.trimmed_epsilon_cover['densities'][:, np.newaxis])
        transition_matrix = np.multiply(transition_matrix.transpose(), self.trimmed_epsilon_cover['densities'][:, np.newaxis]).transpose()

        # Apply degree matrix to normalize transition matrix
        inv_degree_matrix = 1 / (np.sum(transition_matrix, axis=1) + self.weight_to_ground * self.trimmed_epsilon_cover['densities'])
        self.transition_matrix = np.multiply(inv_degree_matrix[:, None], transition_matrix)

        if self.use_sparse:
            self.transition_matrix = csc_matrix(self.transition_matrix)

    def propagate_voltage(self):
        """Propagate voltage by iteratively applying the grounded weight matrix (transition_matrix)"""
        voltage_prev = np.ones(len(self.voltage))*np.inf
        for _ in tqdm(range(0, self.max_iter), desc='propagating labels', disable=self.disable_tqdm):
            self.voltage = self.transition_matrix.dot(self.voltage)
            self._add_source_constraints_to_voltage()

            if self._voltage_monitor(self.voltage, voltage_prev):
                break
            voltage_prev = self.voltage
        self.voltage = self.transition_matrix.dot(self.voltage)
        return

    def get_trimmed_epsilon_cover(self):
        return self.trimmed_epsilon_cover

    def get_voltage(self):
        return self.voltage


class VoltageMapCollection():
    """ This class contains a collection of VoltageMap instances
        defined different sources.

    ## Variables
        - epsilon_cover:
            - Centers in epsilon cover.
            - numpy array shape (n,d).
            - n is the number of centers and d the dimension.
        - source_indices:
            - Index of sources in epsilon cover.
            - numpy array shape (n_s).
            - n_s is the number of sources.
        - kernel_bandwidth: (Optional) (float) Bandwidt of kernel
        - weight_to_ground: (Optional) (float) Weight of ground node
        - source_radius: (Optional) (float) Radius of source region

    ## Public methods
        # propagate_voltage_maps
        # init_voltage_maps_from_rougher_resolution

    ## Private methods
        # _assign_voltage_maps

    ## Get functions
        # get_voltage_map_collection
    """

    def __init__(self, epsilon_cover, source_indices, kernel_bandwidth=None, weight_to_ground=None, source_radius=None):
        self.config = yaml_loader(CONFIG_VOLTAGE_MAPS_PATH)

        self.epsilon_cover = epsilon_cover
        self.source_indices = source_indices
        self.voltage_map_collection = []
        self.voltages = []
        self.transition_matrix = None

        self._assign_voltage_maps(kernel_bandwidth, weight_to_ground, source_radius)
        return

    def _assign_voltage_maps(self, kernel_bandwidth, weight_to_ground, source_radius):
        """Instantiate a voltage map on each source"""
        for source_center_index in self.source_indices:
            self.voltage_map_collection.append(VoltageMap(self.epsilon_cover,
                                                          source_center_index,
                                                          kernel_bandwidth,
                                                          weight_to_ground,
                                                          source_radius
                                                          ))

    def init_voltage_maps_from_rougher_resolution(self, rougher_epsilon_cover, rougher_voltages):
        """Calls the init_voltage_from_rougher_resolution method for
        each voltage_maps instance in the voltage_map_collection.
        :param rougher_epsilon_cover: Dictionary organized as:
            rougher_epsilon_cover[centers] --> Numpy array
            rougher_epsilon_cover[densities] --> Numpy array
        :param rougher_voltages: List of numpy arrays. Each element is a numpy array
        of voltage distribution from a given source
        """
        assert len(self.voltage_map_collection) == len(rougher_voltages), \
            "Error: voltage_map_collection and rougher_voltages must have same length"

        for i in range(len(self.voltage_map_collection)):
            self.voltage_map_collection[i].init_voltage_from_rougher_resolution(rougher_epsilon_cover, rougher_voltages[i])

    def propagate_voltage_maps(self):
        return self.propagate_voltage_maps_timeit()

    @timer_func
    def propagate_voltage_maps_timeit(self):
        """Constructs the graph and propagate voltage on each source using the corresponding voltage_map instance
        in voltage_map_collection"""
        for voltage_map in self.voltage_map_collection:
            voltage_map.trimmer()
            voltage_map.construct_transition_matrix()
            voltage_map.propagate_voltage()
            self.voltages.append(voltage_map.get_voltage())

    def get_voltage_map_collection(self):
        return self.voltage_map_collection

    def get_voltages(self):
        return self.voltages

    def save_voltage_maps(self, filepath):
        np.savez(filepath, voltages=self.voltages)

