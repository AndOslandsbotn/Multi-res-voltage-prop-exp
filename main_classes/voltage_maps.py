from utilities.kernels import select_kernel
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm

from definitions import CONFIG_VOLTAGE_MAPS_PATH
from config.yaml_functions import yaml_loader

class VoltageMap():
    def __init__(self, epsilon_cover, source_index, kernel_bandwidth):
        self.config = yaml_loader(CONFIG_VOLTAGE_MAPS_PATH)
        self.kernel_type =self.config['grounded_graph']['kernel_type']
        self.weight_to_ground = self.config['grounded_graph']['weight_to_ground']
        if kernel_bandwidth == None:
            self.kernel_bandwidth = self.config['grounded_graph']['default_kernel_bandwidth']
        else:
            self.kernel_bandwidth = kernel_bandwidth

        self.disable_tqdm = self.config['propagator']['disable_tqdm']
        self.progress_thr = self.config['propagator']['progress_thr']
        self.max_iter = self.config['propagator']['max_iter']
        self.use_sparse = self.config['propagator']['use_sparse']
        self.voltage_inclusion_thr = self.config['trimming']['thr_voltage_magnitude']

        self.source_index = source_index
        self.epsilon_cover = epsilon_cover
        self.trimmed_epsilon_cover = {}
        self.transition_matrix = None

        self.voltage = None
        self._initialize_voltage_default()
        return

    def _voltage_monitor(self, voltage, voltage_prev):
        stop_condition = False
        if not any(np.abs(voltage - voltage_prev) > self.progress_thr):
            stop_condition = True
        return stop_condition

    def _initialize_voltage_default(self):
        """Initialize voltage on all nodes to zero as a default"""
        self.voltage = np.zeros(self.epsilon_cover['centers'].shape[0])
        self._add_source_constraints_to_voltage()

    def initialize_voltage_from_rougher_resolution(self, rougher_epsilon_cover):
        """Initialize voltage over current epsilon cover from a rougher epsilon cover.
        :param rougher_epsilon_cover: Dictionary with the keys: centers, densities and voltages.
        """
        distances = cdist(self.epsilon_cover['centers'], rougher_epsilon_cover['centers'])
        self.voltage = distances.dot(np.multiply(rougher_epsilon_cover['voltages'], rougher_epsilon_cover['densities']))
        self._add_source_constraints_to_voltage()

    def initialize_voltage_from_same_resolution(self):
        return

    def _add_source_constraints_to_voltage(self):
        """Add source constraints to voltage. """
        """NB! voltage = source density does not seem to work ... Instead perhaps we can try v= number of points inside the
        source node, without dividing by n ..."""
        self.voltage[self.source_index] = self.epsilon_cover['densities'][self.source_index]


    def trimmer(self):
        """Removes the points in the epsilon cover that have voltage below voltage_inclusion_thr"""
        #indices = (self.voltage >= self.voltage_inclusion_thr)
        indices = (self.voltage >= 0)
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

        # Apply source constraints to transition matrix
        self.transition_matrix[self.source_index, :] = 0
        self.transition_matrix[self.source_index, self.source_index] = 1
        #if self.use_sparse:
        #    self.transition_matrix = csc_matrix(self.transition_matrix)

    def propagate_voltage(self):
        """Propagate voltage by iteratively applying the grounded weight matrix (gwm)"""
        voltage_prev = np.ones(len(self.voltage))*np.inf
        # self.transition_matrix = self.transition_matrix.toarray()  # for debug
        for _ in tqdm(range(0, self.max_iter), desc='propagating labels', disable=self.disable_tqdm):
            self.voltage = self.transition_matrix.dot(self.voltage)
            if self._voltage_monitor(self.voltage, voltage_prev):
                break
            voltage_prev = self.voltage
        return

    def get_trimmed_epsilon_cover(self):
        return self.trimmed_epsilon_cover

    def get_voltage(self):
        return self.voltage


class VoltageMapCollection():
    def __init__(self, epsilon_cover, source_indices, kernel_bandwidth=None):
        self.config = yaml_loader(CONFIG_VOLTAGE_MAPS_PATH)

        self.kernel_bandwidth = kernel_bandwidth
        self.epsilon_cover = epsilon_cover
        self.source_indices = source_indices
        self.voltage_map_collection = []
        self.transition_matrix = None

        self._assign_voltage_maps()
        return

    def _assign_voltage_maps(self):
        """Instantiate a voltage map on each source"""
        for source_index in self.source_indices:
            self.voltage_map_collection.append(VoltageMap(self.epsilon_cover,
                                                          source_index,
                                                          self.kernel_bandwidth))

    def initialize_voltage_maps_from_rougher_resolution(self, rougher_epsilon_cover):
        """Calls the initialize_voltage_from_rougher_resolution method for
        each voltage_maps instance in the voltage_map_collection.
        :param rougher_epsilon_cover: Dictionary of dictionaries organized as:
            rougher_epsilon_cover[source_index] = {}
            rougher_epsilon_cover[source_index][centers] --> Numpy array
            rougher_epsilon_cover[source_index][densities] --> Numpy array
            rougher_epsilon_cover[source_index][voltages] --> Numpy array
        Each source indexed by source_index has a dictionary of centers, densities and voltages
        from the rougher epsilon cover.
        """
        for source_index in self.source_indices:
            self.voltage_map_collection.initialize_voltage_from_rougher_resolution(rougher_epsilon_cover[source_index])

    def propagate_voltage_maps(self):
        """Constructs the graph and propagate voltage on each source"""
        for voltage_map in self.voltage_map_collection:
            voltage_map.trimmer()
            voltage_map.construct_transition_matrix()
            voltage_map.propagate_voltage()
        return

    def get_voltage_map_collection(self):
        return self.voltage_map_collection


