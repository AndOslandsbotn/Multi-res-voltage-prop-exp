import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from utilities.util import _dist
from scipy.spatial.distance import cdist

from definitions import CONFIG_COVER_TREE_PATH
from config.yaml_functions import yaml_loader

def estimate_span(dataset):
    """Estimate the span of the dataset"""
    init_radius = np.sqrt(np.sum(np.array([(max(abs(dataset[:, i]))-min(abs(dataset[:, i])))**2 for i in range(0, len(dataset[0, :]))])))
    return init_radius

class CoverTreeNode():
    def __init__(self, center, radius, path, parent=None):
        self.center = center
        self.radius = radius

        # The children, parent, root and path parameters
        # needs to be updated once node is inserted into a cover-tree
        self.children = []
        self.parent = parent
        self.root = None
        self.path = path  # Identifier that indicates the position of the node in the tree
        self.density = 0
        return


    def __str__(self):
        return 'Node path=%s, radius=%5.2f, %d children' % \
               (self.path, self.radius, len(self.children)) + 'center=' + str(self.center)

    def get_center(self):
        return self.center

    def get_children(self):
        return self.children

    def num_children(self):
        return len(self.children)

    def get_level(self):
        return len(self.path)

    def get_downstream_nodes(self):
        """Get all nodes downstream of self"""
        downstream = [self]
        if self.num_children() > 0:
            for child in self.get_children():
                downstream = downstream + child.get_downstream_nodes()
        return downstream


class BasicCoverTree():
    """A cover tree class that link cover tree nodes together to form a forest of
    cover trees. The tree considers the nearest neighbour as the potential new parent
    """

    def __init__(self, init_radius):
        self.config = yaml_loader(CONFIG_COVER_TREE_PATH)
        self.init_radius = init_radius
        self.epsilon_reduce_factor = self.config['covertree']['epsilon_reduce_factor']
        self.max_depth = self.config['covertree']['nlvls']

        assert isinstance(self.epsilon_reduce_factor, int) or isinstance(self.epsilon_reduce_factor, float), f'epsilon_reduce_factor must be int or float'
        assert isinstance(self.max_depth, int), f'max_depth must be int'

        self.nodes = {}
        self.centers = {}
        self.densities = {}
        self.roots = []
        self.tree_depth = 0
        return

    def __str__(self):
        print(f"Number of landmarks in cover tree")
        for lvl in range(1, self.max_depth+1):
              print(f"lvl {lvl}: {len(self.centers[lvl])}")
        return

    def nearest_neighbour(self, x, nodes):
        """Finds the nearest neighbor of x in nodes"""
        nn = None
        dist = np.inf
        for node in nodes:
            if _dist(x, node.center) < dist:
                nn = node
                dist = _dist(x, node.center)
        return nn, dist

    def estimate_densities(self, x):
        n = len(x)
        centers = self.get_centers()
        for lvl in centers.keys():
            index_closest_center = np.argmin(cdist(x, centers[lvl]), axis=1)
            self.densities[lvl] = np.zeros(len(centers[lvl]))
            for center_idx in range(0, len(centers[lvl])):
                self.densities[lvl][center_idx] = np.sum(index_closest_center==center_idx)/n

    def insert(self, x):
        path = self.find_path(x, self.roots)

        if path is None:
            self.add_new_root(x)
        elif not len(path) >= self.max_depth:
            self.add_new_node(x, path)

    def find_path(self, x, nodes):
        """Find path to location in tree, where node should be inserted"""
        if len(nodes) == 0:
            return None

        nn, dist = self.nearest_neighbour(x, nodes)

        if dist < nn.radius:  # If x is inside nn
            node_path = self.find_path(x, nn.get_children())
            if node_path is None:
                return [nn]
            else:
                return [nn] + node_path
        else:  # If we are not inside nn, then we are also not inside any of the other nodes
            return None

    def add_new_node(self, x, path):
        """
        Add a new node as a child to leaf
        :param x: center of new node
        :param path: path to parent of new node
        """
        leaf = path[-1]
        new = CoverTreeNode(x, leaf.radius / 2, leaf.path + (leaf.num_children(),))
        new.parent = leaf
        new.root = leaf.root
        leaf.children.append(new)

        self.update_dictionaries(new)
        if len(new.path) > self.tree_depth:
            self.tree_depth = len(new.path)
        return new

    def add_new_root(self, x):
        """
        Add a new node as root
        :param x: center of new root
        """
        new = CoverTreeNode(x, self.init_radius / 2, (len(self.roots),))
        new.parent = None
        new.root = new
        self.roots.append(new)

        self.update_dictionaries(new)
        return

    def update_dictionaries(self, new_node):
        lvl = len(new_node.path)
        if lvl in self.nodes:
            self.centers[lvl].append(new_node.center)
            self.nodes[lvl].append(new_node)
        else:
            self.centers[lvl] = [new_node.center]
            self.nodes[lvl] = [new_node]

    def get_nodes(self, lvl=None):
        if lvl is None:
            return self.nodes
        else:
            return self.nodes[lvl]

    def get_epsilon_cover(self, max_lvl):
        densities = self.get_densities()
        centers = self.get_centers()
        epsilon_cover = {}
        for lvl in range(1, max_lvl):
            if lvl not in epsilon_cover.keys():
                epsilon_cover[lvl] = {}
            epsilon_cover[lvl]['centers'] = np.array(centers[lvl])
            epsilon_cover[lvl]['densities'] = densities[lvl]
        return epsilon_cover

    def get_densities(self, lvl=None):
        return self.densities

    def get_depth(self):
        return self.max_depth

    def get_centers(self, lvl=None):
        if lvl is None:
            return self.centers
        else:
            return np.array(self.centers[lvl])

    def get_radius(self, lvl):
        return self.init_radius/(2**(lvl-1))

    def clear(self):
        """Clears the cover-tree"""
        self.nodes = {}
        self.centers = {}
        self.roots = []
        self.tree_depth = 0

def run_speed_test(data, init_radius, size, config):
    start_total = perf_counter()
    tree = BasicCoverTree(init_radius, config)
    # tree.set_debug()
    avg_insert_time = []
    for x in data:
        start_average = perf_counter()
        tree.insert(x)
        stop_average = perf_counter()
        avg_insert_time.append(stop_average-start_average)
    avg_insert_time = np.mean(avg_insert_time)
    stop_total = perf_counter()
    print(f'Total time to build cover tree {stop_total - start_total} s, with {size}^2 points')
    print(f'Average time to insert a point {avg_insert_time} s')
    return tree

def run_cover_properties_test(allcenters, epsilon_reduce_factor, init_radius, indices):
    max_lvl = min(5, len(allcenters))
    fig, axes = plt.subplots(1, max_lvl+1, figsize=(15, 3))
    for lvl in range(1, max_lvl+1):
        centers = allcenters[lvl]
        print(f'Plot level {lvl}')
        print(f'Number of centers lvl {lvl} = {len(centers)}')
        radius = init_radius / (epsilon_reduce_factor ** lvl)
        for center in centers:
            axes[lvl].set_title(f'Nodes at level {lvl}')
            axes[lvl].add_patch(plt.Circle(tuple([center[indices[0]], center[indices[1]]]),
                                    radius=radius, color='r', fill=False))
            axes[lvl].scatter(center[indices[0]], center[indices[1]], s=20, c='k', marker='o', zorder=2)

def check(allcenters):
    sum = 0
    for lvl in range(1, len(allcenters)+1):
        centers = allcenters[lvl]
        sum = sum + len(centers)
    print("Check sum of nodes: ", sum)


from utilities.generate_toy_data import generate_2D_plane, non_uniform_1d_experiment
if __name__ == '__main__':
    #data, _, _ = non_uniform_1d_experiment()
    data = generate_2D_plane(datasize=10000)
    init_radius = estimate_span(data)
    epsilon_cover_factory = BasicCoverTree(init_radius)
    for x in data:
        epsilon_cover_factory.insert(x)

    epsilon_reduce_factor = 2
    indices = [0, 1]
    centers = epsilon_cover_factory.get_centers()

    #data_density_estimation, _, _ = non_uniform_1d_experiment()
    data_density_estimation = generate_2D_plane(datasize=10000)
    data_density_estimation += np.random.normal(0, 0.01, size=data_density_estimation.shape)
    densities = epsilon_cover_factory.estimate_densities(data_density_estimation)

    for lvl in range(1, 5):
        print(densities[lvl])
        print(np.sum(densities[lvl]))
    run_cover_properties_test(centers, epsilon_reduce_factor, init_radius, indices)
    plt.show()