
class MultiResolutionEmbedding():
    def __init__(self, results_folder, results_filename):
        self.config = yaml_loader(CONFIG_VOLTAGE_MAPS_PATH)
        self.source_lvl = self.config['embedding']['source_lvl']
        min, max = self.config['embedding']['embedding_lvls']
        self.embedding_lvls = np.arange(min, max+1, 1)

        self.results_folder = os.path.join('Results', results_folder)
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        self.results_filename = results_filename

        self.voltage_map_collections = {}
        self.source_indices = {}
        self.source_centers = None
        self.epsilon_covers = None

        self.time_logg = {}

    def add_epsilon_covers(self, filename):
        self.epsilon_covers = load_pickle_data('Data', filename)
        self.source_centers = self.epsilon_covers[self.source_lvl]['centers']

    def run(self):
        _, exec_time = self.run_with_timeit()
        self.time_logg[f'exec_time_tot'] = exec_time
        self.save_time_logg()

    @timer_func
    def run_with_timeit(self):
        for i, lvl in enumerate(self.embedding_lvls):
            self.source_indices[lvl] = np.where(np.all(np.isin(self.epsilon_covers[lvl]['centers'],
                                                               self.source_centers), axis=1))[0]

            self.voltage_map_collections[lvl] = VoltageMapCollection(self.epsilon_covers[lvl],
                                                                          self.source_indices[lvl])

            if i > 0 and self.config['embedding']['init_from_rougher']:
                rougher_voltages = self.voltage_map_collections[lvl-1].get_voltages()
                self.voltage_map_collections[lvl].init_voltage_maps_from_rougher_resolution(self.epsilon_covers[lvl-1],
                                                                                                  rougher_voltages)

            _, exec_time = self.voltage_map_collections[lvl].propagate_voltage_maps()
            self.time_logg[f'exec_time_lvl{lvl}'] = exec_time

            self.voltage_map_collections[lvl].save_voltage_maps(
                filepath=os.path.join(self.results_folder, self.results_filename + f'_lvl{lvl}.npz'))


    def save_time_logg(self):
        with open(os.path.join(self.results_folder, 'time_logg'), "w") as outfile:
            json.dump(self.time_logg, outfile)


