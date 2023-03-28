from main_classes.voltage_maps import MultiResolutionEmbedding

if __name__ == '__main__':
    n = 10000
    results_folder = f'n{n}'
    results_filename = f'swissroll_voltage_embedding_n{n}'
    filename_epscov = f'swissroll_epscov_n{n}'

    multiResolutionEmbedding = MultiResolutionEmbedding(results_folder, results_filename)
    multiResolutionEmbedding.add_epsilon_covers(filename_epscov)
    multiResolutionEmbedding.run()

