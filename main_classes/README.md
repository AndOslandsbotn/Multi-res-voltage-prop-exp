## VoltageMap class
This class generates a voltage function over 
an epsilon cover with respect to a given source region

###### Variables
 - epsilon_cover: 
    - Centers in epsilon cover. 
    - numpy array shape (n,d). 
    - n is the number of centers and d the dimension.
 - source_center_index: 
    - (int) Index of a particular source in epsilon cover.
 - kernel_bandwidth: (Optional) (float) Bandwidt of kernel
 - weight_to_ground: (Optional) (float) Weight of ground node
 - source_radius: (Optional) (float) Radius of source region
 
        ## Public methods
        # propagate_voltage
        # construct_transition_matrix
        # initialize_voltage_from_rougher_resolution
        # initialize_voltage_from_same_resolution (Not implemented)
        # trimmer
        
        ## Private methods
        # _add_source_constraints_to_voltage
        # _find_source_region
        # _initialize_voltage_default
        
        ## Get functions
        # get_voltage
        # get_trimmed_epsilon_cover
        
 ## VoltageMapCollection
 This class contains a collection of VoltageMap instances 
 defined different sources.
 
 ###### Variables
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
        # initialize_voltage_maps_from_rougher_resolution
        
        ## Private methods
        # _assign_voltage_maps
        
        ## Get functions
        # get_voltage_map_collection