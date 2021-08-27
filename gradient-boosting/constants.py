parent_maps = {}
child_to_parent = {}

training_labels = ["min_surface_downwelling_shortwave_flux_in_air", "max_surface_downwelling_shortwave_flux_in_air",
                   "max_specific_humidity", "min_max_air_temperature", "max_max_air_temperature"]
target_labels = ["max_min_air_temperature"]

# QUERY projection
client_projection = {}
for val in training_labels:
    client_projection[val] = 1
for val in target_labels:
    client_projection[val] = 1
