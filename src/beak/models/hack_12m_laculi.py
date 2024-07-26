"""
This file contains the models used for the magmatic nico experiment.
The BISON model is used as a geophysical baseline model for deeper sources (even shallow, but no surface data).
The JELLYFISH model is most comparable to the google sheet model configuration, but does not include high-res data.
"""

regional_scale_southwest = {
    "BASELINE_BISON": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
    },
    "JITTER_JELLYFISH": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_LITH_PROXIMITY": True,  # TBD
        "SGMC_AGE": True,   # TBD
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley
    },
    "GOOFY_GOPHER": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_LITH": True,  # TBD
        "SGMC_AGE": True,  # TBD
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley
        # Elevation
        "Li_dem": True,
        "Li_slope": True,
        "Li_tpi": True,
    },
    "CAMEL_CASE": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_LITH": True,  # TBD
        "SGMC_AGE": True,  # TBD
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley
        # Elevation
        "Li_dem": True,
        "Li_slope": True,
        "Li_tpi": True,
        # Spectral
        "ASTER_REFLECTANCE": True  # TBD
    },
    "TOPO_TAPIR": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_LITH": True,  # TBD
        "SGMC_AGE": True,  # TBD
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley
        # Elevation
        "Li_dem": True,
        "Li_slope": True,
        "Li_tpi": True,
        "Li_valley_depth"
        # Spectral
        "ASTER_REFLECTANCE": True  # TBD
    },
}

# Set models without using "dense" proximity information
models = ["DROPOUT_DUCK",
          "LOSS_LLAMA",
          "GOOFY_GOPHER",
          "CAMEL_CASE",
          "TOPO_TAPIR",
          ]

update = regional_scale_southwest
update["DROPOUT_DUCK"] = update["BASELINE_BISON"].copy()
update["LOSS_LLAMA"] = update["JITTER_JELLYFISH"].copy()
update["GOOFY_GOPHER"] = update["YOLO_YAK"].copy()
update["CAMEL_CASE"] = update["PRETTY_PENGUIN"].copy()
update["TOPO_TAPIR"] = update["MORPH_MAMBA"].copy()

layers_to_false = [
    "ShallowGravitySources_Worms_Proximity",
    "IsostaticGravity_Worms_Proximity",
    "ShallowMagSources_Worms_Proximity",
]

for model in models:
    for layer in layers_to_false:
        if layer in update[model].keys():
            update[model][layer] = False
