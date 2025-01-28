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
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": False,
        "Mag_LogAnalyticSignal": True,
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
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": False,
        "Mag_LogAnalyticSignal": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_Distance_to_Unconsolidated": True,
        "SGMC_MinAge_Min": True,  # Not Max, maybe even not. Lets try.
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley, maybe even not. Lets try.
        # Geochemistry
        "Geochemistry_StrmSed_Cs_nure_reanalysis": True,
        "Geochemistry_StrmSed_Li_nure_reanalysis": True,
        "Geochemistry_StrmSed_Mo_nure_reanalysis": True,
        "Geochemistry_StrmSed_Rb_nure_reanalysis": True,
        "Geochemistry_StrmSed_Sn_nure_reanalysis": True,
        "Geochemistry_StrmSed_W_nure_reanalysis": True,
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
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": False,
        "Mag_LogAnalyticSignal": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_Distance_to_Unconsolidated": True,
        "SGMC_MinAge_Min": True,  # Not Max, maybe even not. Lets try.
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley, maybe even not. Lets try.
        # Geochemistry
        "Geochemistry_StrmSed_Cs_nure_reanalysis": True,
        "Geochemistry_StrmSed_Li_nure_reanalysis": True,
        "Geochemistry_StrmSed_Mo_nure_reanalysis": True,
        "Geochemistry_StrmSed_Rb_nure_reanalysis": True,
        "Geochemistry_StrmSed_Sn_nure_reanalysis": True,
        "Geochemistry_StrmSed_W_nure_reanalysis": True,
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
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": False,
        "Mag_LogAnalyticSignal": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_Distance_to_Unconsolidated": True,
        "SGMC_MinAge_Min": True,  # Not Max, maybe even not. Lets try.
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley, maybe even not. Lets try.
        # Geochemistry
        "Geochemistry_StrmSed_Cs_nure_reanalysis": True,
        "Geochemistry_StrmSed_Li_nure_reanalysis": True,
        "Geochemistry_StrmSed_Mo_nure_reanalysis": True,
        "Geochemistry_StrmSed_Rb_nure_reanalysis": True,
        "Geochemistry_StrmSed_Sn_nure_reanalysis": True,
        "Geochemistry_StrmSed_W_nure_reanalysis": True,
        # Elevation
        "Li_dem": True,
        "Li_slope": True,
        "Li_tpi": True,
        # Spectral
        "minerals": True,
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
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": False,
        "Mag_LogAnalyticSignal": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        # Geology
        "SGMC_Distance_to_Unconsolidated": True,
        "SGMC_MinAge_Min": True,  # Not Max, maybe even not. Lets try.
        "Geology_Paleolatitude_Period_Minimum": True,  # Lawley, maybe even not. Lets try.
        # Geochemistry
        "Geochemistry_StrmSed_Cs_nure_reanalysis": True,
        "Geochemistry_StrmSed_Li_nure_reanalysis": True,
        "Geochemistry_StrmSed_Mo_nure_reanalysis": True,
        "Geochemistry_StrmSed_Rb_nure_reanalysis": True,
        "Geochemistry_StrmSed_Sn_nure_reanalysis": True,
        "Geochemistry_StrmSed_W_nure_reanalysis": True,
        # Elevation
        "Li_dem": False,    # Substituted by valley depth
        "Li_slope": True,
        "Li_tpi": True,
        "Li_valley_depth": True,
        # Spectral
        "minerals": True,
    },
}

# Set models without using "dense" proximity information
models = [
    "DROPOUT_DUCK",
    "LOSS_LLAMA",
    "YOLO_YAK",
    "PRETTY_PENGUIN",
    "MORPH_MAMBA",
]

update = regional_scale_southwest
update["DROPOUT_DUCK"] = update["BASELINE_BISON"].copy()
update["LOSS_LLAMA"] = update["JITTER_JELLYFISH"].copy()
update["YOLO_YAK"] = update["GOOFY_GOPHER"].copy()
update["PRETTY_PENGUIN"] = update["CAMEL_CASE"].copy()
update["MORPH_MAMBA"] = update["TOPO_TAPIR"].copy()

layers_to_false = [
    "ShallowGravitySources_Worms_Proximity",
    "IsostaticGravity_Worms_Proximity",
    "ShallowMagSources_Worms_Proximity",
]

for model in models:
    for layer in layers_to_false:
        if layer in update[model].keys():
            update[model][layer] = False

# Setup for models without geochemistry
models = [
    "JITTER_JELLYFISH",
    "GOOFY_GOPHER",
    "CAMEL_CASE",
    "TOPO_TAPIR",
    "LOSS_LLAMA",
    "YOLO_YAK",
    "PRETTY_PENGUIN",
    "MORPH_MAMBA",
]

suffix = "NGC"

update = regional_scale_southwest
for model in models:
    update[str(model + "_" + suffix)] = update[model].copy()

layers_to_false = [
    "Geochemistry_StrmSed_Cs_nure_reanalysis",
    "Geochemistry_StrmSed_Li_nure_reanalysis",
    "Geochemistry_StrmSed_Mo_nure_reanalysis",
    "Geochemistry_StrmSed_Rb_nure_reanalysis",
    "Geochemistry_StrmSed_Sn_nure_reanalysis",
    "Geochemistry_StrmSed_W_nure_reanalysis",
]

for model in models:
    model = model + "_" + suffix
    for layer in layers_to_false:
        if layer in update[model].keys():
            update[model][layer] = False
