"""
This file contains the models used for the magmatic nico experiment.
The BISON model is used as a geophysical baseline model for deeper sources (even shallow, but no surface data).
The JELLYFISH model is most comparable to the google sheet model configuration, but does not include high-res data.
"""

regional_scale_upper_midwest = {
    "BASELINE_BISON": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "SatelliteGravity_ShapeIndex": True,
        "CEUS_GRAV_Isostatic_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_HD_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_1VD_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_HD_1VD_CEUSSSC_R0": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        "Mag_LogAnalyticSignal": True,
        "CEUS_MAG_DRTP_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_HD_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        "CONUS_MT2023_48km": True,
        "CONUS_MT2023_92km": True,
    },
    "JITTER_JELLYFISH": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "SatelliteGravity_ShapeIndex": True,
        "CEUS_GRAV_Isostatic_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_HD_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_1VD_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_HD_1VD_CEUSSSC_R0": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        "Mag_LogAnalyticSignal": True,
        "CEUS_MAG_DRTP_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_HD_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        "CONUS_MT2023_48km": True,
        "CONUS_MT2023_92km": True,
        # Geology
        "BGE_ALL_NORILSK": True,                        # TA2
        "BGE_DEPOSIT_NORILSK": True,                    # TA2
        "BGE_ROCK_TYPE_NORILSK": True,                  # TA2
    },
}

# Set models without using "dense" proximity information
models = ["DROPOUT_DUCK", "LOSS_LLAMA"]

update = regional_scale_upper_midwest
update["DROPOUT_DUCK"] = update["BASELINE_BISON"].copy()
update["LOSS_LLAMA"] = update["JITTER_JELLYFISH"].copy()

layers_to_false = [
    "ShallowGravitySources_Worms_Proximity",
    "IsostaticGravity_Worms_Proximity",
    "ShallowMagSources_Worms_Proximity",
]

for model in models:
    for layer in layers_to_false:
        if layer in update[model].keys():
            update[model][layer] = False
