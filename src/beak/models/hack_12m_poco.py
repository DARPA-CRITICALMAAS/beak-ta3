"""
This file contains the models used for the magmatic poco experiment.
The BISON model is used as a geophysical baseline model for deeper sources (even shallow, but no surface data).
The JELLIFISH model is most comparable to the google sheet model configuration, but does not include high-res data.
The PANDA model is an additional configuration for self-organizing maps with high-res data.
"""
from beak.utilities.misc import update_model_config

regional_scale_southwest = {
    "BASELINE_BISON": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "Gravity_Up30km": True,
        "Gravity_Up30km_HGM": True,
        "ShallowGravitySources_Worms_Proximity": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        "IsostaticGravity_HGM": True,
        "IsostaticGravity_Worms_Proximity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "Mag_RTP_VD": True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
    },
    "JITTER_JELLYFISH": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "Gravity_Up30km": True,
        "Gravity_Up30km_HGM": True,
        "ShallowGravitySources_Worms_Proximity": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        "IsostaticGravity_HGM": True,
        "IsostaticGravity_Worms_Proximity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "Mag_RTP_VD": True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        # Radiometrics
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
        # Geology
        "BGE_ALL": True,                                # TA2
        "BGE_DEPOSIT": True,                            # TA2
        "BGE_ROCK_TYPE": True,                          # TA2
        "CEUS_pc_Whitmeyer2007_R0_Unit": True,
        "All_Alteration_Proximity": True,
        "cdr_sgmc_proximity_faults_shear_zones": True,  # CDR, SGMC
        # Geochemistry
        "Geochemistry_StrmSed_Cu_SBR": True,            # Impute mean (SOM and NN inference)
        "Geochemistry_StrmSed_Mo_SBR": True,            # Impute mean (SOM and NN inference)
        "Geochemistry_StrmSed_Zn_SBR": True,            # Impute mean (SOM and NN inference)
    },
}

regional_scale_southwest_swnm = {
    "BASELINE_BISON_PP": regional_scale_southwest["BASELINE_BISON"].copy(),
    "JITTER_JELLYFISH_PP": regional_scale_southwest["JITTER_JELLYFISH"].copy(),
}


# Create high-resolution (HR) configurations for New Mexico Survey (SWNM)
model_name = "BASELINE_BISON_PP"
update = regional_scale_southwest_swnm[model_name]
changes = [("Mag_RTP", "Geophysics_Mag_RTP_SWNM"),
           ("Mag_RTP_HGM", "Geophysics_Mag_RTP_HGM_SWNM"),
           ("Mag_RTP_VD", "Geophysics_Mag_1VD_SWNM")]
regional_scale_southwest_swnm[model_name] = update_model_config(update, changes)


model_name = "JITTER_JELLYFISH_PP"
update = regional_scale_southwest_swnm[model_name]
changes = [("Mag_RTP", "Geophysics_Mag_RTP_SWNM"),
           ("Mag_RTP_HGM", "Geophysics_Mag_RTP_HGM_SWNM"),
           ("Mag_RTP_VD", "Geophysics_Mag_1VD_SWNM"),
           ("NAMrad_K", "Radiometric_K_SWNM"),
           ("NAMrad_Th", "Radiometric_Th_SWNM"),
           ("NAMrad_U", "Radiometric_U_SWNM")]
regional_scale_southwest_swnm[model_name] = update_model_config(update, changes)


# Delete unnecessary stuff
del update,  model_name
