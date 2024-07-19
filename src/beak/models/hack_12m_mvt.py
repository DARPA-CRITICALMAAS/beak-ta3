"""
This file contains the models used for the magmatic mvt experiment.
The BISON model is used as a geophysical baseline model for deeper sources (even shallow, but no surface data).
The JELLIFISH model is most comparable to the google sheet model configuration, but does not include high-res data.
The PUMA model is an additional configuration for self-organizing maps with high-res data.
"""
from beak.utilities.misc import update_model_config

regional_scale_ceus = {
    # Template
    "DUMMY_DICTIONARY": {
        # Some comments
        "Geology_Lithology_Majority": True,
        "Geology_Lithology_Minority": True,
        "...": False,
    },
    "BASELINE_BISON": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Magnetotelluric
        "CONUS_MT2023_30km": True,
    },
    "JITTER_JELLYFISH": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Magnetotelluric
        "CONUS_MT2023_30km": True,
        # Geology
        "BGE_ALL": True,                                # TA2
        "BGE_DEPOSIT": True,                            # TA2
        "BGE_ROCK_TYPE": True,                          # TA2
        "Geology_BlackShale_Proximity": True,           # Lawley
        "Geology_Paleolatitude_Period_Minimum": True,   # Lawley
        "Geology_PassiveMargin_Proximity": True,        # Lawley
        "cdr_sgmc_proximity_faults_shear_zones": True,  # CDR, SGMC
    },
}


# Create high-resolution (HR) configurations for CEUS
regional_scale_southmid_cont = {
    "BASELINE_BISON_PP": regional_scale_ceus["BASELINE_BISON"].copy(),
    "JITTER_JELLYFISH_PP": regional_scale_ceus["JITTER_JELLYFISH"].copy(),
}

model_name = "BASELINE_BISON_PP"
update = regional_scale_southmid_cont[model_name].copy()
changes = [("Mag_RTP", "Mag_RTP_SMidCont"),
           ("Mag_RTP_HGM", "Mag_RTP_PGRV_HGM_SMidCont")]
regional_scale_southmid_cont[model_name] = update_model_config(update, changes)

model_name = "JITTER_JELLYFISH_PP"
update = regional_scale_southmid_cont[model_name].copy()
changes = [("Mag_RTP", "Mag_RTP_SMidCont"),
           ("Mag_RTP_HGM", "Mag_RTP_PGRV_HGM_SMidCont")]
regional_scale_southmid_cont[model_name] = update_model_config(update, changes)


# Delet unnecessary stuff
del update, model_name
