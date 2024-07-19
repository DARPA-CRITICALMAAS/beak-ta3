"""
This file contains the models used for the MVT_REG experiment.
"""

regional_scale_ceus = {
    "DUMMY_DICTIONARY": {
        # Some comments
        "Geology_Lithology_Majority": True,
        "Geology_Lithology_Minority": True,
        "...": False,
    },
    "BASELINE_BISON": {
        # Magnetics
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        "CEUS_MAG_DRTP_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_HD_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_CEUSSSC_R0": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Gravity
        "CEUS_GRAV_RI_HD_1VD_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_HD_CEUSSSC_R0": True,
        "CEUS_GRAV_Isostatic_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_1VD_CEUSSSC_R0": True,
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
    },
    "FEATURE_FOX": {
        # Magnetics
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        "CEUS_MAG_DRTP_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_HD_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_CEUSSSC_R0": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Gravity
        "CEUS_GRAV_RI_HD_1VD_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_HD_CEUSSSC_R0": True,
        "CEUS_GRAV_Isostatic_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_1VD_CEUSSSC_R0": True,
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geology
        "distance_to_geology_macrostrat": True,
        "distance_to_faults_macrostrat": True,
        "cambrian-to-early-ordovician-carbonates_zero_macrostrat": True,
        # Things to wait on: Geochemistry
        "...": False,
    },
    "HACKING_HAMSTER": {  # different order than the FEATURE_FOX
        # Magnetics new
        "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
        "CEUS_MAG_DRTP_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_HD_TDR_CEUSSSC_R0": True,
        "CEUS_MAG_DRTP_CEUSSSC_R0": True,
        # Gravity new
        "CEUS_GRAV_RI_HD_1VD_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_HD_CEUSSSC_R0": True,
        "CEUS_GRAV_Isostatic_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_CEUSSSC_R0": True,
        "CEUS_GRAV_RI_1VD_CEUSSSC_R0": True,
        # Things we have
        "Moho": True,
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Geology
        "distance_to_geology_macrostrat": True,
        "distance_to_faults_macrostrat": True,
        "cambrian-to-early-ordovician-carbonates_zero": True,
        # Things to wait on: Geochemistry
        "...": False,
    },
}
