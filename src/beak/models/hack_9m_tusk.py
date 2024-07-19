"""
This file contains the models used for the MVT_REG experiment.
"""

regional_scale_alaska_ytu = {
    "BASELINE_BISON": {
        # Geophysics: Seismic
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": True,
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "Gravity": True,
        "Gravity_HGM": True,
        # Geophysics: Magnetic
        "Mag_RTP_YTU": True,
        "Mag_RTP_HGM_YTU": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Geophysics: Radiometrics
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
    },
    "FEATURE_FOX": {
        # Geophysics: Seismic
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": True,
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "Gravity": True,
        "Gravity_HGM": True,
        # Geophysics: Magnetic
        "Mag_RTP_YTU": True,
        "Mag_RTP_HGM_YTU": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Geophysics: Radiometrics
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
        # Geochemistry: Stream Sediment
        "Stream_HUC_AGG_W_Q50_Log": True,
        "Stream_HUC_AGG_Sn_Q50_Log": True,
        "Stream_HUC_AGG_Mo_Q50_Log": True,
        # Geology: Lithology
        "Geol_AGDB_Dist_Carbonate": True,
        "Geol_AGDB_Dist_Plutonic": True,
    },
}
