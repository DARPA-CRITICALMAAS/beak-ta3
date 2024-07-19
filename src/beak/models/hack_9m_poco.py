"""
This file contains the models used for the MVT_REG experiment.
"""

national_scale = {
    "BASELINE_BISON": {
        # Model without geochemistry and geological proxies
        # Geophysics: Seismic
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": True,
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        # Geophysics: Magnetic
        "Mag_RTP": True,
        "Mag_LogAnalyticSignal": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Geophysics: Magnetotelluric
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        # Geophysics: Radiometrics
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
    },
    "FEATURE_FOX": {
        # Model with geochemistry and geological proxies
        # Geophysics: Seismic
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": True,
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        # Geophysics: Magnetic
        "Mag_RTP": True,
        "Mag_LogAnalyticSignal": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Geophysics: Magnetotelluric
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        # Geophysics: Radiometrics
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
        # Geochemistry: Stream Sediment
        "Stream_HUC_AGG_Cu_Q50_Log": True,
        # Geology: Lithology
        "Geol_SGMC_Dist_Jurassic_Oligocene_Intrusive": True,
        "Geol_SGMC_Dist_Jurassic_Oligocene_Volcanic": True,
        # Geology: Age
        "Geol_SGMC_MinAge": True,
        "Geol_SGMC_MaxAge": True,
    },
}
