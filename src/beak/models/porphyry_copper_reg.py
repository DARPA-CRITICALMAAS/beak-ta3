"""
This file contains the models used for the MVT_REG experiment.
"""

models = {
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
    # Models for the 12 month hackathon
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
        # Earth-MRI Survey                          # Merging with NAT-scale data ?
        "Geophysics_RTP_SWNM": True,
        "Geophysics_RTP_HGM_SWNM": True,
        "Geophysics_Mag_1VD_SWNM": True,
        "Radiometric_U_SWNM": True,
        "Radiometric_K_SWNM": True,
        "Radiometric_Th_SWNM": True,
        # Magnetotellurics
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        # Geology
        "Precambrian_Units_Whitmeyer": True,         # TBD, query needed
        "QFAULTS_Proximity": True,                   # TBD, data available,
                                                     # only few areas covered within the ROI (no faults or not mapped?)
        "Aster_Alteration_Proximity": True,
        # Geochemistry
        "Stream_HUC_AGG_Cu_Q50_Log": True,           # Decision what to to with nodata/log transform


    }
}
