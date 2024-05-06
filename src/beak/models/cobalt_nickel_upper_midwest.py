"""
This file contains the models used for the MVT_REG experiment.
"""

models = {
    "FEATURE_FOX": {
        # Geophysics: Seismic
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": True,
        "Gravity": True,
        "Gravity_HGM": True,
        "ShallowGravitySources_Worms_Proximity": True,
        "Gravity_Up30km": True,
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        "IsostaticGravity_HGM": True,
        "IsostaticGravity_Worms_Proximity": True,
        # Geophysics: Magnetic
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "ShallowMagSources_Worms_Proximity": True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Geophysics: Magnetotelluric
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        "CONUS_MT2023_48km": True,
        "CONUS_MT2023_92km": True,
        # Geophysics: Radiometrics
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
    },
}
