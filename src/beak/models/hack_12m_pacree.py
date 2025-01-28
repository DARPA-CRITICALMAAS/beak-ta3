"""
This file contains the models used for the peralkaline carbinatite REE experiment.
"""
national_scale = {
    "BASELINE_BISON": {
        # Geophysics: Seismic
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": False,                # experimental
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        "IsostaticGravity_HGM": True,
        # Geophysics: Magnetic
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        "Mag_LogAnalyticSignal": True,
        # Geophysics: Magnetotelluric
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
    },
    "JITTER_JELLYFISH": {
        # Geophysics: Seismic
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": False,  # experimental
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        "IsostaticGravity_HGM": True,
        # Geophysics: Magnetic
        "Mag_RTP": True,
        "Mag_RTP_HGM": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        "Mag_LogAnalyticSignal": True,  # experimental
        # Geophysics: Magnetotelluric
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        # Geophysics: Radiometry
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
        # Geology
        "TA2_beg_rock_type_sgmc_pacree": True,
        "dist_SGMC_Geol_carbonatite_alkali_pegmat": True
        # ADD HERE
    },
    "HACKING_HAMSTER": {
        # Geophysics: Seismic
        "LAB": False,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Geophysics: Gravity
        "SatelliteGravity_ShapeIndex": False,  # experimental
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": False,
        "IsostaticGravity": True,
        "Gravity_HGM": True,
        # Geophysics: Magnetic
        "Mag_RTP": False,
        "Mag_RTP_HGM": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        "Mag_LogAnalyticSignal": False,  # experimental
        "Mag_RTP_VD": True,
        # Geophysics: Magnetotelluric
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_48km": True,
        # Geophysics: Radiometry
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
        # Geology
        "TA2_beg_rock_type_sgmc_pacree": True,
        "dist_SGMC_Geol_carbonatite_alkali_pegmat": True
        # ADD HERE
    },
}

selection = {
    "national": {
        "baseline_bison": {
            # Geophysics: Gravity
            "SatelliteGravity_ShapeIndex": False,
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            # Geophysics: Magnetic
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            "Mag_LogAnalyticSignal": True,
            # Geophysics: Seismic
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Geophysics: Magnetotelluric
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
        },
        "hacking_hamster": {
            # Geophysics: Gravity
            "SatelliteGravity_ShapeIndex": False,
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": False,
            "IsostaticGravity": True,
            "Gravity_HGM": True,
            # Geophysics: Magnetic
            "Mag_RTP": False,
            "Mag_RTP_HGM": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            "Mag_LogAnalyticSignal": False,
            "Mag_RTP_VD": True,
            # Geophysics: Seismic
            "LAB": False,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Geophysics: Magnetotelluric
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_48km": True,
            # Geophysics: Radiometry
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
            # Geology
            "TA2_beg_rock_type_sgmc_pacree": True,
            "dist_SGMC_Geol_carbonatite_alkali_pegmat": True,
        },
    }
}