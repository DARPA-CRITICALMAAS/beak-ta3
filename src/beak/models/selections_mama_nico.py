hackathon_6m = {
    "national": {
        "dropout_duck": {
            # Gravity
            "IsostaticGravity": True,
            "Gravity_Up30km_HGM": True,
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": False,
            # Magnetic
            "Mag_RTP": True,
            "Mag_RTP_HGM_DeepSources": True,
            "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
            "Mag_LogAnalyticSignal": True,
            # Seismic
            "Moho": True,
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            # Magnetotelluric
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
        }
    }
}

hackathon_9m = {
    "regional_upmidwest": {
        "dropout_duck": {
                # Geophysics: Seismic
                "LAB": True,
                "LAB_HGM": True,
                "LAB_Worms_Proximity": True,
                "Moho": True,
                # Geophysics: Gravity
                "SatelliteGravity_ShapeIndex": True,
                "Gravity": True,
                "Gravity_HGM": True,
                "ShallowGravitySources_Worms_Proximity": False,
                "Gravity_Up30km": True,
                "Gravity_Up30km_HGM": True,
                "DeepGravitySources_Worms_Proximity": True,
                "IsostaticGravity": True,
                "IsostaticGravity_HGM": True,
                "IsostaticGravity_Worms_Proximity": False,
                # Geophysics: Magnetic
                "Mag_RTP": True,
                "Mag_RTP_HGM": True,
                "ShallowMagSources_Worms_Proximity": False,
                "Mag_RTP_DeepSources": True,
                "Mag_RTP_HGM_DeepSources": True,
                "DeepMagSources_Worms_Proximity": True,
                # Geophysics: Magnetotelluric
                "CONUS_MT2023_9km": True,
                "CONUS_MT2023_15km": True,
                "CONUS_MT2023_30km": True,
        },
        "mafic_moose": {
            # Gravity
            "SatelliteGravity_ShapeIndex": True,
            "Gravity": True,
            "Gravity_HGM": True,
            "ShallowGravitySources_Worms_Proximity": False,
            "Gravity_Up30km": True,
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            "IsostaticGravity_Worms_Proximity": False,
            # Magnetic
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "ShallowMagSources_Worms_Proximity": False,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismic
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotelluric
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            "CONUS_MT2023_48km": True,
            "CONUS_MT2023_92km": True,
            # Radiometrics
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
            "mafic_macrostrat": True,
        },
    }
}

hackathon_12m = {
    "regional_upmidwest": {
        "dropout_duck": {
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
            "ShallowMagSources_Worms_Proximity": False,
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
        "loss_llama": {
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
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            "Mag_LogAnalyticSignal": True,
            "CEUS_MAG_DRTP_CEUSSSC_R0": True,
            "CEUS_MAG_DRTP_TDR_CEUSSSC_R0": True,
            "CEUS_MAG_DRTP_HD_TDR_CEUSSSC_R0": True,
            "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            "CONUS_MT2023_48km": True,
            "CONUS_MT2023_92km": True,
            # Geology
            "BGE_ALL_NORILSK": True,
            "BGE_DEPOSIT_NORILSK": True,
            "BGE_ROCK_TYPE_NORILSK": True,
        },
    }
}

