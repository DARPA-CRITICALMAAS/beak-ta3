hackathon_9m = {
    "national": {
        "baseline_bison": {
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
            # Geophysics: Seismic
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Geophysics: Magnetotelluric
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            # Geophysics: Radiometrics
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
        },
        "feature_fox": {
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
            # Geophysics: Seismic
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Geophysics: Magnetotelluric
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            # Geophysics: Radiometrics
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
            # Geology: Lithology
            "Geol_SGMC_Dist_Jurassic_Oligocene_Intrusive": True,
            "Geol_SGMC_Dist_Jurassic_Oligocene_Volcanic": True,
            # Geology: Age
            "sgmc_geology_min_ma": True,
            "sgmc_geology_max_ma": True,
            # Geochemistry: Stream Sediment
            "Stream_HUC_AGG_Cu_Q50": True,
        }
    }
}

hackathon_12m = {
    "regional_southwest": {
        "dropout_duck": {
            # Gravity
            "Gravity_Up30km": True,
            "Gravity_Up30km_HGM": True,
            "ShallowGravitySources_Worms_Proximity": False,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            "IsostaticGravity_Worms_Proximity": False,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "Mag_RTP_VD": True,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
        },
        "camel_case": {
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
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            # Radiometrics
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
            # Geology
            "All_Alteration_Proximity": True,
            "cdr_sgmc_proximity_faults_shear_zones": True,
        }
    },
    "regional_southwest_nm": {
        "dropout_duck": {
            # Gravity
            "Gravity_Up30km": True,
            "Gravity_Up30km_HGM": True,
            "ShallowGravitySources_Worms_Proximity": False,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            "IsostaticGravity_Worms_Proximity": False,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "Mag_RTP_VD": True,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
        },
        "loss_llama": {
            # Gravity
            "Gravity_Up30km": True,
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            # Magnetics
            "Geophysics_Mag_RTP_SWNM": True,
            "Geophysics_Mag_RTP_HGM_SWNM": True,
            "Geophysics_Mag_1VD_SWNM": True,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            # Radiometrics
            "Radiometric_K_SWNM": True,
            "Radiometric_Th_SWNM": True,
            "Radiometric_U_SWNM": True,
            # Geology
            "BGE_ALL": True,
            "BGE_DEPOSIT": True,
            "BGE_ROCK_TYPE": True,
            "CEUS_pc_Whitmeyer2007_R0_Unit": True,
            "All_Alteration_Proximity": True,
            "cdr_sgmc_proximity_faults_shear_zones": True,
            # Geochemistry
            "Geochemistry_StrmSed_Cu_SBR": True,
            "Geochemistry_StrmSed_Mo_SBR": True,
            "Geochemistry_StrmSed_Zn_SBR": True,
        }
    }
}