hackathon_12m = {
    "regional_southwest": {
        "dropout_duck": {
            # Gravity
            "IsostaticGravity": True,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "ShallowMagSources_Worms_Proximity": False,
            "CEUS_MAG_TMAG_AAS_CEUSSSC_R0_Log": False,
            "Mag_LogAnalyticSignal": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
        },
        "morph_mamba": {
            # Gravity
            "IsostaticGravity": True,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "Mag_LogAnalyticSignal": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
            # Geology
            "SGMC_Distance_to_Unconsolidated": True,
            "SGMC_MinAge_Min": True,
            "Geology_Paleolatitude_Period_Minimum": True,
            # Geology/Spectral
            "minerals": True,
            # Geochemistry
            "Geochemistry_StrmSed_Cs_nure_reanalysis": True,
            "Geochemistry_StrmSed_Li_nure_reanalysis": True,
            "Geochemistry_StrmSed_Mo_nure_reanalysis": True,
            "Geochemistry_StrmSed_Rb_nure_reanalysis": True,
            "Geochemistry_StrmSed_Sn_nure_reanalysis": True,
            "Geochemistry_StrmSed_W_nure_reanalysis": True,
            # Elevation
            "Li_dem": False,
            "Li_slope": True,
            "Li_tpi": True,
            "Li_valley_depth": True,
        },
    }
}