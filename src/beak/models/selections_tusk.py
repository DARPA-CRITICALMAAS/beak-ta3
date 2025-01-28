hackathon_6m = {
    "national": {
        "baseline_bison": {
            # Gravity
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
            "IsostaticGravity": True,
            "Gravity_Up30km_HGM": True,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM_DeepSources": True,
            "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
            "Mag_LogAnalyticSignal": True,
            # Seismics
            "Moho": True,
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            # Magnetotellurics
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            # Radiometrics
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
        },
        "feature_fox": {
            # Gravity from ScienceBase
            "SatelliteGravity_ShapeIndex": True,
            "Gravity": True,
            "Gravity_HGM": True,
            "Gravity_Up30km": True,
            # Gravity Worms from Datacube
            "Gravity_Bouguer_HGM_Worms_Proximity": True,
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
            # Isostatic Gravity from ScienceBase
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            "IsostaticGravity_HGM_Worms": True,
            # Magnetic from ScienceBase
            "Mag_RTP_HGM": True,
            "Mag_RTP_HGM_DeepSources": True,
            # Gravity Worms from Datacube
            "Magnetic_HGM_Worms_Proximity": True,
            "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
            # Seismic from ScienceBase
            "LAB": True,
            "Moho": True,
            # Aeroradiometry from ScienceBase
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
            # Geology from Datacube
            "Geology_Lithology_Majority": True,
            "Geology_Lithology_Minority": True,
            "Geology_Period_Maximum_Majority": True,
            "Geology_Period_Minimum_Majority": True,
            # Sedimentary dictionaries from Datacube
            "Geology_Dictionary_Calcareous": True,
            "Geology_Dictionary_Carbonaceous": True,
            "Geology_Dictionary_FineClastic": True,
            # Igneous dictionaries from Datacube
            "Geology_Dictionary_Felsic": True,
            "Geology_Dictionary_Intermediate": True,
            "Geology_Dictionary_UltramaficMafic": True,
            # Metamorphic dictionaries from Datacube
            "Geology_Dictionary_Anatectic": True,
            "Geology_Dictionary_Gneissose": True,
            "Geology_Dictionary_Schistose": True,
            # Proximity from Datacube
            "Terrane_Proximity": True,
            "Geology_PassiveMargin_Proximity": True,
            "Geology_BlackShale_Proximity": True,
            "Geology_Fault_Proximity": True,
        },
        "warping_worm": {
            # Gravity from ScienceBase
            "SatelliteGravity_ShapeIndex": True,
            "Gravity": True,
            "Gravity_HGM": True,
            "Gravity_Up30km": True,
            # Isostatic Gravity from ScienceBase
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            # Magnetic from ScienceBase
            "Mag_RTP_HGM": True,
            "Mag_RTP_HGM_DeepSources": True,
            # Seismic from ScienceBase
            "LAB": True,
            "Moho": True,
            # Aeroradiometry from ScienceBase
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
            # Geology from Datacube
            "Geology_Lithology_Majority": True,
            "Geology_Lithology_Minority": True,
            "Geology_Period_Maximum_Majority": True,
            "Geology_Period_Minimum_Majority": True,
            # Sedimentary dictionaries from Datacube
            "Geology_Dictionary_Calcareous": True,
            "Geology_Dictionary_Carbonaceous": True,
            "Geology_Dictionary_FineClastic": True,
            # Igneous dictionaries from Datacube
            "Geology_Dictionary_Felsic": True,
            "Geology_Dictionary_Intermediate": True,
            "Geology_Dictionary_UltramaficMafic": True,
            # Metamorphic dictionaries from Datacube
            "Geology_Dictionary_Anatectic": True,
            "Geology_Dictionary_Gneissose": True,
            "Geology_Dictionary_Schistose": True,
            # Proximity from Datacube
            "Terrane_Proximity": True,
            "Geology_PassiveMargin_Proximity": True,
            "Geology_BlackShale_Proximity": True,
            "Geology_Fault_Proximity": True,
        }
    }
}

hackathon_9m = {
    "regional_alaska_ytu": {
        "baseline_bison": {
            # Geophysics: Gravity
            "SatelliteGravity_ShapeIndex": True,
            "Gravity": True,
            "Gravity_HGM": True,
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            # Geophysics: Magnetic
            "Mag_RTP_YTU": True,
            "Mag_RTP_HGM_YTU": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Geophysics: Seismic
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Geophysics: Radiometrics
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
        },
        "feature_fox": {
            # Geophysics: Gravity
            "SatelliteGravity_ShapeIndex": True,
            "Gravity": True,
            "Gravity_HGM": True,
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            # Geophysics: Magnetic
            "Mag_RTP_YTU": True,
            "Mag_RTP_HGM_YTU": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Geophysics: Seismic
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Geophysics: Radiometrics
            "NAMrad_K": True,
            "NAMrad_Th": True,
            "NAMrad_U": True,
            # Geology: Lithology
            "Geol_AGDB_Dist_Carbonate": True,
            "Geol_AGDB_Dist_Plutonic": True,
            # Geochemistry: Stream Sediment
            "Stream_HUC_AGG_W_Q50_Log": True,
            "Stream_HUC_AGG_Sn_Q50_Log": True,
            "Stream_HUC_AGG_Mo_Q50_Log": True,
        }
    }
}

hackathon_12m = {
    "regional_great_basin": {
        "baseline_bison": {
            "Geophysics_Gravity_Bouguer_HGM": True,
            "Geophysics_Gravity_Bouguer_Up30km": True,
            "Geophysics_Gravity_Bouguer_Up30km_HGM": True,
            "Geophysics_Gravity_Isostatic": True,
            "Geophysics_Gravity_Isostatic_HGM": True,
            "Geophysics_Mag_RTP": True,
            "Geophysics_Mag_RTP_HGM": True,
            "Geophysics_LAB": True,
            "Geophysics_LAB_HGM": True,
            "Geophysics_Moho": True,
            "Geophysics_MT2023_15km": True,
            "Geophysics_MT2023_9km": True,
        },
        "morph_mamba": {
            # Gravity
            "Geophysics_Gravity_Bouguer_HGM": True,
            "Geophysics_Gravity_Bouguer_Up30km": True,
            "Geophysics_Gravity_Bouguer_Up30km_HGM": True,
            "Geophysics_Gravity_Isostatic": True,
            "Geophysics_Gravity_Isostatic_HGM": True,
            # Magnetics
            "Geophysics_Mag_RTP": True,
            "Geophysics_Mag_RTP_HGM": True,
            # Seismics
            "Geophysics_LAB": True,
            "Geophysics_LAB_HGM": True,
            "Geophysics_Moho": True,
            # Magnetotellurics
            "Geophysics_MT2023_15km": True,
            "Geophysics_MT2023_9km": True,
            # Minerals
            "Carb_MRAT_W_proximity": True,
            "Epid_Chlor_MRAT_W_proximity": True,
            "Carbonates": True,
            "Epidote_chlorite": True,
            "Muscovite": True,
            "Muscovite_sil_rich": True,
            "Hydro_silica": True,
            "Kaol_Alun": True,
            "Kaol_Alun_Sil": True,
            # Geology
            "sgmc_geology_min_ma": True,
            "sgmc_geology_max_ma": True,
            "sgmc_preproc_v1.tungsten_skarn_bge_rock_types": False,
            "SGMC_Geology_carbonate_v2_proximity": True,
            "SGMC_Geology_granodiorite_v2_proximity": True,
        }
    },
    "regional_great_basin_geodawn": {
        "baseline_bison": {
            # Gravity
            "Geophysics_Gravity_Bouguer_HGM": True,
            "Geophysics_Gravity_Bouguer_Up30km": True,
            "Geophysics_Gravity_Bouguer_Up30km_HGM": True,
            "Geophysics_Gravity_Isostatic": True,
            "Geophysics_Gravity_Isostatic_HGM": True,
            # Magnetics
            "Geophysics_Mag_RTP": False,
            "Geophysics_Mag_RTP_HGM": True,
            "22103_rtp_a2": True,
            "22103_tmi_a2": True,
            "22103_tmi_vg_a2": True,
            # Seismics
            "Geophysics_LAB": True,
            "Geophysics_LAB_HGM": True,
            "Geophysics_Moho": True,
            # Magnetotellurics
            "Geophysics_MT2023_15km": True,
            "Geophysics_MT2023_9km": True,
        },
        "morph_mamba": {
            # Gravity
            "Geophysics_Gravity_Bouguer_HGM": True,
            "Geophysics_Gravity_Bouguer_Up30km": True,
            "Geophysics_Gravity_Bouguer_Up30km_HGM": True,
            "Geophysics_Gravity_Isostatic": True,
            "Geophysics_Gravity_Isostatic_HGM": True,
            # Magnetics
            "Geophysics_Mag_RTP": False,
            "Geophysics_Mag_RTP_HGM": False,
            "22103_rtp_a2": True,
            "22103_tmi_a2": True,
            "22103_tmi_hg_a2": True,
            "22103_tmi_vg_a2": True,
            # Seismics
            "Geophysics_LAB": True,
            "Geophysics_LAB_HGM": True,
            "Geophysics_Moho": True,
            # Magnetotellurics
            "Geophysics_MT2023_15km": True,
            "Geophysics_MT2023_9km": True,
            # Radiometrics
            "22103_k_a2": True,
            "22103_th_a2": True,
            "22103_u_a2": True,
            # Geology
            "sgmc_geology_min_ma": True,
            "sgmc_geology_max_ma": True,
            "sgmc_preproc_v1.tungsten_skarn_bge_rock_types": False,
            "SGMC_Geology_carbonate_v2_proximity": True,
            "SGMC_Geology_granodiorite_v2_proximity": True,
            # Minerals
            "Carbonates": True,
            "Epidote_chlorite": True,
            "Muscovite": True,
            "Muscovite_sil_rich": True,
            "Hydro_silica": True,
            "Kaol_Alun": True,
            "Kaol_Alun_Sil": True
        }
    }
}