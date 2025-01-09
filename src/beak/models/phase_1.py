"""
This file contains the models used for the MVT_NAT experiment.
"""

mvt = {
    "national": {
        "lawley_preferred": {
            # Geology
            "Geology_Lithology_Majority": True,
            "Geology_Lithology_Minority": True,
            "Geology_Period_Maximum_Majority": True,
            "Geology_Period_Minimum_Majority": True,
            # Sedimentary dictionaries
            "Geology_Dictionary_Calcareous": True,
            "Geology_Dictionary_Carbonaceous": True,
            "Geology_Dictionary_FineClastic": True,
            # Igneous dictionaries
            "Geology_Dictionary_Felsic": True,
            "Geology_Dictionary_Intermediate": True,
            "Geology_Dictionary_UltramaficMafic": True,
            # Metamorphic dictionaries
            "Geology_Dictionary_Anatectic": True,
            "Geology_Dictionary_Gneissose": True,
            "Geology_Dictionary_Schistose": True,
            # Proximity
            "Terrane_Proximity": True,
            "Geology_PassiveMargin_Proximity": True,
            "Geology_BlackShale_Proximity": True,
            "Geology_Fault_Proximity": True,
            # Paleogeography
            "Geology_Paleolatitude_Period_Maximum": True,
            "Geology_Paleolatitude_Period_Minimum": True,
            # Gravity
            "Gravity_GOCE_ShapeIndex": True,
            "Gravity_Bouguer": True,
            "Gravity_Bouguer_HGM": True,
            "Gravity_Bouguer_UpCont30km_HGM": True,
            "Gravity_Bouguer_HGM_Worms_Proximity": True,
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
            # Magnetic
            "Magnetic_HGM": True,
            "Magnetic_LongWavelength_HGM": True,
            "Magnetic_HGM_Worms_Proximity": True,
            "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
            # Seismic
            "Seismic_LAB_Hoggard": True,
            "Seismic_Moho": True,
        },
        "lawley_preferred_isograv": {
            # Geology
            "Geology_Lithology_Majority": True,
            "Geology_Lithology_Minority": True,
            "Geology_Period_Maximum_Majority": True,
            "Geology_Period_Minimum_Majority": True,
            # Sedimentary dictionaries
            "Geology_Dictionary_Calcareous": True,
            "Geology_Dictionary_Carbonaceous": True,
            "Geology_Dictionary_FineClastic": True,
            # Igneous dictionaries
            "Geology_Dictionary_Felsic": True,
            "Geology_Dictionary_Intermediate": True,
            "Geology_Dictionary_UltramaficMafic": True,
            # Metamorphic dictionaries
            "Geology_Dictionary_Anatectic": True,
            "Geology_Dictionary_Gneissose": True,
            "Geology_Dictionary_Schistose": True,
            # Proximity
            "Terrane_Proximity": True,
            "Geology_PassiveMargin_Proximity": True,
            "Geology_BlackShale_Proximity": True,
            "Geology_Fault_Proximity": True,
            # Paleogeography
            "Geology_Paleolatitude_Period_Maximum": True,
            "Geology_Paleolatitude_Period_Minimum": True,
            # Gravity
            "Gravity_GOCE_ShapeIndex": True,
            "IsostaticGravity": True,
            "IsostaticGravity_HGM": True,
            "IsostaticGravity_HGM_Worms": False,
            "Gravity_Bouguer_UpCont30km_HGM": True,
            "Gravity_Bouguer_HGM_Worms_Proximity": True,
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
            # Magnetic
            "Magnetic_HGM": True,
            "Magnetic_LongWavelength_HGM": True,
            "Magnetic_HGM_Worms_Proximity": True,
            "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
            # Seismic
            "Seismic_LAB_Hoggard": True,
            "Seismic_Moho": True,
        },
    },
}

nico = {
    "national": {
        "baseline": {
            "IsostaticGravity": True,
            "Gravity_Up30km_HGM": True,
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
            "CONUS_MT2023_9km": True,
            "CONUS_MT2023_15km": True,
            "CONUS_MT2023_30km": True,
            "Mag_RTP": True,
            "Mag_RTP_HGM_DeepSources": True,
            "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
            "Mag_LogAnalyticSignal": True,
            "Moho": True,
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
        },
    }
}

pacree = {
    "national": {
        "hacking_hamster": {
            # Geophysics: Seismic
            "LAB": False,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Geophysics: Gravity
            "SatelliteGravity_ShapeIndex": False,                   # experimental
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": False,
            "IsostaticGravity": True,
            "Gravity_HGM": True,
            # Geophysics: Magnetic
            "Mag_RTP": False,
            "Mag_RTP_HGM": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            "Mag_LogAnalyticSignal": False,                         # experimental
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
        },
    }
}

poco = {
    "national": {
        "feature_fox": {
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
                "Stream_HUC_AGG_Cu_Q50": True,
                # Geology: Lithology
                "Geol_SGMC_Dist_Jurassic_Oligocene_Intrusive": True,
                "Geol_SGMC_Dist_Jurassic_Oligocene_Volcanic": True,
                # Geology: Age
                "sgmc_geology_min_ma": True,
                "sgmc_geology_max_ma": True,
            },
    }
}
