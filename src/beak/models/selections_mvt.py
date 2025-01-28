hackathon_6m = {
   "national": {
        "lawley_baseline": {
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
        "lawley_preferred": {
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
        },
        "lawley_preferred_isograv": {
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
        },
   }
}

hackathon_12m = {
    "regional_ceus": {
        "dropout_duck": {
            # Gravity
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "ShallowMagSources_Worms_Proximity": False,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotelluric
            "CONUS_MT2023_30km": True,
        },
        "loss_llama": {
            # Gravity
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotelluric
            "CONUS_MT2023_30km": True,
            # Geology
            "BGE_ALL": True,
            "BGE_DEPOSIT": True,
            "BGE_ROCK_TYPE": True,
            "Geology_BlackShale_Proximity": True,               # Lawley
            "Geology_Paleolatitude_Period_Minimum": True,       # Lawley
            "Geology_PassiveMargin_Proximity": True,            # Lawley
            "cdr_sgmc_proximity_faults_shear_zones": True,      # CDR, SGMC
        },
        "goofy_gopher": {
            # Gravity
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            # Magnetics
            "Mag_RTP": True,
            "Mag_RTP_HGM": True,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotelluric
            "CONUS_MT2023_30km": True,
            # Geology
            "Geology_BlackShale_Proximity": True,               # Lawley
            "Geology_PassiveMargin_Proximity": True,            # Lawley
            "cdr_sgmc_proximity_faults_shear_zones": True,      # CDR, SGMC
        },
    },
    "regional_southmid_cont": {
        "dropout_duck": {
            # Gravity
            "Gravity_Up30km_HGM": True,
            "DeepGravitySources_Worms_Proximity": True,
            "IsostaticGravity": True,
            # Magnetics
            "Mag_RTP_SMidCont": True,
            "Mag_RTP_PGRV_HGM_SMidCont": True,
            "Mag_RTP_DeepSources": True,
            "Mag_RTP_HGM_DeepSources": True,
            "DeepMagSources_Worms_Proximity": True,
            # Seismics
            "LAB": True,
            "LAB_HGM": True,
            "LAB_Worms_Proximity": True,
            "Moho": True,
            # Magnetotelluric
            "CONUS_MT2023_30km": True,
        }
    }
}
