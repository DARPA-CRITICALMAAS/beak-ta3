"""
This file contains the models used for the MVT_REG experiment.
"""

models = {
    # Template
    "DUMMY_DICTIONARY": {
        # Some comments
        "Geology_Lithology_Majority": True,
        "Geology_Lithology_Minority": True,
        "...": False,
    },
    # Models for the 12 month hackathon
    "JITTER_JELLYFISH": {
        # Seismics
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
        "Moho": True,
        # Gravity
        "Gravity_Up30km_HGM": True,
        "DeepGravitySources_Worms_Proximity": True,
        "IsostaticGravity": True,
        # Magnetics
        "Mag_RTP": False,
        "Mag_RTP_HGM": False,
        "Mag_RTP_SMIDCONT_MERGE": True,                   # TBD, MERGING (?)
        "Mag_RTP_HGM_SMIDCONT_MERGE": True,               # TBD, MERGING (?)
        "ShallowMagSources_Worms_Proximity":True,
        "Mag_RTP_DeepSources": True,
        "Mag_RTP_HGM_DeepSources": True,
        "DeepMagSources_Worms_Proximity": True,
        # Magnetotelluric
        "CONUS_MT2023_30km": True,
        # Geology
        "SGMC_Geology_Proximity": True,                 # TBD, query needed
        "SGMC_Age_Proximity": False,                    # TBD, query needed
        "Paleo_Latitude_Period_Minimum": True,          # TBD, from Lawley
        "Ancient_Passive_Margins_Proximity": True,      # TBD, from RAW
        "QFAULTS_Proximity": True,                      # TBD, data available bot no overlap with ROI
    }
}
