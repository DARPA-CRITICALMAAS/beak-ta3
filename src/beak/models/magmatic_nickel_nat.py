"""
This file contains the models used for the MAGMATIC_NICKEL experiment.
"""

models = {
    "BASELINE_BISON": {
        # Gravity
        "IsostaticGravity": True,
        "Gravity_Up30km_HGM": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,  # LWL22
        # Magnetic
        "Mag_RTP": True,
        "Mag_RTP_HGM_DeepSources": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,  # LWL22
        "Mag_LogAnalyticSignal": True,
        # Magnetotelluric
        "CONUS_MT2023_9km": True,
        "CONUS_MT2023_15km": True,
        "CONUS_MT2023_30km": True,
        # Seismic
        "Moho": True,
        "LAB": True,
        "LAB_HGM": True,
        "LAB_Worms_Proximity": True,
    },
}
