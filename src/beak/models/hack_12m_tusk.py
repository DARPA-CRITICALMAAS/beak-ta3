"""
This file contains the models used for the magmatic nico experiment.
The BISON model is used as a geophysical baseline model for deeper sources (even shallow, but no surface data).
The JELLYFISH model is most comparable to the google sheet model configuration, but does not include high-res data.
"""

regional_scale_great_basin = {
    "BASELINE_BISON": {
        "Geophysics_Gravity_Bouguer_HGM": True,
        "Geophysics_Gravity_Bouguer_Up30km": True,
        "Geophysics_Gravity_Bouguer_Up30km_HGM": True,
        "Geophysics_Gravity_Isostatic": True,
        "Geophysics_Gravity_Isostatic_HGM": True,
        "Geophysics_MT2023_15km": True,
        "Geophysics_MT2023_9km": True,
        "Geophysics_Radiometric_Potassium": False,
        "Geophysics_Radiometric_Thorium": False,
        "Geophysics_Radiometric_Uranium": False,
        "Geophysics_LAB": True,
        "Geophysics_LAB_HGM": True,
        "Geophysics_Moho": True,
        "Geophysics_Mag_RTP": True,
        "Geophysics_Mag_RTP_HGM": True,
    }
}
