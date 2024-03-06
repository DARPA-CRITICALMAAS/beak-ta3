"""
This file contains the models used for the MVT_NAT experiment.

Currently, the models are:<br>
    `MAGMATIC_NICKEL_V1`: Config V1<p>

"""

models = {
    "MAGMATIC_NICKEL_V1": {
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
        # Gravity from ScienceBase
        "SatelliteGravity_ShapeIndex": True,
        "Gravity": True,
        "Gravity_HGM": True,
        "Gravity_Up30km": True,
        # Gravity Worms from Datacube
        "Gravity_Bouguer_HGM_Worms_Proximity": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        # Isostatic Gravity from ScienceBase
        "US_IsostaticGravity_WGS84": True,
        "US_IsostaticGravity_HGM_WGS84": True,
        "US_IsostaticGravity_HGM_WGS84_Worms": True,
        # Magnetic from ScienceBase
        "MagRTP_HGM": True,
        "MagRTP_HGMDeepSources": True,
        # Gravity Worms from Datacube
        "Magnetic_HGM_Worms_Proximity": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        # Seismic from ScienceBase
        "LAB": True,
        "Moho": True,
    },
    "MAGMATIC_NICKEL_HM6_V1": {
        # Geology Ultramafics
        "US_Ultramafics": True,
        # Isostatic Gravity from ScienceBase
        "US_IsostaticGravity_WGS84": True,
        # Magnetic from ScienceBase
        "MagRTP_HGM": True,
        # Magnetotelluric from ScienceBase
        "CONUS_MT2023_30km_cog": True,
    },
    "SOURCE_V1": {
        # Isostatic Gravity from ScienceBase
        "US_IsostaticGravity_WGS84": True,
        # Magnetic from ScienceBase
        "MagRTP_HGM": True,
        # Magnetotelluric from ScienceBase
        "CONUS_MT2023_30km_cog": True,
    },
    "SOURCE_V2": {
        "US_IsostaticGravity_WGS84": True,
        "MagRTP": True,
        "CONUS_MT2023_9km_cog": True,
        "CONUS_MT2023_15km_cog": True,
        "Mag_AnalyticSignal_cog": True,
        "Moho": True,
        "LAB": True,
    },
    "PATHWAYS_V1": {
        "LAB_HGM_cog": True,
        "LAB_Worms_Proximity": True,
        "Gravity_Up30km_HGM": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        "MagRTP_HGMDeepSources": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        "MagRTP_HGM": True,
        "Magnetic_HGM_Worms_Proximity": True,
    },
    "PATHWAYS_V2": {
        "LAB_HGM_cog": True,
        "LAB_Worms_Proximity": True,
        "Gravity_Up30km_HGM": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        "MagRTP_HGMDeepSources": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        "CONUS_MT2023_30km_cog": True,
    },
    "BASELINE_FINAL": {
        "US_IsostaticGravity_WGS84": True,
        "Gravity_Up30km_HGM": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        "CONUS_MT2023_9km_cog": True,
        "CONUS_MT2023_15km_cog": True,
        "CONUS_MT2023_30km_cog": True,
        "MagRTP": True,
        "MagRTP_HGMDeepSources": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        "Mag_AnalyticSignal_cog": True,
        "Moho": True,
        "LAB": True,
        "LAB_HGM_cog": True,
        "LAB_Worms_Proximity": True,
    },
    "BASELINE_FINAL_SOURCE_BINARY": {
        "SOURCE_TRESHOLDED": False,  # thresholding in code
        "US_IsostaticGravity_WGS84": True,
        "Gravity_Up30km_HGM": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        "CONUS_MT2023_9km_cog": True,
        "CONUS_MT2023_15km_cog": True,
        "CONUS_MT2023_30km_cog": True,
        "MagRTP": True,
        "MagRTP_HGMDeepSources": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        "Mag_AnalyticSignal_cog": True,
        "Moho": True,
        "LAB": True,
        "LAB_HGM_cog": True,
        "LAB_Worms_Proximity": True,
    },
    "BASELINE_FINAL_DICTS": {
        "Geology_Dictionary_Igneous": True,
        "Geology_Dictionary_Intermediate": True,
        "Geology_Dictionary_UltramaficMafic": True,
        "US_IsostaticGravity_WGS84": True,
        "Gravity_Up30km_HGM": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        "CONUS_MT2023_9km_cog": True,
        "CONUS_MT2023_15km_cog": True,
        "CONUS_MT2023_30km_cog": True,
        "MagRTP": True,
        "MagRTP_HGMDeepSources": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        "Mag_AnalyticSignal_cog": True,
        "Moho": True,
        "LAB": True,
        "LAB_HGM_cog": True,
        "LAB_Worms_Proximity": True,
    },
}
