"""
This file contains the models used for the MVT_NAT experiment.

Currently, the models are:<br>
    `TUNGSTEN_SKARN_V1`: Config V1<p>
"""

models = {
    "TUNGSTEN_SKARN_V1": {
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
        # Aeroradiometry from ScienceBase
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
    },
}
