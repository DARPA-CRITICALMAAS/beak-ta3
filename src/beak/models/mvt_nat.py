"""
This file contains the models used for the MVT_NAT experiment.

Currently, the models are:<br>
    `MVT_BASELINE`: The baseline model for MVT deposits from the Lawley et al. 2022 paper, containing only geophysical layers.<p>
    `MVT_PREFERRED`: The preferred model for MVT deposits from the Lawley et al. 2022 paper, containing geophysical and geological layers.
"""

models = {
    "MVT_BASELINE": {
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
        "Seismic_LAB_Priestley": False,
        "Seismic_LAB_Hoggard": True,
        "Seismic_Moho": True,
    },
    "MVT_PREFERRED": {
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
        "Seismic_LAB_Priestley": False,
        "Seismic_LAB_Hoggard": True,
        "Seismic_Moho": True,
    },
    "MVT_PREFERRED_NUMERICAL": {
        # Geology
        "Geology_Lithology_Majority": False,
        "Geology_Lithology_Minority": False,
        "Geology_Period_Maximum_Majority": False,
        "Geology_Period_Minimum_Majority": False,
        # Sedimentary dictionaries
        "Geology_Dictionary_Calcareous": False,
        "Geology_Dictionary_Carbonaceous": False,
        "Geology_Dictionary_FineClastic": False,
        # Igneous dictionaries
        "Geology_Dictionary_Felsic": False,
        "Geology_Dictionary_Intermediate": False,
        "Geology_Dictionary_Ultramafic": False,
        # Metamorphic dictionaries
        "Geology_Dictionary_Anatectic": False,
        "Geology_Dictionary_Gneissose": False,
        "Geology_Dictionary_Schistose": False,
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
        "Seismic_LAB_Priestley": False,
        "Seismic_LAB_Hoggard": True,
        "Seismic_Moho": True,
    },
    "MVT_PREFERRED_ISOGRAV": {
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
        # Paleogeography from Datacube
        "Geology_Paleolatitude_Period_Maximum": True,
        "Geology_Paleolatitude_Period_Minimum": True,
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
}
