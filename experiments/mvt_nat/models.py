"""
This file contains the models used for the MVT_NAT experiment.

Currently, the models are:<br>
    `MVT_PREFERRED`: The preferred model for MVT deposits from the Lawley et al. 2022 paper.
"""

models = {
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
        "Geology_Dictionary_Ultramafic": True,
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
        "Gravity_Bouguer_HGM_Worms_Proximity": True,
        # Magnetic
        "Magnetic_HGM": True,
        "Magnetic_LongWavelength_HGM": True,
        "Magnetic_HGM_Worms_Proximity": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        # Seismic
        "Seismic_LAB_Priestley": True,
        "Seismic_Moho": True,
    }
}
