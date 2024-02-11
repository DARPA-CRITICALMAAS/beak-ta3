"""
This file contains the models used for the Tungsten Skarn experiment.

Currently, the models are:<br>
    `TS_BASELINE`: The baseline model for MVT deposits from the Lawley et al. 2022 paper, containing only geophysical layers.<p>
    `TS_PREFERRED`: The preferred model for MVT deposits from the Lawley et al. 2022 paper, containing geophysical and geological layers.
"""

models = {
    "TS_GEOPHYSICS": {
        #--Numerical Data from LAWLEY 22
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
        #--AeroRadiometric
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
    },
    "TS_PREFERRED": {
        #--Categorical Data from LAWLEY 22
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
        #--Numerical Data from LAWLEY 22
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
        #--AeroRadiometric
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
    },
    "TS_PREFERRED_NUMERICAL": {
        #--Numerical Data from LAWLEY 22
        # Proximity
        "Terrane_Proximity": True,
        "Geology_PassiveMargin_Proximity": True,
        "Geology_BlackShale_Proximity": True,
        "Geology_Fault_Proximity": True,
        # Paleogeography
        "Geology_Paleolatitude_Period_Maximum": True,
        "Geology_Paleolatitude_Period_Minimum": True,
        # Gravity
        "Gravity_GOCE_ShapeIndex": False,
        "Gravity_Bouguer": False,
        "Gravity_Bouguer_HGM": False,
        "Gravity_Bouguer_UpCont30km_HGM": False,
        "Gravity_Bouguer_HGM_Worms_Proximity": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        # Magnetic
        "Magnetic_HGM": False,
        "Magnetic_LongWavelength_HGM": False,
        "Magnetic_HGM_Worms_Proximity": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        # Seismic
        "Seismic_LAB_Priestley": False,
        "Seismic_LAB_Hoggard": False,
        "Seismic_Moho": False,
        #--AeroRadiometric
        "NAMrad_K": True,
        "NAMrad_Th": True,
        "NAMrad_U": True,
    },
    "TS_GROUND_TRUTH": {
        "Training_MVT_Deposit_Present": False,
        "Training_MVT_Occurrence_Present": False,
        "Training_MVT_Present": True,
    },

}
