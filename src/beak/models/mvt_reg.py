"""
This file contains the models used for the MVT_REG experiment.

Currently, the models are:<br>
    `MVT_BASELINE`: The baseline model for MVT deposits from the Lawley et al. 2022 paper, containing only geophysical layers.<p>
    `MVT_PREFERRED`: The preferred model for MVT deposits from the Lawley et al. 2022 paper, containing geophysical and geological layers.
"""

models = {
    "MVT_GEOPHYSICS": {
        #--Numerical Data from LAWLEY 22
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
        #--Numerical Data from MCCAFFERTY 23
        # Gravity
        "Gravity": False,
        "Gravity_HGM": False,
        "Gravity_Up30km": False,
        "Gravity_Up30km_HGM": False,
        "SatelliteGravity_ShapeIndex": False,
        # Magnetic
        "Mag": False,
        "MagRTP": False,
        "MagRTP_DeepSources": False,
        "MagRTP_HGM": False,
        "MagRTP_HGMDeepSources": False,
        "MagRTP_VD": True,
        # Seismic
        "Moho": True,
        "LAB": True,
        #--Aeromagnetics
        "SMidCont_RTP_PGRV_HGM_UTM15": True,
        "SMidCont_RTP_UTM15": True,
        #-- Isostatic Gravimetry
        "US_IsostaticGravity_HGM_WGS84": True,
        "US_IsostaticGravity_WGS84": True,
        "US_IsostaticGravity_HGM_WGS84_Worms": True,
    },
    "MVT_PREFERRED": {
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
        #--Numerical Data from MCCAFFERTY 23
        # Gravity
        "Gravity": True,
        "Gravity_HGM": True,
        "Gravity_Up30km": True,
        "Gravity_Up30km_HGM": True,
        "SatelliteGravity_ShapeIndex": True,
        # Magnetic
        "Mag": False,
        "MagRTP": False,
        "MagRTP_DeepSources": True,
        "MagRTP_HGM": False,
        "MagRTP_HGMDeepSources": True,
        "MagRTP_VD": True,
        # Seismic
        "Moho": True,
        "LAB": True,
        #--Aeromagnetics
        "SMidCont_RTP_PGRV_HGM_UTM15": True,
        "SMidCont_RTP_UTM15": True,
        #-- Isostatic Gravimetry
        "US_IsostaticGravity_HGM_WGS84": True,
        "US_IsostaticGravity_WGS84": True,
        "US_IsostaticGravity_HGM_WGS84_Worms": True,
    },
    "MVT_PREFERRED_NUMERICAL": {
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
        #--Numerical Data from MCCAFFERTY 23
        # Gravity
        "Gravity": True,
        "Gravity_HGM": True,
        "Gravity_Up30km": True,
        "Gravity_Up30km_HGM": True,
        "SatelliteGravity_ShapeIndex": True,
        # Magnetic
        "Mag": False,
        "MagRTP": False,
        "MagRTP_DeepSources": True,
        "MagRTP_HGM": False,
        "MagRTP_HGMDeepSources": True,
        "MagRTP_VD": True,
        # Seismic
        "Moho": True,
        "LAB": True,
        #--Aeromagnetics
        "SMidCont_RTP_PGRV_HGM_UTM15": True,
        "SMidCont_RTP_UTM15": True,
        #-- Isostatic Gravimetry
        "US_IsostaticGravity_HGM_WGS84": True,
        "US_IsostaticGravity_WGS84": True,
        "US_IsostaticGravity_HGM_WGS84_Worms": True,
    },
    "MVT_LABELS": {
        "MVT_EPSG_32615_RES_50_0": True,
    },

}
