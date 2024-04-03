"""
This file contains the models used for the MVT_NAT experiment.

Currently, the models are:<br>
    `MVT_BASELINE`: The baseline model for MVT deposits from the Lawley et al. 2022 paper, containing only geophysical layers.<p>
    `MVT_PREFERRED`: The preferred model for MVT deposits from the Lawley et al. 2022 paper, containing geophysical and geological layers.
"""

models = {
    "MVT_PREFERRED_GEOPHYSICS_ISOGRAV_SOM": {
        # Gravity from ScienceBase
        "Gravity": False,
        "Gravity_HGM": False,
        "Gravity_Up30km": True,
        "SatelliteGravity_ShapeIndex": True,
        # Gravity Worms from Datacube
        "Gravity_Bouguer_HGM_Worms_Proximity": True,
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": True,
        # Isostatic gravity
        "US_IsostaticGravity_HGM_WGS84": True,
        "US_IsostaticGravity_WGS84": True,
        # Magnetics from ScienceBase
        "MagRTP": False,
        "MagRTP_HGM": False,
        "MagRTP_HGMDeepSources": True,
        "Mag_AnalyticSignal_cog": True,
        # Magnetic Worms from Datacube
        "Magnetic_HGM_Worms_Proximity": True,
        "Magnetic_LongWavelength_HGM_Worms_Proximity": True,
        # Magnetics from HighResMagnetic
        "SMidCont_RTP_UTM15": True,
        "SMidCont_RTP_PGRV_HGM_UTM15": True,
        # Seismics from ScienceBase
        "LAB": True,
        "Moho": True,
    },
}
