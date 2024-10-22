from cdr_schemas.features.point_features import PointLegendAndFeaturesResult

# legend only info
test_1 = {
    "id": "1",
    "legend_provenance": {"model": "test", "model_version": "1", "confidence": 0},
    "label": "nickel_deposit",
    "name": "nickel_deposit",
    "map_unit": None,
    "legend_bbox": [1, 2, 3, 4],
    "abbreviation": "ni_deposit",
    "description": "nickel deposit description",
}

def test_point_extraction_1():
    results = PointLegendAndFeaturesResult(**test_1).model_dump_json()
    PointLegendAndFeaturesResult.model_validate_json(results)


# feature only info
test_2 = {
    "id": "1",
    "cdr_projection_id": "1234alksdfjslakfjd",
    "name": "",
    "point_features": {
        "features": [
            {
                "id": "11",
                "geometry": {"coordinates": [23.33, 122]},
                "properties": {
                    "model": "test",
                     "model_version": "1",
                     "bbox":[]},
            }
        ]
    },
}

# extraction info only
def test_point_extraction_2():
    results = PointLegendAndFeaturesResult(**test_2).model_dump_json()
    PointLegendAndFeaturesResult.model_validate_json(results)