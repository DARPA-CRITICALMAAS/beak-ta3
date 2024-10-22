from cdr_schemas.features.line_features import LineLegendAndFeaturesResult

# legend only info
test_1 = {
    "id": "1",
    "legend_provenance": {"model": "test", "model_version": "1", "confidence": 0},
    "name": "nickel",
    "map_unit": None,
    "legend_bbox": [1, 2, 3, 4],
    "abbreviation": "ni",
    "description": "nickel description",
}

def test_line_extraction_1():
    results = LineLegendAndFeaturesResult(**test_1).model_dump_json()
    LineLegendAndFeaturesResult.model_validate_json(results)


# feature only info
test_2 = {
    "id": "1",
    "cdr_projection_id": "1234alksdfjslakfjd",
    "line_features": {
        "features": [
            {
                "id": "11",
                "geometry": {"coordinates": [[23.33, 122],[23.23, 122.6]]},
                "properties": {"model": "test", "model_version": "1"},
            }
        ]
    },
}

# extraction info only
def test_line_extraction_2():
    results = LineLegendAndFeaturesResult(**test_2).model_dump_json()
    LineLegendAndFeaturesResult.model_validate_json(results)
