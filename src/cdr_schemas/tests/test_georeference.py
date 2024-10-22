from cdr_schemas.georeference import GeoreferenceResults, GroundControlPoint

test_1 = {
    "cog_id": "124918591d191e1906990e9913995b49346b34393939317943d81e9a31bc0180",
    "system": "jataware_georef",
    "georeference_results": [
        {
            "projections": [
                {
                    "crs": "EPSG:26713",
                    "gcp_ids": [
                        "6b153e74-ed7a-4612-8314-5d1158b08bc8",
                        "dd639987-696b-408c-a6fb-a6506bc9b15f",
                        "005d0f5f-fa53-48d8-9392-341d1cc83baf",
                        "659712b6-998b-409d-a949-8df9e3472113",
                    ],
                    "file_name": "124918591d191e1906990e9913995b49346b34393939317943d81e9a31bc0180_03b1f83b-42e2-483e-b2f2-107acc6bbf00.pro.cog.tif",
                }
            ],
            "map_area": {
                "model_version": "0.0.1",
                "bbox": [],
                "confidence": None,
                "coordinates": [
                    [[443, 852], [5648, 852], [5648, 7869], [443, 7869], [443, 852]]
                ],
                "model": "jataware_extraction",
                "text": "",
                "type": "Polygon",
                "category": "map_area",
            },
            "likely_CRSs": ["EPSG:26713"],
        }
    ],
    "system_version": "0.1.0",
    "gcps": [
        {
            "model_version": "0.0.1",
            "crs": "EPSG:4267",
            "confidence": None,
            "map_geom": {"latitude": 43.375, "type": "Point", "longitude": -103.875},
            "px_geom": {
                "rows_from_top": 396.3211364746094,
                "type": "Point",
                "columns_from_left": 553.1830596923828,
            },
            "model": "jataware_extraction",
            "gcp_id": "6b153e74-ed7a-4612-8314-5d1158b08bc8",
        },
        {
            "model_version": "0.0.1",
            "crs": "EPSG:4267",
            "confidence": None,
            "map_geom": {"latitude": 43.375, "type": "Point", "longitude": -103.75},
            "px_geom": {
                "rows_from_top": 398.4816436767578,
                "type": "Point",
                "columns_from_left": 5548.023864746094,
            },
            "model": "jataware_extraction",
            "gcp_id": "dd639987-696b-408c-a6fb-a6506bc9b15f",
        },
        {
            "model_version": "0.0.1",
            "crs": "EPSG:4267",
            "confidence": None,
            "map_geom": {"latitude": 43.25, "type": "Point", "longitude": -103.875},
            "px_geom": {
                "rows_from_top": 7210.130844116211,
                "type": "Point",
                "columns_from_left": 542.9927520751953,
            },
            "model": "jataware_extraction",
            "gcp_id": "005d0f5f-fa53-48d8-9392-341d1cc83baf",
        },
        {
            "model_version": "0.0.1",
            "crs": "EPSG:4267",
            "confidence": None,
            "map_geom": {"latitude": 43.25, "type": "Point", "longitude": -103.75},
            "px_geom": {
                "rows_from_top": 7213.253692626953,
                "type": "Point",
                "columns_from_left": 5545.915435791016,
            },
            "model": "jataware_extraction",
            "gcp_id": "659712b6-998b-409d-a949-8df9e3472113",
        },
    ],
}


def test_georef_1():
    results = GeoreferenceResults(**test_1).model_dump_json()
    GeoreferenceResults.model_validate_json(results)


test_2 = {
    "cog_id": "124918591d191e1906990e9913995b49346b34393939317943d81e9a31bc0180",
    "system": "jataware_georef",
    "georeference_results": [
        {
            "projections": [],
        }
    ],
    "system_version": "0.1.0",
    "gcps": [
        {
            "model_version": "0.0.1",
            "crs": "EPSG:4267",
            "map_geom": {"latitude": 43.375, "longitude": -103.875},
            "px_geom": {"rows_from_top": 396, "columns_from_left": 553},
            "model": "test",
            "gcp_id": "6b153e74-ed7a-4612-8314-5d1158b08bc8",
        }
    ],
}


def test_georef_2():
    results = GeoreferenceResults(**test_2).model_dump_json()
    GeoreferenceResults.model_validate_json(results)


test_3 = {
    "model_version": "0.0.1",
    "crs": "EPSG:4267",
    "confidence": None,
    "map_geom": {"latitude": 43.25, "type": "Point", "longitude": -103.875},
    "px_geom": {
        "rows_from_top": 7210.130844116211,
        "type": "Point",
        "columns_from_left": 542.9927520751953,
    },
    "model": "jataware_extraction",
    "gcp_id": "005d0f5f-fa53-48d8-9392-341d1cc83baf",
}


def test_gcp_1():
    results = GroundControlPoint(**test_3).model_dump_json()
    GroundControlPoint.model_validate_json(results)


test_4 = {
    "model_version": "0.0.1",
    "crs": "",
    "map_geom": {"latitude": 43.25, "longitude": None},
    "px_geom": {"rows_from_top": 7210, "columns_from_left": 542},
    "model": "jataware_extraction",
    "gcp_id": "005d0f5f-fa53-48d8-9392-341d1cc83baf",
}


def test_gcp_2():
    results = GroundControlPoint(**test_4).model_dump_json()
    GroundControlPoint.model_validate_json(results)
