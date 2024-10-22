from cdr_schemas.metadata import MapMetaData

test_1 = {
    "title": "Geographical Map of Newland",
    "year": 2020,
    "crs": "EPSG:4267",
    "authors": ["John Doe", "Jane Smith"],
    "organization": "National Geography Institute",
    "scale": 24000,
    "quadrangle_name": "Newland Northeast",
    "map_shape": "rectangular",
    "map_color_scheme": None,
    "publisher": "GeoPublishers Inc.",
    "state": "Newland State",
    "model": "GeoMapping2020",
    "model_version": "v1.2",
}


def test_metadata_1():
    results = MapMetaData(**test_1).model_dump_json()
    MapMetaData.model_validate_json(results)


test_2 = {
    "title": "Geographical Map of Newland",
    "model": "GeoMapping2020",
    "model_version": "v1.2",
}


def test_metadata_2():
    results = MapMetaData(**test_2).model_dump_json()
    MapMetaData.model_validate_json(results)
