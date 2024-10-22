from cdr_schemas.area_extraction import Area_Extraction

test_1 = {
    "model_version": "0.0.1",
    "coordinates": [[[443, 852], [5648, 852], [5648, 7869], [443, 7869], [443, 852]]],
    "model": "jataware_extraction",
    "category": "map_area",
}


def test_area_extraction_1():
    results = Area_Extraction(**test_1).model_dump_json()
    Area_Extraction.model_validate_json(results)


test_2 = {
    "model_version": "0.0.1",
    "model": "jataware_extraction",
    "coordinates": [[[2, 3, 5, 6]]],
    "category": "point_legend_area",
    "text": "hi",
}


def test_area_extraction_2():
    results = Area_Extraction(**test_2).model_dump_json()
    Area_Extraction.model_validate_json(results)
