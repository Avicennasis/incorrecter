from incorrecter.lexicon import EGGCORNS, HOMOPHONES, QWERTY_NEIGHBORS
from incorrecter.noise import semantic_edits


def test_no_entry_maps_to_itself():
    for table in (EGGCORNS, HOMOPHONES):
        for clean, wrong in table.items():
            assert clean.lower() != wrong.lower(), clean


def test_every_entry_is_reachable():
    for clean, wrong in {**EGGCORNS, **HOMOPHONES}.items():
        text = f"Honestly {clean} today"
        assert wrong in {e.replacement for e in semantic_edits(text)}, clean


def test_champing_is_not_listed_as_a_corruption():
    # "champing at the bit" is the original idiom; mapping to it corrects the text instead of breaking it.
    assert "chomping at the bit" not in EGGCORNS


def test_qwerty_neighbours_are_symmetric():
    for key, neighbours in QWERTY_NEIGHBORS.items():
        for other in neighbours:
            assert key in QWERTY_NEIGHBORS[other], (key, other)
