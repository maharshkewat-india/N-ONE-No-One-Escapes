import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd

import app


def test_profile_labels_support_new_and_legacy_prefixes() -> None:
    assert app.get_profile_category("Staff_Maharsh_kewat") == "Staff"
    assert app.get_profile_category("Victim_vivek_patel") == "Victim"
    assert app.get_profile_category("Member_ravendra") == "Staff"
    assert app.get_profile_category("Lost_vivek_patel") == "Victim"
    assert app.get_profile_name("Staff_Maharsh_kewat") == "Maharsh kewat"


def test_registered_inventory_counts_and_categories(monkeypatch) -> None:
    test_dir = Path(__file__).parent / f"_face_inventory_{uuid4().hex}"
    registered_dir = test_dir / "registered_faces"
    registered_dir.mkdir(parents=True)
    try:
        for filename in ["Staff_alpha.jpg", "Victim_beta.png", "Member_legacy.jpeg", "notes.txt"]:
            (registered_dir / filename).touch()

        monkeypatch.setattr(app, "REG_DIR", registered_dir)
        inventory = app.get_registered_profiles_df()

        assert len(inventory) == 3
        assert inventory["category"].value_counts().to_dict() == {"Staff": 2, "Victim": 1}
        assert app.get_registered_profile_count() == 3
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_unknown_count_uses_unique_database_ids(monkeypatch) -> None:
    test_dir = Path(__file__).parent / f"_face_inventory_{uuid4().hex}"
    unknown_dir = test_dir / "unknown_faces"
    unknown_dir.mkdir(parents=True)
    unknown_db = test_dir / "unknown_person_db.csv"
    try:
        (unknown_dir / "unknown_001.jpg").touch()
        (unknown_dir / "unknown_002.jpg").touch()
        pd.DataFrame({"unknown_id": ["unknown_001", "unknown_001", "unknown_002"]}).to_csv(
            unknown_db, index=False
        )

        monkeypatch.setattr(app, "UNKNOWN_DIR", unknown_dir)
        monkeypatch.setattr(app, "UNKNOWN_DB_PATH", unknown_db)

        assert app.get_unknown_face_count() == 2
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_known_face_cache_can_be_limited_to_one_victim(monkeypatch) -> None:
    monkeypatch.setattr(
        app,
        "KNOWN_FACE_ENCODINGS",
        [
            {"profile_id": "Staff_a", "name": "a", "role": "Staff", "encoding": [1.0, 0.0]},
            {"profile_id": "Victim_b", "name": "b", "role": "Victim", "encoding": [1.0, 0.0]},
        ],
    )

    match = app.find_face_in_known_cache(
        [1.0, 0.0], profile_id="Victim_b", role="Victim"
    )

    assert match is not None
    assert match["profile_id"] == "Victim_b"
