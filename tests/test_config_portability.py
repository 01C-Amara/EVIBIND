from pathlib import Path

import tapbench.config as config


def test_missing_repository_defaults_use_packaged_portable_data(
    tmp_path, monkeypatch
) -> None:
    missing_subgrids = tmp_path / "configs" / "hypothesis_subgrids.yaml"
    missing_models = tmp_path / "configs" / "model_pins.yaml"
    missing_map = tmp_path / "analysis" / "hypothesis_map.yaml"
    monkeypatch.setattr(config, "DEFAULT_SUBGRIDS", missing_subgrids)
    monkeypatch.setattr(config, "DEFAULT_MODELS", missing_models)
    monkeypatch.setattr(config, "DEFAULT_HYPOTHESIS_MAP", missing_map)

    loaded = config.load_experiment_config(
        subgrids_path=missing_subgrids,
        models_path=missing_models,
        hypothesis_map_path=missing_map,
    )

    assert loaded.subgrids["schema_version"]
    assert loaded.hypothesis_map["schema_version"]
    assert loaded.models["schema_version"] == "tapbench.portable_model_catalog.v1"
    for model in loaded.models["evaluated_models"]:
        artifact = model["backend_defaults"].get("model_artifact")
        if artifact:
            assert not Path(artifact).is_absolute()
