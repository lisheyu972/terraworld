# AID dataset layout for FluxSR-FTD

[Super-resolution guide](../../README.md#super-resolution-on-aid)

Set `AID_DATA_ROOT` to the directory that contains `json/`. Store one HR and one LR metadata file per sample. The loader discovers the JSON files recursively and matches each pair by sample ID. The paths below show one test pair.

```text
aid/
├── images/
│   ├── hr/scene.tif
│   └── lr/scene.tif
└── json/
    ├── SR_HR_scene.json
    └── SR_LR_scene.json
```

The HR record can provide a text prompt in `description.content_sentence`.

```json
{
  "image_id": "SR_HR_scene",
  "image_path": "images/hr/scene.tif",
  "split": "test",
  "metadata": {"resolution_role": "hr"},
  "description": {"content_sentence": "An aerial image of a residential area."}
}
```

The LR record uses the same ID suffix and split.

```json
{
  "image_id": "SR_LR_scene",
  "image_path": "images/lr/scene.tif",
  "split": "test",
  "metadata": {"resolution_role": "lr"}
}
```

Training reads pairs marked `train` and uses `test` pairs for previews. Inference reads `test` by default. The ×4 configuration uses the paired LR image. The ×2 and ×8 configurations synthesize inputs by bicubic downsampling of the HR image and use the same paired metadata format. Use dataset-relative `image_path` values to keep the dataset portable.
