import json


def read_metadata_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def content_prompt(metadata_json):
    description = metadata_json.get("description", {})
    if isinstance(description, str):
        return description
    if not isinstance(description, dict):
        return "A high-resolution RGB optical remote sensing image."
    return (
        description.get("content_sentence")
        or "A high-resolution RGB optical remote sensing image."
    )


def format_sensor_metadata(metadata, role):
    fields = []
    simple_fields = [
        ("sensor", "sensor"),
        ("platform", "platform"),
        ("modality", "modality"),
        ("gsd_m", "ground sample distance"),
        ("acquisition_datetime", "acquisition time"),
        ("cloud_cover_percent", "cloud cover percent"),
        ("sun_elevation_deg", "sun elevation degrees"),
        ("sun_azimuth_deg", "sun azimuth degrees"),
    ]
    for key, label in simple_fields:
        value = metadata.get(key)
        if value is not None and value != "":
            fields.append(f"{label}: {value}")

    bands = metadata.get("bands")
    if bands:
        fields.append(f"bands used: {', '.join(str(x) for x in bands[:3])}")

    latitude = metadata.get("center_lat")
    longitude = metadata.get("center_lon")
    if latitude is not None and longitude is not None:
        fields.append(f"center latitude: {latitude}")
        fields.append(f"center longitude: {longitude}")

    scale_factor = metadata.get("scale_factor")
    if scale_factor is not None:
        fields.append(f"scale factor: {scale_factor}x")

    if not fields:
        return ""
    return f"{role} metadata: " + "; ".join(fields) + "."


def build_prompt(lr_json_path, hr_json_path, include_metadata=True):
    lr_json = read_metadata_json(lr_json_path)
    hr_json = read_metadata_json(hr_json_path)
    parts = [content_prompt(hr_json)]
    if include_metadata:
        parts.append("Task: optical image super-resolution.")
        parts.append(format_sensor_metadata(lr_json.get("metadata", {}), "Input"))
        parts.append(format_sensor_metadata(hr_json.get("metadata", {}), "Target"))
    return " ".join(part for part in parts if part)


def _clean_sensor_name(value):
    value = str(value or "optical").replace("_", " ")
    if value.lower() == "naip simulated lr":
        return "simulated NAIP"
    return value


def build_clip_prompt(lr_json_path, hr_json_path):
    """Build a compact prompt for CLIP's fixed 77-token context."""
    lr_json = read_metadata_json(lr_json_path)
    hr_json = read_metadata_json(hr_json_path)
    description = hr_json.get("description", {})
    if not isinstance(description, dict):
        description = {}
    lr_metadata = lr_json.get("metadata", {})
    hr_metadata = hr_json.get("metadata", {})

    scene = description.get("scene") or "remote sensing scene"
    objects = description.get("objects") or []
    if not isinstance(objects, (list, tuple)):
        objects = [objects]
    object_text = ", ".join(str(value) for value in objects[:4])
    target_sensor = _clean_sensor_name(hr_metadata.get("sensor", "NAIP"))
    input_sensor = _clean_sensor_name(lr_metadata.get("sensor", "low resolution"))
    input_gsd = lr_metadata.get("gsd_m") or 10
    target_gsd = hr_metadata.get("gsd_m") or 2.5
    scale = hr_metadata.get("scale_factor") or lr_metadata.get("scale_factor") or 4

    prompt = (
        f"High-resolution {target_sensor} RGB aerial image of {scene}"
        + (f", showing {object_text}" if object_text else "")
        + f". {scale}x super-resolution from {input_sensor} RGB at "
        f"{input_gsd} meter GSD to {target_gsd} meter GSD."
    )
    return prompt


def build_prompts(lr_json_path, hr_json_path, include_metadata=True):
    """Return (short CLIP prompt, detailed T5 prompt)."""
    return (
        build_clip_prompt(lr_json_path, hr_json_path),
        build_prompt(lr_json_path, hr_json_path, include_metadata),
    )
