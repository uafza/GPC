"""
Utility functions for SwissCAT+ data handling and file naming.
"""

def construct_filename(
    project_info: dict = None,
    *,
    extension: str = "csv",
    sample: str | None = None,
    method: str | None = None,
    suffix: str | None = None,
) -> str:
    """
    Construct a standardized SwissCAT+ filename.

    Format (fields included only if present after defaults/overrides):
    A##_T##_E##_DEVICE[_B##][_R##][_M##][_SAMPLE][_SUFFIX].extension

    Parameters
    ----------
    project_info : dict, optional
        Keys: Project, Task, Experiment, Equipment
        Optional: Batch, Repeat, Method
    extension : str, default="csv"
        File extension (no dot).
    sample : str, optional
        Sample or barcode (will be trimmed to 10 chars, spaces removed).
    method : str, optional
        Override/force method (e.g., "M01"). If not given, uses project_info["Method"] if present.
    suffix : str, optional
        Optional trailing tag (e.g., "CLBRTN" for calibration).

    Returns
    -------
    str
    """

    defaults = {
        "Project":   "A000",
        "Task":      "T00",
        "Experiment":"E00",
        "Equipment": "DEVICE",
        "Batch":     None,
        "Repeat":    None,
        "Method":    None,
    }

    info = defaults.copy()
    if project_info:
        info.update(project_info)

    # Allow explicit 'method=' arg to override dict
    if method is not None:
        info["Method"] = method

    parts = [
        info["Project"].upper(),
        info["Task"].upper(),
        info["Experiment"].upper(),
        info["Equipment"].upper(),
    ]

    if info.get("Batch"):
        parts.append(info["Batch"].upper())
    if info.get("Repeat"):
        parts.append(info["Repeat"].upper())
    if info.get("Method"):
        parts.append(info["Method"].upper())

    if sample:
        parts.append(str(sample).replace(" ", "")[:10])

    if suffix:
        parts.append(suffix.upper())

    return "_".join(parts) + f".{extension.lower()}"
