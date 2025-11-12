# src/parser/run.py
from parser.workflow_gpc import run_gpc_workflow
from parser.workflow_hplc import run_hplc_workflow

def run_workflow(Project_Info: dict, base_folder: str = "data", out_dir: str = "results"):
    """
    Decide which workflow to run (GPC or HPLC) based on Project_Info['Workflow'].
    """
    mode = Project_Info.get("Workflow", "").upper()
    if mode not in {"GPC", "HPLC"}:
        raise ValueError(f"Project_Info['Workflow'] must be 'GPC' or 'HPLC', got '{mode}'")

    print(f"Selected workflow: {mode}")
    if mode == "GPC":
        plot_flag = bool(Project_Info.get("CalibPlot", False))
        run_gpc_workflow(base_folder=base_folder, out_dir=out_dir, project_info=Project_Info, plot_calibration=plot_flag)
    else:
        run_hplc_workflow(base_folder=base_folder, out_dir=out_dir, project_info=Project_Info)
