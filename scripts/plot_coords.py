from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from synthgen.config import GenerationConfig
from synthgen.banks.anatomy import AnatomyBank
from synthgen.banks.montage import MontageBank

ANATOMY_ID = "mne_sample"

CONFIG_PATH = Path("config/default.yaml")
DEEPSIF_DIR = Path("banks/atlases/deepsif")
ELECTRODE_DIR = Path(f"banks/leadfield/{ANATOMY_ID}/standard_1005_90")

def _region_centers(vertex_coords: np.ndarray, parcellation: np.ndarray, n_regions: int) -> np.ndarray:
    centers = np.zeros((n_regions, 3), dtype=np.float64)
    counts = np.zeros(n_regions, dtype=np.int64)
    for r in range(n_regions):
        m = parcellation == r
        if m.any():
            centers[r] = vertex_coords[m].mean(axis=0)
            counts[r] = m.sum()
    if (counts == 0).any():
        empty = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Empty parcels in parcellation: {empty}")
    return centers.astype(np.float32)

def _load_deepsif_998(deepsif_dir: Path):
    # TVB's Connectivity.from_file resolves relative paths against tvb_data's
    # own root, not the cwd, so always pass an absolute path.
    zip_path = (deepsif_dir / "connectivity_998.zip").resolve()
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Run: python scripts/fetch_atlases.py --deepsif"
        )
    from tvb.datatypes.connectivity import Connectivity as TVBConnectivity
    return TVBConnectivity.from_file(str(zip_path))

def plot_coords():    
    
    cfg = GenerationConfig.from_yaml(Path(CONFIG_PATH))    
    anat = AnatomyBank(cfg.anatomy_bank).load(ANATOMY_ID, scheme="deepsif_994")
    montage = MontageBank(cfg.montage_bank).load("standard_1005_90")
    
    n_regions = int(anat.parcellation.max()) + 1
    assert len(anat.region_labels) == n_regions, "region_labels length must match parcel count"

    base_tag = "deepsif_998"
    vertex_coords = anat.vertex_coords
    centers = _region_centers(anat.vertex_coords, anat.parcellation, n_regions)
    base_conn = _load_deepsif_998(DEEPSIF_DIR)
    base_conn_centres = base_conn.centres
    montage_coords = montage.coords
    
    
    electrode_coords = np.load(ELECTRODE_DIR / "electrode_coords.npz").get("coords")
    assert electrode_coords.shape == montage_coords.shape, "Electrode and montage coords must have same shape"
    
    gap = np.mean(electrode_coords, axis=0) - np.mean(montage_coords, axis=0)
    
    print(f"GAP: X: {gap[0]:.2f} mm, Y: {gap[1]:.2f} mm, Z: {gap[2]:.2f} mm")
    
    electrode_coords -= gap
    
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(projection="3d")
    """
    vertex_coords = vertex_coords[np.random.choice(vertex_coords.shape[0], size=1000, replace=False)]
    centers = centers[np.random.choice(centers.shape[0], size=200, replace=False)]
    base_conn_centres = base_conn_centres[np.random.choice(base_conn_centres.shape[0], size=200, replace=False)]"""
    
    ax.scatter(vertex_coords[:, 0], vertex_coords[:, 1], vertex_coords[:, 2], s=2, alpha=0.5, label="Vertices")
    ax.scatter(electrode_coords[:, 0], electrode_coords[:, 1], electrode_coords[:, 2], s=50, color="orange", label="Electrode Positions")
    ax.scatter(montage_coords[:, 0], montage_coords[:, 1], montage_coords[:, 2], s=50, color="green", label="Montage Points")
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=10, color="red", label="Parcel Centers")
    ax.scatter(base_conn_centres[:, 0], base_conn_centres[:, 1], base_conn_centres[:, 2], s=10, color="blue", label=f"{base_tag} Region Centers")
    ax.legend()
    ax.set_title("Anatomy Vertex Coordinates, Parcel Centers, and Montage Points")
    plt.show()

if __name__ == "__main__":
    plot_coords()