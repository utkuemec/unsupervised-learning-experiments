"""Run every experiment in order. Usage: python src/run_all.py [--quick]"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "01_pca_kernel_pca.py",
    "02_kmeans_dbscan.py",
    "03_hierarchical.py",
    "04_gmm.py",
    "05_autoencoder.py",
]

if __name__ == "__main__":
    here = Path(__file__).parent
    extra = sys.argv[1:]
    for script in SCRIPTS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, str(here / script), *extra], check=True)
