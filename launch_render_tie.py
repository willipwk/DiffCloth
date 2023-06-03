import glob
import os

from tqdm import tqdm

if __name__ == "__main__":
    render_mode = "DIRECT"
    perturbed_obj_files = glob.glob("output/*.obj")
    for fn in tqdm(perturbed_obj_files):
        cmd = f"python3 render_tie.py {fn} {render_mode}"
        os.system(cmd)
