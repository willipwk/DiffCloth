import os
import sys

step_num = int(sys.argv[1])
assert step_num in [0, 13, 26], "Wrong step number"
if step_num == 0:
    att_idx = 347
elif step_num == 13:
    att_idx = 14
elif step_num == 26:
    att_idx = 266
n_output = 500
seed = step_num * 10000 + 3243

i = 0
while i <= n_output:
    print("==>", i)
    os.system(
        f"python3 /home/zyuwei/Projects/DiffCloth/src/python_code/perturb_tie.py -mode 0 -i {i} -seed {seed+i} -n_openmp_thread 9 -obj_fn remeshed/tie_step_data/step{step_num}/start_tie.obj -att_idx {att_idx}"
    )
    output_fn = f"output/perturbed_tie_step{step_num}_{i}.obj"
    if not os.path.isfile(output_fn):
        seed += n_output
    else:
        i += 1
