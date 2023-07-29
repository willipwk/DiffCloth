import os
import shutil
import numpy as np
import glob


def load_obj_and_process(source_filename, dest_filename):        # single side mesh to double side mesh
    f = open(source_filename, 'r')
    df = open(dest_filename, 'w')
    verts = []
    faces = []
    for line in f:
        if line[0:2] == 'v ':
            df.write(line)

        if line[-1] == '\n':
            line = line[:-1]
        wlist = line.split(' ')
        if wlist[0] == 'f':
            faces.append(wlist[1:4])
        elif wlist[0] == 'v':
            verts.append(wlist[1:4])
    f.close()
    df.close()

    with open(dest_filename, 'a') as df:
        for i in faces:
            # print('{} {} {}'.format(i[2], i[1], i[0]))
            df.write('\nf {} {} {}'.format(i[2], i[1], i[0]))   # add mesh in reverse side


    verts = np.array([tuple(map(float, v)) for v in verts])
    faces = np.array([tuple(map(int, map(eval, f))) for f in faces])

    return verts, faces


perturbed_obj_files = glob.glob("output/026/episode3/*/tie_final_*.obj")
count = 0
for filename in perturbed_obj_files:
    print(count)
    filename_short = filename[filename.rfind('/') + 1:]
    key_path = filename[filename.find('/'):filename.rfind('/') + 1]
    output_dir = f"./tmp_double/{key_path}/{count}/"
    
    # every obj is a dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    shutil.copy(filename, output_dir)
    os.rename(output_dir + filename_short, output_dir + "front.obj")

    obj_front_side_path = os.path.join(output_dir, 'front.obj')
    obj_back_side_path = os.path.join(output_dir, 'back.obj')

    load_obj_and_process(obj_front_side_path, obj_back_side_path)
    count += 1