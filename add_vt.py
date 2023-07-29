import bpy
import math
import os
import shutil
from mathutils import Vector
 
 
 
class Render_2D:
    def __init__(self) -> None:
        self.import_file_path = []
        self.import_dir_path = []
        self.import_file_name = []
        self.file_number = 0
 
    def __get_file_path(self, path):
        # get obj file
        import_file_path = []
        filelist = os.listdir(path)
        for i in range(len(filelist)):
            item =os.path.join(path,filelist[i])
            if os.path.isdir(item): # if item is dir, continue searching
                self.__get_file_path(item)
            else:
                if (os.path.splitext(item)[-1][1:] == 'obj'):
                    #import ipdb; ipdb.set_trace()
                    self.import_file_path.append(item)
                    self.import_dir_path.append(os.path.dirname(item))
                    self.import_file_name.append(filelist[i])
                    self.file_number += 1
                 
 
    def process(self, input_root_path, output_root_path):
         
        self.__get_file_path(input_root_path)
        #import ipdb; ipdb.set_trace()
        for i in range(self.file_number):
            print('Start render ',i, ' in', self.file_number)
            # delete all objects in outliner
            bpy.ops.object.select_all(action='SELECT')
            # if i == 0:
            #     bpy.data.objects["Cube"].select_set(True)
            # else:
            #     bpy.data.objects[self.import_file_name[i-1][:-4]].select_set(True)    # 把之前的obj delete掉
            bpy.ops.object.delete(use_global=False)
            bpy.ops.outliner.orphans_purge()
            bpy.ops.outliner.orphans_purge()
            bpy.ops.outliner.orphans_purge()

            # bpy.ops.object.select_all(action='DESELECT')
             
            # import target object
            #import ipdb; ipdb.set_trace()
            bpy.ops.wm.obj_import(
                filepath = self.import_file_path[i],
                directory = self.import_dir_path[i],
                files=[{"name":self.import_file_name[i]}]
                )
            bpy.data.objects[self.import_file_name[i][:-4]].select_set(True) # 寻找名字的时候去掉 .obj

            # add textures
            obj = bpy.data.objects[self.import_file_name[i][:-4]]
            label = self.import_file_name[i][:-4] + "_mat"
            tex_name = "big_108.png"
            mat = bpy.data.materials.new(label)
            mat.use_nodes = True
            if mat.node_tree:
                mat.node_tree.links.clear()
                mat.node_tree.nodes.clear()
            matnodes = mat.node_tree.nodes
            links = mat.node_tree.links
            diffuse = matnodes.new('ShaderNodeBsdfDiffuse')
            output = matnodes.new('ShaderNodeOutputMaterial')
            tex = matnodes.new('ShaderNodeTexImage')
            bpy.ops.image.open(filepath="{}".format(tex_name), directory="./", files=[{"name":tex_name, "name":tex_name}], relative_path=True, show_multiview=False)
        #    bpy.data.images[0].pack(as_png=True)
            bpy.data.images[0].pack()
            tex.image = bpy.data.images[0]
            link = links.new(tex.outputs['Color'], diffuse.inputs['Color'])
            link = links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
        #    disp           = matnodes['Material Output'].inputs['Displacement']
        #    mat.node_tree.links.new(disp, tex.outputs['Color'])
            obj.data.materials.append(mat)
            bpy.data.objects[obj.name].select_set(True)

            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            # bpy.data.objects[self.import_file_name[i][:-4]].select_set(True)    # 选择这个物体
            bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
            
            # bpy.ops.transform.rotate(value=-0.881395, orient_axis='Z', orient_type='VIEW', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='VIEW', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, release_confirm=True)
            # bpy.ops.transform.resize(value=(5.10903, 5.10903, 5.10903), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, release_confirm=True)
            bpy.ops.object.mode_set(mode='OBJECT')

            # bpy.context.space_data.params.filename = "tie_init.obj"    
            bpy.ops.export_scene.obj(
                filepath=output_root_path + self.import_file_name[i],
                filter_glob="*.obj;*.mtl")
 
 
if __name__ == '__main__':
    # my_render.process(input_root_path='/home/ubuntu/chenhn/render/tie_blender/test_input/', output_root_path="/home/ubuntu/chenhn/render/tie_blender/test_output/")
    input_dir = "./tmp_double/026/episode3/results583/"
    output_dir = "./tmp_blender/026/episode3/results583/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    for dir in os.listdir(input_dir):
        my_render = Render_2D()
        print(dir)
        input_root_path = input_dir + dir + "/"
        output_root_path = output_dir + dir + "/"
        os.makedirs(output_root_path)
        my_render.process(input_root_path=input_root_path, output_root_path=output_root_path)
        del my_render
    