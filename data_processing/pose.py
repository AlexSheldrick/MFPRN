    1 import bpy
    2 import os
    3 from mathutils import *
    4 
    5 prefix_pose = "/tmp/camera_poses/"
    6 prefix_image = "/tmp/images/"
    7 
    8 def get_camera_pose(cameraName, objectName, scene, frameNumber):
    9   if not os.path.exists(prefix_pose):
   10     os.makedirs(prefix_pose)
   11 
   12   # OpenGL to Computer vision camera frame convention
   13   M = Matrix().to_4x4()
   14   M[1][1] = -1
   15   M[2][2] = -1
   16 
   17   cam = bpy.data.objects[cameraName]
   18   object_pose = bpy.data.objects[objectName].matrix_world
   19 
   20   #Normalize orientation with respect to the scale
   21   object_pose_normalized = object_pose.copy()
   22   object_orientation_normalized = object_pose_normalized.to_3x3().normalized()
   23   for i in range(3):
   24     for j in range(3):
   25         object_pose_normalized[i][j] = object_orientation_normalized[i][j]
   26 
   27   camera_pose = M*cam.matrix_world.inverted()*object_pose_normalized
   28   print("camera_pose:\n", camera_pose)
   29   
   30   filename = prefix_pose + cameraName + "_%03d" % frameNumber + ".txt"
   31   with open(filename, 'w') as f:
   32     f.write(str(camera_pose[0][0]) + " ")
   33     f.write(str(camera_pose[0][1]) + " ")
   34     f.write(str(camera_pose[0][2]) + " ")
   35     f.write(str(camera_pose[0][3]) + " ")
   36     f.write("\n")
   37 
   38     f.write(str(camera_pose[1][0]) + " ")
   39     f.write(str(camera_pose[1][1]) + " ")
   40     f.write(str(camera_pose[1][2]) + " ")
   41     f.write(str(camera_pose[1][3]) + " ")
   42     f.write("\n")
   43 
   44     f.write(str(camera_pose[2][0]) + " ")
   45     f.write(str(camera_pose[2][1]) + " ")
   46     f.write(str(camera_pose[2][2]) + " ")
   47     f.write(str(camera_pose[2][3]) + " ")
   48     f.write("\n")
   49 
   50     f.write(str(camera_pose[3][0]) + " ")
   51     f.write(str(camera_pose[3][1]) + " ")
   52     f.write(str(camera_pose[3][2]) + " ")
   53     f.write(str(camera_pose[3][3]) + " ")
   54     f.write("\n")
   55 
   56   return
   57 
   58 
   59 def my_handler(scene):
   60   frameNumber = scene.frame_current
   61   print("\n\nFrame Change", scene.frame_current)
   62   get_camera_pose("Camera", "tea_box_02", scene, frameNumber)
   63 
   64 step_count = 250
   65 scene = bpy.context.scene
   66 for step in range(1, step_count):
   67   # Set render frame
   68   scene.frame_set(step)
   69 
   70   # Set filename and render
   71   if not os.path.exists(prefix_image):
   72     os.makedirs(prefix_image)
   73   scene.render.filepath = (prefix_image + '%04d.png') % step
   74   bpy.ops.render.render( write_still=True )
   75 
   76   my_handler(scene)
