import numpy as np

def read_obj(path):
    with open(path, 'r') as f:
        verts = []
        faces = []
        contents = f.readlines()[1:-1]
        for listitem in contents:
            if listitem[0] == 'v':
                verts.append((listitem[2:-1]).split(' '))
            elif listitem[0] == 'f':
                faces.append((listitem[2:-1]).split(' '))
     
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)-1
    return verts, faces