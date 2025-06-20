import yaml
import numpy as np
import rowan as rn

def saveyaml(file_out, data):
    with open(file_out, "w") as f:
        yaml.safe_dump(data,f,default_flow_style=None)

def loadyaml(file_in):
    with open(file_in, "r") as f: 
        file_out = yaml.safe_load(f)
    return file_out


def loadcsv(filename):
    return np.loadtxt(filename, delimiter=",", skiprows=1, ndmin=2)


def derivative(vec, dt):
    dvec = []
    # dvec  =[[0,0,0]]
    for i in range(len(vec) - 1):
        dvectmp = (vec[i + 1] - vec[i]) / dt
        dvec.append(dvectmp.tolist())
    dvec.append([0, 0, 0])
    return np.asarray(dvec)


def reorder_quat(quat):
    """
    Reorder quaternion from [x, y, z, w] to [w, x, y, z]
    """
    quat_reordered =  np.array([quat[3], quat[0], quat[1], quat[2]])
    return quat_reordered