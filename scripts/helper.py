import yaml
import numpy as np

def saveyaml(file_out, data):
    with open(file_out, "w") as f:
        yaml.safe_dump(data,f,default_flow_style=None)

def loadyaml(file_in):
    with open(file_in, "r") as f: 
        file_out = yaml.safe_load(f)
    return file_out


def loadcsv(filename):
    return np.loadtxt(filename, delimiter=",", skiprows=1, ndmin=2)

