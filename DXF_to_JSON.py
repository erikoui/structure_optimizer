# Converts a DXF file to a json that can be read by BNHSA


# README
# - create a 2010 o earlier dxf with the layers MAX MIN and PREF
# - put preferred column sizes in PREF
# - put max in MAX and min in MIN

import sys
import math
import json
import matplotlib.pyplot as plt

from ezdxf import recover, DXFStructureError
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

from shapely.geometry import Point, Polygon

from section import CrossSection

floor_heights = [0, 3, 6, 9, 12]  # for future use
default_E = 3.0e11
default_G = default_E/(2*(1+0.3))

add_diaphragm=True
diaphragm_area=100;

def main():
    # --------------------READ DXF------------------------
    # Safe loading procedure (requires ezdxf v0.14):
    try:
        doc, auditor = recover.readfile('./DXF/test4.dxf')
    except IOError:
        print('Not a DXF file or a generic I/O error.')
        sys.exit(1)
    except DXFStructureError:
        print('Invalid or corrupted DXF file.')
        sys.exit(2)

    # The auditor.errors attribute stores severe errors,
    # which may raise exceptions when rendering.
    if auditor.has_errors:
        print('Auditor detected errors. Exiting...')
        sys.exit(3)

    # Get entities of layers we care about
    msp = doc.modelspace()
    s_max = msp.query('LWPOLYLINE[layer=="MAX"]')
    s_min = msp.query('LWPOLYLINE[layer=="MIN"]')
    s_prf = msp.query('LWPOLYLINE[layer=="PREF"]')

    # Store only their points for simplicity
    raw_max = [list(x.vertices()) for x in s_max.entities]
    raw_min = [list(x.vertices()) for x in s_min.entities]
    raw_prf = [list(x.vertices()) for x in s_prf.entities]
    # -----------------------------------------------------

    # ------------Create data structure [col{points_min[],points_max[],points_pref}]---------------
    # TODO: Check if shapes have same number of points
    columns = []
    print('Parsing columns...')
    if len(raw_prf) != 0 and len(raw_max) == 0:
        print("No plines in MAX layer, assuming only prf columns are to be used.")
        for shape_prf in raw_prf:
            columns.append(
                {'min': shape_prf, 'prf': shape_prf, 'max': shape_prf})
    elif len(raw_prf) == 0:
        print("FATAL: Error in dxf: No columns in prf layer. Exiting")
        sys.exit(3)
    else:
        min_prf_rel = dict()
        found_prf = []
        prf_max_rel = dict()
        found_max = []
        # find which preferred column each min belongs to
        for i, shape_min in enumerate(raw_min):
            found = False
            for j, shape_prf in enumerate(raw_prf):
                if polygon_inside_polygon(shape_min, shape_prf) and j not in found_prf:
                    min_prf_rel[i] = j
                    found_prf.append(j)
                    found = True
                    break
            if not found:
                print("FATAL: Error in dxf: min column "+str(i) +
                      " is not inside any prf column. Exiting")
                sys.exit(3)

        for relation in min_prf_rel.keys():
            found = False
            for i, shape_max in enumerate(raw_max):
                if polygon_inside_polygon(raw_prf[min_prf_rel[relation]], shape_max) and i not in found_max:
                    prf_max_rel[min_prf_rel[relation]] = i
                    found_max.append(i)
                    found = True
                    break
            if not found:
                print("FATAL: Error in dxf: prf column "+str(relation) +
                      " is not inside any max column. Exiting")
                sys.exit(3)

        for relation in min_prf_rel.keys():
            i = relation
            j = min_prf_rel[relation]
            k = prf_max_rel[min_prf_rel[relation]]
            columns.append(
                {'min': raw_min[i], 'prf': raw_prf[j], 'max': raw_max[k]})
    calc_column_data(columns)
    print(str(len(columns)) + ' Columns successfully parsed.')
    # ----------------------------------------------------------------------------------------------

    # Conversion starts here
    col_prefix = "K"
    # nodes have integers as names
    elem_prefix = "E"

    fe_model = {}
    fe_model["cols"] = []
    fe_model["FEnodes"] = []
    fe_model["FEmembers"] = []
    for i, column in enumerate(columns):
        fe_model["cols"].append({
            "name": col_prefix+str(i),
            # 3d display should use points and extrude them
            "start":[column["centroid"].x,column["centroid"].y,column["bottomZ"]],
            "end":[column["centroid"].x,column["centroid"].y,column["topZ"]],
            "limits": {
                "not_implemented": 0
                # TODO use min/max to set limits (this is also not implemented in cpp yet)
                # this might not be nescessary since the cols array is for display only
            },
            "points": column["prf"]
        })
        fe_model["FEnodes"].append({
            "name": i*2,
            "colIndex": i,
            "coords": [column["centroid"].x, column["centroid"].y, column["bottomZ"]],
            "forces": {  # will be changed by the cpp
                "x": 0,
                "y": 0,
                "z": 0
            },
            "moments": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "constraints": {
                "x": column["bottomZ"]<=0.1,
                "y": column["bottomZ"]<=0.1,
                "z": column["bottomZ"]<=0.1,
                "xx": column["bottomZ"]<=0.1,
                "yy": column["bottomZ"]<=0.1,
                "zz": column["bottomZ"]<=0.1
            }
        })
        fe_model["FEnodes"].append({
            "name": i*2+1,
            "colIndex": i,
            "coords": [column["centroid"].x, column["centroid"].y, column["topZ"]],
            "forces": {  # will be changed by the cpp
                "x": 0,
                "y": 0,
                "z": 0
            },
            "moments": {
                "x": 0,
                "y": 0,
                "z": 0
            },
            "constraints": {
                "x": False,
                "y": False,
                "z": False,
                "xx": True,
                "yy": True,
                "zz": False
            }
        })
        fe_model["FEmembers"].append({
            "name": elem_prefix+str(i),
            "from": i*2,
            "to": i*2+1,
            "udl": [0, 0, 0],
            "E": default_E,
            "G": default_G,
            "Ix": column["Ix"],
            "Iy": column["Iy"],
            "J": column["J"],
            "A": column["area"],
            "length": column["topZ"]-column["bottomZ"]
        })
        if(add_diaphragm):
            for j,second_col in enumerate(columns):
                if(i<j):
                    fe_model["FEmembers"].append({
                    "name": "diaphragm",
                    "from": i*2+1,
                    "to": j*2+1,
                    "udl": [0, 0, 0],
                    "E": default_E,
                    "G": default_G,
                    "Ix": 0.001,#0.0026,
                    "Iy": 0.001,#0.0006,
                    "J": 0.002,#0.027,
                    "A": 1,#0.12,
                    "length": 5
                })

    fe_model["FEnodes"][1]["moments"]["z"]=10000#testing
    f = open("./test_json.json", "w")
    f.write(json.dumps(fe_model))
    f.close()

# -----------------------FUNCTIONS------------------------
def calc_column_data(columns):
    for column in columns:
        column["centroid"] = Polygon(column['prf']).centroid
        column["area"] = Polygon(column['prf']).area
        # In the future we might have multiple floors
        column["bottomZ"] = floor_heights[0]
        # In the future we might have multiple floors
        column["topZ"] = floor_heights[1]
        column["Ix"]=get_Ix(column["prf"],column["centroid"])
        column["Iy"]=get_Iy(column["prf"],column["centroid"])
        column["J"]=column["Ix"]+column["Iy"]
        # column["Ixy"]=get_Ixy(column["prf"],column["centroid"])
        # column["Rx"]=get_Rx(column["prf"],column["area"])
        # column["Ry"]=get_Ry(column["prf"],column["area"])

# Save DXF (include .dxf in filename)
def save_dxf(ezdxf_doc, filename):
    print('Saving dxf to ' + filename)
    ezdxf_doc.saveas(filename)

# save PNG (inglude .png in filename)
def save_dxf_to_image(mpl_figure, filename):
    print('Saving image to ' + filename)
    mpl_figure.savefig(filename, dpi=300)

# Check if s1 is inside or touching s2 (s1,s2 are arrays of (x,y) tuples)
def polygon_inside_polygon(s1, s2):
    p1 = Polygon(s1)
    p2 = Polygon(s2)
    return p2.covers(p1)

# Calculate moment of inertia x (shape coords are global)
def get_Ix(shape, centroid):
    cs=CrossSection(shape)
    moi=cs.MomentOfInertia()
    return moi[0]

# Calculate moment of inertia y (shape coords are global)
def get_Iy(shape,centroid):
    cs=CrossSection(shape)
    moi=cs.MomentOfInertia()
    return moi[1]

# Calculate moment of inertia xy (shape coords are global)
def get_Ixy(shape,centroid):
    cs=CrossSection(shape)
    moi=cs.MomentOfInertia()
    return moi[2]

main()
