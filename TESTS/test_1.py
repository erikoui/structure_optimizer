# Opens and displays a R2010 dxf file

# TODO
# - Add functionality for lift or L walls
# - Take into account the mass of the slabs and beams for COM
# - Take into account the stiffness of slabs and beams for COR
# - Radius of gyration<total moment of inertia (not the sum but the Iy*y^2 thing)

# NOTE
# - Rigidity (or lack thereof) of flat slab has minimal effect(~5%) to the center of rigidity
# - Rigidity of beams reduces this effect even more sice we assume perfectly rigid diaphragm

# README
# - create a 2010 o earlier dxf with the layers MAX MIN and PREF
# - put preferred column sizes in PREF
# - put max in MAX and min in MIN

import sys
import math
import matplotlib.pyplot as plt

from ezdxf import recover, DXFStructureError
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

from shapely.geometry import Point, Polygon

from section import CrossSection

def main():
    # Safe loading procedure (requires ezdxf v0.14):
    try:
        doc, auditor = recover.readfile('./DXF/C2.dxf')
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
    s_max=msp.query('LWPOLYLINE[layer=="MAX"]')
    s_min=msp.query('LWPOLYLINE[layer=="MIN"]')
    s_prf=msp.query('LWPOLYLINE[layer=="PREF"]')

    # Store only their points for simplicity
    raw_max=[list(x.vertices()) for x in s_max.entities]
    raw_min=[list(x.vertices()) for x in s_min.entities]
    raw_prf=[list(x.vertices()) for x in s_prf.entities]

    # Create data structure [col{points_min[],points_max[],points_pref}]
    # TODO: Check if shapes have same number of points

    columns=[]
    print('Parsing columns...')
    if len(raw_prf)!=0 and len(raw_max)==0:
        print("No plines in MAX layer, assuming only prf columns are to be used.")
        for shape_prf in raw_prf:
            columns.append({'min':shape_prf,'prf':shape_prf,'max':shape_prf})
    elif len(raw_prf)==0:
        print("FATAL: Error in dxf: No columns in prf layer. Exiting")
        sys.exit(3)
    else:
        min_prf_rel=dict()
        found_prf=[]
        prf_max_rel=dict()
        found_max=[]
        for i,shape_min in enumerate(raw_min):# find which preferred column each min belongs to
            found=False
            for j,shape_prf in enumerate(raw_prf):
                if polygon_inside_polygon(shape_min,shape_prf) and j not in found_prf:
                    min_prf_rel[i]=j
                    found_prf.append(j)
                    found=True
                    break
            if not found:
                print("FATAL: Error in dxf: min column "+str(i)+" is not inside any prf column. Exiting")
                sys.exit(3)

        for relation in min_prf_rel.keys():
            found=False
            for i,shape_max in enumerate(raw_max):
                if polygon_inside_polygon(raw_prf[min_prf_rel[relation]],shape_max) and i not in found_max:
                    prf_max_rel[min_prf_rel[relation]]=i
                    found_max.append(i)
                    found=True
                    break
            if not found:
                print("FATAL: Error in dxf: prf column "+str(relation)+" is not inside any max column. Exiting")
                sys.exit(3)

        for relation in min_prf_rel.keys():
            i=relation
            j=min_prf_rel[relation]
            k=prf_max_rel[min_prf_rel[relation]]
            columns.append({'min':raw_min[i],'prf':raw_prf[j],'max':raw_max[k]})

    print(str(len(columns)) +' Columns successfully parsed.')
    
    calc_column_data(columns)
    total_area=calc_total_area(columns)
    center_of_mass=calc_global_com(columns)
    center_of_rigidity=calc_global_cor(columns)
    global_Ix=calc_global_Ix(columns,center_of_mass)
    global_Iy=calc_global_Iy(columns,center_of_mass)
    global_radius_gyration=math.sqrt(min(global_Iy,global_Ix)/total_area)
    global_J=global_Ix+global_Iy
    global_rx=math.sqrt(global_Ix/total_area)# torsional_radius_x
    global_ry=math.sqrt(global_Iy/total_area)# torsional_radius_y
    ex=abs(center_of_mass[0]-center_of_rigidity[0])# eccentricity
    ey=abs(center_of_mass[1]-center_of_rigidity[1])# eccentricity
    print("ex/rx:"+str(ex/global_rx))
    print("ey/ry:"+str(ey/global_ry))
    if(ex/global_rx<=0.3 and ey/global_ry<0.3):
        print("eccentricity ok")
    else:
        print("eccentricity NOT ok")
    print("rx:"+str(global_rx))
    print("ry:"+str(global_ry))
    print("floor Ix: "+str(global_Ix))
    print("floor Iy: "+str(global_Iy))
    print("Is:"+str(global_radius_gyration))
    if(global_rx>=global_radius_gyration and global_ry>global_radius_gyration):
        print("Torsional radius OK")
    else:
        print("Torsional radius NOT OK")
            

    msp.add_line(center_of_mass,center_of_rigidity)
    msp.add_line((0,0),(1,0))
    msp.add_line((0,0),(0,1))
    for i, column in enumerate(columns):
        msp.add_text(str(i),dxfattribs={'height':0.5}).set_pos(column["prf"][0],align='RIGHT')
    msp.add_text("COM",dxfattribs={'height':0.5}).set_pos(center_of_mass,align='MIDDLE')
    msp.add_text("COR",dxfattribs={'height':0.5}).set_pos(center_of_rigidity,align='MIDDLE')
    preview(doc)

def calc_column_data(columns):
    for column in columns:
        column["centroid"]=Polygon(column['prf']).centroid
        column["area"]=Polygon(column['prf']).area
        column["Ix"]=get_Ix(column["prf"],column["centroid"])
        column["Iy"]=get_Iy(column["prf"],column["centroid"])
        column["Ixy"]=get_Ixy(column["prf"],column["centroid"])
        column["Rx"]=get_Rx(column["prf"],column["area"])
        column["Ry"]=get_Ry(column["prf"],column["area"])

def calc_total_area(columns):
    sum=0
    for c in columns:
        sum+=c["area"]
    return sum

# Center of mass
def calc_global_com(columns):
    # Find COMass:
    sum_x=0
    sum_y=0
    sum_w=0
    for column in columns:
        sum_x+=column['centroid'].x*column['area']
        sum_y+=column['centroid'].y*column['area']
        sum_w+=column['area']
    com=(sum_x/sum_w,sum_y/sum_w)
    print("Center of mass: "+str(com))
    return com

# Center of rigidity
def calc_global_cor(columns):
    # Assuming constant E and L
    sum_Iy=0
    sum_Ix=0
    sum_Iy_cx=0
    sum_Ix_cy=0
    for column in columns:
        sum_Iy+=column['Iy']
        sum_Ix+=column['Ix']
        sum_Iy_cx+=column['Iy']*column['centroid'].x
        sum_Ix_cy+=column['Ix']*column['centroid'].y
    cor=(sum_Iy_cx/sum_Iy,sum_Ix_cy/sum_Ix)
    print("Center of rigidity: "+str(cor))
    return cor

# Floor Ix
def calc_global_Ix(columns, global_com):
    sum=0
    for column in columns:
        sum+=column["Ix"]+column["area"]*(global_com[1]-column["centroid"].y)**2
    return sum

# Floor Iy
def calc_global_Iy(columns, global_com):
    sum=0
    for column in columns:
        sum+=column["Iy"]+column["area"]*(global_com[0]-column["centroid"].x)**2
    return sum
    
def calc_torsional_radius(columns):
    ...

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

def get_Rx(shape,area):
    cs=CrossSection(shape)
    rx=math.sqrt(cs.MomentOfInertia()[0]/area)
    return rx

def get_Ry(shape,area):
    cs=CrossSection(shape)
    ry=math.sqrt(cs.MomentOfInertia()[1]/area)
    return ry

# Find nearest point of shape to source
def get_nearest_point(source,shape):
    md=math.sqrt((source[0]-shape[0][0])*(source[0]-shape[0][0])+(source[1]-shape[0][1])*(source[1]-shape[0][1]))
    mdi=0
    for i in range(len(shape)):
        d=math.sqrt((source[0]-shape[i][0])*(source[0]-shape[i][0])+(source[1]-shape[i][1])*(source[1]-shape[i][1]))
        if(d<md):
            md=d
            mdi=i
    return shape[mdi]

# Check if s1 is inside or touching s2 (s1,s2 are arrays of (x,y) tuples)
def polygon_inside_polygon(s1,s2):
    p1=Polygon(s1)
    p2=Polygon(s2)
    return p2.covers(p1)

# Display the drawing
def preview(doc):
    fig1 = plt.figure()
    ax = fig1.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)
    plt.show()

# Save DXF (include .dxf in filename)
def save_dxf(ezdxf_doc,filename):
    print('Saving dxf to ' + filename)
    ezdxf_doc.saveas(filename)

#save PNG (inglude .png in filename)
def save_dxf_to_image(mpl_figure,filename):
    print('Saving image to ' + filename)
    mpl_figure.savefig(filename, dpi=300)

main()