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

from copy import copy, deepcopy
import statistics
import sys
import math
import random
import matplotlib.pyplot as plt
from pyparsing import col
import tensorflow as tf

from ezdxf import recover, DXFStructureError
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from numpy import delete

from shapely.geometry import Point, Polygon

from section import CrossSection

#hyperparameters
max_iters=10000
save_interval=math.ceil(max_iters/350)#Some error makes max images possible to save=355
area_weight=10
stiffness_weight=1
learn_rate=0.015


def sigmoid(x,a):
    #returns from 0 to 1. The more a, the smaller the range where it is not 0 or 1. (tighter curve)
    if x >= 0:
        z = math.exp(-a*x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(a*x)
        sig = z / (1 + z)
        return sig

def error_func(ex,global_rx,ey,global_ry,global_Ix,global_Iy,total_area):
    #add (ex/global_rx+ey/global_ry)/
    ex_err=(ex/global_rx)+(ey/global_ry)
    if((ex/global_rx)>0.3):
        ex_err*=2
    elif((ey/global_ry)>0.3):
        ex_err*=2
    else:
        ex_err=max(0.05,ex_err)
    
    return tf.multiply(ex_err,tf.divide(tf.multiply(total_area,area_weight),tf.multiply(tf.add(global_Ix,global_Iy),stiffness_weight)))
   # tf.divide(tf.multiply((tf.add(tf.divide(ex,global_rx),tf.divide(ey,global_ry)),tf.divide(tf.multiply(total_area,area_weight),first_area)),tf.divide(tf.multiply(tf.multiply(global_Ix,global_Iy),stiffness_weight),first_IxIy)))

def calc_error(cols,maxes,mines,first_error):
    ex,global_rx,ey,global_ry,global_Ix,global_Iy,total_area=calc_data(cols)
    
    score_to_minimize=error_func(ex,global_rx,ey,global_ry,global_Ix,global_Iy,total_area)/first_error
    #check if cols in bounds
    for i,c in enumerate(cols):
        score_to_minimize*=tf.cond(tf.less(c[0][0],maxes[i][0][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][0],maxes[i][0][0])),+1.0),2),lambda: 1)
        score_to_minimize*=tf.cond(tf.greater(c[0][1],maxes[i][0][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][1],maxes[i][0][1])),+1.0),2),lambda: 1)
        score_to_minimize*=tf.cond(tf.less(c[1][1],maxes[i][1][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][1],maxes[i][1][1])),+1.0),2),lambda: 1)
        score_to_minimize*=tf.cond(tf.greater(c[1][0],maxes[i][1][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][0],maxes[i][1][0])),+1.0),2),lambda: 1)
        score_to_minimize*=tf.cond(tf.greater(c[0][0],mines[i][0][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][0],mines[i][0][0])),+1.0),2),lambda: 1)
        score_to_minimize*=tf.cond(tf.less(c[0][1],mines[i][0][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][1],mines[i][0][1])),+1.0),2),lambda: 1)
        score_to_minimize*=tf.cond(tf.greater(c[1][1],mines[i][1][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][1],mines[i][1][1])),+1.0),2),lambda: 1)
        score_to_minimize*=tf.cond(tf.less(c[1][0],mines[i][1][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][0],mines[i][1][0])),+1.0),2),lambda: 1)
    
    mse=tf.reduce_mean([score_to_minimize])
    print("Loss:",mse)
    return mse
    
def run_gradient_descent(columns, learning_rate,maxes,mines,first_error):
 
    # Any values to be part of gradient calcs need to be vars/tensors
    tf_cols = tf.convert_to_tensor(columns, dtype='float32') 
    
    # Hardcoding 25 iterations of gradient descent
    for i in range(max_iters):

        # Do all calculations under a "GradientTape" which tracks all gradients
        with tf.GradientTape() as tape:
            tape.watch(tf_cols)
            loss = calc_error(tf_cols,maxes,mines,first_error)

        # Auto-diff magic!  Calcs gradients between loss calc and params
        dloss_dparams = tape.gradient(loss, tf_cols)
       
        # Gradients point towards +loss, so subtract to "descend"
        tf_cols = tf_cols - learning_rate * dloss_dparams
        if(i%save_interval==0):
            try:
                save_specific_sample(i,tf_cols)
            except:
                print("Could not save image")


def gen_b_box(points):
    #points: list of points
    min_x=points[0][0]
    max_x=points[0][0]
    min_y=points[0][1]
    max_y=points[0][1]
    for p in points:
        if p[0]<min_x:
            min_x=p[0]
        if p[0]>max_x:
            max_x=p[0]
        if p[1]<min_y:
            min_y=p[1]
        if p[1]>max_y:
            max_y=p[1] 
    return [[min_x,max_y],[max_x,min_y]]

def gen_points(b_box):
    #b_box: [[min_x,max_y],[max_x,min_y]]
    min_x=b_box[0][0]
    max_x=b_box[1][0]
    min_y=b_box[1][1]
    max_y=b_box[0][1]
    return [[min_x,max_y],[min_x,min_y],[max_x,min_y],[max_x,max_y]]

def main():
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
    limits_exist=False
    print('Parsing columns...')
    if len(raw_prf)!=0 and len(raw_max)==0:
        print("No plines in MAX layer, assuming only prf columns are to be used.")
        for shape_prf in raw_prf:
            for v in shape_prf:
                v=list(v)
            #generate bounding box
            shape_prf=gen_b_box(shape_prf)
            columns.append({'min':shape_prf,'prf':shape_prf,'max':shape_prf})
    elif len(raw_prf)==0:
        print("FATAL: Error in dxf: No columns in prf layer. Exiting")
        sys.exit(3)
    else:
        limits_exist=True
        min_prf_rel=dict()
        found_prf=[]
        prf_max_rel=dict()
        found_max=[]
        for i,shape_min in enumerate(raw_min):# find which preferred column each min belongs to
            found=False
            for j,shape_prf in enumerate(raw_prf):
                for k,v in enumerate(raw_prf[j]):
                    raw_prf[j][k]=list(raw_prf[j][k])
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
            columns.append({'min':gen_b_box(raw_min[i]),'prf':gen_b_box(raw_prf[j]),'max':gen_b_box(raw_max[k])})

    print(str(len(columns)) +' Columns successfully parsed.')
    

    if(not limits_exist):
        exit()

    #randomize_layout(columns)
    tfcolumns=[i['prf'] for i in columns]
    maxes=[i['max'] for i in columns]
    mines=[i['min'] for i in columns]
    ex,global_rx,ey,global_ry,global_Ix,global_Iy,total_area=calc_data(tfcolumns)
    
    first_error=error_func(ex,global_rx,ey,global_ry,global_Ix,global_Iy,total_area)
    run_gradient_descent(tfcolumns,learn_rate,maxes,mines,first_error)
        
def randomize_layout(columns):
    for column in columns:#for each column
        column['prf'][0][0]=random.random()*(column['max'][0][0]-column['min'][0][0])+column['min'][0][0]
        column['prf'][1][0]=random.random()*(column['max'][1][0]-column['min'][1][0])+column['min'][1][0]
        column['prf'][1][1]=random.random()*(column['max'][1][1]-column['min'][1][1])+column['min'][1][1]
        column['prf'][0][1]=random.random()*(column['max'][0][1]-column['min'][0][1])+column['min'][0][1]
    return columns

def mutate(columns):
    indices=random.sample(range(len(columns)),(math.floor(random.random()*len(columns))))
    for i in indices:
        columns[i]['prf'][0][0]=random.random()*(columns[i]['max'][0][0]-columns[i]['min'][0][0])+columns[i]['min'][0][0]
        columns[i]['prf'][1][0]=random.random()*(columns[i]['max'][1][0]-columns[i]['min'][1][0])+columns[i]['min'][1][0]
        columns[i]['prf'][1][1]=random.random()*(columns[i]['max'][1][1]-columns[i]['min'][1][1])+columns[i]['min'][1][1]
        columns[i]['prf'][0][1]=random.random()*(columns[i]['max'][0][1]-columns[i]['min'][0][1])+columns[i]['min'][0][1]
    

def calc_data(columns):
    centroids=[]
    areas=[]
    Ixs=[]
    Iys=[]
    Rxs=[]
    Rys=[]
    for column in columns:
        #TODO: eliminate gen_points as much as possible (hint: columns are all rectangles
        centroids.append(((column[0][0]+column[1][0])/2,(column[0][1]+column[1][1])/2))#Polygon(gen_points(column['prf'])).centroid
        areas.append(abs((column[0][0]-column[1][0])*(column[0][1]-column[1][1])))#Polygon(gen_points(column['prf'])).area
        Ixs.append(get_Ix(column,centroids[-1]))
        Iys.append(get_Iy(column,centroids[-1]))
        Rxs.append(get_Rx(gen_points(column),areas[-1]))
        Rys.append(get_Ry(gen_points(column),areas[-1]))

    total_area=calc_total_area(areas)
    center_of_mass=calc_global_com(centroids,areas)
    center_of_rigidity=calc_global_cor(Iys,Ixs,centroids)
    global_Ix=calc_global_Ix(Ixs,areas,centroids,center_of_mass)
    global_Iy=calc_global_Iy(Iys,areas,centroids,center_of_mass)
    global_rx=math.sqrt(global_Ix/total_area)# torsional_radius_x
    global_ry=math.sqrt(global_Iy/total_area)# torsional_radius_y
    ex=abs(center_of_mass[0]-center_of_rigidity[0])# eccentricity
    ey=abs(center_of_mass[1]-center_of_rigidity[1])# eccentricity
    # print("ex/rx:"+str(ex/global_rx))
    # print("ey/ry:"+str(ey/global_ry))
    # print("global Ix:"+str(global_Ix))
    # print("global Iy:"+str(global_Iy))
    # print("global A:"+str(total_area))
    return ex,global_rx,ey,global_ry,global_Ix,global_Iy,total_area

def calc_total_area(areas):
    sum=0
    for c in areas:
        sum+=c
    return sum

# Center of mass
def calc_global_com(centroids,areas):
    # Find COMass:
    sum_x=0
    sum_y=0
    sum_w=0
    for i in range(len(centroids)):
        sum_x+=centroids[i][0]*areas[i]
        sum_y+=centroids[i][1]*areas[i]
        sum_w+=areas[i]
    com=(sum_x/sum_w,sum_y/sum_w)
    #print("Center of mass: "+str(com))
    return com

# Center of rigidity
def calc_global_cor(Iys,Ixs,centroids):
    # Assuming constant E and L
    sum_Iy=0
    sum_Ix=0
    sum_Iy_cx=0
    sum_Ix_cy=0
    for i in range(len(Ixs)):
        sum_Iy+=Iys[i]
        sum_Ix+=Ixs[i]
        sum_Iy_cx+=Iys[i]*centroids[i][0]
        sum_Ix_cy+=Ixs[i]*centroids[i][1]
    cor=(sum_Iy_cx/sum_Iy,sum_Ix_cy/sum_Ix)
    #print("Center of rigidity: "+str(cor))
    return cor

# Floor Ix
def calc_global_Ix(Ixs,areas,centroids,center_of_mass):
    sum=0
    for i in range(len(Ixs)):
        sum+=Ixs[i]+areas[i]*(center_of_mass[1]-centroids[i][1])**2
    return sum

# Floor Iy
def calc_global_Iy(Iys,areas,centroids,center_of_mass):
    sum=0
    for i in range(len(Iys)):
        sum+=Iys[i]+areas[i]*(center_of_mass[0]-centroids[i][1])**2
    return sum
    
def calc_torsional_radius(columns):
    ...

# Calculate moment of inertia x (shape coords are global)
def get_Ix(shape, centroid):
    # cs=CrossSection(shape)
    # moi=cs.MomentOfInertia()
    b=shape[0][0]-shape[1][0]
    h=shape[0][1]-shape[1][1]
    return abs(b*h**3/12)

# Calculate moment of inertia y (shape coords are global)
def get_Iy(shape,centroid):
    # cs=CrossSection(shape)
    # moi=cs.MomentOfInertia()
    # return moi[1]
    h=shape[0][0]-shape[1][0]
    b=shape[0][1]-shape[1][1]
    return abs(b*h**3/12)

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

def preview_random_sample(popu,doc):
    msp = doc.modelspace()
    lines=[]
    col=popu[math.floor(random.random()*len(popu))]
    for co in col['columns']:
        c=gen_points(co['prf'])
        for i,v in enumerate(c):
            lines.append(msp.add_line(v,c[(i+1)%len(c)]))
    preview(doc)
    for line in lines:
            msp.delete_entity(line)

def preview_specific_sample(popu,doc,index):
    msp = doc.modelspace()
    lines=[]
    col=popu[index]
    for co in col['columns']:
        c=gen_points(co['prf'])
        for i,v in enumerate(c):
            lines.append(msp.add_line(v,c[(i+1)%len(c)]))
    preview(doc)
    for line in lines:
            msp.delete_entity(line)

def save_specific_sample(iter,cols):
    msp = doc.modelspace()
    lines=[]
    for co in cols:
        c=gen_points(co)
        for i,v in enumerate(c):
            lines.append(msp.add_line(v,c[(i+1)%len(c)]))
    save_dxf_to_image(doc,"iter"+str(iter)+".png")
    for line in lines:
            msp.delete_entity(line)

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
def save_dxf_to_image(doc,filename):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)
    print('Saving image to ' + filename)
    fig.savefig(filename, dpi=300)
    #plt.show()

# Safe loading procedure (requires ezdxf v0.14):
try:
    doc, auditor = recover.readfile('./test.dxf')
except IOError:
    print('Not a DXF file or a generic I/O error.')
    sys.exit(1)
except DXFStructureError:
    print('Invalid or corrupted DXF file.')
    sys.exit(2)

main()