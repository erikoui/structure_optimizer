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
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np

from ezdxf import recover, DXFStructureError
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

import tensorflow as tf

from copy import deepcopy

# Can be 'GA' or 'AI'
MODE='AI'
# GA is broken btw

#hyperparameters AI
learn_rate=0.1

#hyperparameters GA
pop_size=3000
survivor_ratio=1
selection_ratio=0.1#how likely is for the first to be parents <1
mutation_ratio=0.05

#hyperparameters general
max_iters=5000
save_interval=math.ceil(max_iters/10)#Some error makes max images possible to save=355
area_weight=0.001# How GOOD more area is
stiffness_weight=0.1
Ix_ratio=1# used to convert Ix to [0,1]
Iy_ratio=1# used to convert Iy to [0,1]
area_ratio=1# used to convert area to [0,1]

# Generate bounding box from a list of points
def gen_b_box(points):
    # points: list of points
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

def calculate_ratios(columns):
    centroids=[]
    areas=[]
    Ixs=[]
    Iys=[]
    Rxs=[]
    Rys=[]
    for column in columns:
        centroids.append(((column[0][0]+column[1][0])/2,(column[0][1]+column[1][1])/2))
        areas.append(abs((column[0][0]-column[1][0])*(column[0][1]-column[1][1])))
        Ixs.append(get_Ix(column))
        Iys.append(get_Iy(column))
        # Rxs.append(get_Rx(column))
        # Rys.append(get_Ry(column))

    total_area=sum(areas)
    center_of_mass=calc_global_com(centroids,areas)
    center_of_rigidity=calc_global_cor(Iys,Ixs,centroids)
    global_Ix=calc_global_Ix(Ixs,areas,centroids,center_of_mass)
    global_Iy=calc_global_Iy(Iys,areas,centroids,center_of_mass)

    global Ix_ratio
    Ix_ratio=1/global_Ix
    global Iy_ratio
    Iy_ratio=1/global_Iy
    global area_ratio
    area_ratio=1/total_area**2

# Calculates how good the structure is
def error_func(columns,maxes,mines,iter):
    global minimum_loss
    centroids=[]
    areas=[]
    Ixs=[]
    Iys=[]
    Rxs=[]
    Rys=[]
    for column in columns:
        centroids.append(((column[0][0]+column[1][0])/2,(column[0][1]+column[1][1])/2))
        areas.append(abs((column[0][0]-column[1][0])*(column[0][1]-column[1][1])))
        Ixs.append(get_Ix(column))
        Iys.append(get_Iy(column))
        # Rxs.append(get_Rx(column))
        # Rys.append(get_Ry(column))

    total_area=sum(areas)
    center_of_mass=calc_global_com(centroids,areas)
    center_of_rigidity=calc_global_cor(Iys,Ixs,centroids)
    global_Ix=calc_global_Ix(Ixs,areas,centroids,center_of_mass)
    global_Iy=calc_global_Iy(Iys,areas,centroids,center_of_mass)
    global_rx=math.sqrt(global_Ix/total_area)# torsional_radius_x
    global_ry=math.sqrt(global_Iy/total_area)# torsional_radius_y
    ex=abs(center_of_mass[0]-center_of_rigidity[0])/global_rx# eccentricity
    ey=abs(center_of_mass[1]-center_of_rigidity[1])/global_ry# eccentricity

    score_to_minimize=((e_sig(ex)+e_sig(ey))/2)/sigmoid((stiffness_weight*(global_Ix*Ix_ratio+global_Iy*Iy_ratio)/2),1)/(total_area**2*area_ratio*area_weight)

    mse=score_to_minimize
    if(MODE=='AI'):
        #check if cols in bounds
        for i,c in enumerate(columns):
            score_to_minimize*=tf.cond(tf.less(c[0][0],maxes[i][0][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][0],maxes[i][0][0])),+1.0),2),lambda: 1)
            score_to_minimize*=tf.cond(tf.greater(c[0][1],maxes[i][0][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][1],maxes[i][0][1])),+1.0),2),lambda: 1)
            score_to_minimize*=tf.cond(tf.less(c[1][1],maxes[i][1][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][1],maxes[i][1][1])),+1.0),2),lambda: 1)
            score_to_minimize*=tf.cond(tf.greater(c[1][0],maxes[i][1][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][0],maxes[i][1][0])),+1.0),2),lambda: 1)
            score_to_minimize*=tf.cond(tf.greater(c[0][0],mines[i][0][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][0],mines[i][0][0])),+1.0),2),lambda: 1)
            score_to_minimize*=tf.cond(tf.less(c[0][1],mines[i][0][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[0][1],mines[i][0][1])),+1.0),2),lambda: 1)
            score_to_minimize*=tf.cond(tf.greater(c[1][1],mines[i][1][1]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][1],mines[i][1][1])),+1.0),2),lambda: 1)
            score_to_minimize*=tf.cond(tf.less(c[1][0],mines[i][1][0]),lambda: tf.pow(tf.add(tf.abs(tf.subtract(c[1][0],mines[i][1][0])),+1.0),2),lambda: 1)
        
        mse=tf.reduce_mean([score_to_minimize])
        print("Iter "+str(iter)+" | Loss:",str(mse.numpy()),end='\r')
    return mse
    
def sigmoid(x,a):
    #returns from 0 to 1. The more a, the smaller the range where it is not 0 or 1. (tighter curve)
    if x >= 0:
        z = math.exp(-a*x-6)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(a*x-6)
        sig = z / (1 + z)
        return sig

# my weird eccentricity sigmoid 
def e_sig(x):
    return 1/(1+math.exp(-36*x+6))
    
def run_gradient_descent(columns, learning_rate,maximums,minimums,doc):
 
    # Any values to be part of gradient calcs need to be vars/tensors
    tf_cols = tf.convert_to_tensor(columns, dtype='float32') 
    
    for i in range(max_iters+1):

        # Do all calculations under a "GradientTape" which tracks all gradients
        with tf.GradientTape() as tape:
            tape.watch(tf_cols)
            loss = error_func(tf_cols,maximums,minimums,i)

        # Auto-diff magic!  Calcs gradients between loss calc and params
        dloss_dparams = tape.gradient(loss, tf_cols)
       
        # Gradients point towards +loss, so subtract to "descend"
        tf_cols = tf_cols - learning_rate * dloss_dparams
        if(i%save_interval==0):
            try:
                save_columns_to_png(i,tf_cols,doc,loss)
            except:
                print("Could not save image")

def randomize_layout(columns,maximums,minimums):
    for i in range(len(columns)):#for each column
        columns[i][0][0]=random.random()*(maximums[i][0][0]-minimums[i][0][0])+minimums[i][0][0]
        columns[i][1][0]=random.random()*(maximums[i][1][0]-minimums[i][1][0])+minimums[i][1][0]
        columns[i][1][1]=random.random()*(maximums[i][1][1]-minimums[i][1][1])+minimums[i][1][1]
        columns[i][0][1]=random.random()*(maximums[i][0][1]-minimums[i][0][1])+minimums[i][0][1]
    return columns

def mutate(columns,maximums,minimums):
    indices=random.sample(range(len(columns)),(math.floor(random.random()*len(columns))))
    for i in indices:
        columns[i][0][0]=random.random()*(maximums[i][0][0]-minimums[i][0][0])+minimums[i][0][0]
        columns[i][1][0]=random.random()*(maximums[i][1][0]-minimums[i][1][0])+minimums[i][1][0]
        columns[i][1][1]=random.random()*(maximums[i][1][1]-minimums[i][1][1])+minimums[i][1][1]
        columns[i][0][1]=random.random()*(maximums[i][0][1]-minimums[i][0][1])+minimums[i][0][1]
 
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

# TODO: I think this function needs to use parallel axis theorem
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

# Calculate moment of inertia x
def get_Ix(b_box):
    b=b_box[0][0]-b_box[1][0]
    h=b_box[0][1]-b_box[1][1]
    return abs(b*h**3/12)

# Calculate moment of inertia y
def get_Iy(b_box):
    h=b_box[0][0]-b_box[1][0]
    b=b_box[0][1]-b_box[1][1]
    return abs(b*h**3/12)

# Calculate moment of inertia xy
def get_Ixy(b_box):
    ...

def get_Rx(b_box):
    ...

def get_Ry(b_box):
    ...

# Check if b1 is inside b2
def box_inside_box(b1,b2):
    #          x1        x1          y1      y1            x2         x2           y2      y2
    return (b1[0][0]>b2[0][0] and b1[0][1]<b2[0][1] and b1[1][0]<b2[1][0] and b1[1][1]>b2[1][1])

# Converts a bounding box to a list of 4 points
def gen_points(b_box):
    #b_box: [[min_x,max_y],[max_x,min_y]]
    min_x=b_box[0][0]
    max_x=b_box[1][0]
    min_y=b_box[1][1]
    max_y=b_box[0][1]
    return [[min_x,max_y],[min_x,min_y],[max_x,min_y],[max_x,max_y]]

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
def save_dxf_to_image(doc,filename, loss):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)
    #print('Saving image to ' + filename +' (loss: '+str(loss)+')',end='\r')
    fig.savefig(filename, dpi=300)

def preview_columns(col,doc):
    msp = doc.modelspace()
    lines=[]
    for co in col:
        c=gen_points(co)
        for i,v in enumerate(c):
            lines.append(msp.add_line(v,c[(i+1)%len(c)]))
    preview(doc)
    for line in lines:
            msp.delete_entity(line)

def save_columns_to_png(iter,cols,doc,loss):
    msp = doc.modelspace()
    lines=[]
    for co in cols:
        c=gen_points(co)
        for i,v in enumerate(c):
            lines.append(msp.add_line(v,c[(i+1)%len(c)]))
    try:
        save_dxf_to_image(doc,"iter"+str(iter)+"_"+"{:.3g}".format(loss.numpy())+".png",loss)
    except:
        save_dxf_to_image(doc,"iter"+str(iter)+"_"+"{:.3g}".format(loss)+".png",loss)
    for line in lines:
        msp.delete_entity(line)

# ------------ SCRIPT STARTS HERE ---------------
path_to_file='./DXF/test_maxminonly.dxf'
print("CWD: "+os.getcwd())
# Safe loading procedure (requires ezdxf v0.14):
try:
    if os.path.exists(path_to_file):
        doc, auditor = recover.readfile(path_to_file)
    else:
        print('Cannot find '+path_to_file)
        sys.exit(1)
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

# Store only their points for simplicity using incomprehensible list comprehension
raw_max=[gen_b_box(x) for x in [list(x.vertices()) for x in s_max.entities]]
raw_min=[gen_b_box(x) for x in [list(x.vertices()) for x in s_min.entities]]

# Create data structure
columns=[]
raw_data=[]
minimums=[]
maximums=[]

print('Parsing columns...')
if len(raw_max)==0 or len(raw_min)==0:
    print("FATAL: Error in dxf: No columns in MAX or MIN layer. Exiting")
    sys.exit(3)
else:
    min_max_rel=dict()
    found_max=[]

    # find which max column each min belongs to
    for i,box_min in enumerate(raw_min):# for all minimums
        found=False
        for j,box_max in enumerate(raw_max):
            if box_inside_box(box_min,box_max) and j not in found_max:
                min_max_rel[i]=j
                found_max.append(j)
                found=True
                break
        if not found:
            print("FATAL: Error in dxf: min column "+str(i)+" is not inside any max column. Exiting")
            sys.exit(3)
    
    for relation in min_max_rel.keys():
        i=relation
        k=min_max_rel[relation]
        minimums.append(gen_b_box(raw_min[i]))
        columns.append(gen_b_box(raw_min[i]))
        maximums.append(gen_b_box(raw_max[k]))

print(str(len(columns)) +' Columns successfully parsed.')

calculate_ratios(columns)
start_loss=error_func(columns,maximums,minimums,0)
print("Starting loss:"+str(start_loss))



if(MODE=='GA'):
    # Genesis
    population=[]
    for i in range(1,pop_size):
        population.append({"columns":deepcopy(columns)})
    for p in population:
        randomize_layout(p['columns'],maximums,minimums)

    #reproduction
    for iter in range(0,max_iters+1):
        for p in population:
            loss=error_func(p['columns'],maximums,minimums,iter)
            p['score']=loss
        list.sort(population, key=lambda f:f['score'])
        print("iteration\t"+str(iter)+"/"+str(max_iters)+": median score "+"{:.3g}".format(population[math.floor(len(population)*0.5)]['score'])+"\tbest score "+"{:.3g}".format(population[0]['score'])+"              ",end='\r')
        population=population[:math.floor(len(population)*survivor_ratio)]
        if(iter%save_interval==0):
            save_columns_to_png(iter,population[0]['columns'],doc,population[0]['score'])
        next_gen=[]
        for i in range(pop_size):
            #breed
            indices=list(range(len(population[0]['columns'])))
            parents=[]
            mini3=[]
            for j in range(len(population[0]['columns'])):
                parents.append(math.floor(np.random.geometric(p=selection_ratio,size=len(population))[0]))
            for index,p in enumerate(parents):
                mini3.append(population[p]['columns'][index])
            # random.shuffle(indices)

            # parent1=math.floor(np.random.geometric(p=selection_ratio,size=len(population))[0])
            # parent2=math.floor(np.random.geometric(p=selection_ratio,size=len(population))[0])
            # while(parent2==parent1):
            #     parent2=math.floor(np.random.geometric(p=selection_ratio,size=len(population))[0])
            # mini1=[population[parent1]['columns'][i] for i in indices[:math.floor(len(indices)/2)]]
            # mini2=[population[parent2]['columns'][i] for i in indices[math.floor(len(indices)/2):]]
            # mini1.extend(mini2)
            # #[1 3 4 2 5 8]
            # #[1 2 3 4 5 8]
            # # restore order of columns 
            # mini3=list(range(len(mini1)))
            # j=0
            # for index in indices:
            #     mini3[index]=mini1[j]
            #     j=j+1

            next_gen.append({'columns':mini3})
            if(random.random()<mutation_ratio):
                mutate(next_gen[-1]['columns'],maximums,minimums)
        population=deepcopy(next_gen)

elif(MODE=='AI'):
    first_error=error_func(columns,maximums,minimums,0)
    run_gradient_descent(columns,learn_rate,maximums,minimums,doc)

else:
    print("Set mode to AI or GA. Exiting")
    exit(1)