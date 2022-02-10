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

from ezdxf import recover, DXFStructureError
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from numpy import delete

from shapely.geometry import Point, Polygon

from section import CrossSection

#hyperparameters. Adjust such tat initial score is close to 0
max_iters=1
pop_size=100000
survivor_ratio=0.01
mutation_ratio=0.00
save_interval=math.ceil(max_iters/350)

area_weight=20
stiffness_weight=1

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

def error_func(ex,global_rx,ey,global_ry,total_area,global_Ix,global_Iy):
    ex_err=(ex/global_rx)+(ey/global_ry)
    if((ex/global_rx)>0.3):
        ex_err*=2
    elif((ey/global_ry)>0.3):
        ex_err*=2
    else:
        ex_err=max(0.05,ex_err)
    
    return ex_err*(total_area*area_weight)/((global_Ix+global_Iy)*stiffness_weight)

def main():
    # Safe loading procedure (requires ezdxf v0.14):
    try:
        doc, auditor = recover.readfile('./test.dxf')
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

    calc_column_data(columns)
    total_area=calc_total_area(columns)
    center_of_mass=calc_global_com(columns)
    center_of_rigidity=calc_global_cor(columns)
    global_Ix=calc_global_Ix(columns,center_of_mass)
    global_Iy=calc_global_Iy(columns,center_of_mass)
    # global_radius_gyration=math.sqrt(min(global_Iy,global_Ix)/total_area)
    # global_J=global_Ix+global_Iy
    global_rx=math.sqrt(global_Ix/total_area)# torsional_radius_x
    global_ry=math.sqrt(global_Iy/total_area)# torsional_radius_y
    ex=abs(center_of_mass[0]-center_of_rigidity[0])# eccentricity
    ey=abs(center_of_mass[1]-center_of_rigidity[1])# eccentricity
    print("ex/rx:"+str(ex/global_rx))
    print("ey/ry:"+str(ey/global_ry))
    print("global Ix:"+str(global_Ix))
    print("global Iy:"+str(global_Iy))
    print("global A:"+str(total_area))

    score_to_minimize=error_func(ex,global_rx,ey,global_ry,total_area,global_Ix,global_Iy)
    print("Starting score to minimize:"+str(score_to_minimize))
    
    # Genesis
    population=[]
    for i in range(1,pop_size):
        population.append({"columns":deepcopy(columns)})
    for p in population:
        randomize_layout(p['columns'])

    #reproduction
    for iter in range(0,max_iters):
        for p in population:
                calc_column_data(p['columns'])
                total_area=calc_total_area(p['columns'])
                center_of_mass=calc_global_com(p['columns'])
                center_of_rigidity=calc_global_cor(p['columns'])
                global_Ix=calc_global_Ix(p['columns'],center_of_mass)
                global_Iy=calc_global_Iy(p['columns'],center_of_mass)
                # global_radius_gyration=math.sqrt(min(global_Iy,global_Ix)/total_area)
                # global_J=global_Ix+global_Iy
                global_rx=math.sqrt(global_Ix/total_area)# torsional_radius_x
                global_ry=math.sqrt(global_Iy/total_area)# torsional_radius_y
                ex=abs(center_of_mass[0]-center_of_rigidity[0])# eccentricity
                ey=abs(center_of_mass[1]-center_of_rigidity[1])# eccentricity
                # print("ex/rx:"+str(ex/global_rx))
                # print("ey/ry:"+str(ey/global_ry))
                # print("Ix:"+str(global_Ix))
                score_to_minimize=error_func(ex,global_rx,ey,global_ry,total_area,global_Ix,global_Iy)
                #preview_random_sample(population,doc)
                p['score']=score_to_minimize
                #print("Score:"+str(score_to_minimize)) 
        list.sort(population,key=lambda f:f['score'])
        print("iteration "+str(iter)+"/"+str(max_iters)+": median score "+str(population[math.floor(len(population)*0.5)]['score']))
        print("best score "+str(population[0]['score']))
        population=population[:math.floor(len(population)*survivor_ratio)]
        if(iter%save_interval==0):
            save_specific_sample(iter,population,doc,0)
        next_gen=[]
        for i in range(pop_size):
            #breed
            indices=list(range(len(population[0]['columns'])))
            random.shuffle(indices)
            parent1=random.sample(list(range(len(population))),1)[0]
            parent2=random.sample(list(range(len(population))),1)[0]
            mini1=[population[parent1]['columns'][i] for i in indices[:math.floor(len(indices)/2)]]
            mini2=[population[parent2]['columns'][i] for i in indices[math.floor(len(indices)/2):]]
            mini1.extend(mini2)
            #[1 3 4 2 5 8]
            #[1 2 3 4 5 8]
            # restore order of columns 
            mini3=list(range(len(mini1)))
            j=0
            for index in indices:
                mini3[index]=mini1[j]
                j=j+1

            next_gen.append({'columns':mini3})
            if(random.random()<mutation_ratio):
                mutate(next_gen[-1]['columns'])
        population=next_gen[:]
        
def randomize_layout(columns):
    for column in columns:#for each column
        column['prf'][0][0]=random.random()*(column['max'][0][0]-column['min'][0][0])+column['min'][0][0]
        column['prf'][1][0]=random.random()*(column['max'][1][0]-column['min'][1][0])+column['min'][1][0]
        column['prf'][1][1]=random.random()*(column['max'][1][1]-column['min'][1][1])+column['min'][1][1]
        column['prf'][0][1]=random.random()*(column['max'][0][1]-column['min'][0][1])+column['min'][0][1]
    return columns

def mutate(columns):
    indices=random.sample(range(len(columns)),1)
    for i in indices:
        columns[i]['prf'][0][0]=random.random()*(columns[i]['max'][0][0]-columns[i]['min'][0][0])+columns[i]['min'][0][0]
        columns[i]['prf'][1][0]=random.random()*(columns[i]['max'][1][0]-columns[i]['min'][1][0])+columns[i]['min'][1][0]
        columns[i]['prf'][1][1]=random.random()*(columns[i]['max'][1][1]-columns[i]['min'][1][1])+columns[i]['min'][1][1]
        columns[i]['prf'][0][1]=random.random()*(columns[i]['max'][0][1]-columns[i]['min'][0][1])+columns[i]['min'][0][1]
    

def calc_column_data(columns):
    for column in columns:
        #TODO: eliminate gen_points as much as possible (hint: columns are all rectangles)
        column["centroid"]=((column['prf'][0][0]+column['prf'][1][0])/2,(column['prf'][0][1]+column['prf'][1][1])/2)#Polygon(gen_points(column['prf'])).centroid
        column["area"]=abs((column['prf'][0][0]-column['prf'][1][0])*(column['prf'][0][1]-column['prf'][1][1]))#Polygon(gen_points(column['prf'])).area
        column["Ix"]=get_Ix(column["prf"],column["centroid"])
        column["Iy"]=get_Iy(column["prf"],column["centroid"])
        #column["Ixy"]=get_Ixy(gen_points(column["prf"]),column["centroid"])
        column["Rx"]=get_Rx(gen_points(column["prf"]),column["area"])
        column["Ry"]=get_Ry(gen_points(column["prf"]),column["area"])

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
        sum_x+=column['centroid'][0]*column['area']
        sum_y+=column['centroid'][1]*column['area']
        sum_w+=column['area']
    com=(sum_x/sum_w,sum_y/sum_w)
    #print("Center of mass: "+str(com))
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
        sum_Iy_cx+=column['Iy']*column['centroid'][0]
        sum_Ix_cy+=column['Ix']*column['centroid'][1]
    cor=(sum_Iy_cx/sum_Iy,sum_Ix_cy/sum_Ix)
    #print("Center of rigidity: "+str(cor))
    return cor

# Floor Ix
def calc_global_Ix(columns, global_com):
    sum=0
    for column in columns:
        sum+=column["Ix"]+column["area"]*(global_com[1]-column["centroid"][1])**2
    return sum

# Floor Iy
def calc_global_Iy(columns, global_com):
    sum=0
    for column in columns:
        sum+=column["Iy"]+column["area"]*(global_com[0]-column["centroid"][0])**2
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

def save_specific_sample(iter,popu,doc,index):
    msp = doc.modelspace()
    lines=[]
    col=popu[index]
    for co in col['columns']:
        c=gen_points(co['prf'])
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

main()