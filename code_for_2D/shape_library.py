from scipy import sparse
#import matplotlib.pyplot as plt
import os

import tensorflow as tf
import numpy as np
import scipy

from mpl_toolkits.mplot3d import Axes3D

from plyfile import PlyData, PlyElement

from IPython.display import clear_output
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import vispy.geometry
from matplotlib import path
from scipy.spatial.distance import cdist



def totuple(a):
    return [ tuple(i) for i in a]
        
def plotly_trisurf(V, TRIV):
    p3.clear()
    p3.plot_trisurf(V[:,0], V[:,1], V[:,2], triangles=TRIV)
    p3.scatter(V[:,0], V[:,1], V[:,2], marker='sphere', color='blue', size=0.33)
    p3.squarelim()
    p3.show()
    
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
def tfeval(X):
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0}, 
    allow_soft_placement = False, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    tf.global_variables_initializer().run(session=sess)
    #[tfEvals, tfEvecs] = tf.self_adjoint_eig(X)
    #[evals, evecs]  = sess.run( [tfEvals, tfEvecs] );
    x = sess.run(tf.identity(X) );
    sess.close();
    return x

def tfeig(X):
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0}, 
    allow_soft_placement = False, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    tf.global_variables_initializer().run(session=sess)
    #[tfEvals, tfEvecs] = tf.self_adjoint_eig(X)
    #[evals, evecs]  = sess.run( [tfEvals, tfEvecs] );
    LAP = sess.run(tf.identity(X) );
    sess.close();
    
    [evals, evecs] = scipy.linalg.eigh(LAP);
    evals = np.diag(evals)
    return evals, evecs, LAP

def load_mesh(path):
    VERT = np.loadtxt(path+'/mesh.vert')
    TRIV = np.loadtxt(path+'/mesh.triv',dtype='int32')-1
    
    return VERT, TRIV

def totuple(a):
    return [ tuple(i) for i in a]
    
def save_ply(V,T,filename):
    if(V.shape[1]==2):
        Vt = np.zeros((V.shape[0],3))
        Vt[:,0:2] = V
        V = Vt
        
    vertex = np.array(totuple(V),dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
#     print([ tuple([i]) for i in T])
    face = np.array([ tuple([i]) for i in T],dtype=[('vertex_indices', 'i4', (3,))])
    el1 = PlyElement.describe(vertex, 'vertex')
    el2 = PlyElement.describe(face, 'face')
    PlyData([el1,el2]).write(filename)
    #PlyData([el2]).write('some_binary.ply')    
    
def load_ply(fname):
    plydata = PlyData.read(fname)
    VERT = np.asarray([ (v[0],v[1],v[2]) for v in plydata.elements[0].data])
    TRIV = np.asarray([ t[0] for t in plydata.elements[1].data])
    return VERT,TRIV

    
def ismember(T, pts):
    out = np.zeros(np.shape(T)[0])
    for r in range(np.shape(T)[0]):
        s=0
        for c in range(np.shape(T)[1]):
            if np.sum(T[r,c]==pts)>0: s=s+1;
        out[r] = s>0;
    return out


def prepare_mesh(VERT,TRIV,dtype='float32'):
    edges = np.ones(shape=(np.shape(VERT)[0],np.shape(VERT)[0],2),dtype='int32')*-1
    edges_count = np.zeros(shape=(np.shape(VERT)[0],np.shape(VERT)[0]),dtype='int32')

    def setedg(i,j,k):
        _setedg(i,j,k)
        _setedg(j,i,k)

    def _setedg(i,j,k):
        edges_count[i,j] +=1
        if edges[i,j,0]==k: 
            return
        if edges[i,j,1]==k: 
            return
        if edges[i,j,0]==-1: 
    #         print(edges[i,j,0])
            edges[i,j,0]=k
        else:        
            edges[i,j,1]=k


    for ti in range(np.shape(TRIV)[0]):
        setedg(TRIV[ti,0],TRIV[ti,1],TRIV[ti,2])
        setedg(TRIV[ti,2],TRIV[ti,0],TRIV[ti,1])
        setedg(TRIV[ti,1],TRIV[ti,2],TRIV[ti,0])

    n = np.shape(VERT)[0]
    m = int(np.sum( ((edges[:,:,0]>=0) + (edges[:,:,1]>=0)) >0)/2);

    map_ = np.ones(shape=(n,n),dtype='int32')*-1;
    invmap = np.ones(shape=(m,2),dtype='int32')*-1;
    iM = np.zeros(shape=(m,n),dtype=dtype);
    bound_edges = np.zeros(shape=(m,1),dtype='bool');

    idx=0
    for i in range(n):
        for j in range(i+1,n):
            if(edges[i,j,0]==-1 and edges[i,j,1]==-1): continue;
            map_[i,j]  = idx;
            map_[j,i]  = idx;
            invmap[idx,:] = [i,j]
            iM[idx,i] = 1;
            iM[idx,j] = -1;   
            bound_edges[idx,0] = edges_count[i,j]<2
            idx=idx+1;
#     print(idx)

    Ael = np.zeros(shape=(n,m),dtype=dtype);
    for i in range(n):
        Ael[i,map_[i,np.nonzero(map_[i,:]+1)]]=1

    Ik  = np.zeros(shape=(m,m),dtype=dtype);
    Ih  = np.zeros(shape=(m,m),dtype=dtype);
    Ik_k  = np.zeros(shape=(m,m,3),dtype=dtype);
    Ih_k  = np.zeros(shape=(m,m,3),dtype=dtype);
    for i in range(n):
        for j in range(i+1,n):        
            if(edges[i,j,0]==-1): continue        

            k = edges[i,j,0]
            Ik[map_[i,j],map_[i,j]]=-1;
            Ik[map_[i,j],map_[j,k]]=1;
            Ik[map_[i,j],map_[k,i]]=1

            Ik_k[map_[i,j],map_[i,j],0] = 1;
            Ik_k[map_[i,j],map_[j,k],1] = 1;
            Ik_k[map_[i,j],map_[k,i],2] = 1;

            if(edges[i,j,1]==-1): continue    

            k = edges[i,j,1]
            Ih[map_[i,j],map_[i,j]]=-1;
            Ih[map_[i,j],map_[j,k]]=1;
            Ih[map_[i,j],map_[k,i]]=1;

            Ih_k[map_[i,j],map_[i,j],0] = 1;
            Ih_k[map_[i,j],map_[j,k],1] = 1;
            Ih_k[map_[i,j],map_[k,i],2] = 1;


    Tni =  np.zeros(shape=(np.shape(TRIV)[0],n),dtype=dtype);
    Tpi =  np.zeros(shape=(np.shape(TRIV)[0],n),dtype=dtype);
    Txi =  np.zeros(shape=(np.shape(TRIV)[0],n),dtype=dtype);
    for i in range(np.shape(TRIV)[0]):
        Tni[i,TRIV[i,0]] = -1;
        Tni[i,TRIV[i,1]] =  1;
        Tpi[i,TRIV[i,0]] =  1;
        Txi[i,TRIV[i,2]] =  1;

    Windices = np.zeros(shape=(m,2),dtype=dtype)
    for i in range(m):    
        #Windices[i,:] = [invmap[i,0],invmap[i,1]];
        Windices[i,:] = [invmap[i,0]*n+invmap[i,1], i];


    def calc_adj_matrix(VERT,TRIV):
        n = np.shape(VERT)[0]
        A = np.zeros((n,n))    
        A[TRIV[:,0],TRIV[:,1]] = 1
        A[TRIV[:,1],TRIV[:,2]] = 1
        A[TRIV[:,2],TRIV[:,0]] = 1
        return A

    A = calc_adj_matrix(VERT, TRIV)
    A = np.matmul(np.diag(1/np.sum(A, axis=1)),A);
    Bary = A - np.eye(np.shape(VERT)[0]);

    nb = bound_edges.shape[0]
    ord_be  = np.zeros(shape=(nb,2),dtype=dtype);

    bed=invmap[bound_edges[:,0],:]
    avail = np.ones(shape=(bed.shape[0],), dtype='bool')
    ord_list = []
    ord_list.append(bed[0,:])
    avail[0] = False
    for i in range(bed.shape[0]-1):
        nx = np.logical_and(np.sum(bed==ord_list[-1][1],axis=1) ,avail)
        if(np.sum(nx)==0):
            nx = np.logical_and(np.sum(bed==ord_list[-1][0],axis=1) ,avail)
        avail = np.logical_and(avail, 1-nx)
        nx_e = bed[nx,:][0]
        if(nx_e[0] != ord_list[-1][1]):
            nx_e = nx_e[[1,0]]
        ord_list.append(nx_e)
    ord_list=np.asarray(ord_list)  

    return np.asarray(VERT,dtype),TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list



def fps_euclidean(X,nsamp, seed=1):
    pts = np.zeros((nsamp,2))
    pts[range(np.size(seed)),:] = X[seed,:];
    for i in range(np.size(seed),nsamp):
        d = np.min(cdist(X,pts),axis=1)
        index_max = np.argmax(d)
        pts[i,:] = X[index_max,:]; 
    return pts


def resample(VERT, TRIV, npts=-1):
    if(npts==-1):
        npts=int(np.shape(VERT)[0])
    
    minx = np.min(VERT)
    maxx = np.max(VERT)

    dpts = int(npts/5);
    xv, yv = np.meshgrid(np.asarray(range(dpts))/dpts*(maxx-minx)+minx,np.asarray(range(dpts))/dpts*(maxx-minx)+minx, sparse=False, indexing='xy')
    xv = np.reshape(xv,xv.shape[0]*xv.shape[1])
    yv = np.reshape(yv,yv.shape[0]*yv.shape[1])

    xv = xv + (np.random.rand(xv.shape[0])-0.5)*0.9*(maxx-minx)/dpts
    yv = yv + (np.random.rand(yv.shape[0])-0.5)*0.9*(maxx-minx)/dpts
    
    mesh = prepare_mesh(VERT,TRIV)
    [VERT, TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh
    
    pts = np.stack((xv,yv),axis=1)
    
    p = path.Path(VERT[ord_list[:,0],:2])  # square with legs length 1 and bottom left corner at the origin
    inside = p.contains_points(pts)

    #resample boundary
    ord_list = np.vstack([ord_list, ord_list[:1,:]])
    
    bound_pts = [VERT[ord_list[0,0],:2]]
    for i in range(1,ord_list.shape[0]):
        sp = bound_pts[-1]
        pv = VERT[ord_list[i,0],:2]-sp
        l = np.linalg.norm(pv)
        toadd = int(l/0.05)        
        for j in range(1,toadd+1):
            pta = sp+pv*j/toadd
            bound_pts.append(pta)

    bound_pts = np.asarray(bound_pts) 

    pts = np.concatenate( (bound_pts,pts[inside,:] ), axis=0)
    pts = fps_euclidean(pts, npts, np.asarray(range(bound_pts.shape[0])) )

    sg = np.stack( ( np.asarray(range(bound_pts.shape[0]-1)),np.asarray(range(1,bound_pts.shape[0])) ), axis=1)
    sg = np.concatenate( (sg, [ [bound_pts.shape[0]-1, 0]] ) )
    
    dt = vispy.geometry.Triangulation(pts, sg)
    dt.triangulate()

    VV = dt.pts
    TT = dt.tris
    
    valid_idx = np.unique(TT)
    vV = VV[valid_idx,:]
    map_v = np.ones( (VV.shape[0]), np.int32)*-1
    map_v[valid_idx] =  np.asarray(range(valid_idx.shape[0]))
    vT = map_v[TT]

    n = np.cross( vV[vT[:,0],:]-vV[vT[:,1],:], vV[vT[:,0],:]-vV[vT[:,2],:]  )
    vT[n<0,1:] = np.flip(vT,1)[n<0,:2]

    return vV, vT
