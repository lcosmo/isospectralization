from scipy import sparse
#import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import scipy

from plyfile import PlyData, PlyElement
import time


def totuple(a):
    return [ tuple(i) for i in a]

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
    face = np.array([ tuple([i]) for i in T],dtype=[('vertex_indices', 'i4', (3,))])
    el1 = PlyElement.describe(vertex, 'vertex')
    el2 = PlyElement.describe(face, 'face')
    PlyData([el1,el2]).write(filename)
    
    
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
    #print(idx)

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

    #Windices = np.zeros(shape=(n*n,m),dtype=dtype)
    #for i in range(m):    
    #    Windices[invmap[i,0]*n+invmap[i,1],i] = -1;
    
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


    return np.asarray(VERT,dtype),TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary
