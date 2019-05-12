from scipy import sparse
#import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import scipy
from IPython.display import clear_output

from shape_library import *
    
class OptimizationParams:
    def __init__(self, smoothing='displacement'):
        self.checkpoint = 100
        self.numsteps = 2000
        self.evals = [10,20,30]
        self.smoothing = smoothing
        self.decay_target = 0.01
        
        if(smoothing=='displacement'):
            self.curvature_reg = 2e3
            self.smoothness_reg = 2e3
        else:
            self.curvature_reg = 1e5
            self.smoothness_reg = 5e4
        
        self.volume_reg = 1e1
        self.l2_reg = 2e6
        
        self.opt_step = 0.00025
        self.min_eval_loss = 0.05
        
        
def tf_calc_lap(mesh,VERT): 
    [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = mesh
    
    dtype='float32'
    if(VERT.dtype=='float64'):
        dtype='float64'
    if(VERT.dtype=='float16'):
        dtype='float16'

    L2 = tf.expand_dims(tf.reduce_sum(tf.matmul(iM,VERT)**2,axis=1),axis=1)
    L=tf.sqrt(L2);

    def  fAk(Ik,Ik_k):
        Ikp=np.abs(Ik);
        Sk = tf.matmul(Ikp,L)/2    
        SkL = Sk-L;    
        Ak = Sk*(tf.matmul(Ik_k[:,:,0],Sk)-tf.matmul(Ik_k[:,:,0],L))\
                       *(tf.matmul(Ik_k[:,:,0],Sk)-tf.matmul(Ik_k[:,:,1],L))\
                       *(tf.matmul(Ik_k[:,:,0],Sk)-tf.matmul(Ik_k[:,:,2],L))
        return tf.sqrt(tf.abs(Ak)+1e-20)

    Ak = fAk(Ik,Ik_k)
    Ah = fAk(Ih,Ih_k)

    #sparse representation of the Laplacian matrix
    W = -tf.matmul(Ik,L2)/(8*Ak)-tf.matmul(Ih,L2)/(8*Ah);


    #compute indices to build the dense Laplacian matrix
    Windtf = tf.SparseTensor(indices=Windices, values=-np.ones((m),dtype), dense_shape=[n*n, m])
    Wfull  = -tf.reshape(tf.sparse_tensor_dense_matmul(Windtf,W),(n,n))
    Wfull = (Wfull + tf.transpose(Wfull))

    #actual Laplacian
    Lx = Wfull-tf.diag(tf.reduce_sum(Wfull,axis=1))
    S = (tf.matmul(Ael,Ak)+tf.matmul(Ael,Ah))/6;
    
    return Lx,S,L,Ak;
 
    
def calc_evals(VERT,TRIV):
    mesh = prepare_mesh(VERT,TRIV,'float64')
    Lx,S,L,Ak = tf_calc_lap(mesh,mesh[0])
    Si = tf.diag(tf.sqrt(1/S[:,0]))
    Lap =  tf.matmul(Si,tf.matmul(Lx,Si));
    [evals,evecs]  = tf.self_adjoint_eig( Lap )
    return tfeval(evals)

    
def build_graph(mesh, evals, nevals,nfix, step=1.0, params=OptimizationParams()): #smoothing='absolute', numsteps=40000):
        """Build the tensorflow graph
        
        Input arguments:
        - mesh: structure representing the triangulated mesh
        - nevals: number of eigenvalues to optimize
        - nfix: number of vertices to keep fixed (for partial shape optimization, 0 otherwise)
        """
        [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = mesh

        dtype='float32'
        if(Xori.dtype=='float64'):
            dtype='float64'
        if(Xori.dtype=='float16'):
            dtype='float16'
        print(dtype)
        graph = lambda: None
        
        #model the shape deformation as a displacement vector field
        dX = tf.Variable((0*Xori).astype(dtype) );
        scaleX = tf.Variable(1,dtype=dtype); #not used in shape alignment
                
        graph.input_X = tf.placeholder(shape=dX.shape,dtype=dtype);
        graph.assign_X = tf.assign(dX, graph.input_X-Xori*scaleX).op;
                
        graph.X=Xori*scaleX+dX;
        
        Lx,S,L,Ak = tf_calc_lap(mesh,graph.X)

        #Normalized Laplacian
        Si = tf.diag(tf.sqrt(1/S[:,0]))
        Lap =  tf.matmul(Si,tf.matmul(Lx,Si));

        
        #Spectral decomposition approach
        [s_,v]  = tf.self_adjoint_eig( Lap )
        graph.cost_evals_f1 = 1e2*tf.nn.l2_loss( (s_[0:nevals]-evals[0:nevals])* (1/np.asarray(range(1,nevals+1),dtype)) )/nevals # \
         
            
        #Approach avoiding spectral decomposition - NOT USED
        # [_,EigsOpt,lap] = tfeig(Lap)
        # v = tf.Variable(EigsOpt[:,0:nevals].astype(dtype) );
        # cost_evals_a = 1e3*tf.nn.l2_loss(tf.matmul(tf.transpose(v),v)-tf.eye(nevals,dtype=dtype));
        # cost_evals_b = 1e1*tf.nn.l2_loss( (tf.matmul(Lap,v) - tf.matmul(v,np.diag(evals[0:nevals]).astype(dtype))) )/nevals
        # graph.cost_evals_f2 = cost_evals_a + cost_evals_b
         
            
        meanA, varA = tf.nn.moments(Ak, axes=[0])
        meanL, varL = tf.nn.moments(L, axes=[0])

        graph.global_step = tf.Variable(step+1.0, name='global_step',trainable=False, dtype=dtype)
        graph.global_step_val = tf.placeholder(dtype)
        graph.set_global_step = tf.assign(graph.global_step, graph.global_step_val).op        
        
        #regularizers decay factor
        cosine_decay = 0.5 * (1 + tf.cos(3.14 * tf.minimum(np.asarray(params.numsteps/2.0,dtype=dtype),graph.global_step) / (params.numsteps/2.0)))
        graph.decay= (1 - params.decay_target) * cosine_decay + params.decay_target
        
        if(params.smoothing=='displacement'):    
            graph.vcL = params.curvature_reg*graph.decay * tf.nn.l2_loss( tf.matmul(Bary.astype(dtype),dX)[nfix:,:]);
            graph.vcW = params.smoothness_reg*graph.decay *tf.nn.l2_loss( tf.matmul(Lx,dX)[nfix:,:]) 
        if(params.smoothing=='absolute'):
            graph.vcL = params.curvature_reg*graph.decay * tf.nn.l2_loss( tf.matmul(Bary.astype(dtype),S*graph.X)[nfix:,:]);
            graph.vcW = params.smoothness_reg**graph.decay *tf.nn.l2_loss( tf.matmul(Lx,graph.X)[nfix:,:]) 
       
        #Volume compuation
        T1 =  tf.gather(graph.X, TRIV[:,0])
        T2 =  tf.gather(graph.X, TRIV[:,1])
        T3 =  tf.gather(graph.X, TRIV[:,2])
        XP = tf.cross(T2-T1, T3-T2)
        T_C = (T1+T2+T3)/3
        graph.Volume = params.volume_reg*graph.decay*tf.reduce_sum(XP*T_C/2)/3


        #L2 regularizer on total displacement weighted by area elements
        graph.l2_reg = params.l2_reg*tf.nn.l2_loss(S*dX)

            
        graph.cost_spectral = graph.cost_evals_f1 + graph.vcW + graph.vcL -  graph.Volume + graph.l2_reg

        optimizer = tf.train.AdamOptimizer(params.opt_step)
        
        #gradient clipping  
        gvs = optimizer.compute_gradients(graph.cost_spectral)
        capped_gvs = [(tf.clip_by_value(grad, -0.0001, 0.0001), var) for grad, var in gvs if grad!=None]
        graph.train_op_spectral = optimizer.apply_gradients(capped_gvs, global_step=graph.global_step)

        [graph.s_,v]  = tf.self_adjoint_eig( Lap )        
        return graph
        
        

        
    
def run_optimization(mesh, target_evals, out_path, params = OptimizationParams() ):
    
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 1}, allow_soft_placement = False, gpu_options=gpu_options)
    config.gpu_options.allow_growth=True

    try:
        os.makedirs('%s/' % (out_path))
    except OSError:
        pass

    [VERT, TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = mesh
    pstart = 0;           
    Xori = VERT[:,0:3]
    Xopt = VERT[:,0:3]
    
    #Optimize the shape increasing the number of eigenvalue to be taken into account
    iterations = [];
    for nevals in params.evals:
        step=0
        tf.reset_default_graph()

        graph = build_graph(mesh, target_evals, nevals, pstart, step, params)

        with tf.Session(config=config) as session:
            tf.global_variables_initializer().run()

            _ = session.run(graph.assign_X,feed_dict = {graph.input_X: Xopt})

            while(step<params.numsteps-1):
              tic()
              for step in range(step,params.numsteps):
                try:
                    if ( (step) % params.checkpoint == 0 or step==(params.numsteps-1) ):  
                        toc()
                        tic()
                        er, erE, ervcL, Xopt2, evout, errcW, vol, l2reg = session.run([graph.cost_spectral, graph.cost_evals_f1, graph.vcL, graph.X, graph.s_, graph.vcW, graph.Volume, graph.l2_reg])
                        print('Iter %f, cost: %f(e %f, l %f, w %f - vol: %f + l2reg: %f)' % (step, er, erE,  ervcL, errcW, vol, l2reg))

                        save_ply(Xopt,TRIV,'%s/evals_%d_iter%d.ply' % (out_path,nevals,step))
                        np.savetxt('%s/evals_%d_iter%d.txt' % (out_path,nevals,step),evout)
                        
                        np.savetxt('%s/iterations.txt' % (out_path),iterations)
                        #early stop
                        if(erE<params.min_eval_loss):
                            step=params.numsteps
                            print('Minimum eighenvalues loss reached')
                            break
                        
                    #Optimization step
                    _, er, ee, Xopt_t = session.run([graph.train_op_spectral,graph.cost_spectral,graph.cost_evals_f1,graph.X])
                    iterations.append((step, nevals, er, ee))
                except KeyboardInterrupt:
                    step = params.numsteps
                    break;
                #except:
                #    ee=float('nan')

                #If something went wrong with the spectral decomposition perturbate the last valid state and start over
                if(ee!=ee):
                  print('iter %d: Perturbating vertices position' % step)
                  tf.global_variables_initializer().run()
                  Xopt=Xopt+(np.random.rand(np.shape(Xopt)[0],np.shape(Xopt)[1])-0.5)*1e-3
                  _ = session.run(graph.assign_X,feed_dict = {graph.input_X: Xopt})                    
                  _ = session.run(graph.set_global_step, feed_dict = {graph.global_step_val: step})  
#                   session.run(tf.variables_initializer(optimizer.variables()))
                else:
                  Xopt=Xopt_t   
