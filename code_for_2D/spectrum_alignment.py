from scipy import sparse
#import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import numpy as np
import scipy
from shape_library import *
from IPython.display import clear_output

import matplotlib.pyplot as plt
    
class OptimizationParams:
    def __init__(self, smoothing='displacement'):
        
        self.checkpoint = 100
        self.plot=False
        
        self.evals = [20]
        self.numsteps = 5000
        self.remesh_step = 500
        
        self.decay_target = 0.05
        self.learning_rate = 0.005
        self.min_eval_loss = 0.05
        
        self.flip_penalty_reg = 1e10
        self.inner_reg = 1e0
        self.bound_reg = 2e1
        
def tf_calc_lap(mesh,VERT): 
    [_,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh
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
    mesh = prepare_mesh(VERT,TRIV)
    Lx,S,L,Ak = tf_calc_lap(mesh,mesh[0])
    Si = tf.diag(tf.sqrt(1/S[:,0]))
    Lap =  tf.matmul(Si,tf.matmul(Lx,Si));
    [evals,evecs]  = tf.self_adjoint_eig( Lap )
    return tfeval(evals)

    
def build_graph(mesh, evals, nevals, step=1.0, params=OptimizationParams()): 
        """Build the tensorflow graph
        
        Input arguments:
        - mesh: structure representing the triangulated mesh
        - nevals: number of eigenvalues to optimize
        """
        graph = lambda: None
        
        [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh
        dtype='float32'
        if(Xori.dtype=='float64'):
            dtype='float64'
        if(Xori.dtype=='float16'):
            dtype='float16'
            
            
        #setup cosine decay
        global_step = tf.Variable(step+1.0, name='global_step',trainable=False)
        
        graph.global_step_val = tf.placeholder(dtype)
        graph.set_global_step = tf.assign(global_step, graph.global_step_val).op
        
        cosine_decay = 0.5 * (1 + tf.cos(3.14 * tf.minimum(np.asarray(params.numsteps/2.0,dtype=dtype),global_step) / (params.numsteps/2.0)))
        graph.decay= (1 - params.decay_target) * cosine_decay + params.decay_target
               
        
        #model the shape deformation as a displacement vector field
        bound_vert = np.zeros((n,1),np.float32)
        bound_vert[ord_list] = 1
        
        dXb = tf.Variable((0*Xori).astype(dtype));
        dXi = tf.Variable((0*Xori).astype(dtype));
        scaleX = tf.Variable(1,dtype=dtype); #not used in shape alignment
       
    
        graph.input_X = tf.placeholder(shape=dXb.shape,dtype=dtype);        
        graph.assign_X = tf.assign(dXb, graph.input_X-Xori*scaleX).op;
                
        graph.X=(Xori + dXb*bound_vert + dXi*(1-bound_vert))*scaleX;

        Lx,S,L,Ak = tf_calc_lap(mesh,graph.X)

        #Normalized Laplacian
        Si = tf.diag(tf.sqrt(1/S[:,0]))
        Lap =  tf.matmul(Si,tf.matmul(Lx,Si));
        
        #Spectral decomposition
        [graph.evals,v]  = tf.self_adjoint_eig( Lap )
        graph.cost_evals = 1e1*tf.nn.l2_loss( (graph.evals[0:nevals]-evals[0:nevals]) * (1/np.asarray(range(1,nevals+1),'float32'))) # \
#       
        # triangle flip penalty
        tp = tf.matmul(Tpi[:,:],graph.X)
        tx = tf.matmul(Txi[:,:],graph.X)
        tn = tf.matmul(Tni[:,:],graph.X)
        Rot = np.asarray([[0, 1],[-1, 0]],dtype)
        cp = tf.reduce_sum(tf.matmul(tn,Rot)*(tx-tp),axis=1)
        graph.cp = cp-1e-4
        graph.flip_cost =  params.flip_penalty_reg*tf.nn.l2_loss(graph.cp-tf.abs(graph.cp))
        
        #inner points regularizer
        meanA, varA = tf.nn.moments(Ak, axes=[0])
        graph.inner_reg_cost = params.inner_reg*(tf.nn.l2_loss(L) + tf.nn.l2_loss(varA));
        #boundary points regularizer
        graph.bound_reg_cost = params.bound_reg*graph.decay* tf.reduce_sum(bound_edges*L)

        #inner and outer points cost functions
        graph.cost_bound = graph.cost_evals + graph.flip_cost + graph.bound_reg_cost
        graph.cost_inner = graph.inner_reg_cost + graph.flip_cost
        
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        
        def clipped_grad_minimize(cost, variables):
            gvs = optimizer.compute_gradients(cost, var_list=variables)
            clipped_gvs = [(tf.clip_by_value(grad, -0.0001, 0.0001), var) for grad, var in gvs if grad!=None]
            return optimizer.apply_gradients(clipped_gvs,global_step=global_step)
        
        graph.train_op_bound = clipped_grad_minimize(graph.cost_bound, [dXb])
        graph.train_op_inner = clipped_grad_minimize(graph.cost_inner, [dXi])
                
        return graph
        
    
    
def run_optimization(mesh, target_evals, out_path, params = OptimizationParams() ):
    
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 1}, allow_soft_placement = False, gpu_options=gpu_options)
    config.gpu_options.allow_growth=True

    try:
        os.makedirs('%s/' % (out_path))
    except OSError:
        pass

    [Xopt,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh
    
    iterations = [];
    for nevals in params.evals:
      tf.reset_default_graph()
      with tf.Session(config=config) as session:   
        
        step=0    
        while(step<params.numsteps-1):
          mesh = prepare_mesh(Xopt,TRIV)
         
          [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh
          edg_v = np.zeros((n,1),np.float32)
          edg_v[ord_list] = 1

          #Buil tensorflow graph
          graph = build_graph(mesh,target_evals, nevals,step)#,smoothing,numsteps,edg_v)
          tf.global_variables_initializer().run()

          tic()
          for step in range(step+1,params.numsteps):
            
            if((step)%params.remesh_step==0):
                print("RECOMPUTING TRIANGULATION at step %d" % step)
                break;

            try:
                feed_dict = {}

                #alternate optimization of inner and boundary vertices
                if(int(step/10)%2==0):
                    _, er, ee, Xopt_t = session.run([graph.train_op_inner,graph.cost_inner,graph.cost_evals,graph.X], feed_dict=feed_dict)
                else:
                    _, er, ee, Xopt_t = session.run([graph.train_op_bound,graph.cost_bound,graph.cost_evals,graph.X], feed_dict=feed_dict)
                
                iterations.append((step, nevals, er, ee,int(step/10)%2))
                
                
                if ( (step) % params.checkpoint == 0 or step==(params.numsteps-1) or step==1): 
                    toc()
                    tic()

                    cost, cost_evals, cost_vcL, cost_vcW, decay, flip, evout = session.run([graph.cost_bound, graph.cost_evals, graph.inner_reg_cost,graph.bound_reg_cost, graph.decay, graph.cp,graph.evals], feed_dict=feed_dict)

                    print('Iter %f, cost: %f(evals cost: %f (%f) (%f), smoothness weight: %f). Flip: %d' %
                          (step, cost, cost_evals, cost_vcL, cost_vcW, decay, np.sum(flip<0)))  

                    if params.plot:
#                         fig=plt.figure(figsize=(9, 4), dpi= 80, facecolor='w', edgecolor='k')
                        plt.plot(target_evals[0:nevals],'-r')
                        plt.plot(evout[0:nevals],'-b')
                        plt.show()

                        plt.triplot(Xopt[:,0],Xopt[:,1],TRIV)   
                        plt.axis('equal')
                        plt.show()
                        
                    save_ply(Xopt,TRIV,'%s/evals_%d_iter%d.ply' % (out_path,nevals,step))
                    np.savetxt('%s/evals_%d_iter%d.txt' % (out_path,nevals,step),evout)

                    np.savetxt('%s/iterations.txt' % (out_path),iterations)
                    #early stop
                    if(ee<params.min_eval_loss):
                        step=params.numsteps
                        print('Minimum eighenvalues loss reached')
                        break
                    
            except KeyboardInterrupt:
                step = params.numsteps
                break;
            except:
                print(sys.exc_info()[0])
                ee=float('nan')

            #If something went wrong with the spectral decomposition perturbate the last valid state and start over
            if(ee!=ee):
              print('iter %d. Perturbating initial condition' % step)
              tf.global_variables_initializer().run()
              Xopt=Xopt+(np.random.rand(np.shape(Xopt)[0],np.shape(Xopt)[1])-0.5)*1e-3                  
              _ = session.run(graph.set_global_step, feed_dict = {graph.global_step_val: step})  
            else:
              Xopt=Xopt_t     
          if(step<params.numsteps-1):
              [Xopt,TRIV] = resample(Xopt, TRIV)
                
  
