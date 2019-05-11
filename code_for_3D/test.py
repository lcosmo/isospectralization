from shape_library import *
from spectrum_alignment import *
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

params = OptimizationParams()
params.min_eval_loss = 0.0001
params.evals = [20]
params.numsteps = 2000


[VERT, TRIV, _] = load_mesh('data/round_cuber_1000/');
[_, _, evals_t] = load_mesh('data/round_cuber_out_1000/');
mesh = prepare_mesh(VERT,TRIV,'float32')
run_optimization(mesh = mesh, target_evals = evals_t, out_path = 'results/test_cube_out_disp', params = params)
