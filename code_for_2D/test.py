from shape_library import *
from spectrum_alignment import *
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

params = OptimizationParams()
params.evals = [20]
params.numsteps = 5000
params.plot=False

[VERT, TRIV, _] = load_mesh('data/oval/');
[VERT,TRIV] = resample(VERT, TRIV, 300)

# [_, _, evals_t] = load_mesh('data/mickey/');
# mesh = prepare_mesh(VERT,TRIV,'float32')
# run_optimization(mesh = mesh, target_evals = evals_t, out_path = 'results/mickey', params = params)

[_, _, evals_t] = load_mesh('data/bell/');
mesh = prepare_mesh(VERT,TRIV,'float32')
run_optimization(mesh = mesh, target_evals = evals_t, out_path = 'results/mickey', params = params)