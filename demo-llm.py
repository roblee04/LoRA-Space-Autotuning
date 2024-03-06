#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#


"""
Example of invocation of this script:

mpirun -n 1 python ./demo.py -nrun 20 -ntask 5 -perfmodel 0 -optimization GPTune

where:
    -ntask is the number of different matrix sizes that will be tuned
    -nrun is the number of calls per task
    -perfmodel is whether a coarse performance model is used
    -optimization is the optimization algorithm: GPTune,opentuner,hpbandster
"""


################################################################################
import sys
import os
# import mpi4py
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import * # import all


import argparse
# from mpi4py import MPI
import numpy as np
import time

from callopentuner import OpenTuner
from callhpbandster import HpBandSter

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-nodes', type=int, default=1,help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=2,help='Number of cores per machine node')
    parser.add_argument('-machine', type=str,default='-1', help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-nrun', type=int, default=20, help='Number of runs per task')
    parser.add_argument('-perfmodel', type=int, default=0, help='Whether to use the performance model')
    parser.add_argument('-tvalue', type=float, default=1.0, help='Input task t value')

    args = parser.parse_args()

    return args

def objectives(point):

    # from the input space (tuning task)
    input_model_to_run = point['input_model']

    ## from the parameter space (tuning parameter)
    lora_r = point['lora_r'] ##
    lora_alpha = point['lora_alpha']
    ###

    print ("input_model_to_run: ", input_model_to_run)
    print ("lora_r: ", lora_r)
    print ("lora_alpha: ", lora_alpha)

    #def dump_to_config_file( ... ):
    #    with open("file.txt", "w"):
    #        ....

    #run ...

    #...wait_for_it_to_be_finished

    #query_the_objective.. (accuracy, time, etc.)

    import numpy as np
    accuracy = np.random.random()

    #YOUR_LORA = CALL_YOUR_LORA(input_model_to_run,
    #        lora_r,
    #        lora_alpha)

    #accuracy = EVALUATE_YOUR_LORA(YOUR_LORA)

    return [accuracy]

# if need to return multiple objectives, ... return [accuracy, time, etc..]


def main():

    import matplotlib.pyplot as plt
    global nodes
    global cores

    # Parse command line arguments
    args = parse_args()
    ntask = args.ntask
    nrun = args.nrun
    tvalue = args.tvalue
    TUNER_NAME = args.optimization

    (machine, processor, nodes, cores) = GetMachineConfiguration()
    print ("machine: " + machine + " processor: " + processor + " num_nodes: " + str(nodes) + " num_cores: " + str(cores))
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME

    #### should ADAPT this..
    input_models = ["modelA.data", "modelB.data", "modelC.data"]
    input_model = Categoricalnorm(input_models, transform="onehot", name="input_model")
    input_space = Space([input_model])

    lora_r = Integer(1, 64, transform="normalize", name="lora_r")
    lora_alpha = Integer(1, 32, transform="normalize", name="lora_alpha")
    lora_dropout = Real(0., 1., transform="normalize", name="lora_dropout")
    lora_target_modules = Categoricalnorm(["moduleA","moduleB","moduleC"], transform="onehot", name="lora_target_modules")
    lora_target_linear = Categoricalnorm(["false","true"], transform="onehot", name="lora_target_linear")
    lora_fan_in_fan_out = Categoricalnorm(["in","out"], transform="onehot", name="lora_fan_in_fan_out")
    parameter_space = Space([lora_r, lora_alpha, lora_dropout, lora_target_modules, lora_target_linear, lora_fan_in_fan_out])

    ## for now, it's just coded to consider one metric output
    ## but ... should handle multiple metrics?
    accuracy = Real(float("-Inf"), float("Inf"), name="accuracy")
    output_space = Space([accuracy])

    constraints = {}

    problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None)

    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    options = Options()
    
    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False

    # options['objective_evaluation_parallelism'] = True
    # options['objective_multisample_threads'] = 1
    # options['objective_multisample_processes'] = 4
    # options['objective_nprocmax'] = 1

    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1
    options['model_optimzier'] = 'lbfgs'

    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16

    # options['sample_algo'] = 'MCS'

    # Use the following two lines if you want to specify a certain random seed for the random pilot sampling
    options['sample_class'] = 'SampleLHSMDU'
    options['sample_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for surrogate modeling
    options['model_class'] = 'Model_GPy_LCM' #'Model_LCM'
    options['model_random_seed'] = 0
    # Use the following two lines if you want to specify a certain random seed for the search phase
    # options['search_class'] = 'SearchSciPy'
    options['search_random_seed'] = 0

    # options['search_class'] = 'SearchSciPy'
    # options['search_algo'] = 'l-bfgs-b'

    options['verbose'] = False
    options.validate(computer=computer)

    print(options)

    giventask = [["modelA.data"]]

    NI = len(giventask)

    NS = nrun

    TUNER_NAME = 'GPTune'

    if(TUNER_NAME=='GPTune'):
        data = Data(problem)
        gt = GPTune(problem, computer=computer, data=data, options=options,driverabspath=os.path.abspath(__file__))

        #### THIS IS CALLING GPTUNE TUNER, THE BAYESIAN OPTIMIZATION PROCESS
        #### NS = number of samples to be evaluated
        #### NS1 = number of "random" samples, rather than from BO's iteration
        #### NI = number of "tasks"
        #### MLA = Multitask learning autotuning, so if the number of tasks is greater than "1", the multitask learning technique will be used (the technique is called "LCM"). But if the number of tasks is "1", then this is just BO with standard GP.
        (data, modeler, stats) = gt.MLA(NS=NS, NS1=int(NS/2), NI=NI, Tgiven=giventask)

        ## printed output
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    input .. model:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='opentuner'):
        (data,stats)=OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='cgp'):
        from callcgp import cGP
        options['EXAMPLE_NAME_CGP']='GPTune-Demo'
        options['N_PILOT_CGP']=int(NS/2)
        options['N_SEQUENTIAL_CGP']=NS-options['N_PILOT_CGP']
        (data,stats)=cGP(T=giventask, tp=problem, computer=computer, options=options, run_id="cGP")
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    t:%f " % (data.I[tid][0]))
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

if __name__ == "__main__":
    main()
