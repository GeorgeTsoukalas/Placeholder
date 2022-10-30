from __future__ import print_function
import math
from functools import reduce
import os
import sys
import numpy as np
import torch
import random
import torch.optim as optim
from itertools import chain
from tqdm import tqdm

import argparse
import pickle
import time
import heapq

# import program_learning
from dataset import Dataset
from algorithms import NAS
from program_graph import ProgramGraph
from utils.data_loader import CustomLoader, IOExampleLoader
from utils.evaluation import label_correctness, value_correctness
from utils.logging import init_logging, log_and_print, print_program
from utils.loss import SoftF1LossWithLogits

from dsl.library_functions import LibraryFunction

from metal.common.utils import CEHolder
from metal.common.constants import CE_KEYS
from metal.parser.sygus_parser import SyExp

import pdb



def parse_args():
    parser = argparse.ArgumentParser()
    # cmd_args for experiment setup
    # parser.add_argument('-t', '--trial', type=int, required=True,
    #                     help="trial ID")
    # parser.add_argument('--exp_name', type=str, required=True,
    #                     help="experiment_name")
    parser.add_argument('--save_dir', type=str, required=False, default="results/",
                        help="directory to save experimental results")

    # cmd_args for data
    # parser.add_argument('--train_data', type=str, required=True,
    #                     help="path to train data")
    # parser.add_argument('--test_data', type=str, required=True,
    #                     help="path to test data")
    # parser.add_argument('--valid_data', type=str, required=False, default=None,
    #                     help="path to val data. if this is not provided, we sample val from train.")
    # parser.add_argument('--train_labels', type=str, required=True,
    #                     help="path to train labels")
    # parser.add_argument('--test_labels', type=str, required=True,
    #                     help="path to test labels")
    # parser.add_argument('--valid_labels', type=str, required=False, default=None,
    #                     help="path to val labels. if this is not provided, we sample val from train.")
    # parser.add_argument('--input_type', type=str, required=True, choices=["atom", "list"],
    #                     help="input type of data")
    # parser.add_argument('--output_type', type=str, required=True, choices=["atom", "list"],
    #                     help="output type of data")
    # parser.add_argument('--input_size', type=int, required=True,
    #                     help="dimenion of features of each frame")
    # parser.add_argument('--output_size', type=int, required=True,
    #                     help="dimension of output of each frame (usually equal to num_labels")
    # parser.add_argument('--num_labels', type=int, required=True,
    #                     help="number of class labels")

    # cmd_args for program graph
    parser.add_argument('--max_num_units', type=int, required=False, default=16,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--min_num_units', type=int, required=False, default=4,
                        help="max number of hidden units for neural programs")
    parser.add_argument('--max_num_children', type=int, required=False, default=10,
                        help="max number of children for a node")
    parser.add_argument('--max_depth', type=int, required=False, default=8,
                        help="max depth of programs")
    parser.add_argument('--penalty', type=float, required=False, default=0.0,
                        help="structural penalty scaling for structural cost of edges")
    parser.add_argument('--ite_beta', type=float, required=False, default=1.0,
                        help="beta tuning parameter for if-then-else")
    parser.add_argument('--sem', type=str, required=False, choices=["arith","minmax"], default="arith",
                        help="discrete semantics approximation")

    # cmd_args for training
    parser.add_argument('--train_valid_split', type=float, required=False, default=0.8,
                        help="split training set for validation."+\
                        " This is ignored if validation set is provided using valid_data and valid_labels.")
    parser.add_argument('--normalize', action='store_true', required=False, default=False,
                        help='whether or not to normalize the data')
    parser.add_argument('--batch_size', type=int, required=False, default=50,
                        help="batch size for training set")
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('-search_lr', '--search_learning_rate', type=float, required=False, default=0.02,
                        help="learning rate")
    parser.add_argument('--neural_epochs', type=int, required=False, default=4,
                        help="training epochs for neural programs")
    parser.add_argument('--symbolic_epochs', type=int, required=False, default=6,
                        help="training epochs for symbolic programs")
    # parser.add_argument('--lossfxn', type=str, required=True, choices=["crossentropy", "bcelogits", "softf1"],
    #                     help="loss function for training")
    parser.add_argument('--f1double', action='store_true', required=False, default=False,
                        help='whether use double for soft f1 loss')
    parser.add_argument('--class_weights', type=str, required=False, default = None,
                        help="weights for each class in the loss function, comma separated floats")
    parser.add_argument('--topN_select', type=int, required=False, default=2,
                        help="number of candidates remain in each search")
    parser.add_argument('--resume_graph', type=str, required=False, default=None,
                        help="resume graph from certain path if necessary")
    parser.add_argument('--sec_order', action='store_true', required=False, default=False,
                        help='whether use second order for architecture search')
    parser.add_argument('--spec_design', action='store_true', required=False, default=False,
                        help='if specific, train process is defined manually')
    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help="manual seed")
    parser.add_argument('--finetune_epoch', type=int, required=False, default=10, #CHANGED --finetune_epoch to --finetune_epochs
                        help='Epoch for finetuning the result graph.')
    parser.add_argument('--finetune_lr', type=float, required=False, default=0.01,
                        help='Epoch for finetuning the result graph.')

    # cmd_args for algorithms
    # parser.add_argument('--algorithm', type=str, required=True,
    #                     choices=["mc-sampling", "mcts", "enumeration", "genetic", "astar-near", "iddfs-near", "rnn", 'nas'],
    #                     help="the program learning algorithm to run")
    parser.add_argument('--frontier_capacity', type=int, required=False, default=float('inf'),
                        help="capacity of frontier for A*-NEAR and IDDFS-NEAR")
    parser.add_argument('--initial_depth', type=int, required=False, default=1,
                        help="initial depth for IDDFS-NEAR")
    parser.add_argument('--performance_multiplier', type=float, required=False, default=1.0,
                        help="performance multiplier for IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--depth_bias', type=float, required=False, default=1.0,
                        help="depth bias for  IDDFS-NEAR (<1.0 prunes aggressively)")
    parser.add_argument('--exponent_bias', type=bool, required=False, default=False,
                        help="whether the depth_bias is an exponent for IDDFS-NEAR"+
                        " (>1.0 prunes aggressively in this case)")
    parser.add_argument('--num_mc_samples', type=int, required=False, default=10,
                        help="number of MC samples before choosing a child")
    parser.add_argument('--max_num_programs', type=int, required=False, default=100,
                        help="max number of programs to train got enumeration")
    parser.add_argument('--population_size', type=int, required=False, default=10,
                        help="population size for genetic algorithm")
    parser.add_argument('--selection_size', type=int, required=False, default=5,
                        help="selection size for genetic algorithm")
    parser.add_argument('--num_gens', type=int, required=False, default=10,
                        help="number of genetions for genetic algorithm")
    parser.add_argument('--total_eval', type=int, required=False, default=100,
                        help="total number of programs to evaluate for genetic algorithm")
    parser.add_argument('--mutation_prob', type=float, required=False, default=0.1,
                        help="probability of mutation for genetic algorithm")
    parser.add_argument('--max_enum_depth', type=int, required=False, default=7,
                        help="max enumeration depth for genetic algorithm")
    parser.add_argument('--cell_depth', type=int, required=False, default=3,
                        help="max depth for each cell for nas algorithm")


    parser.add_argument('-data_root', default=None, help='root of dataset')
    parser.add_argument('-file_list', default=None, help='list of programs')
    parser.add_argument('-single_sample', default=None, type=str, help='tune single program')
    parser.add_argument('-use_interpolation', default=0, type=int, help='whether use interpolation')
    parser.add_argument('-top_left', type=bool, default=False, help="set to true to use top-left partition")
    parser.add_argument('-GM', type=bool, default=False, help="set to true to use Gradient Matching")
    return parser.parse_args()

#Print program in Sygus format.
def convert_to_sygus(program):
    if not isinstance(program, LibraryFunction):
        return SyExp(program.name, [])
    else:
        print("The program is " + str(program))
        if program.has_params:
            return SyExp(program.name, [])
            #print(program.parameters)
            #assert False # Fix this implementation
        else:
            collected_names = []
            for submodule, functionclass in program.submodules.items():
                collected_names.append(convert_to_sygus(functionclass))
            return SyExp(program.name, collected_names)


def reward_w_interpolation(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # check if it passes
    status, key, ce = lambda_new_ce()
    if status > 0:
        return 1.0

    # interpolate ce and add neary ones into the buffer
    holder.interpolate_ce(ce)

    #harmonic mean
    scores = []
    for key in CE_KEYS:
        score = lambda_holder_eval(key)
        scores.append(score)
    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0

    return -2.0 + hm_t


def reward_1(sample_index, holder, lambda_holder_eval, lambda_new_ce):
    # print("\n\nsample_index:", sample_index)
    # holder.show_stats()
    # ct = 0
    # s = 0
    scores = []
    for key in CE_KEYS:
        score = lambda_holder_eval(key)
        # print("key:", key,  "score: ", score, "ce_per_key:", holder.ce_per_key)
        # if key in holder.ce_per_key:
        #     ct += len(holder.ce_per_key[key].ce_list)
        #     s += 0.99
        scores.append(score)
    t = sum(scores) # t \in [0, 2.0]
    if t > 0:
        hm_t = 4.0 * scores[0] * scores[1] / t
    else:
        hm_t = 0.0
    # print("ct=",ct, "t=", t, "s=",s)

    return -2.0 + hm_t


def evaluate(algorithm, graph, train_loader, train_config, device):
    validset = train_loader.validset
    with torch.no_grad():
        metric = algorithm.eval_graph(graph, validset, train_config['evalfxn'], train_config['num_labels'], device)
    return metric

if __name__ == '__main__':
    cmd_args = parse_args()

    # manual seed all random for debug
    log_and_print('random seed {}'.format(cmd_args.random_seed))
    torch.random.manual_seed(cmd_args.random_seed)
    #torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.random_seed)
    random.seed(cmd_args.random_seed)

    dataset = Dataset(cmd_args)
    specsample_ls = dataset.sample_minibatch(1, replacement=True)
    #print (f'spec: {specsample_ls[0].spectree.spec}')
    #print (f'grammar: {specsample_ls[0].spectree.grammar}')
    #print (f'vars: {specsample_ls[0].spectree.vars}')

    #print (f'node_seq: {specsample_ls[0].spectree.node_seq}')
    #print (f'node_type_seq: {specsample_ls[0].spectree.node_type_seq}')
    #print (f'numOf_nodes: {specsample_ls[0].spectree.numOf_nodes}')
    #print (f'nodename2ind: {specsample_ls[0].spectree.nodename2ind}')

    print (f'all_tests: {specsample_ls[0].spectree.all_tests}')

    # Context free grammar for synthesis
    cfg = specsample_ls[0].spectree.grammar
    root_symbol = cfg.start
    #print (f'Grammar root: {root_symbol}')
    # IO examples holder.
    g = specsample_ls[0]
    holder = CEHolder(g)

    # Variables of the to-synthesize program.
    vars = specsample_ls[0].spectree.vars
    var_ids = {}
    for id, var in enumerate(vars):
        var_ids[var] = id

    full_exp_name = 'Test'
    save_path = os.path.join(cmd_args.save_dir, full_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # init log
    init_logging(save_path)
    log_and_print("Starting experiment {}\n".format(full_exp_name))

    #///////////////////////////////
    #///////////////////////////////
    #///////////////////////////////

    # TODO allow user to choose device
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    lossfxn = torch.nn.MSELoss()

    if device != 'cpu':
        lossfxn = lossfxn.cuda()

    max_depth = len(specsample_ls[0].spectree.grammar.productions)
    input_size = len(specsample_ls[0].spectree.vars)
    output_size = 1
    num_labels = 1
    input_type = output_type = "atom"

    train_config = {
        'arch_lr' : cmd_args.search_learning_rate,
        'model_lr' : cmd_args.search_learning_rate,
        'train_lr' : cmd_args.learning_rate,
        'search_epoches' : cmd_args.neural_epochs,
        'finetune_epoches' : cmd_args.symbolic_epochs,
        'arch_optim' : optim.Adam,
        'model_optim' : optim.Adam,
        'lossfxn' : lossfxn,
        'evalfxn' : value_correctness,
        'num_labels' : num_labels,
        'save_path' : save_path,
        'topN' : cmd_args.topN_select,
        'arch_weight_decay' : 0,
        'model_weight_decay' : 0,
        'penalty' : cmd_args.penalty,
        'secorder' : cmd_args.sec_order,
        'specific' : [#[None, 2, 0.01, 5], [4, 2, 0.01, 5], [3, 2, 0.01, 5], [2, 2, 0.01, 5], \
                [None, max_depth, 0.1, 10]]#, ["astar", max_depth, 0.1, 5]]#, [4, 4, 0.01, 500], [3, 4, 0.01, 500], [2, 4, 0.01, 500]]#, ["astar", 4, 0.01, cmd_args.neural_epochs]] todo: here is where the epochs are defined for the main training session
    }

    # Initialize program graph
    if cmd_args.resume_graph is None:
        program_graph = ProgramGraph(None, input_type, output_type, input_size, output_size,
                                    cmd_args.max_num_units, cmd_args.min_num_units, max_depth,
                                    device, ite_beta=cmd_args.ite_beta, cfg=cfg, var_ids=var_ids, root_symbol=root_symbol)
        start_depth = 0
    else:
        assert os.path.isfile(cmd_args.resume_graph)
        program_graph = pickle.load(open(cmd_args.resume_graph, "rb"))
        program_graph.max_depth = max_depth
        start_depth = program_graph.get_current_depth()
        # start_depth = 3

    # Initialize algorithm
    algorithm = NAS(frontier_capacity=cmd_args.frontier_capacity)

    #///////////////////////////////
    #///////////////////////////////
    #///////////////////////////////


    iteri = 0
    partition_num = 0
    all_graphs = [[0, program_graph]]
    while(True):
        _, program_graph = heapq.heappop(all_graphs)
        examples = holder.all_ces
        # train_data must be sorted by var_ids
        train_data, train_labels = [], []
        #assert False
        #print(examples)
        #print(len(examples))
        for ex in examples:
            #print(ex.config)
            item = [float(ex.config['X']), float(ex.config['Y'])]
            #item = [None] * len(vars)
            #for k in ex.config: # problem with the implementation here - assymetry in the inputs as iterating through a dictionary is already unorder - have to fix this for later implmenetation
                #item[var_ids[k]] = float( ex.config[k] )
                #item[var_ids[k]] = 1. if ex.config[k] else 0.
            #print(str(item) + " " + str(var_ids))
            print(item)
            train_data.append(item)
            if ex.kind == 'T':
                train_labels.append([1.])
            else:
                train_labels.append([0.])
        # examples = holder.additional_ces
        # for ex in examples:
        #     item = [None] * len(vars)
        #     for k in ex[0]:
        #         item[var_ids[k]] = ex[0][k]
        #     train_data.append(item)
        #     train_labels.append([ex[1]])
        # for model & architecture
        #assert False
        search_loader = IOExampleLoader(train_data, train_labels, batch_size=cmd_args.batch_size, shuffle=False)
        batched_trainset = search_loader.get_batch_trainset()
        batched_validset = search_loader.get_batch_validset()

        log_and_print('data for architecture search')
        log_and_print('batch num of train: {}'.format(len(batched_trainset)))
        log_and_print('batch num of valid: {}'.format(len(batched_validset)))

        # for program train
        train_loader = IOExampleLoader(train_data, train_labels, batch_size=cmd_args.batch_size, shuffle=False)
        batched_prog_trainset = train_loader.get_batch_trainset()
        prog_validset = train_loader.get_batch_validset()
        testset = train_loader.testset

        log_and_print('data for architecture search')
        log_and_print('batch num of train: {}'.format(len(batched_prog_trainset)))
        log_and_print('batch num of valid: {}'.format(len(prog_validset)))
        log_and_print('total num of test: {}'.format(len(testset)))

        # Run program learning algorithm
        best_graph, program_graph = algorithm.run_specific(program_graph,\
                                    search_loader, train_loader,
                                    train_config, device, start_depth=start_depth, warmup=False, cegis=(iteri>0), sem=cmd_args.sem)

        best_program = best_graph.extract_program()
        program_graph.show_graph()
        # print program
        log_and_print("Best Program Found:")
        program_str = print_program(best_program)
        log_and_print(program_str)

        # Save best program
        pickle.dump(best_graph, open(os.path.join(save_path, "graph.p"), "wb"))

        # Finetune
        if cmd_args.finetune_epoch is not None:
            train_config = {
                'train_lr' : cmd_args.finetune_lr,
                'search_epoches' : cmd_args.neural_epochs,
                'finetune_epoches' : cmd_args.finetune_epoch, # changed from cmd_args.finetune_epochs as this could not be found
                'model_optim' : optim.Adam,
                'lossfxn' : lossfxn,
                'evalfxn' : label_correctness,
                'num_labels' : num_labels,
                'save_path' : save_path,
                'topN' : cmd_args.topN_select,
                'arch_weight_decay' : 0,
                'model_weight_decay' : 0,
                'secorder' : cmd_args.sec_order
            }
            log_and_print('Finetune')
            # start time
            start = time.time()
            best_graph = algorithm.train_graph_model(best_graph, train_loader, train_config, device, lr_decay=1.0)
            # calculate time
            total_spend = time.time() - start
            log_and_print('finetune time spend: {} \n'.format(total_spend))
            # store
            pickle.dump(best_graph, open(os.path.join(save_path, "finetune_graph.p"), "wb"))

            # debug
            testset = train_loader.testset
            best_program = best_graph.extract_program()

        # Convert program to sygus for cegis.
        best_program = best_program.submodules["program"]
        print("Now here is a preliminary printing of the program, ignoring the later stuff")
        def lcm(denominators):
            return reduce(lambda a,b: a*b // math.gcd(a,b), denominators)
        def printNumericalInvariant(parameters):
            return str(float(parameters["weights"][0][0].detach())) + "*X + " + str(float(parameters["weights"][0][1].detach())) + "*Y + " + str(float(parameters["bias"][0].detach())) + " > 0"
        def printNumericalInvariantSmoothed(params):
            parameters = (params["weights"][0].detach()).numpy()
            biggest_weight = abs(np.min(parameters))
            bias = float(params["bias"][0].detach())/biggest_weight
            new_weights = [weight/biggest_weight for weight in parameters]#[float(params["bias"][0].detach())/biggest_weight] + [weight/biggest_weight for weight in parameters]
            approximations = []
            for new_weight in new_weights:
                closest_approx_values = (-100, -100)
                closest_approx = 1000
                for i in range(-5, 5): # K = 5 like in the paper
                    for j in range(1,5): #K = 5
                        if (abs(new_weight - i/j) < closest_approx and np.sign(new_weight) == np.sign(i)):
                            closest_approx = abs(new_weight - i/j)
                            closest_approx_values = (i,j)
                approximations.append(closest_approx_values)
            least_common_multiple = lcm([frac[1] for frac in approximations])
            #print("approximations are " + str(approximations) + " and the lcm is " + str(least_common_multiple))
            return str(least_common_multiple * approximations[0][0]/approximations[0][1]) + "*X + " + str(least_common_multiple * approximations[1][0]/approximations[1][1]) + "*Y + " + str(math.ceil(least_common_multiple * bias)) + " > 0"
        def printProgram(program, smoothed=False):
            if program.name == "affine":
                if smoothed:
                    print("(" + program.name + " " + printNumericalInvariantSmoothed(program.parameters))
                else:
                    print(program.parameters)
                    print("(" + program.name + " " + printNumericalInvariant(program.parameters))
            else:
                print("(" + program.name)
                for submodule, function in program.submodules.items():
                    printProgram(function, smoothed)
                print(" )")
        printProgram(best_program)
        print(" and the smoothed version of the program is ")
        printProgram(best_program, True)
        assert False # ending program because the rest here is not functional - the convert to sygus method does not support affine functions currently
        sygus_program = convert_to_sygus(best_program)

        passed, all = holder.eval_both(sygus_program)
        if passed == all:
            log_and_print("Found a solution: " + sygus_program.to_py())
            break
        else:
            print (f'passes: {passed}, all: {all}')
            holder.weigh_failed_ce(sygus_program, w=1)
            train_loader = IOExampleLoader(train_data, train_labels, batch_size=cmd_args.batch_size, shuffle=False)
            for pair in all_graphs:
                pair[0] = evaluate(algorithm, pair[1], train_loader, train_config, device)
            splited_subgraph = program_graph.partition(cmd_args.top_left, cmd_args.GM)
            partition_num += 1
            if splited_subgraph is not None:
                for subgraph in splited_subgraph:
                    all_graphs.append([evaluate(algorithm, subgraph, train_loader, train_config, device), subgraph])

            heapq.heapify(all_graphs)

        iteri += 1
        # res = None
        # lambda_holder_eval = lambda key: holder.eval(key, sygus_program)
        # #lambda_new_ce = lambda: get_ce( g, generated_tree)
        # lambda_new_ce = lambda: holder.get_failed_ce(sygus_program)
        #
        # if cmd_args.use_interpolation:
        #     res = reward_w_interpolation(g.sample_index, holder, lambda_holder_eval, lambda_new_ce)
        # else:
        #     res = reward_1(g.sample_index, holder, lambda_holder_eval, lambda_new_ce)
        #
        # if res > -0.000001:
        #     log_and_print("Found a solution: " + sygus_program.to_py())
        #     break

    print("number of partitions: ", partition_num)
