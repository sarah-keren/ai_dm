__author__ = 'sarah'

import ai_dm.Search.utils as utils
import ai_dm.Search.defs as defs
import ai_dm.Search.heuristic as heuristic
import ai_dm.Search.constraint as constraint
import logging, time

# TODO: take care of logs
# TODO: take care of transition function and the duplication of env for each node (and for the termination criteria)
def best_first_search (problem, frontier, closed_list = None, termination_criteria = None, evaluation_criteria = None, prune_func = None, constraints = None, log=False, log_file=None, iter_limit = defs.NA, time_limit = defs.NA):


    """Search for the design sequence with the maximal value.
       The search problem must specify the following elements:

       - problem - problem that specifies for each node its value and its successors
         (taking into account available actions and constraints).

       The search related elements that need to be defined are the following:

       - frontier (fringe) - keeps the open nodes and defines the order by which they are examined (e.g., queue) see Search.utils for options
                    if this is a heuristic search, the heuristic is used by the frontier when add is evoked
       - closed_list - maintains the states that have been explored (if set to None, no list is maintained)
       - termination_criteria - a termination condition for which the current best result is returned.
         For example, for a goal directed search the search will stop once the goal condition is satisfied
       - prune_func - given the successors of a node, the pruning function returns only the nodes to be further explored
    """
    if log:
        logging.info('Starting: best_first_search')
        results_log = {}

    start_time = time.time()

    # init the search node
    root_node = utils.Node(problem.get_current_state(), None, None, 0)
    # the frontier sets the order by which nodes are explored (e.g.FIFO, LIFO etc.)
    # we are assuming the root node is valid, i.e., it doesn't violate the constraints
    frontier.add(root_node)

    # keeping the best value found so far
    best_value = None
    # keeping the best solution found so far
    best_node = None
    # counting number of explored nodes
    explored_count = 0

    # a flag used to indicate that the termination criteria has not yet been reached
    continue_search = True

    # continue while there are still nodes to explore and the termination condition has not been met
    ex_terminated = False

    try:

        while not frontier.is_empty() and continue_search:

            # count explored nodes
            explored_count += 1

            # check resource limit has not been reached
            if defs.NA != iter_limit and explored_count > iter_limit:
                ex_terminated = True
                break

            if start_time is not None and defs.NA != time_limit :
                cur_time = time.time()
                time_elapsed = cur_time - start_time
                if time_elapsed > time_limit:
                    ex_terminated = True
                    break

            # get the current node
            cur_node = frontier.extract()
            if log:
                log_string = 'InMethod best_first_design(node): explored_count:%d' % explored_count

                log_string += ' cur_node:%s'%cur_node
                if log_file:
                    logging.info('best_first_design(node) node number %d cur_node:%s' % (explored_count,cur_node))
                else:
                    print('best_first_design(node) node number %d cur_node:%s' % (explored_count,cur_node))

            # add the node to the closed list
            if closed_list:
                closed_list.add(cur_node)

            # verify the model is valid (this is done here to support non-persistent model (i.e. models where non-valid modification sequences can be a prefix of valid ones)
            # and update the best value found so far
            start_time_evaluate = time.time()
            cur_value = problem.evaluate(cur_node)

            if best_node is None or evaluation_criteria.is_better_or_equal(cur_value, cur_node, best_value, best_node, problem):
                best_value = cur_value
                best_node = cur_node

            if log:
                log_string += ', node_eval_time:%.3f' % (time.time() - start_time_evaluate)
                #log_string += ', cur_value:%.3f, best_node:%s, best_value:%.3f'%(cur_value, best_node, best_value)
                if log_file:
                    log_progress(results_log, cur_node, cur_value, best_node, best_value, problem, log_file, start_time, explored_count)
                else:
                    print(log_string)


            # check if termination criteria had been met - and stop the search if it has
            if termination_criteria is not None and termination_criteria.isTerminal(best_node, best_value, problem):
                if log:
                    logging.info(log_string)
                break

            # get the succsessors of the node
            succs = problem.successors(cur_node)
            if constraints:
                succs = utils.apply_constraints(constraints, succs)
            if log:
                log_string += ', pre_prune_succ_count:%d' % (len(succs))

            # if pruning is applied - prune the set of successors
            start_time_prune = time.time()
            if prune_func is not None:
                succs = prune_func(succs, cur_node)
            if log:
                log_string += ', prun_func_time:%.3f' %(time.time() - start_time_prune)

            # sort succesors to make sure goal is reached at the same time for all approaches
            succs = sorted(succs, key=lambda x: x.get_transition_path_string(), reverse=False)

            if log:
                log_string += ', succ_count:%d' % (len(succs))

            # evaluate each child
            if succs is None or len(succs) == 0:
                continue

            # add children to the frontier
            # if the closed list was specified - add it only if it's not already there
            start_time_succ = time.time()
            for child in succs:
                already_in_closed_list = False
                if closed_list is not None:
                    if closed_list.is_in_list(child):
                        already_in_closed_list = True
                if not already_in_closed_list and child not in frontier:
                    frontier.add(child)
                    if closed_list:
                        closed_list.add(child)
            succ_calc_time =  time.time() - start_time_succ

            if log:
                log_string += ', succ_calc_time:%.3f' % (succ_calc_time)
                logging.info(log_string)


        calc_time = time.time() - start_time  # , "seconds"
        #logging.info('Ending: best_first_design: {best_value:%d, best_node:%s, explored_count:%d, ex_terminated:%s, calc_time:%.3f}'%(best_value,best_node, explored_count,ex_terminated, calc_time))

        # return the best solution found
        if log:
            print('solution is: %s'%(best_node.get_transition_path_string()))
        return [best_value, best_node, best_node.get_transition_path_string(), explored_count, ex_terminated]

    except Exception as e:
        if log_file is not None:
            log_file.write('Exception occurred: %s'%str(e))
        raise e

def breadth_first_search(problem, log=False, log_file=None, iter_limit=defs.NA, time_limit=defs.NA):
    return best_first_search(problem,
                             frontier=utils.FIFOQueue(),
                             closed_list=utils.ClosedListOfKeys(),
                             termination_criteria=utils.TerminationCriteriaGoalStateReached(),
                             evaluation_criteria=utils.EvaluationCriteriaGoalCondition(),
                             prune_func=None,
                             constraints = None,
                             log=log,
                             log_file=log_file,
                             iter_limit=iter_limit,
                             time_limit=time_limit)


def depth_first_search(problem, log=False, log_file=None, iter_limit=defs.NA, time_limit=defs.NA):
    return best_first_search(problem,
                             frontier=utils.LIFOQueue(),
                             closed_list=utils.ClosedListOfKeys(),
                             termination_criteria=utils.TerminationCriteriaGoalStateReached(),
                             evaluation_criteria=utils.EvaluationCriteriaGoalCondition(),
                             prune_func=None,
                             constraints=None,
                             log=log,
                             log_file=log_file,
                             iter_limit=iter_limit,
                             time_limit=time_limit)


def depth_first_search_l(problem, max_depth, log=False, log_file=None, iter_limit=defs.NA, time_limit=defs.NA):
    return best_first_search(problem,
                             frontier=utils.LIFOQueue(),
                             closed_list=utils.ClosedListOfKeys(),
                             termination_criteria=utils.TerminationCriteriaGoalStateReached(),
                             evaluation_criteria=utils.EvaluationCriteriaGoalCondition(),
                             prune_func=None,
                             constraints=[constraint.DepthConstraint(max_depth)],
                             log=log,
                             log_file=log_file,
                             iter_limit=iter_limit,
                             time_limit=time_limit)

def a_star(problem, heuristic_func=heuristic.zero_heuristic , log=False, log_file=None, iter_limit=defs.NA, time_limit=defs.NA):
    f_func = lambda x: x.get_path_cost(problem)[0]+heuristic_func(x)
    return best_first_search(problem=problem,
                             frontier=utils.PriorityQueue(f_func),
                             closed_list=utils.ClosedListOfKeys(),
                             termination_criteria=utils.TerminationCriteriaGoalStateReached(),
                             evaluation_criteria=utils.EvaluationCriteriaGoalCondition(),
                             prune_func=None,
                             constraints=None,
                             log=log,
                             log_file=log_file,
                             iter_limit=iter_limit,
                             time_limit=time_limit)


def greedy_best_first_search(problem, heuristic_func=heuristic.zero_heuristic , log=False, log_file=None, iter_limit=defs.NA, time_limit=defs.NA):
    return best_first_search(problem=problem,
                             frontier=utils.PriorityQueue(heuristic_func),
                             closed_list=utils.ClosedListOfKeys(),
                             termination_criteria=utils.TerminationCriteriaGoalStateReached(),
                             evaluation_criteria=utils.EvaluationCriteriaGoalCondition(),
                             prune_func=None,
                             constraints=None,
                             log=log,
                             log_file=log_file,
                             iter_limit=iter_limit,
                             time_limit=time_limit)

def log_progress(results_log, cur_node, cur_value, best_node, best_value, umd_problem, log_file, start_time, explored_count):

    # get current time
    cur_time = time.time() - start_time


    # current resource (cost)
    cur_modification_cost = cur_node.cost()[0]

    # keep track of the best value for each resource allocation and the number of explored nodes
    if cur_modification_cost in results_log:
        [max_val_for_cost,max_node_for_cost, latest_tick_per_cost, explored_count_per_cost] = results_log[cur_modification_cost]
        if umd_problem.is_better(cur_value, max_val_for_cost):
            results_log[cur_modification_cost] = [cur_value, cur_node, cur_time, explored_count_per_cost+1]
        else: # update node count and time
            results_log[cur_modification_cost] = [max_val_for_cost, max_node_for_cost, cur_time, explored_count_per_cost+1]

    else:
        results_log[cur_modification_cost] = [cur_value, cur_node, cur_time, 1]

    log_message = 'cur_value::' + str(cur_value) + defs.SEPARATOR + 'cur_node::' + cur_node.__repr__() + defs.SEPARATOR + 'cur_time::' + '%.5f' % cur_time + defs.SEPARATOR + 'cur_cost::' + '%.3f' % cur_modification_cost + defs.SEPARATOR + 'best_node::' + best_node.__repr__() + defs.SEPARATOR + 'best_value::' + str(best_value) + '\n'
    print(log_message)
    log_file.write(log_message)
    log_file.flush()












