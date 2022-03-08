__author__ = 'sarah'

import Search.search_utils
import Search.defs
import logging, time

def best_first_search(problem, frontier,  closed_list = [], termination_criteria = None, prune_func = None, log_file=None, iter_limit = defs.NA, time_limit = defs.NA, use_search_node_for_evaluation = False):


    """Search for the design sequence with the maximal value.
       The UMD problem must specify the following elements:

       - umd_problem - utility maximizing design problem that specifies for each node (modification sequence) its value and its successors
         (taking into account available modifications and constraints).

       The search related elements that need to be defined are the following:

       - frontier - keeps the open nodes and defines the order by which they are examined (e.g., queue) see search for options
                    if this is a heuristic search, the heuristic is used by the frontier when add is evoked
       - closed_list - maintains the states that have been explored (if set to None, no list is maintained) 
       - termination_criteria - a termination condition for which the current best result is returned.
         For example, for GRD the search will stop once wcd =0
       - prune_func - given the successors of a node, the pruning function returns only the nodes to be further explored
    """

    logging.info('Starting: best_first_search')

    # init the search node
    root_node = search_utils.Node(problem.initial_state, None, None,0, problem)
    # the frontier sets the order by which nodes are explored (e.g.FIFO, LIFO etc.)
    # we are assuming the root node is valid, i.e., it doesn't violate the design constraints 
    frontier.add(root_node)
    # check for node validity 
        
    # keeping the best value found so far
    best_value = None
    # keeping the best solution found so far
    best_node  = None
    # counting number of explored nodes
    explored_count = 0

    
    # a flag used to indicate that the termination criteria has not yet been reached
    continue_search = True

    results_log = {}
    start_time = None
    start_time = time.time()


    # continue while there are still nodes to explore and the termination condition has not been met
    ex_terminated = False

    try:

        while not frontier.isEmpty() and continue_search:
            explored_count += 1

            #LOG
            log_string = 'InMethod best_first_design(node): explored_count:%d'%explored_count

            if defs.NA != iter_limit and explored_count > iter_limit:
                continue_search = False
                ex_terminated = True
                break

            if start_time is not None and defs.NA != time_limit :
                cur_time = time.time()
                time_elapsed = cur_time - start_time
                if time_elapsed > time_limit:
                    continue_search = False
                    ex_terminated = True
                    break

            # get the current node
            cur_node = frontier.extract()
            log_string += ' cur_node:%s'%cur_node
            logging.info('best_first_design(node) node number %d cur_node:%s' % (explored_count,cur_node))
            # add the node to the closed list
            if closed_list is not None:
                closed_list.add(cur_node)

            # verify the model is valid (this is done here to support non-persistent model (i.e. models where non-valid modification sequences can be a prefix of valid ones)
            # and update the best value found so far

            start_time_evaluate = time.time()
            if cur_node.value is None:
                if use_search_node_for_evaluation:
                    cur_value = umd_problem.evaluate(cur_node)
                else:
                    cur_value = umd_problem.evaluate(cur_node.state)
                cur_node.value = cur_value
              
            else:
                cur_value = cur_node.value
            log_string += ', node_eval_time:%.3f' % (time.time() - start_time_evaluate)

            if best_node is None or umd_problem.is_better(cur_value, best_value):
                best_value = cur_value
                best_node = cur_node

            if log_file is not None:
                log_progress(results_log, cur_node, cur_value, best_node, best_value, umd_problem, log_file, start_time, explored_count)
            log_string += ', cur_value:%.3f, best_node:%s, best_value:%.3f'%(cur_value, best_node, best_value)

            # check if termination criteria had been met - and stop the search if it has
            if termination_criteria is not None and termination_criteria.isTerminal(best_node, best_value):
                logging.info(log_string)
                break

            # get the succsessors of the node
            succs = umd_problem.successors(cur_node)
            log_string += ', pre_prune_succ_count:%d' % (len(succs))

            # if pruning is applied - prune the set of successors
            start_time_prune = time.time()
            if prune_func is not None:
                succs = prune_func(succs,cur_node)
            log_string += ', prun_func_time:%.3f' %(time.time() - start_time_prune)

            # sort succesors to make sure wcd = 0 is reached at the same time for all approaches
            succs = sorted(succs, key=lambda x: x.str_modification_seq(), reverse=False)

            log_string += ', succ_count:%d' % (len(succs))
            # evaluate each child
            if succs is None:
                continue

            # add childs to the frontier
            # if the closed list was specified - add it only if it's not already there
            start_time_succ = time.time()
            for child in succs:
                already_in_closed_list = False
                if closed_list is not None:
                    if closed_list.isInList(child):
                        already_in_closed_list = True
                if not already_in_closed_list and child not in frontier:
                    frontier.add(child)
                    closed_list.add(child)
            succ_calc_time =  time.time() - start_time_succ
            log_string += ', succ_calc_time:%.3f' % (succ_calc_time)
                # TODO SARAH: Decide how to support modification sets
                #elif child in frontier:
                #    incumbent = frontier[child]
                #    if f(child) < f(incumbent):
                #        del frontier[incumbent]
                #        frontier.append(child)

            logging.info(log_string)


        calc_time = time.time() - start_time  # , "seconds"
        logging.info('Ending: best_first_design: {best_value:%d, best_node:%s, explored_count:%d, ex_terminated:%s, calc_time:%.3f}'%(best_value,best_node, explored_count,ex_terminated, calc_time))

        # return the best solution found
        return [best_value,best_node, explored_count,ex_terminated, results_log]

    except Exception as e:
        ex_terminated = True
        if log_file is not None:
            log_file.write('Exception occurred: %s'%str(e))
        raise e
        #return [best_value, best_node, explored_count, ex_terminated, results_log]


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

    log_message = 'cur_value::'+str(cur_value) +defs.SEPARATOR+ 'cur_node::'+cur_node.__repr__() +defs.SEPARATOR+ 'cur_time::' + '%.5f'%cur_time +defs.SEPARATOR+ 'cur_cost::'+'%.3f'%cur_modification_cost+defs.SEPARATOR+ 'best_node::'+best_node.__repr__() +defs.SEPARATOR+ 'best_value::'+str(best_value)+'\n'
    print(log_message)
    log_file.write(log_message)
    log_file.flush()

    

class TerminationCriteria:
    
    def isTerminal(self, node, value):
        raise NotImplementedError    
    def __str__(self):
        raise NotImplementedError
    

class TerminationCriteriaOptimalValue(TerminationCriteria):
    
    def __init__(self, optimal_value, orSmaller = True):
        self.optimal_value = optimal_value
        self.orSmaller = orSmaller

    def isTerminal(self, node, node_value):
        
        if self.orSmaller: 
            if node_value <= self.optimal_value :
                return True
            else:
                return False 
            
        else: #or bigger
            if node_value >= self.optimal_value :
                return True
            else:
                return False
            
    def __str__(self):
        raise NotImplementedError













