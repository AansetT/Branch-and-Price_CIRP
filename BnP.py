from copy import deepcopy
import pandas as pd
from math import floor, ceil
import queue
import CG
from Multitours import Visited_Customers, Penalize_MultiTours, Select_Trip
import time

class MaxPriorityQueue(queue.PriorityQueue):
    def __init__(self):
        super().__init__()

    def put(self, item, priority):
        super().put((-priority, item))

    def get(self):
        _, item = super().get()
        return item

    def empty(self):
        return super().empty()

class Node:
    instance_parameters: dict = None
    upper_bound = None
    def __init__(self, parent = None, depth = None, TC = None, solution = None, Node_Data : dict = None):
        self.parent = parent
        self.depth = depth
        self.TC = TC
        self.solution = solution
        self.children = []
        self.is_leaf = False
        self.is_feasible = self._is_feasible()
        self.is_IPfeasible = self._is_IPfeasible()
        self.state = {"leaf": self.is_leaf, "feasible": self.is_feasible, "IPfeasible": self.is_IPfeasible}

        # Data from the tree. MultiTours, Cost, T, X, and constraints that come from the parent node
        self.Node_Data = Node_Data
    
    def solve(self):
        constraints = dict(list(self.Node_Data.items())[-2:])
        self.Node_Data['X'], self.TC, self.solution = CG.ColumnGeneration(self.Node_Data['MultiTours'], self.Node_Data['Cost'], 
                                                                          self.Node_Data['T'], self.Node_Data['X'], 
                                                                          Node.instance_parameters, constraints=constraints, verbose=False)
        self.update_state()
        # CG.print_solution_RMP(self.TC, self.solution)
   
    def Number_Vehicles_Used(self):
        l = sum(x[1] for x in self.solution)
        round_l = round(l)
        if l.is_integer():
            return None
        elif abs(l - round_l) < 1e-5:
            return None
        else:
            return l    
    
    def _is_feasible(self):
        if self.TC is not None:
            return True
        else:
            return False

    def _is_IPfeasible(self):
        if self.is_feasible:
            return all(x[1].is_integer() for x in self.solution)
        else:
            return False
    
    def _is_leaf(self):
        if not self.is_feasible:
            return True
        elif self.is_feasible and not self.is_IPfeasible:
            if self.TC >= Node.upper_bound - 1e-5:
                return True
            else :
                return False
        elif self.is_IPfeasible:
            return True
 
    def update_state(self):
        self.is_feasible = self._is_feasible()
        self.is_IPfeasible = self._is_IPfeasible()
        self.is_leaf = self._is_leaf()
        self.state = {"leaf": self.is_leaf, "feasible": self.is_feasible, "IPfeasible": self.is_IPfeasible}

    def __repr__(self):
        return '<Node state="{}">'.format(self.state)

class Tree:
    def __init__(self, root_node: Node, UB = None, LB = None, UB_solution = None, instance_parameters: dict = None):
        self.root = root_node
        self.nodes = [self.root]
        self.UB = UB
        self.LB = LB
        self.UB_solution = UB_solution
        self.UB_node = self.root
        self.candidates = MaxPriorityQueue()

        # Update the instance parameters and UB for every node
        Node.instance_parameters = instance_parameters
        Node.upper_bound = self.UB
    
    def update_UB(self, node: Node):
        if node.is_IPfeasible and node.TC < self.UB:
            self.UB = node.TC
            self.UB_solution = node.solution.copy()
            self.UB_node = deepcopy(node)
            Node.upper_bound = self.UB

    def update_LB(self, node: Node):
        if node.is_feasible and not node.is_leaf and node.TC > self.LB and node.TC < self.UB - 1e-5:
            self.LB = node.TC
    
    def priority(self, parent_node: Node):
        return parent_node.depth + 1
    
    def max_H(self, parent_node: Node):
        solution = [sol for sol in parent_node.solution if 0 < sol[1] < 1]
        C = parent_node.Node_Data['Cost']

        max_sol = max(solution, key=lambda sol: C[sol[0]] * min(sol[1], 1 - sol[1]))
        return max_sol[0]

    def AddNode(self, parent_node: Node, parent_node_idx: int, data_child: dict, Priority):
        child = Node(parent = parent_node_idx, depth=Priority, Node_Data = data_child)
        self.nodes.append(child)
        self.candidates.put(len(self.nodes)-1, Priority)
        parent_node.children.append(len(self.nodes)-1)

    def branch(self, node: Node, node_idx: int, Strategy: str):
        # Do not branch if the node is a leaf or if it has not been solved
        if node.is_leaf:
            return None
        
        # Level 1: Branch on the number of vehicles
        l = node.Number_Vehicles_Used()
        if l is not None:
            self.Branch_Number_Vehicles(node, node_idx,l)
        else:
            # Level 2: Branch on the MultiTours or on the trips
            if Strategy == "MultiTours":
                self.Branch_MultiTours(node, node_idx)
            elif Strategy == "Trips":
                self.Branch_Trips(node, node_idx)

    def Branch_Number_Vehicles(self, node: Node, node_idx: int,l):
        Priority = self.priority(node)

        # Branching information
        dataL = deepcopy(node.Node_Data)
        dataL['level1'].append(("less", floor(l)))
        dataR = deepcopy(node.Node_Data)
        dataR['level1'].append(("greater", ceil(l)))

        # Create the left and right children
        self.AddNode(parent_node=node, parent_node_idx=node_idx, data_child=dataL, Priority=Priority)
        self.AddNode(parent_node=node, parent_node_idx=node_idx, data_child=dataR, Priority=Priority)
    
    def Branch_MultiTours(self, node: Node, node_idx: int):
        Priority = self.priority(node)
        # Select the MultiTour with the highest H in the solution
        multi_tour_idx = self.max_H(node)
        m_customers = Visited_Customers(node.Node_Data['MultiTours'][multi_tour_idx])
        n_trips = len(node.Node_Data['MultiTours'][multi_tour_idx])-1

        # Branching information
        dataL = deepcopy(node.Node_Data)
        dataL['level2'].append(("Multitours", multi_tour_idx, 1, m_customers))
        Penalize_MultiTours(dataL['MultiTours'],dataL['Cost'],penalty=self.UB,customers=m_customers, tour_idx=multi_tour_idx)
        dataR = deepcopy(node.Node_Data)
        dataR['level2'].append(("Multitours", multi_tour_idx, 0, n_trips))
        Penalize_MultiTours(dataR['MultiTours'],dataR['Cost'],penalty=self.UB,tour_idx=multi_tour_idx)

        # Create the left and right children
        self.AddNode(parent_node=node, parent_node_idx=node_idx, data_child=dataL, Priority=Priority)
        self.AddNode(parent_node=node, parent_node_idx=node_idx, data_child=dataR, Priority=Priority)
    
    def Branch_Trips(self, node: Node, node_idx: int):
        Priority = self.priority(node)
        # Select the MultiTour with the highest H in the solution
        multi_tour_idx = self.max_H(node)
        multi_tour = node.Node_Data['MultiTours'][multi_tour_idx]

        # Trip to branch on
        trip = Select_Trip(node.Node_Data['MultiTours'], node.Node_Data['level2'], node.solution, multi_tour)

        #Branching information
        dataL = deepcopy(node.Node_Data)
        dataL['level2'].append(("Trips", trip, 0))
        Penalize_MultiTours(dataL['MultiTours'],dataL['Cost'],penalty=self.UB,trip = trip)
        dataR = deepcopy(node.Node_Data)
        dataR['level2'].append(("Trips", trip, 1))

        for j in range(Node.instance_parameters['ncustomers']+1):
            if j != trip[1]:
                penalized_trip = [trip[0],j]
                Penalize_MultiTours(dataR['MultiTours'],dataR['Cost'],penalty=self.UB,trip = penalized_trip)
            if j != trip[0]:
                penalized_trip = [j,trip[1]]
                Penalize_MultiTours(dataR['MultiTours'],dataR['Cost'],penalty=self.UB,trip = penalized_trip)

        # Create the left and right children
        self.AddNode(parent_node=node, parent_node_idx=node_idx, data_child=dataL, Priority=Priority)
        self.AddNode(parent_node=node, parent_node_idx=node_idx, data_child=dataR, Priority=Priority)
  
def BnP(MultiTours,Cost,T,X, instance_parameters: dict, strategy, initial_exec_time = 0):
    initime = time.time()
    time_limit = 180
    results = []
    # Run the Column Generation Heuristic
    Min_TC, solutionMP, Min_TC_RMP, solutionRMP, X = CG.ColumnGeneration_heuristic(MultiTours,Cost,T,X, instance_parameters)
    if Min_TC == Min_TC_RMP:
        CG.print_solution_heuristic(MultiTours,T,Cost,Min_TC,solutionMP)
        return Min_TC, solutionMP, None
    
    # Step 1: Create the root node
    node_data = {"MultiTours": MultiTours, "Cost": Cost, "T": T, "X": X, "level1": [], "level2": []}
    root_node = Node(depth=0, TC = Min_TC_RMP, solution = solutionRMP, Node_Data = node_data)

    # Create the tree
    tree = Tree(root_node, UB = Min_TC, LB = Min_TC_RMP, UB_solution = solutionMP, instance_parameters = instance_parameters)
    print_results(results, tree.root.depth, tree.UB, tree.LB, tree.root.TC, tree.root.is_leaf)

    # Step 2: Initialize the candidates queue
    tree.branch(node=tree.root, node_idx=0, Strategy = strategy)

    exec_time = time.time() - initime + initial_exec_time

    # Step 3 and 4: Main steps of the B&P algorithm
    max_nodes = 0
    while not tree.candidates.empty() and max_nodes < 100 and exec_time/60 <= time_limit:
        # Select the node with the highest priority
        node_index = tree.candidates.get()
        node = tree.nodes[node_index]

        # Solve the node (prune if necessary)
        node.solve()

        # Update bounds
        tree.update_UB(node)
        tree.update_LB(node)

        # Branch
        tree.branch(node=node, node_idx=node_index, Strategy = strategy)

        max_nodes += 1
        exec_time = time.time() - initime + initial_exec_time
        print_results(results, node.depth, tree.UB, tree.LB, node.TC, node.is_leaf)
    
    if exec_time/60 > time_limit:
        print("Time limit reached: ", exec_time/60, "minutes")
    
    print_solution_tree(tree)
    
    return tree.UB, tree.UB_solution, tree

def print_solution_tree(tree: Tree):
    print()
    print("Optimal solution found")
    print("Min total cost:", tree.UB)
    print("Solution:", tree.UB_solution)

    for sol in tree.UB_solution:
        print(tree.UB_node.Node_Data['MultiTours'][sol[0]], tree.UB_node.Node_Data['T'][sol[0]], tree.UB_node.Node_Data['Cost'][sol[0]])

def print_results(results, depth, upperbound, lowerbound,TC, leaf):
    cost = round(TC,3) if TC is not None else "Infeasible"
    values = [len(results),depth, cost, str(leaf), round(upperbound,3), round(lowerbound,3), round((upperbound-lowerbound)/lowerbound*100,3)]
    results.append(values)

    if len(results) == 1:  # If it's the first iteration, print the headers
        print()
        print("Branch and Price Results")
        print('{:<4} {:<5} {:<12} {:<5} {:<12} {:<12} {:<10}'.format('node','Depth', 'TC', 'Leaf', 'UB', 'LB', 'Gap'))
    # Print the values
    print('{:<4} {:<5} {:<12} {:<5} {:<12} {:<12} {:<10}'.format(*values))
    



