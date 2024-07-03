import gurobipy as gp
import numpy as np
# Imports of Master problem and subproblem
import IRP_LPref
import IRP_MGSP
import Multitours as MT

def ColumnGeneration(MultiTours,Cost,T,X, instance_parameters: dict, constraints: dict = None, verbose = True):
    ncustomers, nvehicles, psi, delta, phi, h, d, k, t, vehicleCapacity = get_instance_parameters(instance_parameters)
    results = []
    RC_star = -1
    iteration = 0
    count = 0
    while RC_star < -1e-10 and iteration < 50:
        # Restrincted Master Problem
        RMP, w_var = IRP_LPref.MasterProblem(MultiTours, ncustomers, nvehicles, Cost, X, LPrelaxation=True)
        RMP = AddConstraints(MultiTours=MultiTours, constraints=constraints, MPmodel=RMP, w=w_var)
        Min_TC = IRP_LPref.MinTC_LPref(RMP)
        if Min_TC is None:
            solutionRMP = None
            return X, Min_TC, solutionRMP
        solutionRMP = IRP_LPref.SolutionMP(RMP, w_var)
        DualPrices, DualZero = IRP_LPref.DualPricesMP(RMP, ncustomers)

        # Multi-tour Generation Subproblem
        SP, x_var, T_var = IRP_MGSP.SubProblem(ncustomers, psi[0], delta, phi, h, d, k, t, vehicleCapacity[0], DualPrices, DualZero)
        SP = AddConstraints(MultiTours=MultiTours, ncustomers=ncustomers, constraints=constraints, SPmodel=SP, x=x_var)
        RC_star = IRP_MGSP.MinRC(SP)
        if RC_star > 0:
            return X, Min_TC, solutionRMP
        new_MultiTour, T_new = IRP_MGSP.SolutionSP(SP, x_var, T_var)

        # Binary representation of the new MultiTour
        new_x = MT.Multitours_Binary([new_MultiTour], ncustomers)

        # Calculate cost of the new MultiTour
        new_cost = MT.Multitour_Cost(1, ncustomers, psi, [T_new], delta, t, new_x, phi, h, d)

        # Check if the new MultiTour is already in the MultiTours list
        idx = MT.MultiTour_index(MultiTours, new_MultiTour)

        # Add the new MultiTour
        X, count = Add_Column(MultiTours, Cost, T, X, new_MultiTour, new_cost, T_new, new_x, idx, count, RC_star)
        if count > 1:
            if verbose:
                print("No improvement in the last 1 iterations")
            break

        if verbose:
            print_iteration(iteration, RC_star, new_MultiTour, Min_TC)
        # results.append([iteration, RC_star, new_MultiTour, T_new, Min_TC])
        iteration += 1
    return X, Min_TC, solutionRMP

def ColumnGeneration_heuristic(MultiTours,Cost,T,X, instance_parameters: dict):
    # Steps 1 to 4 of the Column Generation Heuristic
    X, Min_TC_RMP, solutionRMP = ColumnGeneration(MultiTours,Cost,T,X, instance_parameters)
    print_solution_RMP(Min_TC_RMP,solutionRMP)

    # Step 5: Solve the Master Problem as a MILP
    MP, w_var_notused = IRP_LPref.MasterProblem(MultiTours, instance_parameters['ncustomers'], instance_parameters['nvehicles'], Cost, X, LPrelaxation=False)
    Min_TC = IRP_LPref.MinTC_LPref(MP)
    solutionMP = IRP_LPref.SolutionMP(MP, w_var_notused)
    print_solution_heuristic(MultiTours,T,Cost,Min_TC,solutionMP)
    
    return Min_TC, solutionMP, Min_TC_RMP, solutionRMP, X

def Add_Column(MultiTours, Cost, T, X, new_MultiTour, new_cost, T_new, new_x, idx, count, RC):
    # Only add the new MultiTour if it is not already in the MultiTours list or if it has a lower cost
    if idx == "" or (new_cost[0] < Cost[idx] and T[idx]-T_new > 1e-10):
        # Add new MultiTour to the MultiTours list
        MultiTours.append(new_MultiTour)

        # Add bianry representation of the new MultiTour to X
        X = np.append(X, new_x, axis=0)

        # Add new cost and cycle time to the Cost and T arrays
        Cost.append(new_cost[0])
        # T = np.append(T, T_new)
        T.append(T_new)

        # reset count of iterations without improvement
        count = 0
    else:
        count +=1
    return X, count # We return these because they are immutable in python so they need to be reassigned

def AddConstraints(MultiTours: list = None, ncustomers: int = None, constraints: dict = None, MPmodel: gp.Model = None, w = None, SPmodel: gp.Model = None, x = None):
    '''Additional constraints that come from branch and price. Added to the Master Problem and Subproblem'''
    if constraints:
        level1_constraints = constraints.get('level1')
        if level1_constraints and MPmodel:
            MultiTours_indeces = [i for i in range(len(MultiTours))]
            MPmodel = Level1_constraints(MultiTours_indeces, level1_constraints, MPmodel, w)
        level2_constraints = constraints.get('level2')
        if level2_constraints and MPmodel:
            Level2_constraintsMP(level2_constraints, MPmodel, w)
        if level2_constraints and SPmodel:
            Level2_constraintsSP(MultiTours, ncustomers, level2_constraints, SPmodel, x)

    if MPmodel:
        return MPmodel
    elif SPmodel:
        return SPmodel

def Level1_constraints(multitours_indeces, constraints: list, MPmodel: gp.Model, w):
    for i, c in enumerate(constraints):
        if c[0] == 'less':
            MPmodel.addConstr(gp.quicksum(w[theta] for theta in multitours_indeces) <= c[1], name = "Branch_nvehicles"+str(i))
        elif c[0] == 'greater':
            MPmodel.addConstr(gp.quicksum(w[theta] for theta in multitours_indeces) >= c[1], name = "Branch_nvehicles"+str(i))
    return MPmodel

def Level2_constraintsMP(constraints: list, MPmodel: gp.Model, w):
    for i, c in enumerate(constraints):
        typeConstr, multi_tour_t, value  = c[0], c[1], c[2]
        if typeConstr == 'Multitours':
            if value == 1:
                MPmodel.addConstr( w[multi_tour_t] == 1)
        elif typeConstr == 'Trips':
            pass
    return MPmodel

def Level2_constraintsSP(MultiTours, ncustomers, constraints: list, SPmodel: gp.Model, x):
    Rplus_indices = [i for i in range(ncustomers+1)]
    for it, c in enumerate(constraints):
        typeConstr = c[0]
        if typeConstr == 'Multitours':
            multi_tour_t, value  = c[1], c[2]
            if value == 1:
                m_customers = c[3]
                for j in Rplus_indices:
                    for i in m_customers:
                        SPmodel.addConstr(x[i,j] == 0, name = "Branch_Multitours"+str(it))
                        SPmodel.addConstr(x[j,i] == 0, name = "Branch_Multitours"+str(it))
            if value == 0: # TODO consider all the variations of the multi-tour
                n_trips = c[3]
                m_tour = MultiTours[multi_tour_t]
                SPmodel.addConstr(gp.quicksum(x[m_tour[q],m_tour[q+1]] for q in range(n_trips)) <= n_trips-1, name = "Branch_Multitours"+str(it))
                SPmodel.addConstr(gp.quicksum(x[m_tour[q+1],m_tour[q]] for q in range(n_trips)) <= n_trips-1, name = "Branch_Multitours"+str(it))
        elif typeConstr == 'Trips':
            trip, value  = c[1], c[2]
            if value == 0:
                SPmodel.addConstr(x[trip[0],trip[1]] == 0, name = "Branch_Trips"+str(it))
                SPmodel.addConstr(x[trip[1],trip[0]] == 0, name = "Branch_Trips"+str(it))
            if value == 1:
                for j in Rplus_indices:
                    if j != trip[1]:
                        SPmodel.addConstr(x[trip[0],j] == 0, name = "Branch_Trips"+str(it))
                        SPmodel.addConstr(x[j,trip[0]] == 0, name = "Branch_Trips"+str(it))
                    if j != trip[0]:
                        SPmodel.addConstr(x[j,trip[1]] == 0, name = "Branch_Trips"+str(it))
                        SPmodel.addConstr(x[trip[1],j] == 0, name = "Branch_Trips"+str(it))        
    return SPmodel

def get_instance_parameters(parameters: dict):
    ncustomers = parameters['ncustomers']
    nvehicles = parameters['nvehicles']
    psi = parameters['psi']
    delta = parameters['delta']
    phi = parameters['phi']
    h = parameters['h']
    d = parameters['d']
    k = parameters['k']
    t = parameters['t']
    vehicleCapacity = parameters['vehicleCapacity']
    return ncustomers, nvehicles, psi, delta, phi, h, d, k, t, vehicleCapacity

def print_iteration(iteration, RC_star, new_MultiTour, Min_TC):
    print("Iteration", iteration, ":")
    print("Min reduced cost",RC_star)
    print("Generated MultiTour", new_MultiTour)
    print("Min total cost RMP",Min_TC)

def print_solution_heuristic(MultiTours,T,Cost,Min_TC,solutionMP, BnP = False):
    if BnP:
        print()
        print("Optimal solution found")
    else:
        print()
        print("Integer Master Problem:")
    print("Solution of the Master Problem", solutionMP)
    print("Min total cost MP",Min_TC)

    for sol in solutionMP:
        print(MultiTours[sol[0]], T[sol[0]], Cost[sol[0]])

def print_solution_RMP(Min_TC_RMP,solutionRMP):
    print()
    print("Relaxed Master Problem:")
    print("Solution of the Relaxed Master Problem", solutionRMP)
    print("Min total cost RMP",Min_TC_RMP)