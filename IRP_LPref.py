import gurobipy as gp
import pandas as pd
from gurobipy import GRB

def MasterProblem(MultiTours, ncustomers, nvehicles,C, x, LPrelaxation=False):
    #model implementation
    model = gp.Model('LPref model')

    # param to not show the output log
    model.setParam('OutputFlag', 0)

    # Indeces
    Customers_indeces = [i for i in range(1, ncustomers+1)]
    Rplus_indices = [i for i in range(ncustomers+1)]
    MultiTours_indeces = [i for i in range(len(MultiTours))]
    w_indices = [theta for theta in MultiTours_indeces]

    #Variables
    if LPrelaxation:
        w = model.addVars(w_indices, name = 'w', vtype = GRB.CONTINUOUS, lb = 0, ub = 1)
    else:
        w = model.addVars(w_indices, name = 'w', vtype = GRB.BINARY)
    
    #Objective Function
    model.setObjective(gp.quicksum(C[theta]*w[theta] for theta in MultiTours_indeces), GRB.MINIMIZE)

    # Constraint 1
    for j in Customers_indeces:
        model.addConstr(gp.quicksum(gp.quicksum(x[theta,i,j] for i in Rplus_indices) * w[theta] for theta in MultiTours_indeces) == 1, name = 'CustomerVisits')
    
    # Constraint 2
    model.addConstr(gp.quicksum(w[theta] for theta in MultiTours_indeces) <= nvehicles, name = 'NumberOfVehicles')
    
    #Solve
    # model.optimize()

    return model, w

def MinTC_LPref(model: gp.Model):
    model.update()
    model.reset()
    model.optimize()
    try:
        return model.objVal
    except:
        # print('Model is infeasible')
        return None

def SolutionMP(model: gp.Model, w: gp.Var):
    solution = []

    for index, value in w.items():
        if value.x > 1e-6:
            solution.append((index, value.x))

    return solution

def DualPricesMP(model: gp.Model, ncustomers):
    '''Dualzero is going to represent all the additional dual multipliers: Pi_0, sigma, Pi_1'''
    duals = []
    for c in model.getConstrs():
        duals.append(c.pi)
    # dual_zero = duals.pop(-1)
    dual_zero = duals[ncustomers:]
    duals = duals[:ncustomers]
    duals = pd.Series(duals, index=range(1, ncustomers+1))
    return duals, dual_zero
