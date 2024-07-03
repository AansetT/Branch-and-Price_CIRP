import gurobipy as gp
from gurobipy import GRB

def SubProblem(ncustomers, psi, delta, phi, h, d, k, t, vehicleCapacity, DualPrices, DualZero):
    #model implementation
    model = gp.Model('MGSP model')

    # param to not show the output log
    model.setParam('OutputFlag', 0)

    model.setParam(GRB.Param.PoolSolutions, 5)
    model.setParam(GRB.Param.PoolSearchMode, 2)

    # Runtime limit of 30 seconds !!!!
    # model.setParam('TimeLimit', 60*3)
    # model.setParam('MIPGap', 0.2)

    # Indeces
    Customers_indeces = [i for i in range(1, ncustomers+1)]
    Rplus_indices = [i for i in range(ncustomers+1)]
    x_indeces = [(i, j) for i in Rplus_indices for j in Rplus_indices]
    z_indeces = [(i, j) for i in Rplus_indices for j in Rplus_indices]

    #Variables
    x = model.addVars(x_indeces, name = 'x', vtype = GRB.BINARY)
    z = model.addVars(z_indeces, name = 'z', vtype = GRB.CONTINUOUS, lb=0)
    T = model.addVar(name = 'T', vtype = GRB.CONTINUOUS, lb=0, ub=1000)
    T_reciprocal = model.addVar(name = 'T_reciprocal', vtype = GRB.CONTINUOUS, lb=1/1000)

    #Params Gurobi
    model.setParam("NonConvex", 2)

    #Objective Function
    model.setObjective(psi + gp.quicksum(T_reciprocal * delta * t[i, j] * x[i, j] for j in Rplus_indices for i in Rplus_indices) + 
                       gp.quicksum((T_reciprocal * phi[i] + 0.5 * h[i] * d[i] * T) * x[i, j] for j in Rplus_indices for i in Customers_indeces) - 
                       gp.quicksum(DualPrices[j] * x[i, j] for j in Customers_indeces for i in Rplus_indices) - sum(DualZero), GRB.MINIMIZE)
    
    # Constraint 0
    model.addConstr(T * T_reciprocal == 1, name = 'Reciprocal')

    # Constraint 1
    for j in Customers_indeces:
        model.addConstr(gp.quicksum(x[i,j] for i in Rplus_indices) <= 1, name = '15')
    
    # Constraint 2
    for j in Rplus_indices:
        model.addConstr(gp.quicksum(x[i,j] for i in Rplus_indices if i != j) - gp.quicksum(x[j,k] for k in Rplus_indices if j!=k) == 0, name = '16')

    # Constraint 3
    model.addConstr(gp.quicksum(x[i,j] * t[i,j] for j in Rplus_indices for i in Rplus_indices if i!=j) - T <= 0, name = '17')

    # Constraint 4
    for j in Customers_indeces:
        model.addConstr(gp.quicksum(z[i,j] for i in Rplus_indices if i!=j) - gp.quicksum(z[j,k] for k in Rplus_indices if j!=k) == gp.quicksum(d[j] * x[i, j] for i in Rplus_indices), name = '18')

    # Constraint 5
    for j in Customers_indeces:
        model.addConstr(T * z[0,j] <= vehicleCapacity, name = '19')

    # Constraint 6
    for j in Customers_indeces:
        model.addConstr(T * d[j] * gp.quicksum( x[i,j] for i in Customers_indeces) <= k[j], name = '20')

    # Constraint 7
    for i in Rplus_indices:
        for j in Rplus_indices:
            if i!=j:
                model.addConstr(z[i,j] <= x[i,j]*gp.quicksum(d[k] for k in Customers_indeces), name = '21')

    return model, x, T

def MinRC(model: gp.Model):
    model.update()
    model.reset()
    model.optimize()

    try:
        return model.objVal
    except:
        return None

def SolutionSP( model: gp.Model, x: gp.Var, T: gp.Var):
    solution = []
    
    for index_tuple, value in x.items():
        if value.x != 0:
            solution.append(index_tuple)
    
    cycle_time = T.x

    try:
        multitour = edge_to_multitour(solution)
    except:
        print("problematic solution", solution)
        print(model.ObjVal)
    return multitour, cycle_time

def edge_to_multitour(edges):
    # Create a dictionary to store the adjacency list
    adjacency_list = {}

    # Populate the adjacency list
    for edge in edges:
        if edge[0] not in adjacency_list:
            adjacency_list[edge[0]] = []
        adjacency_list[edge[0]].append(edge[1])

    # Start with the first node and construct the sequence
    start_node = edges[0][0]
    sequence = [start_node]

    while len(sequence) < len(edges) + 1:
        next_node = adjacency_list[sequence[-1]].pop(0)
        sequence.append(next_node)
    
    return sequence
