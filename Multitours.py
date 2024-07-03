import numpy as np
import math
import networkx as nx
import itertools

# Random multitour generation
def generate_random_tour(Time):
    current_node = 0
    tour = [current_node]  # Start and end at node 0
    visited = []
    while True:
        possible_destinations = Time[Time['Origin'] == current_node]['Destination'].unique()      
        
        possible_destinations = list(set(possible_destinations) - set(visited))
        if len(possible_destinations) == 0:
            break  # If there are no more possible destinations, end the tour
        next_node = np.random.choice(possible_destinations)
        
        tour.append(next_node)
        current_node = next_node

        # Remove node 0 from the list of tour nodes
        visited = list(set(tour) - set([0]))

        if current_node == 0:
            if np.random.rand() < 0.5:
                break  # End the tour when back to node 0
    return tour

def generate_random_multi_tour(n, Network):
    tours = [generate_random_tour(Time=Network) for i in range(n)]
    return tours

# Identify tours in each multitour. A tour are the nodes between two zeros
def identify_tours(multitour):
    result = []
    start_index = None

    for i, value in enumerate(multitour):
        if value == 0:
            if start_index is not None:
                result.append(multitour[start_index + 1:i])
            start_index = i
    return result


# Functions to generate a set of basic multi-tours

# Find the simple path between two nodes
def dijkstra_path(graph, start, end, visited):
    nodes = set(graph.nodes()) - set(visited) 
    if nx.has_path(graph.subgraph(nodes), start, end):
        return nx.dijkstra_path(graph.subgraph(nodes), start, end)
    else:
        return None

# function to Generate basic tour for a single node
def generate_basic_tour(df, start, visit):
    G = nx.from_pandas_edgelist(df, 'Origin', 'Destination', create_using=nx.DiGraph())
    path1 = dijkstra_path(G, start, visit, [])
    if path1 is None:
        return "No path to " + str(visit) + "."
    visits = list(set(path1) - set([0, visit]))
    path2 = dijkstra_path(G, visit, start, visits)
    if path2 is None:
        return "No path to 0."
    return path1 + path2[1:]

# function to Generate the set of basic tours (each multitour is not mutually exclusive with the others)
def generate_basic_tours(df, Time, Customers, ncustomers,vehicleCapacity, delta, phi, h, d):
    basic_tours = []
    for i in range(1, ncustomers + 1):
        basic_tours.append(generate_basic_tour(df, 0, i))

    Tours = []
    current_tour = basic_tours[0]
    for i in range(1, len(basic_tours)):
        visited_current = list(set(current_tour) - set([0]))
        next_tour = basic_tours[i]
        visited_next = list(set(next_tour) - set([0]))

        # check if any element in visited_next is in visited_current
        if set(visited_next) & set(visited_current):
            Tours.append(current_tour)
            current_tour = next_tour
        else:
            temporal_tour = current_tour + next_tour[1:]
            temporal_single_tour = identify_tours(temporal_tour)
            Tmin, Tmax, T_EOQ, T, checkedTour = get_cycle_time(Time, Customers, [temporal_tour], [temporal_single_tour], 1, vehicleCapacity, delta, phi, h, d)
            if checkedTour != []:
                current_tour = temporal_tour
            else:
                Tours.append(current_tour)
                current_tour = next_tour
    if current_tour not in Tours:
        Tours.append(current_tour)
    
    for tour in basic_tours:
        if tour not in Tours:
            Tours.append(tour)
    return Tours

# Generate set of basic tours where each multitour is mutually exclusive with the others
def generate_basic_tours2(df, Time, Customers, ncustomers,vehicleCapacity, delta, phi, h, d):
    basic_tours = []
    for i in range(1, ncustomers + 1):
        basic_tours.append(generate_basic_tour(df, 0, i))

    Tours = [tour for tour in basic_tours if len(tour) != 3]
    for tour in Tours:
        visited = list(set(tour) - set([0]))
        basic_tours.remove(tour)

    next_tour = basic_tours[0]
    current_tour = [0]
    for i in range(1, len(basic_tours)):
        visited_current = list(set(next_tour) - set([0]))
        if set(visited_current).issubset(set(visited)):
            next_tour = basic_tours[i]
        else:
            temporal_tour = current_tour + next_tour[1:]
            temporal_single_tour = identify_tours(temporal_tour)
            Tmin, Tmax, T_EOQ, T, checkedTour = get_cycle_time(Time, Customers, [temporal_tour], [temporal_single_tour], 1, vehicleCapacity, delta, phi, h, d)
            if checkedTour != []:
                current_tour = temporal_tour
                next_tour = basic_tours[i]
                visited += visited_current
            else:
                Tours.append(current_tour)
                current_tour = next_tour  
    if current_tour not in Tours:
        Tours.append(current_tour)

    for tour in basic_tours:
        if tour not in Tours:
            Tours.append(tour)         

    return Tours

# Initial basic tours used in the heuristic of the subproblem
def initial_basic_tours(ncustomers, Network):
    basic_tours = []
    for i in range(1, ncustomers + 1):
        basic_tours.append(generate_basic_tour(Network, 0, i))
    return basic_tours

# Cycle Time calculations
def calculate_travel_time(tour, Time):
    return sum(Time.loc[(Time['Origin'] == tour[i]) & (Time['Destination'] == tour[i+1]), 'Time'].iloc[0] for i in range(len(tour)-1))

def calculate_max_time(tour, Customers):
    return min(Customers.loc[Customers['Customer'] == customer, 'Tmax'].iloc[0] for customer in set(tour) - {0})

def calculate_max_single_tour_time(tour, Customers, vehicleCapacity, vehicle = 0):
    return vehicleCapacity[vehicle] / sum(Customers.loc[Customers['Customer'] == customer, 'Demand'].iloc[0] for customer in tour)

def calculate_t_eoq(tour, tmin, delta, phi, h, d):
    visited_customers = set(tour) - {0}
    return math.sqrt(delta * tmin * sum(phi[customer] for customer in visited_customers) / sum(0.5 * h[customer] * d[customer] for customer in visited_customers))

def calculate_t(tmin, tmax, t_eoq):
    if tmin <= tmax:
        if t_eoq > tmax:
            return tmax
        elif t_eoq < tmin:
            return tmin
        else:
            return t_eoq
    else:
        return math.inf
    
def get_cycle_time(Time, Customers, MultiTours, SingleTours, n_MultiTours, vehicleCapacity, delta, phi, h, d):
    # Calculate Tmin for each MultiTour
    Tmin = [calculate_travel_time(tour, Time) for tour in MultiTours]

    # Maximum time for each customer (considering customer capacity)
    Tmax_Cust = [calculate_max_time(tour, Customers) for tour in MultiTours]

    # Maximum time of each single tour
    Tmax_SingleTour = [[calculate_max_single_tour_time(tour, Customers, vehicleCapacity) for tour in SingleTours[i]] for i in range(n_MultiTours)]

    # Maximum time of each multitour
    Tmax_Multitour = [min(Tmax_SingleTour[i]) for i in range(n_MultiTours)]

    # Real maximum time of each multitour (considering customer capacity)
    Tmax = [min(Tmax_Cust[i], Tmax_Multitour[i]) for i in range(n_MultiTours)]

    # Compute T based on the EOQ
    T_EOQ = [calculate_t_eoq(tour, Tmin[i], delta, phi, h, d) for i, tour in enumerate(MultiTours)]

    # Calculate cycle time T (considerint Tmin, Tmax and T_EOQ)
    T = [calculate_t(Tmin[i], Tmax[i], T_EOQ[i]) for i, tour in enumerate(MultiTours)]

    T = np.array(T)

    # If multitour is infeasible update Multitours, Tmin, Tmax, T_EOQ and T
    infeasibles = np.where(T == math.inf)[0]

    MultiTours = [multitour for i, multitour in enumerate(MultiTours) if i not in infeasibles]
    Tmin = np.delete(Tmin, infeasibles)
    Tmax = np.delete(Tmax, infeasibles)
    T_EOQ = np.delete(T_EOQ, infeasibles)
    T = np.delete(T, infeasibles)
    T = T.tolist()

    return Tmin, Tmax, T_EOQ, T, MultiTours

# Multitour Binary Representation
def Multitours_Binary(MultiTours, ncustomers):
    X = np.zeros((len(MultiTours), ncustomers+1, ncustomers+1))

    for i, tour in enumerate(MultiTours):
        for j in range(len(tour)-1):
            X[i, tour[j], tour[j+1]] = 1
    
    return X

# Multitour Cost Calculation
def Multitour_Cost(n_MultiTours, ncustomers, psi, T, delta, t, X, phi, h, d):
    Rplus = [i for i in range(ncustomers+1)]
    Customers_indeces = [i for i in range(1, ncustomers+1)]
    cost = [0 for i in range(n_MultiTours)]
    for theta in range(n_MultiTours):
        if T[theta] == 0:
            cost[theta] = 0
        else:
            cost[theta] = psi[0] + 1/T[theta] * sum(delta * t[i,j] * X[theta,i,j] for j in Rplus for i in Rplus) + sum((phi[i] * 1/T[theta] + 0.5 * h[i] * d[i] * T[theta]) * sum(X[theta,i,j] for j in Rplus) for i in Customers_indeces)

    return cost

# Multitour Combinations based on the simmetry of the single tours
def Simmetry_Combinations_NOT_USED(multitour):
    singletours = identify_tours(multitour)
    num_single_tours = len(singletours)

    singletours = [[0] + tour + [0] for tour in singletours]
    singletours2 = singletours.copy()
    singletours2 = [tour[::-1] for tour in singletours2 if len(tour) > 3] # Reverse the single tours

    singletours += singletours2

    if len(singletours) == 2:
        combinations = singletours
        return combinations
    else:
        temp_combinations = list(itertools.permutations(singletours,num_single_tours))

        combinations = []
        for comb in temp_combinations:
            if all_combinations_different(comb):
                test = [x for sublist in comb[:-1] for x in sublist[:-1]]
                test.extend(comb[-1])
                combinations.append(test)

        return combinations

# Function that checks that a single tour and its reversed version are not in the included in a combination at the same time
def all_combinations_different(combination):
    for i in range(len(combination)):
        for j in range(i+1,len(combination)):
            if combination[i] == combination[j][::-1]:
                return False
    return True

def Simmetry_Combinations(multitour):
    singletours = identify_tours(multitour)

    singletours = [[0] + tour + [0] for tour in singletours]
    singletours_reversed = [tour[::-1] for tour in singletours if len(tour) > 3] # Reverse the single tours

    if singletours_reversed:
        temp_combined = [[t, tr] for t, tr in zip(singletours, singletours_reversed)]
    else:
        temp_combined = [[t] for t in singletours]


    if len(singletours) == 1:
        combinations = singletours
        return combinations
    else:
        combinations = []
        for p in itertools.permutations(temp_combined): # Genera todas las permutaciones de las listas combinadas
            for comb in itertools.product(*p): # genera todas las combinaciones de las versiones normales e invertidas de las listas
                
                test = [x for sublist in comb[:-1] for x in sublist[:-1]]
                test.extend(comb[-1])
                combinations.append(test)

        return combinations

# Function that checks if a tours (and its symmetrical versions) is already in the MultiTours list
def MultiTour_index(MultiTours, tour):
    tour_combinations = Simmetry_Combinations(tour)
    # tour_combinations = [tour]
    
    for t in tour_combinations:
        idx = [index for index, elem in enumerate(MultiTours) if elem == t]
        if idx:
            return idx[-1]
        else:
            idx = ""
    return idx

# List of visited customers in a multitour
def Visited_Customers(tour):
    return list(set(tour) - {0})

# Check if a trip is in a tour
def trip_in_tour(trip, tour):
    return any(trip == tour[p:p+2] for p in range(len(tour)-1))

# Check if any of the customers in the list are in the tour
def customers_in_tour(customers, tour):
    return any(customer in tour for customer in customers)

# Index of the multitours that contain any of the customers in the list
def MultiTour_index_customers(MultiTours, customers):
    indices = [i for i, tour in enumerate(MultiTours) if customers_in_tour(customers, tour)]
    return indices

# Index of the multitours that contain a specific trip
def MultiTour_index_trip(MultiTours, trip):
    indices = [i for i, tour in enumerate(MultiTours) if trip_in_tour(trip, tour)]
    return indices

# Penalize cost of multitours that contain any of the customers in the list, or a specific multitour
def Penalize_MultiTours(MultiTours, Cost, penalty, customers = None, tour_idx = None, trip = None):
    if customers and tour_idx:
        indices = MultiTour_index_customers(MultiTours, customers)
        for i in indices:
            if i != tour_idx:
                Cost[i] += penalty
    if tour_idx and not customers and not trip:
        Cost[tour_idx] += penalty
    if trip:
        indices = MultiTour_index_trip(MultiTours, trip)
        for i in indices:
            Cost[i] += penalty

# Multitours in a solution
def Multitours_in_Solution(MultiTours, solution):
    indeces = [x[0] for x in solution]
    multitours = [MultiTours[i] for i in indeces]
    return multitours

# Select the trip to branch
def Select_Trip(Multitours, constraints, solution, tour):
    customers = Visited_Customers(tour)
    solution_tours = Multitours_in_Solution(Multitours, solution) 
    solution_tours = [mtour for mtour in solution_tours if mtour != tour] #eliminate the tour from this list
    indices = MultiTour_index_customers(solution_tours, customers)

    # Previous trips that are in the constraints
    trips_constrained = [constr[1] for constr in constraints]
    trip = trips_constrained[0] if trips_constrained else None
    while (trip in trips_constrained and indices) or trip is None:
        # Select the first multitour in the solution that shares a customer with the tour
        shared_customer_multitour = solution_tours[indices[0]]
        indices = indices[1:]

        # Find the first trip in the tour that is not in the shared multitour
        for q in range(1,len(tour)-1):
            # Check if the trip is in the shared multitour
            trip = [tour[q], tour[q+1]]
            # if not any(trip == shared_customer_multitour[p:p+2] for p in range(len(shared_customer_multitour)-1)):
            if not trip_in_tour(trip, shared_customer_multitour):
                break
    return trip

def Add_MultiTours_Manually(parameters, MultiTours,Cost,T,X, newtour, newTime):
    x0 = Multitours_Binary([newtour], parameters['ncustomers'])

    cost0 = Multitour_Cost(1, parameters['ncustomers'], parameters['psi'], [newTime], parameters['delta'], parameters['t'], x0, parameters['phi'], parameters['h'], parameters['d'])

    MultiTours.append(newtour)

    T.append(newTime)

    X = np.append(X, x0, axis=0)

    Cost.append(cost0[0])
    return MultiTours,Cost,T,X
