import Multitours as MT
from itertools import combinations
import networkx as nx
import math


def SubProblem_Heuristic(Initial_Basic_Tours, ncustomers, psi, delta, phi, h, d, k, t, vehicleCapacity, Time, Customers, DualPrices, DualZero):
    # Initial basic tours comes from the function in the multitours.py file

    #Reduced cost function
    Reduced_Cost = Reduced_Cost_generator(ncustomers, psi, delta, phi, h, d, t, DualPrices, DualZero)

    # Temporary list of multi-tours
    temporary_list = Initial_Basic_Tours
    T_temporary_list = [MT.get_cycle_time(Time, Customers, [tour], [MT.identify_tours(tour)], 1, vehicleCapacity, delta, phi, h, d)[0][0] for tour in temporary_list] #it is possible that with Tmin we get better results
    RC_temporary_list = [Reduced_Cost(tour, T_temporary_list[i]) for i, tour in enumerate(temporary_list)]

    # params
    parameters = {'Time': Time, 'Customers': Customers, 'VehicleCapacity': vehicleCapacity, 'delta': delta, 'phi': phi, 'h': h, 'd': d}

    #Savings Step
    max_saving = 1
    while max_saving > 0:
        max_saving, temporary_list, RC_temporary_list, T_temporary_list = Saving_Step(temporary_list, RC_temporary_list, T_temporary_list, Reduced_Cost, parameters)
    
    best_RC = min(RC_temporary_list)
    index_best = RC_temporary_list.index(best_RC)
    best_multi_tour = temporary_list[index_best]
    best_multi_tour_Time = T_temporary_list[index_best]

    return best_multi_tour, best_RC, best_multi_tour_Time


def Saving_Step(temporary_list, RC_temporary_list, T_temporary_list, Reduced_Cost, params: dict= None):
    # params
    Time = params['Time']
    Customers = params['Customers']
    vehicleCapacity = params['VehicleCapacity']
    delta = params['delta']
    phi = params['phi']
    h = params['h']
    d = params['d']

    # All combinations of two multi-tours
    # multi_tour_combinations = generate_combinations(temporary_list)
    multi_tour_combinations = []
    Generated_tours = []
    Time_Generated_tours = []
    RC_generated_tours = []
    Savings_tours = []
    # for v1, v2 in multi_tour_combinations:
    for v1, v2 in combinations(temporary_list, 2):
        multi_tour_combinations.append((v1, v2))
        single_tours_v1 = MT.identify_tours(v1)
        n = len(single_tours_v1)
        idx_v1 = temporary_list.index(v1)
        RC_v1 = RC_temporary_list[idx_v1]
        single_tours_v2 = MT.identify_tours(v2)
        m = len(single_tours_v2)
        idx_v2 = temporary_list.index(v2)
        RC_v2 = RC_temporary_list[idx_v2]

        list_of_tours = single_tours_v1 + single_tours_v2

        # create multi-tour v*
        v_star = [0]
        for tour in list_of_tours:
            v_star += tour + [0]

        Time_v_star = MT.get_cycle_time(Time, Customers, [v_star], [MT.identify_tours(v_star)], 1, vehicleCapacity, delta, phi, h, d)[3]

        # Checking for feasibility
        if Time_v_star != []:
            Time_v_star = Time_v_star[0]
            # Calculate the reduced cost of the multi-tour v*
            RC_v_star = Reduced_Cost(v_star, Time_v_star)
        else:
            RC_v_star = math.inf
            v_star = []

        for i, tour1 in enumerate(list_of_tours):
            for j, tour2 in enumerate(list_of_tours):
                if (i < j) and (tour1 in single_tours_v1) and (tour2 in single_tours_v2):
                    C_plus = TSP_tour (tour1, tour2, [0], Time)
                    v_star_temp = Combine_tours(C_plus, i, j , list_of_tours)
                    Time_v_star_temp = MT.get_cycle_time(Time, Customers, [v_star_temp], [MT.identify_tours(v_star_temp)], 1, vehicleCapacity, delta, phi, h, d)[3]
                    if Time_v_star_temp != []:
                        Time_v_star_temp = Time_v_star_temp[0]
                        RC_v_star_temp = Reduced_Cost(v_star_temp, Time_v_star_temp)
                        v_star, RC_v_star, Time_v_star = Update_v_star(v_star, RC_v_star, Time_v_star, v_star_temp, RC_v_star_temp, Time_v_star_temp)
        
        # calcualte savings
        saving = Savings(RC_v1, RC_v2, RC_v_star)
        Generated_tours.append(v_star)
        Savings_tours.append(saving)
        Time_Generated_tours.append(Time_v_star)
        RC_generated_tours.append(RC_v_star)
    
    # Identify largest saving
    max_saving = max(Savings_tours)
    index = Savings_tours.index(max_saving)
    best_v_star = Generated_tours[index]
    RC_best_v_star = RC_generated_tours[index]
    Time_best_v_star = Time_Generated_tours[index]

    #Update the temporary list
    if max_saving > 0:
        temporary_list, RC_temporary_list, T_temporary_list = Update_temporary_list(temporary_list, RC_temporary_list, T_temporary_list, best_v_star, index, multi_tour_combinations, Time_best_v_star, RC_best_v_star)
    
    return max_saving, temporary_list, RC_temporary_list, T_temporary_list

def generate_combinations(input_list):
    return list(combinations(input_list, 2))

def Reduced_Cost_generator(ncustomers, psi, delta, phi, h, d, t, DualPrices, DualZero):
    Rplus = [i for i in range(ncustomers+1)]
    Customers_indeces = [i for i in range(1, ncustomers+1)]
    def Reduced_Cost(tour, Time):
        x = MT.Multitours_Binary([tour], ncustomers)
        RC = MT.Multitour_Cost(1, ncustomers, psi, [Time], delta, t, x, phi, h, d)[0] - sum(DualPrices[j] * x[0, i, j] for j in Customers_indeces for i in Rplus) - sum(DualZero)
        return RC
    return Reduced_Cost

def TSP_tour(tour1, tour2, supply_centre, Time):
    nodos = supply_centre + tour1 + tour2
    Time = Time.rename(columns={'Time': 'weight'})
    G = nx.from_pandas_edgelist(Time, 'Origin', 'Destination', edge_attr='weight', create_using=nx.DiGraph())
    sub_G = G.subgraph(nodos)
    
    tour = nx.approximation.greedy_tsp(sub_G, weight='weight', source=nodos[0])
    
    return tour

def Combine_tours(C_plus, i, j , list_of_tours):
    # Get all the elements of list of tours except the ith and jth elements
    tours = list_of_tours.copy()
    tours = tours[:i] + tours[i+1:j] + tours[j+1:]

    v = [0]
    for tour in tours:
        v += tour + [0]
    v+=C_plus[1:]
    return v

def Update_v_star(v_star, RC_v_star, Time_v_star, v_star_temp, RC_v_star_temp, Time_v_star_temp):
    if RC_v_star_temp < RC_v_star:
        v_star = v_star_temp.copy()
        RC_v_star = RC_v_star_temp
        Time_v_star = Time_v_star_temp

    return v_star, RC_v_star, Time_v_star

def Savings(RC_v1, RC_v2, RC_v_star):
    if RC_v_star >=0:
        return - math.inf
    if any(RC_v_star > Red_cost for Red_cost in [RC_v1, RC_v2]):
        return - math.inf
    return RC_v1 + RC_v2 - RC_v_star

def Update_temporary_list(temporary_list, RC_temporary_list, T_temporary_list, best_v_star, max_saving_index, multi_tour_combinations, Time_best_v_star, RC_best_v_star):
    temporary_list.append(best_v_star)
    RC_temporary_list.append(RC_best_v_star)
    T_temporary_list.append(Time_best_v_star)
    
    v1, v2 = multi_tour_combinations[max_saving_index]
    index_v1 = temporary_list.index(v1)
    index_v2 = temporary_list.index(v2)
    temporary_list = temporary_list[:index_v1] + temporary_list[index_v1+1:index_v2] + temporary_list[index_v2+1:]
    RC_temporary_list = RC_temporary_list[:index_v1] + RC_temporary_list[index_v1+1:index_v2] + RC_temporary_list[index_v2+1:]
    T_temporary_list = T_temporary_list[:index_v1] + T_temporary_list[index_v1+1:index_v2] + T_temporary_list[index_v2+1:]

    return temporary_list, RC_temporary_list, T_temporary_list
