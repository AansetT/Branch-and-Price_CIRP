import Multitours as MT
import numpy as np
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
    savings_matrix = np.zeros((len(temporary_list), len(temporary_list)))
    Time_matrix = np.zeros((len(temporary_list), len(temporary_list)))
    RC_matrix = np.zeros((len(temporary_list), len(temporary_list)))
    tour_matrix = np.empty((len(temporary_list), len(temporary_list)), dtype=object)

    # params
    parameters = {'Time': Time, 'Customers': Customers, 'VehicleCapacity': vehicleCapacity, 'delta': delta, 'phi': phi, 'h': h, 'd': d}

    #Savings Step
    max_saving = 1
    while max_saving > 0:
        max_saving, temporary_list, RC_temporary_list, T_temporary_list, savings_matrix, Time_matrix, RC_matrix, tour_matrix = Saving_Step(temporary_list, RC_temporary_list, T_temporary_list, Reduced_Cost, savings_matrix,Time_matrix,RC_matrix, tour_matrix,parameters)
    
    # best_RC = min(RC_temporary_list)
    # index_best = RC_temporary_list.index(best_RC)
    # best_multi_tour = temporary_list[index_best]
    # best_multi_tour_Time = T_temporary_list[index_best]

    # return best_multi_tour, best_RC, best_multi_tour_Time

    # filter the temporary list to get all the ones that have negative reduced cost
    tours = [temporary_list[i] for i, RC in enumerate(RC_temporary_list) if RC < 0]
    RCs = [RC_temporary_list[i] for i, RC in enumerate(RC_temporary_list) if RC < 0]
    Times = [T_temporary_list[i] for i, RC in enumerate(RC_temporary_list) if RC < 0]

    if RCs == []:
        tours=temporary_list
        RCs=RC_temporary_list
        Times=T_temporary_list

    return tours, RCs, Times


def Saving_Step(temporary_list, RC_temporary_list, T_temporary_list, Reduced_Cost,  savings_matrix, Time_matrix,RC_matrix, tour_matrix, params: dict= None):
    # params
    Time = params['Time']
    Customers = params['Customers']
    vehicleCapacity = params['VehicleCapacity']
    delta = params['delta']
    phi = params['phi']
    h = params['h']
    d = params['d']

    # All combinations of two multi-tours
    for idx_v1 in range(len(temporary_list)):
        for idx_v2 in range(len(temporary_list)):
            if idx_v1 < idx_v2 and savings_matrix[idx_v1, idx_v2] == 0:
                v1 = temporary_list[idx_v1]
                v2 = temporary_list[idx_v2]
                single_tours_v1 = MT.identify_tours(v1)
                RC_v1 = RC_temporary_list[idx_v1]
                single_tours_v2 = MT.identify_tours(v2)
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
                tour_matrix[idx_v1, idx_v2] = v_star
                savings_matrix[idx_v1, idx_v2] = saving
                if Time_v_star != []:
                    Time_matrix[idx_v1, idx_v2] = Time_v_star
                else:
                    Time_matrix[idx_v1, idx_v2] = math.inf
                RC_matrix[idx_v1, idx_v2] = RC_v_star
    
    # Identify largest saving
    max_saving = np.max(savings_matrix)
    indexes = np.where(savings_matrix == max_saving)
    index_i = indexes[0][0]
    index_j = indexes[1][0]
    best_v_star = tour_matrix[index_i, index_j]
    RC_best_v_star = RC_matrix[index_i, index_j]
    Time_best_v_star = Time_matrix[index_i, index_j]

    #Update the temporary list
    if max_saving > 0:
        temporary_list, RC_temporary_list, T_temporary_list, savings_matrix, Time_matrix, RC_matrix, tour_matrix = Update_temporary_list(temporary_list, RC_temporary_list, T_temporary_list, best_v_star, index_i, index_j, savings_matrix, Time_matrix,RC_matrix, tour_matrix, Time_best_v_star, RC_best_v_star)
    
    return max_saving, temporary_list, RC_temporary_list, T_temporary_list, savings_matrix, Time_matrix, RC_matrix, tour_matrix

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
        return 0
    if any(RC_v_star > Red_cost for Red_cost in [RC_v1, RC_v2]):
        return 9
    return RC_v1 + RC_v2 - RC_v_star

def Update_temporary_list(temporary_list, RC_temporary_list, T_temporary_list, best_v_star, max_saving_index_i, max_saving_index_j, savings_matrix, Time_matrix,RC_matrix, tour_matrix, Time_best_v_star, RC_best_v_star):
    temporary_list.append(best_v_star)
    RC_temporary_list.append(RC_best_v_star)
    T_temporary_list.append(Time_best_v_star)
    
    # v1, v2 = multi_tour_combinations[max_saving_index]
    v1 = temporary_list[max_saving_index_i]
    v2 = temporary_list[max_saving_index_j]
    index_v1 = max_saving_index_i
    index_v2 = max_saving_index_j

    # Eliminate v1 and v2 from the temporary list
    temporary_list = temporary_list[:index_v1] + temporary_list[index_v1+1:index_v2] + temporary_list[index_v2+1:]
    RC_temporary_list = RC_temporary_list[:index_v1] + RC_temporary_list[index_v1+1:index_v2] + RC_temporary_list[index_v2+1:]
    T_temporary_list = T_temporary_list[:index_v1] + T_temporary_list[index_v1+1:index_v2] + T_temporary_list[index_v2+1:]

    # Eliminate row and column of v1 and v2 in the RC matrix
    RC_matrix = eliminate_row_column(RC_matrix, index_v1)
    RC_matrix = eliminate_row_column(RC_matrix, index_v2-1)

    # Add the row and column of the new multi-tour to the RC matrix
    RC_matrix = add_row_column(RC_matrix)

    # Eliminate row and column of v1 and v2 in the Time matrix
    Time_matrix = eliminate_row_column(Time_matrix, index_v1)
    Time_matrix = eliminate_row_column(Time_matrix, index_v2-1)

    # Add the row and column of the new multi-tour to the Time matrix
    Time_matrix = add_row_column(Time_matrix)

    # Eliminate row and columnof v1 in the saving matrix
    savings_matrix = eliminate_row_column(savings_matrix, index_v1)
    savings_matrix = eliminate_row_column(savings_matrix, index_v2-1)

    # Add the row and column of the new multi-tour to the saving matrix
    savings_matrix = add_row_column(savings_matrix)

    # Eliminate row and column of v1 and v2 in the tour matrix
    tour_matrix = eliminate_row_column(tour_matrix, index_v1)
    tour_matrix = eliminate_row_column(tour_matrix, index_v2-1)

    # Add the row and column of the new multi-tour to the tour matrix
    tour_matrix = add_row_column(tour_matrix, tours=True)

    return temporary_list, RC_temporary_list, T_temporary_list, savings_matrix, Time_matrix, RC_matrix, tour_matrix

def eliminate_row_column(matrix, tour_index):
    matrix_modified = np.delete(matrix, tour_index, axis=0)
    matrix_modified = np.delete(matrix_modified, tour_index, axis=1)
    return matrix_modified

def add_row_column(matrix, tours = False):
    if tours:
        new_row = np.empty((1, matrix.shape[1]), dtype=object)
    else:
        new_row = np.zeros((1, matrix.shape[1]))
    matrix = np.append(matrix, new_row, axis=0)
    if tours:
        new_column = np.empty((matrix.shape[0], 1), dtype=object)
    else:
        new_column = np.zeros((matrix.shape[0], 1))  # Create a column of zeros with the same number of rows as the modified array
    matrix = np.append(matrix, new_column, axis=1)
    return matrix