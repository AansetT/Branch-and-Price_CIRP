import pandas as pd
import numpy as np
import Multitours as MT
import os
from Multitours import Multitours_Binary
from scipy.spatial.distance import cdist

def Artificial_MultiTour(ncustomers):
    artificial = [0]
    for i in range(1, ncustomers + 1):
        artificial.append(i)
    artificial.append(0)

    return artificial

def Artificial_Variable(ncustomers, BigM):
    artificial = Artificial_MultiTour(ncustomers)

    # Multitours
    n_MultiTours = 1
    MultiTours = [artificial]
    T = [100]
    Cost = [BigM] # Big M
    X = Multitours_Binary(MultiTours, ncustomers)

    return MultiTours,Cost,T,X

def Add_Artificial_Variables(MultiTours_in,Cost_in,T_in,X_in,ncustomers, BigM):
    MultiTours,Cost,T,X = Artificial_Variable(ncustomers, BigM)
    MultiTours_in.extend(MultiTours)
    Cost_in.extend(Cost)
    T_in.extend(T)
    X_in = np.concatenate((X_in, X), axis=0)
    return X_in

def Initial_Tours(Random = False, Basic1 = False, Basic2 = False, n_radom_multitours = 10, Network = None, Time = None, Customers = None, instance_parameters: dict = None, BigM = 1000000):
    ncustomers, nvehicles, psi, delta, phi, h, d, k, t, vehicleCapacity = get_instance_parameters(instance_parameters)
    #Random Multitours
    if Random:
        n_MultiToursR = n_radom_multitours
        MultiToursR = MT.generate_random_multi_tour(n_MultiToursR, Network)
        SingleToursR = [MT.identify_tours(multitour) for multitour in MultiToursR]

    # Basic MultiTours
    if Basic1:
        MultiToursB = MT.generate_basic_tours(Network, Time, Customers, ncustomers, vehicleCapacity, delta, phi, h, d)
    if Basic2:
        MultiToursB = MT.generate_basic_tours2(Network, Time, Customers, ncustomers, vehicleCapacity, delta, phi, h, d)
    SingleToursB = [MT.identify_tours(multitour) for multitour in MultiToursB]
    n_MultiToursB = len(MultiToursB)

    # Cycle time and bounds. Update the MultiTours list with only feasible tours (Possibility to mix basic and random tours)
    if Random:
        Tmin, Tmax, T_EOQ, T, MultiTours = MT.get_cycle_time(Time, Customers, MultiToursR, SingleToursR, n_MultiToursR, vehicleCapacity, delta, phi, h, d)
    if Basic1 or Basic2:
        Tmin, Tmax, T_EOQ, T, MultiTours = MT.get_cycle_time(Time, Customers, MultiToursB, SingleToursB, n_MultiToursB, vehicleCapacity, delta, phi, h, d)
    if Random and (Basic1 or Basic2):
        Tmin, Tmax, T_EOQ, T, MultiTours = MT.get_cycle_time(Time, Customers, MultiToursR + MultiToursB, SingleToursR + SingleToursB, n_MultiToursR + n_MultiToursB, vehicleCapacity, delta, phi, h, d)
    n_MultiTours = len(MultiTours) 

    # Binary representation of the MultiTours
    X = MT.Multitours_Binary(MultiTours, ncustomers)

    # Multitour cost
    Cost = MT.Multitour_Cost(n_MultiTours, ncustomers, psi, T, delta, t, X, phi, h, d)

    # Add artificial variable just to ensure a feasible solution
    # X = Add_Artificial_Variables(MultiTours,Cost,T,X,ncustomers,BigM=BigM)

    return MultiTours,Cost,T,X


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

# Function to read the data
def read_data(filename, ncustomers, nvehicles):
    notebook_directory = os.getcwd()
    folder_path = os.path.dirname(notebook_directory)
    file_path = os.path.join(folder_path, "Data", filename+".xlsx")

    Customers = pd.read_excel(file_path, sheet_name='Sheet1', nrows=ncustomers, usecols="A:E")
    Time = pd.read_excel(file_path, sheet_name='Sheet1', skiprows=ncustomers+2, usecols="A:C")
    Vehicles = pd.read_excel(file_path, sheet_name='Sheet1', nrows=nvehicles, usecols="G:I")
    delta = pd.read_excel(file_path, sheet_name='Sheet1', nrows=1, usecols="K:K")
    Network = Time[["Origin", "Destination"]]
    Customers["Tmax"] = Customers["Capacity"] / Customers["Demand"]
    Vehicles.rename(columns={"Capacity.1": "Capacity"}, inplace=True)

    return Customers, Time, Vehicles, Network, delta

def instance_parameters(ncustomers, nvehicles, Customers, Time, Vehicles, delta):
    # Vehicle cost parameter
    psi = [int(i) for i in Vehicles["psi"].to_list()]

    # Vehicle capacity
    vehicleCapacity = [int(i) for i in Vehicles["Capacity"].to_list()]

    # Transportation cost
    delta = delta.values[0][0]

    # Delivery cost
    # column Phi of df Costumers with indeces 1 to ncustomers
    phi = Customers["Phi"].copy()
    phi.index = phi.index + 1

    # Holding Cost
    h = Customers["Holding"].copy()
    h.index = h.index + 1

    # Demand
    d = Customers["Demand"].copy()
    d.index = d.index + 1

    # Capacity
    k = Customers["Capacity"].copy()
    k.index = k.index + 1

    # Travel time parameter
    shape = (ncustomers + 1, ncustomers + 1)
    t = np.full(shape, 100.0)

    for index, row in Time.iterrows():
        i = int(row['Origin'])
        j = int(row['Destination'])
        t[i, j] = row['Time']

    # Create a dictionary with all the parameters
    parameters = {
        "ncustomers": ncustomers,
        "nvehicles": nvehicles,
        "vehicleCapacity": vehicleCapacity,
        "psi": psi,
        "delta": delta,
        "phi": phi,
        "h": h,
        "d": d,
        "k": k,
        "t": t
    }

    return parameters

def read_instance(n_vehicles, set_number, filename):
    notebook_directory = os.getcwd()
    folder_path = os.path.dirname(notebook_directory)
    file_path = os.path.join(folder_path, "Data", "Instances",str(n_vehicles)+"V-CIRP","Set "+str(set_number),filename+".txt")

    data = read_txt_file(file_path)

    columns_table2 = ['Customer', 'x', 'y', 'Phi', 'Demand', 'Holding', 'Reward']
    table2_df = pd.DataFrame(data['table2']['data'], columns=columns_table2).astype(float)

    # Number of customers
    ncustomers = len(data['table2']['data']) - 1
    vehicle_cap = int(data['table1']['data'][0][1])

    # Trasportation cost delta
    d = float(data['table1']['data'][0][2])
    speed = float(data['table1']['data'][0][3])
    delta = speed * d
    delta = pd.DataFrame([delta], columns=['delta'])

    # Fixed cost of the vehicles psi
    psi = int(data['table1']['data'][0][4])

    # Vehicles DataFrame
    Vehicles = pd.DataFrame([[i+1, vehicle_cap, psi] for i in range(n_vehicles)],
                            columns=['Vehicle', 'Capacity', 'psi'])
    
    # Customers DataFrame
    Customers = table2_df.iloc[1:].copy()
    Customers.reset_index(drop=True, inplace=True)
    Customers = Customers[['Customer', 'Demand', 'Phi', 'Holding']]
    Customers['Customer'] = Customers['Customer'].astype(int)
    Customers['Capacity'] = vehicle_cap * 3
    Customers["Tmax"] = Customers["Capacity"] / Customers["Demand"]

    # Distance/time matrix
    coordinates = table2_df[['x', 'y']].to_numpy()
    distance_matrix = cdist(coordinates, coordinates, 'euclidean')
    time_matrix = np.around(distance_matrix / speed, 3)

    # Time DataFrame
    data_time = [(i, j, time_matrix[i, j]) for i in range(ncustomers + 1) for j in range(ncustomers + 1) if i != j]
    Time = pd.DataFrame(data_time, columns=['Origin', 'Destination', 'Time'])

    # Network DataFrame
    Network = Time[["Origin", "Destination"]]

    return ncustomers, Customers, Time, Vehicles, Network, delta


def read_txt_file(file_path):
    data = {}  # Dictionary to store the data

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Read the first table
        table1_header = lines[0].strip().split()
        table1_data = [line.strip().split() for line in lines[1:2]]  
        data['table1'] = {'header': table1_header, 'data': table1_data}

        # Read the second table
        table2_header = lines[3].strip().split()
        table2_data = [line.strip().split('\t') for line in lines[4:]]  # Assuming second table starts from line 4
        data['table2'] = {'header': table2_header, 'data': table2_data}

    return data
    