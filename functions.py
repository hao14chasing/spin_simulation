from spirit import state, simulation, configuration, quantities, parameters, log, geometry, io, system, hamiltonian
from spirit.parameters import *
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pickle

"""
Energy
====================
This module contains functions to set up the energy test parameters, 
extract and visualize the energy data from the log file, and check the convergence of the energy.

"""


"""Input file path"""
config_file_path = "C:/Users/hao14/Desktop/Simulation/Co3Sn2S2.cfg"

"""Energy log folder path"""
energy_log_folder = "C:/Users/hao14/Desktop/Simulation/Energy Log"

"""Energy log file tag"""
#energy_log_tag = "Test"


"""Energy log file path"""
#energy_file = energy_log_folder + "/" + energy_log_tag + "_Image-00_Energy-archive.txt"

"""Magnetization log folder path"""
magnetization_log_folder = "C:/Users/hao14/Desktop/Simulation/Magnetization Log"



"""A function to set up the llg parameters"""
"""Usage Function"""
def energy_test_parameters(p_state, energy_log_tag , n_iterations = 1000000, n_iterations_log = 2000):
    ###Initial configuration
    configuration.plus_z(p_state)
    
    ### Output folder
    global energy_log_folder
    directory_path = energy_log_folder + "/" + energy_log_tag
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    energy_log_folder = directory_path
    
    global energy_file
    energy_file = energy_log_folder + "/" + energy_log_tag + "_Image-00_Energy-archive.txt"
    
    ### Output settings
    llg.set_output_general(p_state, True)
    llg.set_output_folder(p_state, energy_log_folder)
    llg.set_output_energy(p_state, False)
    llg.set_output_tag(p_state, energy_log_tag)
    
    ### Iterations settings
    llg.set_iterations(p_state, n_iterations, n_iterations_log)


"""A function to compute the average of every n consecutive data points"""
"""Helper Function"""
def average_energy_consecutive_data(iterations, e_tot_values, n):
    avg_iterations = []
    avg_e_tot = []
    for i in range(0, len(iterations) - n + 1, n):
        avg_iterations.append(np.mean(iterations[i:i+n]))
        avg_e_tot.append(np.mean(e_tot_values[i:i+n]))
    return np.array(avg_iterations), np.array(avg_e_tot)



"""A function to check the convergence of the standard deviation of energy"""
"""Helper Function"""
def check_convergence_std(e_tot, window_size=20, std_threshold=0.01):
    moving_avg = np.convolve(e_tot, np.ones(window_size)/window_size, mode='valid')
    moving_std = np.array([np.std(e_tot[i:i+window_size]) for i in range(len(e_tot)-window_size+1)])
    
    for i in range(len(moving_std)):
        if moving_std[i] < std_threshold:
            return True, i + window_size  # Return True and the iteration index where convergence starts
    return False, None


"""A function to check the convergence of the average value of energy"""
"""Helper Function"""
def check_convergence_avg(e_tot, window_size=20, avg_threshold=0.01):
    # Calculate the moving average
    moving_avg = np.convolve(e_tot, np.ones(window_size)/window_size, mode='valid')
    
    # Check for convergence based on moving average differences
    for i in range(1, len(moving_avg)):
        if abs(moving_avg[i] - moving_avg[i-1]) < avg_threshold:
            return True, i + window_size  # Return True and the iteration index where convergence starts
    return False, None




"""A function to extract and visualize the energy data from the log file"""
"""Usage Function"""
def energy_visualization(energy_file, num_average = 1, window_size=20, std_threshold=0.1):
    iterations = []
    e_tot_values = []
    
    with open(energy_file, 'r') as file:
        for line in file:
            match = re.match(r'^\s*(\d+)\s*\|\|\s*(-?\d+\.\d+)', line)
            if match:
                iterations.append(int(match.group(1)))
                e_tot_values.append(float(match.group(2)))
    
    avg_iterations, avg_e_tot = average_energy_consecutive_data(iterations, e_tot_values, num_average)
    
    converged, convergence_iteration = check_convergence_avg(e_tot_values, window_size, std_threshold)
    
    if converged:
        print(f"Convergence detected at iteration {iterations[convergence_iteration]}")
    else:
        print("No convergence detected within the given parameters")
    
    plt.figure(figsize=(10, 6))
    plt.plot(avg_iterations, avg_e_tot, marker='o', linestyle='-', color='b')
    plt.title('Iteration vs E_tot')
    plt.xlabel('Iteration')
    plt.ylabel('E_tot')
    plt.grid(True)
    plt.show()
    
    
"""A function to read the energy data from the file"""
"""Usage Function"""
def energy_reader(file_path):
    with open(file_path, 'r') as file:
        iterations = []
        e_tot_values = []
        for line in file:
            match = re.match(r'^\s*(\d+)\s*\|\|\s*(-?\d+\.\d+)', line)
            if match:
                iterations.append(int(match.group(1)))
                e_tot_values.append(float(match.group(2)))
    return iterations, e_tot_values    
    


"""
Magnetization
====================

This module contains functions to set up the magnetization test parameters,
"""

"""A function to calculate the magnitude of the magnetization"""
"""Helper Function"""
def magnetization_magnitude(p_state):
    return np.linalg.norm(quantities.get_magnetization(p_state))


"""A function that calculates the average of magnetization after convergence at certain temperature"""
"""Helper Function"""
def magnetization_average_after_convergence(p_state, temperature , convergence_iterations, total_iteration, n_iterations_log):
    
    ### Log settings
    llg.set_output_general(p_state, False)
    llg.set_iterations(p_state, convergence_iterations, convergence_iterations)
    llg.set_temperature(p_state, temperature)
    
    ### Start simulation until convergence
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB)
    
    ### After convergence, calculate the average magnetization
    llg.set_iterations(p_state, n_iterations_log, total_iteration)
    
    iterations_left = total_iteration - convergence_iterations
    num_batches = int(iterations_left / n_iterations_log)
    
    batch_magnetization = []
    
    for _ in range(num_batches):
        simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB)
        batch_magnetization.append(magnetization_magnitude(p_state))
        
    return np.mean(batch_magnetization)
    
    
"""A function to collect and visualize the warm up magnetization data"""
"""Usage Function"""
def magnetization_visualization_warmup(p_state, start_temp, end_temp, 
                                    num_temperature_points, convergence_iterations, 
                                    total_iteration, n_iterations_log):
    ### Warming up temperature scale
    T_range = np.arange(start_temp, end_temp, (end_temp - start_temp) / num_temperature_points)
    
    magnetization = []
    
    ### Get the magnetization data for each temperature 
    for temp in T_range:
        mag = magnetization_average_after_convergence(p_state, temp, convergence_iterations, total_iteration, n_iterations_log)
        magnetization.append(mag)
    
    ### Plot the magnetization data for each temperature
    plt.figure(figsize=(10, 6))
    plt.plot(T_range, magnetization, marker='o', linestyle='-', color='b')
    plt.title('Temperature vs Magnetization (Warm up)')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.grid(True)
    plt.show()
    
    ### Save the magnetization data and temperature data
    
    ###File name
    cell = geometry.get_n_cells(p_state) #An array of shape(3)
    cell_str = '_'.join([str(c) for c in cell])
    
    anisotropy_info = hamiltonian.get_anisotropy(p_state) #Tuple
    anisotropy = anisotropy_info[0]
    anisotropy_str = "K="+str(anisotropy)+'meV'
    
    file_name = cell_str + "_" + anisotropy_str + "_warmup_Magnetization.pkl"
    
    magnetization_file_path = os.path.join(magnetization_log_folder, file_name)
    
    list_dict = {"Temperature": T_range, "Magnetization": magnetization}
    
    with open(magnetization_file_path, 'wb') as file:
        pickle.dump(list_dict, file)
    

"""A function to collect and visualize the cool down magnetization data"""
"""Usage Function"""
def magnetization_visualization_cooldown(p_state, start_temp, end_temp, 
                                        num_temperature_points, convergence_iterations, 
                                        total_iteration, n_iterations_log):
    ### Warming up temperature scale
    T_range = np.arange(start_temp, end_temp, (end_temp - start_temp) / num_temperature_points)
    
    magnetization = []
    
    ### Get the magnetization data for each temperature 
    for temp in T_range:
        mag = magnetization_average_after_convergence(p_state, temp, convergence_iterations, total_iteration, n_iterations_log)
        magnetization.append(mag)
    
    ### Plot the magnetization data for each temperature
    plt.figure(figsize=(10, 6))
    plt.plot(T_range, magnetization, marker='o', linestyle='-', color='b')
    plt.title('Temperature vs Magnetization (Cool down)')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.grid(True)
    plt.show()
    
    ### Save the magnetization data and temperature data
    
    ###File name
    cell = geometry.get_n_cells(p_state) #An array of shape(3)
    cell_str = '_'.join([str(c) for c in cell])
    
    anisotropy_info = hamiltonian.get_anisotropy(p_state) #Tuple
    anisotropy = anisotropy_info[0]
    anisotropy_str = "K="+str(anisotropy)+'meV'
    
    magnetic_field_info = hamiltonian.get_field(p_state) #Tuple
    magnetic_field = magnetic_field_info[0]
    magnetic_field_str = "B="+str(magnetic_field)+'T'
    
    file_name = cell_str + "_" + anisotropy_str + "_" + magnetic_field_str + "_cooldown_Magnetization.pkl"
    
    magnetization_file_path = os.path.join(magnetization_log_folder, file_name)
    
    list_dict = {"Temperature": T_range, "Magnetization": magnetization}
    
    with open(magnetization_file_path, 'wb') as file:
        pickle.dump(list_dict, file)
    

"""A function to read the magnetization data from the file"""
"""Usage Function"""
def magnetization_reader(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    T_range = data["Temperature"]
    magnetization = data["Magnetization"]
    
    return T_range, magnetization





"""
Spin
====================
This module contains functions to access the spin data from the state object and visualize the spin data.
This module only works for 2D square lattice.
"""



spin_general_log_folder = "C:/Users/hao14/Desktop/Simulation/Spin Log"


"""A function to run the simulation and save the average spin data"""
"""Helper Function"""
def spin_average_after_convergence(p_state, temperature, convergence_iterations, total_iteration, n_iterations_log):
    ### Log settings 
    llg.set_output_general(p_state, False)
    llg.set_iterations(p_state, convergence_iterations, convergence_iterations)
    llg.set_temperature(p_state, temperature)
    
    ### Start simulation until convergence
    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB)
    
    ### After convergence, calculate the average spin
    llg.set_iterations(p_state, n_iterations_log, total_iteration)
    
    iterations_left = total_iteration - convergence_iterations
    num_batches = int(iterations_left / n_iterations_log)
    
    batch_spin = []
    
    for _ in range(num_batches):
        simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_SIB)
        spin_directions = system.get_spin_directions(p_state) ### A list of shape (n_spins, 3)
        batch_spin.append(spin_directions)
        
    ### batch_spin has shape (num_batches, n_spins, 3)
    ### get a list of shape (n_spins, 3) by averaging over the batches
    
    batch_spin_array = np.array(batch_spin)
    average = np.mean(batch_spin_array, axis=0)
    
    return average.tolist()



"""A function to collect the warm up spin data"""
"""Usage Function"""
def spin_warmup(p_state, start_temp, end_temp, num_temperature_points, convergence_iterations, total_iteration, n_iterations_log):
    ### Warming up temperature scale
    T_range = np.arange(start_temp, end_temp, (end_temp - start_temp) / num_temperature_points)
    
    spin_data = []
    
    ### Get the spin data for each temperature 
    for temp in T_range:
        spin = spin_average_after_convergence(p_state, temp, convergence_iterations, total_iteration, n_iterations_log)
        spin_data.append(spin)
    
    ### Save the spin data and temperature data
    
    ###File name
    cell = geometry.get_n_cells(p_state) #An array of shape(3)
    cell_str = '_'.join([str(c) for c in cell])
    
    anisotropy_info = hamiltonian.get_anisotropy(p_state) #Tuple
    anisotropy = anisotropy_info[0]
    anisotropy_str = "K="+str(anisotropy)+'meV'
    
    file_name = cell_str + "_" + anisotropy_str + "_warmup_Spin.pkl"
    
    spin_file_path = os.path.join(spin_general_log_folder, file_name)
    
    spin_dict = {"Temperature": T_range, "Spin": spin_data}
    
    with open(spin_file_path, 'wb') as file:
        pickle.dump(spin_dict, file)
        


"""A function to visualize spin data"""
"""Usage Function"""
def spin_total_visualization(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    T_range = data["Temperature"]
    spin_data = data["Spin"]
    
    spin_data = np.array(spin_data)
    
    # for i in range(len(T_range)):
    #     fig = plt.figure(figsize=(10, 6))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     ax.quiver(0, 0, 0, spin_data[i][:, 0], spin_data[i][:, 1], spin_data[i][:, 2])
        
    #     ax.set_xlim([-1, 1])
    #     ax.set_ylim([-1, 1])
    #     ax.set_zlim([-1, 1])
        
    #     ax.set_title(f"Temperature = {T_range[i]}")
        
    #     plt.show()
    
    ### Plot the spin data for each temperature Sx, Sy, Sz
    ### Make a scatter of the spin data for each temperature Sx, Sy, Sz
    ### The y-axis is the amplitude of the spin data    
    ### The x-axis is all the individual spins
    
    for i in range(len(T_range)):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].scatter(np.arange(len(spin_data[i])), spin_data[i][:, 0], color='b', alpha=0.7)
        ax[0].set_title(f"Temperature = {T_range[i]} K")
        ax[0].set_xlabel('Spin Index')
        ax[0].set_ylabel('<Sx>')
        
        ax[1].scatter(np.arange(len(spin_data[i])), spin_data[i][:, 1], color='g', alpha=0.7)
        ax[1].set_title(f"Temperature = {T_range[i]} K")
        ax[1].set_xlabel('Spin Index')
        ax[1].set_ylabel('<Sy>')
        
        ax[2].scatter(np.arange(len(spin_data[i])), spin_data[i][:, 2], color='r', alpha=0.7)
        ax[2].set_title(f"Temperature = {T_range[i]} K")
        ax[2].set_xlabel('Spin Index')
        ax[2].set_ylabel('<Sz>')
        
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
        
        plt.show()
        
        
    return T_range, spin_data





"""A function to transform the linear spin data to 2D spin data at a certain temperature"""
"""Usage Function"""
def spin_2D_transform(file_path, temperature):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    T_range = data["Temperature"]
    spin_data = data["Spin"]
    
    indices = np.where(T_range == temperature)
    index = indices[0][0]
    
    spin_original = spin_data[index]   ### An array of shape (n_spins, 3) 
    spin_original_array = np.array(spin_original) ### 1D array spin
    num_of_spins = spin_original_array.shape[0]
    
    ### length of the grid
    length = int(np.sqrt(num_of_spins))
    
    ### Transform the spin data to 2D
    ### The original spin data orders by x index increasing first, then y index increasing, then z index increasing
    
    reshaped_spin_array = spin_original_array.reshape(length, length, 3)
    result_spin_array = np.zeros_like(reshaped_spin_array) ### Each item is a spin chain in the y direction
    
    for i in range(length):
        for j in range(length):
            result_spin_array[i, j] = reshaped_spin_array[j, i]
    
    dict_spin = {"Temperature": temperature, "Spin": result_spin_array}
    
    return dict_spin



"""A function to visualize the y-axis spin data chain""" ### Input column should be a list of indices of the columns
"""Usage Function"""
def spin_y_visualization(dict_spin, column=None):
    temperature = dict_spin["Temperature"]
    spin_data = dict_spin["Spin"]
    
    ### If a column is specified, plot the spin data chain for column
    ### Otherwise, plot the spin data chain for all columns
    ### Note: the column index is 1 lager than the actual index (Being discussed)
    
    if column is not None:
        for i in column:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].scatter(np.arange(spin_data.shape[0]), spin_data[i][:,0], color='b', alpha=0.7)
            ###ax[0].plot(np.arange(spin_data.shape[0]), spin_data[i-1][:,0], color='b', alpha=0.7)
            ax[0].set_title(f"Temperature = {temperature} K, Column = {i}")
            ax[0].set_xlabel('Spin Index')
            ax[0].set_ylabel('<Sx>')
            
            ax[1].scatter(np.arange(spin_data.shape[0]), spin_data[i][:,1], color='g', alpha=0.7)
            ax[1].set_title(f"Temperature = {temperature} K, Column = {i}")
            ax[1].set_xlabel('Spin Index')
            ax[1].set_ylabel('<Sy>')
            
            ax[2].scatter(np.arange(spin_data.shape[0]), spin_data[i][:,2], color='r', alpha=0.7)
            ax[2].set_title(f"Temperature = {temperature} K, Column = {i}")
            ax[2].set_xlabel('Spin Index')
            ax[2].set_ylabel('<Sz>')
            
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
        
        plt.show()
        
    else:
        for i in range(spin_data.shape[0]):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            
            ax[0].scatter(np.arange(spin_data.shape[0]), spin_data[i][:,0], color='b', alpha=0.7)
            ax[0].set_title(f"Temperature = {temperature} K, Column = {i+1}")
            ax[0].set_xlabel('Spin Index')
            ax[0].set_ylabel('<Sx>')
            
            ax[1].scatter(np.arange(spin_data.shape[0]), spin_data[i][:,1], color='g', alpha=0.7)
            ax[1].set_title(f"Temperature = {temperature} K, Column = {i+1}")
            ax[1].set_xlabel('Spin Index')
            ax[1].set_ylabel('<Sy>')
            
            ax[2].scatter(np.arange(spin_data.shape[0]), spin_data[i][:,2], color='r', alpha=0.7)
            ax[2].set_title(f"Temperature = {temperature} K, Column = {i+1}")
            ax[2].set_xlabel('Spin Index')
            ax[2].set_ylabel('<Sz>')
            
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
        
        plt.show()
    
    

"""A function to visualize given rows of a specific column of the spin data chain in y direction"""
"""Usage Function"""
def spin_y_visualization_rows(dict_spin, dict_position, same_y_axis = False): 
    ### dict_position is a dictionary with keys as column index and values as a list of row indices
    ### Column is a number, Row is a list of indices
    ### First item in row list is the first row, second item is the final row
    ### dict_position = {1: [1, 2], 2: [1, 2], 3: [1, 2]}
    ### Key is the column index, value is a list of row indices
    ### Note: Both column and row index is 1 lager than the actual index (Being discussed, not implemented yet)
    
    temperature = dict_spin["Temperature"]
    spin_data = dict_spin["Spin"]
    
    columns = dict_position.keys()
    
    for column in columns:
        row = dict_position[column] ### Example, dict_position={1: [1, 2]}, column = 1, row = dict_position[1] = [1, 2]
    
    
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].scatter(np.arange(row[1]-row[0]+1), spin_data[column][row[0] : row[1]+1, 0], color='b', alpha=0.7)
        ### ax[0].plot(np.arange(row[1]-row[0]+1), spin_data[column-1][row[0]-1 : row[1], 0], color='b', alpha=0.7)
        ### row[0] is the first row, row[1] is the second row
        ax[0].plot(np.arange(row[1]-row[0]+1), spin_data[column][row[0] : row[1]+1, 0], color='b', alpha=0.7)
        ax[0].set_title(f"Temperature = {temperature} K, Column = {column}, Row = {row[0]}-{row[1]}")
        ax[0].set_xlabel('Spin Index')
        ax[0].set_ylabel('<Sx>')
        if same_y_axis:
            ax[0].set_ylim(-1, 1)
        
        ax[1].scatter(np.arange(row[1]-row[0]+1), spin_data[column][row[0] : row[1]+1, 1], color='g', alpha=0.7)
        ax[1].plot(np.arange(row[1]-row[0]+1), spin_data[column][row[0] : row[1]+1, 1], color='g', alpha=0.7)
        ax[1].set_title(f"Temperature = {temperature} K, Column = {column}, Row = {row[0]}-{row[1]}")
        ax[1].set_xlabel('Spin Index')
        ax[1].set_ylabel('<Sy>')
        if same_y_axis:
            ax[1].set_ylim(-1, 1)
        
        ax[2].scatter(np.arange(row[1]-row[0]+1), spin_data[column][row[0] : row[1]+1, 2], color='r', alpha=0.7)
        ax[2].plot(np.arange(row[1]-row[0]+1), spin_data[column][row[0] : row[1]+1, 2], color='r', alpha=0.7)
        ax[2].set_title(f"Temperature = {temperature} K, Column = {column}, Row = {row[0]}-{row[1]}")
        ax[2].set_xlabel('Spin Index')
        ax[2].set_ylabel('<Sz>')
        if same_y_axis:
            ax[2].set_ylim(-1, 1)
        
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
    
    plt.show()
    


"""A function to visualize given rows of a specific column of the spin data chain in x direction"""
"""Usage Function"""
def spin_x_visualization_columns(dict_spin, dict_position, same_y_axis = False): 
    ### dict_position is a dictionary with keys as row index and values as a list of column indices
    ### Row is a number, Column is a list of indices
    ### First item in column list is the first column, second item is the final column
    ### dict_position = {1: [1, 2], 2: [1, 2], 3: [1, 2]}
    ### Key is the row index, value is a list of column  indices
    ### Note: Both column and row index is 1 lager than the actual index (Being discussed, not implemented yet)
    
    temperature = dict_spin["Temperature"]
    spin_data = dict_spin["Spin"]
    
    rows = dict_position.keys()
    
    for row in rows:
        column = dict_position[row] ### Example, dict_position={1: [1, 2]}, row = 1, column = dict_position[1] = [1, 2]
    
    
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].scatter(np.arange(column[1]-column[0]+1), spin_data[column[0] : column[1]+1, row, 0], color='b', alpha=0.7)
        ax[0].plot(np.arange(column[1]-column[0]+1), spin_data[column[0] : column[1]+1, row, 0], color='b', alpha=0.7)
        ax[0].set_title(f"Temperature = {temperature} K, Row = {row}, Column = {column[0]}-{column[1]}")
        ax[0].set_xlabel('Spin Index')
        ax[0].set_ylabel('<Sx>')
        if same_y_axis:
            ax[0].set_ylim(-1, 1)
        
        ax[1].scatter(np.arange(column[1]-column[0]+1), spin_data[column[0] : column[1]+1, row, 1], color='g', alpha=0.7)
        ax[1].plot(np.arange(column[1]-column[0]+1), spin_data[column[0] : column[1]+1, row, 1], color='g', alpha=0.7)
        ax[1].set_title(f"Temperature = {temperature} K, Row = {row}, Column = {column[0]}-{column[1]}")
        ax[1].set_xlabel('Spin Index')
        ax[1].set_ylabel('<Sy>')
        if same_y_axis:
            ax[1].set_ylim(-1, 1)
        
        ax[2].scatter(np.arange(column[1]-column[0]+1), spin_data[column[0] : column[1]+1, row, 2], color='r', alpha=0.7)
        ax[2].plot(np.arange(column[1]-column[0]+1), spin_data[column[0] : column[1]+1, row, 2], color='r', alpha=0.7)
        ax[2].set_title(f"Temperature = {temperature} K, Row = {row}, Column = {column[0]}-{column[1]}")
        ax[2].set_xlabel('Spin Index')
        ax[2].set_ylabel('<Sz>')
        if same_y_axis:
            ax[2].set_ylim(-1, 1)
        
        
        
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3)
    
    plt.show()



    
    
"""A function to visualize the whole spin grid for Sz at a certain temperature"""
"""Usage Function"""
def spin_grid_Sz_visualization(dict_spin):
    temperature = dict_spin["Temperature"]
    spin_data = dict_spin["Spin"]
    length = spin_data[0].shape[0]
    
    # Initialize a 100x100 grid to store the results
    grid = np.zeros((length, length))
    
    # Process each item in the 2D array
    for i in range(length):
        for j in range(length):
            x, y, z = spin_data[i, j]
            grid[i, j] = 1 if z > 0 else 0
            
            
    
    # Get coordinates for 1s and 0s
    x_ones, y_ones = [], []
    x_zeros, y_zeros = [], []

    for i in range(length):
        for j in range(length):
            if grid[i, j] == 1:
                x_ones.append(j)
                y_ones.append(i)
            elif grid[i, j] == 0:
                x_zeros.append(j)
                y_zeros.append(i)

    # Plot the grid using scatter
    plt.figure(figsize=(10, 10))
    plt.scatter(y_ones, x_ones, color='blue', label='1', marker='s', s=10)
    plt.scatter(y_zeros, x_zeros, color='red', label='0', marker='s', s=10)
    
    final_grid_length = length + 1
    
    plt.title(f'Spin Grid for Sz at Temperature = {temperature} K')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.gca().invert_yaxis()  # To match the matrix indexing
    plt.grid(True)
    plt.xticks(np.arange(0, final_grid_length, 10))  # Set x-axis ticks at intervals of 10
    plt.yticks(np.arange(0, final_grid_length, 10))  # Set y-axis ticks at intervals of 10
    plt.gca().invert_yaxis()
    plt.show()
    
    return grid


"""A function to find the y-axis transitions in the spin grid"""
"""Usage Function"""
def find_transitions(grid):
    transitions_dict = {}
    for x in range(grid.shape[0]):
        transitions = []
        for y in range(1, grid.shape[1]):
            if grid[x, y] != grid[x, y-1]:
                transitions.append(y)
                #transitions.append(y + 1)  # Adding 1 to the index
        transitions_dict[x] = transitions
    return transitions_dict
