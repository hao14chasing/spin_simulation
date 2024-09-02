# Link
Hello everyone, please refer to this [link](https://drive.google.com/drive/folders/1Zpf90ZBy9BOea7TSxAk99REi_fS5m7G9?usp=sharing) for a more detailed simulation package :)

# Main Goal
The main goal is to use a publicity-accessible spin simulation code called *Spirit* to simulate the results obtained from the paper *Observation of a phase transition within the domain walls of ferromagnetic Co<sub>3</sub>Sn<sub>2</sub>S<sub>2</sub>*. In this paper, Lee suggested that "what sets this compound apart is the giant value of its dimensionless anisotropy factor, *K*", I want to use the code to reproduce the observation that spin transforms **Bloch Wall** to **Linear Wall** and verify if it is truly caused by giant value of dimensionless anisotropy factor. (●'◡'●)
![DancingDogGIF](https://github.com/user-attachments/assets/dad0b02c-72d9-4526-b841-2c8f42119f0c)


# Simulation
## Main Folder
The main directory contains the core material used for simulation, including useful functions, documentation, and an overall usage example. **Updates are still underway. Stay Tuned!**

## Logging Folder
The logging folder contains the Spin Log, Magnetization Log, and Energy Log to be accessed for future reference.

## Analysis Folder
This folder is a thorough analysis of the result of some parameters used for simulation. The main example folder is
### J=6.79, K=6.7 Analysis
By setting Anisotropy and Exchange energy both to 6.79 meV, I see clear information of **Bloch Wall** 😃
### 100_100_2 Lattice
If dipole-dipole interaction is taken into consideration, then we should have more spins(i.e. layers of atoms) for a more accurate simulation. However, this simulation takes a very long long time(even just testing the curie temperature takes more than one day to run😒), so the folder is still under updating. **Stay Tuned!**

## Reference Folder
This folder contains necessary papers that could be useful for simulation. It includes example simulation, simulation theory, and *Spirit* documentations, etc.

## Visualization Folder
This folder contains the overall spin structure with a 3D image, provided by *Spirit* built-in visualization tool. Example folder: Warmup Graph, Field Cool Graph, etc.

## Testing Folder
The testing folder contains unnecessary packages and code that I used only to understand the code better.



