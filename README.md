# Link
Hello everyone, please refer to this [link](https://drive.google.com/file/d/1lKIHMaaIsfRNZ7yES82_02Rk1B9-WIXC/view?usp=sharing) for a detailed description of my work :)  
Please refer to this [link](https://1drv.ms/p/c/aa1d360d744c3259/EWvUJgMmNIRApNO0yVku-U0Boyw3FFoagYQ5U2HESM_3PA?e=FHG5W8) for a detailed visualization.  
Please refer to this [link](https://drive.google.com/drive/folders/1HZoNb6WUklaA5iSivsKnnZN-_pv9sPwQ?usp=sharing) for a more detailed simulation package.


# Main Goal
The main goal is to use a publicity-accessible spin simulation code called *Spirit* ([Müller *et al*](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.224414)) to simulate the results obtained from the paper *Observation of a phase transition within the domain walls of ferromagnetic Co<sub>3</sub>Sn<sub>2</sub>S<sub>2</sub>* ([Lee *et al*](https://www.nature.com/articles/s41467-022-30460-y)). In this paper, Lee *et al* suggested that "what sets this compound apart is the giant value of its dimensionless anisotropy factor, *K*", I want to use the code to reproduce the observation that spin transforms **Bloch Wall** to **Linear Wall** and verify if it is truly caused by giant value of dimensionless anisotropy factor. (●'◡'●)

I'm able to see a Bloch Wall

![image](https://github.com/user-attachments/assets/acf06419-80a5-412e-bdd4-5ead51bf306d) ![image](https://github.com/user-attachments/assets/f5aafe6f-5ed1-47c2-bca2-2ef9b602f3ba)

And Yes Sir! Linear Wall is also be observed
![linear_domain_wall](https://github.com/user-attachments/assets/5e271bfb-f392-4f13-8277-aad9718e2044)

![DancingDogGIF](https://github.com/user-attachments/assets/dad0b02c-72d9-4526-b841-2c8f42119f0c)

Stay Tuned for a comprehensive analysis of phase transition


# Hamiltonian
In the absence of an applied magnetic field, ferromagnetic materials can be described by the following Hamiltonian:

![image](https://github.com/user-attachments/assets/36df5be3-b9f5-4766-b5b7-791ec8589b76)


Here, **$\mathbf{s}$** means the individual spin of unit length, **$i$** and **$j$** are lattice sites. The Hamiltonian form includes:
1. The single-ion magnetic anisotropy, where **$\hat{K}_{j}$** are the axes of the uniaxial anisotropies of the basis cell with the anisotropy strength **$K_{j}$**.
2. The symmetric exchange interaction, where **$J_{ij}$** denotes the symmetric exchange energy and **$\langle ij \rangle$** denotes the unique pairs of interacting spins **$i$** and **$j$**.
3. The dipolar interaction, where **$\mu_{0}$** is vacuum permeability, **$\mu_{i}$** is dipole moment, and **$\hat{r}_{ij}$** denotes the unit vector of the bond connecting two spins.



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



