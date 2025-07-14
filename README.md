# Optimality-guarantees-for-crystal-structure-prediction

## Step 1: Use simplified potential to initiate integer programming and generate the first batch of candidate structures. 

### Version 1(Jun/13/2025 by Jian Sun)：Optimize to find least energy model with specific number of atoms. 

Need to specify the data for each atom, including charge, A, pho, c for calculate Buckingham energy, Shannon radius. Need to specify atom pair potential parameters, lattice length, composition(numbers of atoms), and grid divisions.

### Version 2(Jun/23/2025 by Jian Sun): Add symmetry orbits and constraints. Need to specify space group.

Update 1(Jul/8/2025): Add get data part, store paramters for Buckingham Energy calculation(a, pho, c). Update test code for complex structure.

### Version 3(Jul/13/2025 by Jian Sun): Add file lattice_grid.py, parameter_loader.py, potentials.py.

lattice_grid.py: Construct geometry for calculation, suppport 7 kinds of crystal system.

parameter_loader.py: Allow main to extract data from files in data folder, and only select elements provided by user.

potentials.py: Allow multiple kinds of potential energy calculation, auto select between Buckingham & Morse.

## Step 2: Perform DFT calculations on the candidate structures to collect energy and atomic force data.


(End here while calculating small models.)
## Step 3: Train a machine learning potential function using DFT data to replace the original potential function. 
(Note: ml part is not mandatory, but helpful to escape local minima and find more potential second-least energy form, in case of models contain more atoms and are more complex.)

## Step 4: Embed the new potential function into the energy objective function of the integer programming and resolve it again.