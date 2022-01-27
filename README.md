# InvariantAncestrySearch
This repository contains python code necessary to replicated the experiments performed in our paper "Invariant Ancestry Search".

## Structure of the repository
The repository is structured in the following manner:
* In the folder `/InvariantAncestrySearch` there are two important files:
  * `utils.py` contains a class `DataGenerator` which we use for sampling SCMs and data from said sampled SCMs. This, can for instance be done by the sequence
  ```{python}
  from InvariantAncestrySearch import DataGenerator
  
  SCM1 = DataGenerator(d = 10, N_interventions = 5, p_conn = 2 / 10, InterventionStrength = 1) # This is an SCM generator
  SCM1.SampleDAG()  # Generates a DAG with d = 10 predictor nodes, 5 interventions and roughly d + 1 edges between the (d + 1)-sized subgraph of (X, Y)
  SCM1.BuildCoefMatrix  # Samples coefficients for the linear assignments -- interventions have strength 1
  data1 = SCM1.MakeData(100)  # Generates 100 samples from SCM1
  
  SCM2 = DataGenerator(d = 6, N_interventions = 1, p_conn = 2 / 6, InterventionStrength = 0.5) # And this is also an SCM generator
  SCM2.SampleDAG()  # Generates a DAG with d = 6 predictor nodes, 1 intervention and roughly d + 1 edges between the (d + 1)-sized subgraph of (X, Y)
  SCM2.BuildCoefMatrix  # Samples coefficients for the linear assignments -- interventions have strength 1
  data2 = SCM2.MakeData(1000)  # Generates 1000 samples from SCM2
  ```
  * `IASfunctions.py` includes all relevant functions used in the scripts, e.g., to test for minimal invariance or compute the set of all minimally invariant sets. All functions are documentated.
* In the folder `/simulation_scripts` there are scripts to reproduce all experiments performed in the paper. These too documentation inside them. The functions run out-of-the-box, if all necessary libraries are installed and do not need to be run in a certain order.
* In the folder `/output/` there are database files, saved from running the scripts in `/simulation_scripts/`. These contain the data used to make all figures in the paper and can be opened with the python library `shelve`.
* The file `requirements.txt` contains info on which modules are required to run the code. Note also that an R installation is required as well as the R package `dagitty`
