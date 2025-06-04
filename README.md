# Simulation Environment for GreCCo EC control

This python framework offers a testbed for energy community control within in the GreCCo project.

Its core is a simulator.

Starting a simulation is done using the script
grecco_sim/scripts/start_simulation.py

# Documentation

Generate the Documentation using `sphinx-build doc _build`.
If in doubt about style:

https://google.github.io/styleguide/pyguide.html


## Installation

Create a python environment using the shipped requirements.txt
```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

In general, it should be sufficient to clone the project and execute the start script.

There is sample simulation data in the `data` folder 




### Gurobi (Academic licence) - may be needed for on/off flexibilities

DO NOT INSTALL CASADI FROM CONDA CHANNELS!
Instead, use pip to install casadi. This way, solver binaries are included.

Install Gurobi on your system.
Some environment variables have to be set. See them here:

https://github.com/casadi/casadi/wiki/FAQ%3A-how-to-get-third-party-solvers-to-work%3F

Hint: the LD_LIBRARY_PATH must be set *before* starting the python script.
In pycharm this can be achieved using the run configurations.
