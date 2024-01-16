Reaction-Diffusion PDEs Solver
========================

This module solves the Reaction-Diffusion PDEs for one or more species in one or two-dimensional domain. The ``Crank-Nicolson`` finite difference schema is implemented, and Periodic, Dirichlet, and Neumann boundary conditions are supported.

How to Install
-----------------------
After downloading the project zip file and extracting its folder or cloning the project by ``git``, go to its directory and install it by executing the following command line

``pip install .``

(Ensure you don't forget the dot at the end of the command). After that, you can import the module into your Python environment, like

``from solvers.rd import *``


Where to start
-----------------------

- Check the [HowTo.ipynb](HowTo.ipynb) notebook to see an example of a Brusselator model solution.

- The [Circuit_3954.ipynb](Circuit_3954.ipynb) explains the PDE of the three-node gene circuit, its topology, and how to load the sample parameters. Similar to the Brusselator in the [HowTo.ipynb](HowTo.ipynb) example, it provides the kinetic function for integration, but you need to write the rest yourself. Not that you need the "Pandas" to be able to load the sampled parameters.
  
- The source code is in the _solvers_ folder if you need to delve into more details of how the solver works. 
