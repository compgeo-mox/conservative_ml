# Deep learning based reduced order modeling of Darcy flow systems with local mass conservation
### Wietse M. Boon, Nicola R. Franco, Alessio Fumagalli and Paolo Zunino

The [examples](./examples/) folder contains the source code for replicating the three test cases. See [arXiv pre-print](https://arxiv.org/abs/2311.14554).

# Abstract
We propose a new reduced order modeling strategy for tackling parametrized Partial Differential Equations (PDEs) with linear constraints, 
in particular Darcy flow systems in which the constraint is given by mass conservation. Our approach employs classical neural network 
architectures and supervised learning, but it is constructed in such a way that the resulting Reduced Order Model (ROM) is guaranteed to 
satisfy the linear constraints exactly. The procedure is based on a
splitting of the PDE solution into a particular solution satisfying the constraint and a homogenous solution. The homogeneous solution is 
approximated by mapping a suitable potential function, generated by a neural network model, onto the kernel of the constraint operator;
for the particular solution, instead, we propose an efficient spanning tree algorithm.
Starting from this paradigm, we present three approaches that follow this methodology, obtained by exploring different choices of the 
potential spaces: from empirical ones, derived via Proper Orthogonal Decomposition (POD), to more abstract ones based on differential complexes.
All proposed approaches combine computational efficiency with rigorous mathematical interpretation, thus guaranteeing the 
explainability of the model outputs. To demonstrate the efficacy of the proposed strategies and to emphasize their advantages over vanilla 
black-box approaches, we present a series of numerical experiments on fluid flows in porous media, ranging from mixed-dimensional problems to nonlinear systems. 
This research lays the foundation for further exploration and development in the realm of model order reduction, potentially unlocking new capabilities and solutions in computational geosciences and beyond.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv pre-print](https://arxiv.org/abs/2311.14554).

# PorePy and PyGeoN version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and [PyGeoN](https://github.com/compgeo-mox/pygeon) and might revert them.
Newer versions of may not be compatible with this repository.<br>
PorePy valid commit: XXXX <br>
PyGeoN valid tag: XXXX

# License
See [license](./LICENSE).
