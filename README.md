This is the implementation of fully first-order bilevel gradient to replace the non-fully first-order methods in differentiable optimization.
The main comparison is ``qpth`` and ``cvxpylayer``. Our method provides a fully first-order method to differentiate through optimization layers approximately.
It has the advantage in computation and memory efficiency, but in the tradeoff of the accuracy of the gradient.

Please install the required packages:
- torch
- qpth
- cvxpylayer
- gurobi
- cvxpy

To run the code, please use:
```
python main.py --method=ffoqp --eps=0.1 --lr=0.001
python main.py --method=cvxpylayer --eps=0.1 --lr=0.001
python main.py --method=qpth --eps=0.1 --lr=0.001
```


I ran a few simple experiments to compare the performance of the three methods. Each epoch includes 2048 samples (split into training and testing) to run decision-focused leanring (end-to-end learning). The computation results are as follows:

For optimization problems of size 256,
- ffoqp takes roughly 25s per epoch.
- qpth takes roughly 23s per epoch.
- cvxpylayer takes roughly 57s per epoch.

For optimization problems of size 512,
- ffoqp takes roughly 62s per epoch.
- qpth takes roughly 132s per epoch.
- cvxpylayer takes roughly 231s per epoch.


The computation improvement can only be seen in larger instances of the optimization problem. This is likely due to the dominance of the overhead in the forward process to solve the optimization problem. Only when the size goes larger, the backward process of the optimization problem becomes more expensive than the forward process. In smaller instances, the overhead in the forward process dominates the computation time, and thus the improvement in the backward process is not significant enough to be observed.
qpth also uses a faster algorithm to parallelize and implement the forward pass, which is why qpth is faster than cvxpylayer (in theory they are the same).
