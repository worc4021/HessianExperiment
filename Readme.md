# Experiment summary

When implementing approximations for Hessians I observed that I wasn't getting anywhere near the performance of the inbuilt ipopt lbfgs method. This repository attempts to visualise the performance producible by naive implementations of the various methods.

The results below all illustrate the [Goldstein Price](https://www.sfu.ca/~ssurjano/goldpr.html) test function in the range -2 <= x <= 2 and -2<=y<=2.

# Some results

For each approximation we show an illustration of the number of iterations for each initial guess. As well as a classification of which point the solver converged to.

The following colour scheme applies:
- <p style="color:#0072BD">The global minimum (0,-1) with function value 3.</p>
- <p style="color:#D95319">The local minimum (-0.6,-0.4) with function value 30.</p>
- <p style="color:#EDB120">The local minimum (1.8,0.2) with function value 84.</p>
- <p style="color:#7E2F8E">The local minimum (1.2,0.8) with function value 840.</p>
- <p style="color:#77AC30">Failed solves - iteration count exceeded etc.</p>

## Ipopt's lbfgs

![Ipopt lbfgs attained solutions](/img/ipopt_lbfgs_solution.png)*The region of attraction for ipopt's lbfgs as a benchmark.*

![Ipopt lbfgs iteration count](/img/ipopt_lbfgs_iterations.png)*Vast majority of points solves very quickly, when initialised too close to minimum the bfgs gets lost.*

The respective basins of attraction are large and compared with the other methods hardly any off shoots. The rate of convergence is hard to beat.

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| MAXITER_EXCEEDED | 0.65787 | 3000 |
| STOP_AT_ACCEPTABLE_POINT | 0.31616 | 29 |
| SUCCESS | 99.026 | 15 |

## Automatic differentiation

![Autodiff attained solutions](/img/autodiff_solution.png)*Critical point converged to over initial value, green is a failed point.*

![Autodiff with exact hessian as a benchmark, number of iterations](/img/autodiff_iterations.png)*Number of iterations over initial value.*

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| MAXITER_EXCEEDED | 10.2737 | 3000 |
| STOP_AT_ACCEPTABLE_POINT | 15.5431 | 76 |
| SUCCESS | 74.1832 | 54 |

Despite supplying the exact hessian, the problem does not perform anywhere near the ipopt limited memory BFGS approach. The basins of attraction are _mostly_ connected


## BFGS

![BFGS attained solutions](/img/bfgs_solution.png)*Various disjoint basins of attraction for the multiple optima*

![BFGS iteration count](/img/bfgs_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION | 4.8766 | 3 |
| STOP_AT_ACCEPTABLE_POINT | 2.3984 | 25 |
| SUCCESS | 92.7251 | 19 |

The basins of attraction here are a mess, small variations in initial guess likely lead to a different solution in this problem. The pictures look kind of cool though.


## SR1

![SR1 attained solutions](/img/sr1_solution.png)

![SR1 iteration count](/img/sr1_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION | 3.9951 | 0 |
| STOP_AT_ACCEPTABLE_POINT | 1.6 | 18 |
| STOP_AT_TINY_STEP | 0.083033 | 21 |
| SUCCESS | 94.3218 | 14 |

## DFP

![DFP attained solutions](/img/dfp_solution.png)

![DFP iteration count](/img/dfp_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION | 6.0039 | 11 |
| MAXITER_EXCEEDED | 1.501 | 3000 |
| STOP_AT_ACCEPTABLE_POINT | 3.9536 | 65 |
| SUCCESS | 88.5415 | 50 |

## Limited-Memory BFGS

![LBFGS attained solutions](/img/lbfgs_solution.png)

![LBFGS iteration count](/img/lbfgs_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION | 1.9608 | 16 |
| MAXITER_EXCEEDED | 0.0095807 | 3000 |
| STOP_AT_ACCEPTABLE_POINT | 4.4518 | 26 |
| SUCCESS | 93.5777 | 17 |

## Limited-Memory SR1

![LSR1 attained solutions](/img/lsr1_solution.png)

![LSR1 iteration count](/img/lsr1_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION | 1.4403 | 13 |
| STOP_AT_ACCEPTABLE_POINT | 1.5936 | 19 |
| STOP_AT_TINY_STEP | 0.067065 | 21 |
| SUCCESS | 96.8991 | 14 |

## Limited-Memory DFP

![LDFP attained solutions](/img/ldfp_solution.png)

![LDFP iteration count](/img/ldfp_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION | 2.4079 | 18 |
| MAXITER_EXCEEDED | 0.012774 | 3000 |
| STOP_AT_ACCEPTABLE_POINT | 3.7812 | 29 |
| SUCCESS | 93.7981 | 20 |