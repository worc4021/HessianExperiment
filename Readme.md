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
| MAXITER_EXCEEDED|0.66745|3000 |
| STOP_AT_ACCEPTABLE_POINT|0.14371|32 |
| SUCCESS|99.1888|16 |

## Automatic differentiation

![Autodiff attained solutions](/img/autodiff_solution.png)*Critical point converged to over initial value, green is a failed point.*

![Autodiff with exact hessian as a benchmark, number of iterations](/img/autodiff_iterations.png)*Number of iterations over initial value.*

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| MAXITER_EXCEEDED|64.181|3000 |
| STOP_AT_ACCEPTABLE_POINT|33.7783|531 |
| SUCCESS|2.0407|25 |

Despite supplying the exact hessian, the problem does not perform anywhere near the ipopt limited memory BFGS approach. The basins of attraction are _mostly_ connected


## BFGS

![BFGS attained solutions](/img/bfgs_solution.png)*Various disjoint basins of attraction for the multiple optima*

![BFGS iteration count](/img/bfgs_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION|11.5735|369 |
| MAXITER_EXCEEDED|52.2243|3000 |
| STOP_AT_ACCEPTABLE_POINT|31.9931|426 |
| STOP_AT_TINY_STEP|2.6475|65 |
| SUCCESS|1.5617|23 |

The basins of attraction here are a mess, small variations in initial guess likely lead to a different solution in this problem. The pictures look kind of cool though.


## SR1

![SR1 attained solutions](/img/sr1_solution.png)

![SR1 iteration count](/img/sr1_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION|7.0929|399 |
| MAXITER_EXCEEDED|59.3811|3000 |
| STOP_AT_ACCEPTABLE_POINT|28.5089|444 |
| STOP_AT_TINY_STEP|3.4203|691 |
| SUCCESS|1.5968|21 |

## DFP

![DFP attained solutions](/img/dfp_solution.png)

![DFP iteration count](/img/dfp_iterations.png)

| Ipopt Status | % of solves | Median iteration count |
| ----------- | ----------- | ----------- |
| ERROR_IN_STEP_COMPUTATION|10.5164|498 |
| MAXITER_EXCEEDED|63.2485|3000 |
| STOP_AT_ACCEPTABLE_POINT|22.5625|419 |
| STOP_AT_TINY_STEP|2.2419|209 |
| SUCCESS|1.4307|27 |