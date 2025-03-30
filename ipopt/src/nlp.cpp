// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-11-05

#include "nlp.hpp"
#include "IpOrigIpoptNLP.hpp"
#include "IpIpoptCalculatedQuantities.hpp"
#include "IpTNLPAdapter.hpp"
#include <cassert>
#include <GoldsteinPrice.hpp>
#include <iostream>
#include <algorithm>
#include "fmt/format.h"


std::string getStatus(Ipopt::SolverReturn status)
{
    std::string retVal;
    switch (status)
    {
    case Ipopt::SUCCESS:
        retVal = "SUCCESS";
        break;
    case Ipopt::MAXITER_EXCEEDED:
        retVal = "MAXITER_EXCEEDED";
        break;
    case Ipopt::CPUTIME_EXCEEDED:
        retVal = "CPUTIME_EXCEEDED";
        break;
    case Ipopt::STOP_AT_TINY_STEP:
        retVal = "STOP_AT_TINY_STEP";
        break;
    case Ipopt::STOP_AT_ACCEPTABLE_POINT:
        retVal = "STOP_AT_ACCEPTABLE_POINT";
        break;
    case Ipopt::LOCAL_INFEASIBILITY:
        retVal = "LOCAL_INFEASIBILITY";
        break;
    case Ipopt::USER_REQUESTED_STOP:
        retVal = "USER_REQUESTED_STOP";
        break;
    case Ipopt::FEASIBLE_POINT_FOUND:
        retVal = "FEASIBLE_POINT_FOUND";
        break;
    case Ipopt::DIVERGING_ITERATES:
        retVal = "DIVERGING_ITERATES";
        break;
    case Ipopt::RESTORATION_FAILURE:
        retVal = "RESTORATION_FAILURE";
        break;
    case Ipopt::ERROR_IN_STEP_COMPUTATION:
        retVal = "ERROR_IN_STEP_COMPUTATION";
        break;
    case Ipopt::INVALID_NUMBER_DETECTED:
        retVal = "INVALID_NUMBER_DETECTED";
        break;
    case Ipopt::TOO_FEW_DEGREES_OF_FREEDOM:
        retVal = "TOO_FEW_DEGREES_OF_FREEDOM";
        break;
    case Ipopt::INVALID_OPTION:
        retVal = "INVALID_OPTION";
        break;
    case Ipopt::OUT_OF_MEMORY:
        retVal = "OUT_OF_MEMORY";
        break;
    case Ipopt::INTERNAL_ERROR:
        retVal = "INTERNAL_ERROR";
        break;
    case Ipopt::UNASSIGNED:
        retVal = "UNASSIGNED";
        break;
    case Ipopt::WALLTIME_EXCEEDED:
        retVal = "WALLTIME_EXCEEDED";
        break;
    default:
        retVal = "UNKOWN";
        break;
    }
    return retVal;
}

GoldsteinPrice::GoldsteinPrice(Ipopt::Number _x0, Ipopt::Number _y0, HessianMode _mode)
    : x0({_x0, _y0}), exit_status(getStatus(Ipopt::UNASSIGNED)), mode(_mode)
{
    if (statefulMode(mode))
    {
        nHorizon = 2;
    }
    previousCalls.reserve(nHorizon);
    previousCalls.emplace_back(GoldsteinPriceModel<Ipopt::Number>(x0));
    iHorizon = 1;
    gradientScaling = previousCalls.back().g.lpNorm<Eigen::Infinity>() > 1e-9 ? 100. / previousCalls.back().g.lpNorm<Eigen::Infinity>() : 1.;
}


void GoldsteinPrice::update(std::span<const Ipopt::Number> xSolver) {
    GoldsteinPriceModel currentCall(xSolver);
    if (previousCalls.size() < iHorizon)
        previousCalls.emplace_back(currentCall);
    else
        previousCalls.back() = currentCall;
    
    if (statefulMode(mode) && !acceptedInitial)
    {
        Eigen::Vector2d s = previousCalls.back().x - previousCalls.front().x;
        Eigen::Vector2d y = (previousCalls.back().g - previousCalls.front().g)*gradientScaling;
        Ipopt::Number factor = s.dot(y) > 1e-9 ? y.dot(y)/s.dot(y): 1.;
        previousCalls.front().B = Eigen::Matrix2d::Identity()*factor;
    }

    if (HessianMode::bfgs == mode)
        previousCalls.back().B = damped_bfgs(previousCalls.back(), previousCalls.front(), gradientScaling, damping_threshold);
    else if (HessianMode::dfp == mode)
        previousCalls.back().B = damped_dfp(previousCalls.back(), previousCalls.front(), gradientScaling, damping_threshold);
    else if (HessianMode::sr1 == mode)
        previousCalls.back().B = sr1_update(previousCalls.back(), previousCalls.front(), gradientScaling);
    
}

bool GoldsteinPrice::get_nlp_info(
    Ipopt::Index &n,
    Ipopt::Index &m,
    Ipopt::Index &nnz_jac_g,
    Ipopt::Index &nnz_h_lag,
    IndexStyleEnum &index_style)
{

    n = 2;
    m = 0;
    nnz_jac_g = 0;
    nnz_h_lag = 3;
    index_style = TNLP::C_STYLE;

    return true;
}

bool GoldsteinPrice::get_bounds_info(
    [[maybe_unused]]Ipopt::Index n,
    Ipopt::Number *x_l,
    Ipopt::Number *x_u,
    [[maybe_unused]]Ipopt::Index m,
    Ipopt::Number *,
    Ipopt::Number *)
{
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    assert(n == 2);
    assert(m == 0);

    // x1 has a lower bound of -1 and an upper bound of 1
    x_l[0] = -2.;
    x_u[0] = 2.;

    // x2 has no upper or lower bound, so we set them to
    // a large negative and a large positive Ipopt::Number.
    // The value that is interpretted as -/+infinity can be
    // set in the options, but it defaults to -/+1e19
    x_l[1] = -2.;
    x_u[1] = 2.;

    return true;
}

bool GoldsteinPrice::get_starting_point(
    Ipopt::Index ,
    [[maybe_unused]]bool init_x,
    Ipopt::Number *x,
    [[maybe_unused]]bool init_z,
    Ipopt::Number *,
    Ipopt::Number *,
    Ipopt::Index ,
    [[maybe_unused]]bool init_lambda,
    Ipopt::Number *)
{
    // Here, we assume we only have starting values for x, if you code
    // your own NLP, you can provide starting values for the others if
    // you wish.
    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // we initialize x in bounds, in the upper right quadrant
    x[0] = x0[0];
    x[1] = x0[1];

    return true;
}

bool GoldsteinPrice::eval_f(
    Ipopt::Index n,
    const Ipopt::Number *xSolver,
    bool ,
    Ipopt::Number &obj_value)
{
    if (previousCalls.back().isNewX(std::span{xSolver,static_cast<std::size_t>(n)}))
    {
        update(std::span{xSolver,static_cast<std::size_t>(n)});
    }

    obj_value = previousCalls.back().f;

    return true;
}

bool GoldsteinPrice::eval_grad_f(
    Ipopt::Index n,
    const Ipopt::Number xSolver[],
    bool ,
    Ipopt::Number grad_f[])
{
    if (previousCalls.back().isNewX(std::span{xSolver,static_cast<std::size_t>(n)}))
    {
        update(std::span{xSolver,static_cast<std::size_t>(n)});
    }
    grad_f[0] = previousCalls.back().g[0] * gradientScaling;
    grad_f[1] = previousCalls.back().g[1] * gradientScaling;
    return true;
}

bool GoldsteinPrice::eval_g(
    Ipopt::Index ,
    const Ipopt::Number *,
    bool ,
    Ipopt::Index ,
    Ipopt::Number *)
{

    return true;
}

bool GoldsteinPrice::eval_jac_g(
    Ipopt::Index ,
    const Ipopt::Number *,
    bool ,
    Ipopt::Index ,
    Ipopt::Index ,
    Ipopt::Index *,
    Ipopt::Index *,
    Ipopt::Number *)
{
    return true;
}

bool GoldsteinPrice::eval_h(
    Ipopt::Index n,
    const Ipopt::Number *xSolver,
    bool ,
    Ipopt::Number ,
    Ipopt::Index ,
    const Ipopt::Number *,
    bool ,
    Ipopt::Index ,
    Ipopt::Index *iRow,
    Ipopt::Index *jCol,
    Ipopt::Number *values)
{

    if (nullptr == values)
    {
        iRow[0] = 0;
        jCol[0] = 0;
        iRow[1] = 1;
        jCol[1] = 0;
        iRow[2] = 1;
        jCol[2] = 1;
    }
    else
    {
        // return the values
        if (HessianMode::auto_diff == mode)
        {
            Eigen::Matrix2d hessian = previousCalls.back().hessian();
            hessian *= gradientScaling;
            hessian *= gradientScaling;
            values[0] = hessian(0, 0);
            values[1] = hessian(1, 0);
            values[2] = hessian(1, 1);
        }
        else {
            if (previousCalls.back().isNewX(std::span{xSolver,static_cast<std::size_t>(n)}))
            {
                update(std::span{xSolver,static_cast<std::size_t>(n)});
            }
            if (limitedMemoryMode(mode) && previousCalls.size() > 1)
            {
                std::function<Eigen::Matrix<Ipopt::Number, 2, 2>(GoldsteinPriceModel<Ipopt::Number> &, GoldsteinPriceModel<Ipopt::Number> &)> update;
                if (HessianMode::lbfgs == mode)
                    update = [this](GoldsteinPriceModel<Ipopt::Number> &currentCall, GoldsteinPriceModel<Ipopt::Number> &previousCall) {return damped_bfgs(currentCall, previousCall, gradientScaling, damping_threshold);};
                else if (HessianMode::ldfp == mode)
                    update = [this](GoldsteinPriceModel<Ipopt::Number> &currentCall, GoldsteinPriceModel<Ipopt::Number> &previousCall) {return damped_dfp(currentCall, previousCall, gradientScaling, damping_threshold);};
                else if (HessianMode::lsr1 == mode)
                    update = [this](GoldsteinPriceModel<Ipopt::Number> &currentCall, GoldsteinPriceModel<Ipopt::Number> &previousCall) {return sr1_update(currentCall, previousCall, gradientScaling);};
                
                Eigen::Vector2d s = previousCalls[1].x - previousCalls[0].x;
                Eigen::Vector2d y = (previousCalls[1].g - previousCalls[0].g)*gradientScaling;
                Ipopt::Number factor = s.dot(y) > 1e-9 ? y.dot(y)/s.dot(y): 1.;
                previousCalls[0].B = Eigen::Matrix2d::Identity()*factor;
                for (std::size_t i = 0; i < previousCalls.size()-1; ++i)
                {
                    previousCalls[i+1].B = update(previousCalls[i+1], previousCalls[i]);
                }
            }
            
            Eigen::Matrix2d hessian = previousCalls.back().B;
            values[0] = hessian(0, 0);
            values[1] = hessian(1, 0);
            values[2] = hessian(1, 1);
        }
    }

    return true;
}

bool GoldsteinPrice::intermediate_callback(Ipopt::AlgorithmMode ,
                                           Ipopt::Index ,
                                           Ipopt::Number ,
                                           Ipopt::Number ,
                                           Ipopt::Number ,
                                           Ipopt::Number ,
                                           Ipopt::Number ,
                                           Ipopt::Number ,
                                           Ipopt::Number ,
                                           Ipopt::Number ,
                                           [[maybe_unused]] Ipopt::Index ls_trials,
                                           [[maybe_unused]] const Ipopt::IpoptData *ip_data,
                                           [[maybe_unused]] Ipopt::IpoptCalculatedQuantities *ip_cq)
{
    acceptedInitial |= (statefulMode(mode) && 1 < previousCalls.size());
    if (previousCalls.size() >= nHorizon)
    {
        previousCalls.erase(previousCalls.begin());
    } else {
        iHorizon = std::min(iHorizon + 1, nHorizon);
    }
    
    return true;
}

void GoldsteinPrice::finalize_solution(
    Ipopt::SolverReturn status,
    Ipopt::Index ,
    const Ipopt::Number *x,
    const Ipopt::Number *,
    const Ipopt::Number *,
    Ipopt::Index ,
    const Ipopt::Number *,
    const Ipopt::Number *,
    Ipopt::Number ,
    const Ipopt::IpoptData *,
    Ipopt::IpoptCalculatedQuantities *)
{
    x0[0] = x[0];
    x0[1] = x[1];
    exit_status = getStatus(status);
}

