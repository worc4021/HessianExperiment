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
    previousCalls.push_back(GoldsteinPriceModel<Ipopt::Number>(x0));
}


void GoldsteinPrice::update(std::span<const Ipopt::Number> xSolver) {
    GoldsteinPriceModel currentCall(xSolver);
    if (previousCalls.size()==1)
        previousCalls.emplace_back(currentCall);
    else
        previousCalls[1] = currentCall;

    if (HessianMode::bfgs == mode)
        previousCalls.back().B = damped_bfgs(previousCalls[1], previousCalls[0],damping_threshold);
    else if (HessianMode::dfp == mode)
        previousCalls.back().B = damped_dfp(previousCalls[1], previousCalls[0],damping_threshold);
    else if (HessianMode::sr1 == mode)
        previousCalls.back().B = sr1_update(previousCalls[1], previousCalls[0]);
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
    Ipopt::Index n,
    Ipopt::Number *x_l,
    Ipopt::Number *x_u,
    Ipopt::Index m,
    Ipopt::Number *g_l,
    Ipopt::Number *g_u)
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
    Ipopt::Index n,
    bool init_x,
    Ipopt::Number *x,
    bool init_z,
    Ipopt::Number *z_L,
    Ipopt::Number *z_U,
    Ipopt::Index m,
    bool init_lambda,
    Ipopt::Number *lambda)
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
    bool new_x,
    Ipopt::Number &obj_value)
{
    if (previousCalls.back().isNewX(std::span{xSolver,n}))
    {
        update(std::span{xSolver,n});
    }

    obj_value = previousCalls.back().f;

    return true;
}

bool GoldsteinPrice::eval_grad_f(
    Ipopt::Index n,
    const Ipopt::Number xSolver[],
    bool new_x,
    Ipopt::Number grad_f[])
{
    if (previousCalls.back().isNewX(std::span{xSolver,n}))
    {
        update(std::span{xSolver,n});
    }
    grad_f[0] = previousCalls.back().g[0];
    grad_f[1] = previousCalls.back().g[1];
    return true;
}

bool GoldsteinPrice::eval_g(
    Ipopt::Index n,
    const Ipopt::Number *x,
    bool new_x,
    Ipopt::Index m,
    Ipopt::Number *g)
{

    return true;
}

bool GoldsteinPrice::eval_jac_g(
    Ipopt::Index n,
    const Ipopt::Number *x,
    bool new_x,
    Ipopt::Index m,
    Ipopt::Index nele_jac,
    Ipopt::Index *iRow,
    Ipopt::Index *jCol,
    Ipopt::Number *values)
{
    return true;
}

bool GoldsteinPrice::eval_h(
    Ipopt::Index n,
    const Ipopt::Number *xSolver,
    bool new_x,
    Ipopt::Number obj_factor,
    Ipopt::Index m,
    const Ipopt::Number *lambda,
    bool new_lambda,
    Ipopt::Index nele_hess,
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
        if (mode == HessianMode::auto_diff)
        {
            Eigen::Matrix2d hessian = previousCalls.back().hessian();
            values[0] = hessian(0, 0);
            values[1] = hessian(1, 0);
            values[2] = hessian(1, 1);
        }
        else {
            if (previousCalls.back().isNewX(std::span{xSolver,n}))
            {
                update(std::span{xSolver,n});
            }
            Eigen::Matrix2d hessian = previousCalls.back().B;
            values[0] = hessian(0, 0);
            values[1] = hessian(1, 0);
            values[2] = hessian(1, 1);
        }
    }

    return true;
}

bool GoldsteinPrice::intermediate_callback(Ipopt::AlgorithmMode mode,
                                           Ipopt::Index iter,
                                           Ipopt::Number obj_value,
                                           Ipopt::Number inf_pr,
                                           Ipopt::Number inf_du,
                                           Ipopt::Number mu,
                                           Ipopt::Number d_norm,
                                           Ipopt::Number regularization_size,
                                           Ipopt::Number alpha_du,
                                           Ipopt::Number alpha_pr,
                                           [[maybe_unused]] Ipopt::Index ls_trials,
                                           [[maybe_unused]] const Ipopt::IpoptData *ip_data,
                                           [[maybe_unused]] Ipopt::IpoptCalculatedQuantities *ip_cq)
{

    // Ipopt::OrigIpoptNLP *orignlp{nullptr};
    // orignlp = dynamic_cast<Ipopt::OrigIpoptNLP *>(GetRawPtr(ip_cq->GetIpoptNLP()));
    // Ipopt::TNLPAdapter *tnlp_adapter{nullptr};
    // if (nullptr != orignlp)
    //     tnlp_adapter = dynamic_cast<Ipopt::TNLPAdapter *>(GetRawPtr(orignlp->nlp()));

    // Ipopt::Number x[2];
    // tnlp_adapter->ResortX(*ip_data->curr()->x(), x);
    if (previousCalls.size()>1 && previousCalls[0].B.isApprox(Eigen::Matrix2d::Identity()))
    {
        Eigen::Vector2d s = previousCalls[1].x - previousCalls[0].x;
        Eigen::Vector2d y = previousCalls[1].g - previousCalls[0].g;
        previousCalls[0].B = Eigen::Matrix2d::Identity()*y.dot(y)/s.dot(y);
    }
    if (previousCalls.size()>1)
        previousCalls.erase(previousCalls.begin());
    return true;
}

void GoldsteinPrice::finalize_solution(
    Ipopt::SolverReturn status,
    Ipopt::Index n,
    const Ipopt::Number *x,
    const Ipopt::Number *z_L,
    const Ipopt::Number *z_U,
    Ipopt::Index m,
    const Ipopt::Number *g,
    const Ipopt::Number *lambda,
    Ipopt::Number obj_value,
    const Ipopt::IpoptData *ip_data,
    Ipopt::IpoptCalculatedQuantities *ip_cq)
{
    x0[0] = x[0];
    x0[1] = x[1];
    exit_status = getStatus(status);
}




// struct GoldsteinPrice::HessianStore
// {
//     HessianMode mode{HessianMode::dfp};
//     Eigen::Matrix2d hessian = Eigen::Matrix2d::Identity();
//     Eigen::Vector2d gradPre = Eigen::Vector2d::Zero();
//     Eigen::Vector2d grad = Eigen::Vector2d::Zero();
//     Eigen::Vector2d xPre = Eigen::Vector2d::Zero();
//     Eigen::Vector2d x = Eigen::Vector2d::Random();
//     std::size_t nUpdates{0};
    
//     HessianStore(const Ipopt::Number x0, const Ipopt::Number y0, HessianMode _mode = HessianMode::bfgs) : mode(_mode) {
//         x << x0, y0;
//         autodiff::dual _x = x[0];
//         autodiff::dual _y = x[1];
//         grad[0] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y));
//         grad[1] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));
//     }

//     Eigen::Matrix2d getHessian(const Ipopt::Number xSolver[])
//     {
//         Eigen::Vector2d xTmp;
//         xTmp[0] = xSolver[0];
//         xTmp[1] = xSolver[1];

//         if ((xTmp - x).norm())
//         {
//             x = xTmp;
//             autodiff::dual _x = x[0];
//             autodiff::dual _y = x[1];

//             grad[0] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y));
//             grad[1] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));

//             Eigen::Matrix2d _hessian = hessian;

//             Eigen::Vector2d s = x - xPre;
//             Eigen::Vector2d y = grad - gradPre;
//             double sy = s.dot(y);
//             if (nUpdates<1)
//                 _hessian *= y.dot(y) / sy;
                
//             if (sy)
//             {
//                 if (HessianMode::bfgs == mode)
//                 {
//                     Eigen::Vector2d Bs = _hessian * s;
//                     double sqrtsBs = sqrt(s.dot(Bs));
//                     double sqrtsy = sqrt(s.dot(y));
//                     Bs /= sqrtsBs;
//                     y /= sqrtsy;
//                     _hessian -= Bs * Bs.transpose();
//                     _hessian += y * y.transpose();
//                 }
//             }

//             if (HessianMode::sr1 == mode)
//             {
//                 Eigen::Vector2d deltaX = y - _hessian * s;
//                 _hessian += deltaX * deltaX.transpose() / deltaX.dot(s);
//             }
//             else if (HessianMode::dfp == mode)
//             {
//                 double rho = 1. / s.dot(y);
//                 Eigen::Matrix2d L = Eigen::Matrix2d::Identity() - rho * y * s.transpose();
//                 _hessian = L * _hessian * L.transpose();
//                 _hessian += rho * y * y.transpose();
//             }
//             return _hessian;
//         }
//         else
//         {
//             return hessian;
//         }
//     }

//     void update(const Ipopt::Number xSolver[])
//     {

//         Eigen::Vector2d xTmp;
//         xTmp << xSolver[0], xSolver[1];
//         if ((xTmp - x).norm())
//         {
//             xPre = x;
//             gradPre = grad;
//             hessian = getHessian(xSolver);
//             ++nUpdates;
//         }
        
//     }

//     void reset()
//     {
//         hessian = Eigen::Matrix2d::Identity();
//         gradPre = Eigen::Vector2d::Zero();
//         xPre = Eigen::Vector2d::Zero();
//         nUpdates = 0;
//     }
// };

// struct GoldsteinPrice::LimitedMemoryHessianStore
// {
//     Eigen::Matrix<double, 2, 6> x = Eigen::Matrix<double, 2, 6>::Zero();
//     Eigen::Matrix<double, 2, 6> grad = Eigen::Matrix<double, 2, 6>::Zero();

//     std::size_t numel{0};

//     HessianMode mode{HessianMode::ldfp};

//     Eigen::Matrix2d getHessian(const Ipopt::Number xSolver[])
//     {
//         Eigen::Matrix2d hessian = Eigen::Matrix2d::Identity();

//         if (numel > 1)
//         {

//             x.col(0) << xSolver[0], xSolver[1];
//             autodiff::dual _x = x(0, 0);
//             autodiff::dual _y = x(1, 0);

//             grad.col(0) << autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y)),
//                 autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));

//             std::vector<Eigen::Vector2d> ai(numel - 1);
//             std::vector<Eigen::Vector2d> bi(numel - 1);

//             Eigen::Vector2d s = x.col(0) - x.col(1);
//             Eigen::Vector2d y = grad.col(0) - grad.col(1);
//             double delta = y.dot(y) / y.dot(s);
//             hessian *= delta;

//             for (std::size_t i = numel - 1; i > 0; --i)
//             {
//                 s = x.col(i - 1) - x.col(i);
//                 y = grad.col(i - 1) - grad.col(i);
//                 if (HessianMode::lsr1 == mode)
//                 {
//                     Eigen::Vector2d deltaX = y - hessian * s;
//                     hessian += deltaX * deltaX.transpose() / deltaX.dot(s);
//                 }
//                 else if (HessianMode::lbfgs == mode)
//                 {

//                     Eigen::Vector2d Bs = hessian * s;
//                     double sqrtsBs = sqrt(s.dot(Bs));
//                     double sqrtsy = sqrt(s.dot(y));
//                     bi[i - 1] = y / sqrtsy;
//                     ai[i - 1] = Bs / sqrtsBs;
//                     for (std::size_t j = numel - 2; j >= i; --j)
//                     {
//                         ai[i - 1] += bi[j].dot(s) * bi[j];
//                         ai[i - 1] -= ai[j].dot(s) * ai[j];
//                     }
//                     ai[i - 1] /= sqrtsy;
//                 }
//                 else if (HessianMode::ldfp == mode)
//                 {
//                 }
//             }

//             for (std::size_t i = numel - 1; i > 0; --i)
//             {
//                 hessian += bi[i - 1] * bi[i - 1].transpose();
//                 hessian -= ai[i - 1] * ai[i - 1].transpose();
//             }
//         }
//         return hessian;
//     }
//     void update(const Ipopt::Number xSolver[])
//     {
//         for (std::size_t i = 5; i > 0; --i)
//         {
//             x.col(i) = x.col(i - 1);
//             grad.col(i) = grad.col(i - 1);
//         }
//         x.col(0) << xSolver[0], xSolver[1];
//         autodiff::dual _x = x(0, 0);
//         autodiff::dual _y = x(1, 0);

//         grad.col(0) << autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y)),
//             autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));
//         numel = std::min<std::size_t>(numel + 1, 6ul);
//     }

//     void reset()
//     {
//         x = Eigen::Matrix<double, 2, 6>::Zero();
//         grad = Eigen::Matrix<double, 2, 6>::Zero();
//         numel = 0;
//     }
// };
