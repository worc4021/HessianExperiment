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
#include <autodiff/forward/dual.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>

template <typename T>
T model(T x, T y)
{
    return (1. + pow(x + y + 1., 2) * (19. - 14. * x + 3. * pow(x, 2) - 14. * y + 6. * x * y + 3. * pow(y, 2))) * (30. + pow(2. * x - 3. * y, 2) * (18. - 32. * x + 12. * pow(x, 2) + 48. * y - 36. * x * y + 27. * pow(y, 2)));
}

struct GoldsteinPrice::HessianStore
{
    HessianMode mode{HessianMode::dfp};
    Eigen::Matrix2d hessian = Eigen::Matrix2d::Identity();
    Eigen::Vector2d gradPre = Eigen::Vector2d::Zero();
    Eigen::Vector2d grad = Eigen::Vector2d::Zero();
    Eigen::Vector2d xPre = Eigen::Vector2d::Zero();
    Eigen::Vector2d x = Eigen::Vector2d::Zero();

    Eigen::Matrix2d getHessian(const Ipopt::Number *xSolver)
    {
        x[0] = xSolver[0];
        x[1] = xSolver[1];

        autodiff::dual _x = x[0];
        autodiff::dual _y = x[1];

        grad[0] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y));
        grad[1] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));

        Eigen::Matrix2d _hessian = hessian;

        if (xPre.norm() > 0.)
        {
            Eigen::Vector2d s = x - xPre;
            Eigen::Vector2d y = grad - gradPre;
            if (HessianMode::sr1 == mode)
            {
                Eigen::Vector2d deltaX = y - _hessian * s;
                _hessian += deltaX * deltaX.transpose() / deltaX.dot(s);
            }
            else if (HessianMode::dfp == mode || HessianMode::bfgs == mode)
            {
                double phi = HessianMode::bfgs == mode ? 1. : 0.;
                Eigen::Vector2d Bs = _hessian*s;
                double sBs = s.dot(Bs);
                double sy = s.dot(y);
                double theta = 1.;
                if (sy<0.2*sBs)
                    theta = 0.8*sBs/(sBs-sy);
                
                Eigen::Vector2d r = theta*y + (1-theta)*Bs;
                double sr = s.dot(r);
                Eigen::Vector2d v = r / sr - Bs / sBs;                
                Eigen::Matrix2d U = r*r.transpose();
                Eigen::Matrix2d V = Bs*Bs.transpose();
                Eigen::Matrix2d W = v*v.transpose();
                _hessian += U/sr - V/sBs + phi*sBs * W;
            }
        }

        return _hessian;
    }

    void update(const Ipopt::Number *xSolver)
    {
        hessian = getHessian(xSolver);
        gradPre = grad;
        xPre = x;
    }

    void reset() {
        hessian = Eigen::Matrix2d::Identity();
        gradPre = Eigen::Vector2d::Zero();
        grad = Eigen::Vector2d::Zero();
        xPre = Eigen::Vector2d::Zero();
        x = Eigen::Vector2d::Zero();
    }
};

struct GoldsteinPrice::LimitedMemoryHessianStore
{
    Eigen::Matrix<double, 2, 6> x = Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Matrix<double, 2, 6> grad = Eigen::Matrix<double, 2, 6>::Zero();

    std::size_t numel{0};

    HessianMode mode{HessianMode::ldfp};

    Eigen::Matrix2d getHessian(const Ipopt::Number *xSolver)
    {
        Eigen::Matrix2d hessian = Eigen::Matrix2d::Identity();

        if (numel > 1)
        {

            x.col(0) << xSolver[0], xSolver[1];
            autodiff::dual _x = x(0, 0);
            autodiff::dual _y = x(1, 0);

            grad.col(0) << autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y)),
                autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));

            for (std::size_t i = numel - 1; i > 0; --i)
            {
                Eigen::Vector2d s = x.col(i - 1) - x.col(i);
                Eigen::Vector2d y = grad.col(i - 1) - grad.col(i);
                if (HessianMode::lsr1 == mode)
                {
                    Eigen::Vector2d deltaX = y - hessian * s;
                    hessian += deltaX * deltaX.transpose() / deltaX.dot(s);
                }
                else if (HessianMode::ldfp == mode || HessianMode::lbfgs == mode)
                {
                    double phi = HessianMode::lbfgs == mode ? 1. : 0.;
                    Eigen::Vector2d Bs = hessian*s;
                    double sBs = s.dot(Bs);
                    double sy = s.dot(y);
                    double theta = 1.;
                    if (sy<0.2*sBs)
                        theta = 0.8*sBs/(sBs-sy);
                    
                    Eigen::Vector2d r = theta*y + (1-theta)*Bs;
                    double sr = s.dot(r);
                    Eigen::Vector2d v = r / sr - Bs / sBs;                
                    Eigen::Matrix2d U = r*r.transpose();
                    Eigen::Matrix2d V = Bs*Bs.transpose();
                    Eigen::Matrix2d W = v*v.transpose();
                    hessian += U/sr - V/sBs + phi*sBs * W;
                }
            }
        }
        return hessian;
    }
    void update(const Ipopt::Number *xSolver)
    {
        for (std::size_t i = 5; i > 0; --i)
        {
            x.col(i) = x.col(i - 1);
            grad.col(i) = grad.col(i - 1);
        }
        x.col(0) << xSolver[0], xSolver[1];
        autodiff::dual _x = x(0, 0);
        autodiff::dual _y = x(1, 0);

        grad.col(0) << autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y)),
            autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));
        numel = std::min<std::size_t>(numel + 1, 6ul);
    }

    void reset() {
        x = Eigen::Matrix<double, 2, 6>::Zero();
        grad = Eigen::Matrix<double, 2, 6>::Zero();
        numel = 0;
    }
};

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

/* Constructor. */
GoldsteinPrice::GoldsteinPrice()
{
}

GoldsteinPrice::GoldsteinPrice(Ipopt::Number x0, Ipopt::Number y0, HessianMode _mode)
    : x0({x0, y0}), exit_status(getStatus(Ipopt::UNASSIGNED)), mode(_mode), hessianStore(std::make_unique<HessianStore>())
{
    hessianStore->mode = _mode;
}

GoldsteinPrice::~GoldsteinPrice()
{
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
    index_style = FORTRAN_STYLE;

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
    assert(m == 1);

    // x1 has a lower bound of -1 and an upper bound of 1
    x_l[0] = -2.;
    x_u[0] = 2.;

    // x2 has no upper or lower bound, so we set them to
    // a large negative and a large positive Ipopt::Number.
    // The value that is interpretted as -/+infinity can be
    // set in the options, but it defaults to -/+1e19
    x_l[1] = -3.;
    x_u[1] = 1.;

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
    // return the value of the objective function
    Ipopt::Number x = xSolver[0];
    Ipopt::Number y = xSolver[1];

    obj_value = model(x, y);

    return true;
}

bool GoldsteinPrice::eval_grad_f(
    Ipopt::Index n,
    const Ipopt::Number *xSolver,
    bool new_x,
    Ipopt::Number *grad_f)
{
    // return the gradient of the objective function grad_{x} f(x)

    autodiff::dual x = xSolver[0];
    autodiff::dual y = xSolver[1];

    grad_f[0] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(x), autodiff::at(x, y));
    grad_f[1] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(y), autodiff::at(x, y));
    ;

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
    if (nullptr == values)
    {
    }
    else
    {
    }

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

    if (obj_factor == 0.)
    {
        if (mode == HessianMode::sr1 || mode == HessianMode::dfp || mode == HessianMode::bfgs)
            hessianStore->reset();
        else if (mode == HessianMode::lsr1 || mode == HessianMode::ldfp || mode == HessianMode::lbfgs)
            limitedMemoryHessianStore->reset();
    }

    if (nullptr == values)
    {
        // return the structure. This is a symmetric matrix, fill the lower left
        // triangle only.

        // element at 1,1: grad^2_{x1,x1} L(x,lambda)
        iRow[0] = 1;
        jCol[0] = 1;
        iRow[1] = 2;
        jCol[1] = 1;
        iRow[2] = 2;
        jCol[2] = 2;
    }
    else
    {
        // return the values
        if (mode == HessianMode::auto_diff)
        {
            autodiff::dual2nd x = xSolver[0];
            autodiff::dual2nd y = xSolver[1];

            {
                auto [u0, u1, der] = autodiff::derivatives(model<autodiff::dual2nd>, autodiff::wrt(x, x), autodiff::at(x, y));
                values[0] = der;
            }
            {
                auto [u0, u1, der] = autodiff::derivatives(model<autodiff::dual2nd>, autodiff::wrt(x, y), autodiff::at(x, y));
                values[1] = der;
            }
            {
                auto [u0, u1, der] = autodiff::derivatives(model<autodiff::dual2nd>, autodiff::wrt(y, y), autodiff::at(x, y));
                values[2] = der;
            }
        }
        else if (mode == HessianMode::sr1 || mode == HessianMode::dfp || mode == HessianMode::bfgs)
        {
            Eigen::Matrix2d hessian = hessianStore->getHessian(xSolver);
            values[0] = hessian(0, 0);
            values[1] = hessian(1, 0);
            values[2] = hessian(1, 1);
        }
        else if (mode == HessianMode::lsr1 || mode == HessianMode::ldfp || mode == HessianMode::lbfgs)
        {
            Eigen::Matrix2d hessian = limitedMemoryHessianStore->getHessian(xSolver);
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
    if (Ipopt::AlgorithmMode::RestorationPhaseMode == mode){
        if (this->mode == HessianMode::sr1 || this->mode == HessianMode::dfp || this->mode == HessianMode::bfgs)
            hessianStore->reset();
        else if (this->mode == HessianMode::lsr1 || this->mode == HessianMode::ldfp || this->mode == HessianMode::lbfgs)
            limitedMemoryHessianStore->reset();
    }

    Ipopt::OrigIpoptNLP *orignlp{nullptr};
    orignlp = dynamic_cast<Ipopt::OrigIpoptNLP *>(GetRawPtr(ip_cq->GetIpoptNLP()));
    Ipopt::TNLPAdapter *tnlp_adapter{nullptr};
    if (nullptr != orignlp)
        tnlp_adapter = dynamic_cast<Ipopt::TNLPAdapter *>(GetRawPtr(orignlp->nlp()));

    Ipopt::Number x[2];
    tnlp_adapter->ResortX(*ip_data->curr()->x(), x);
    if (this->mode == HessianMode::sr1 || this->mode == HessianMode::dfp)
        hessianStore->update(x);
    else if (this->mode == HessianMode::lsr1 || this->mode == HessianMode::ldfp)
        limitedMemoryHessianStore->update(x);
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