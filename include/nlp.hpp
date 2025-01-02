// Copyright (C) 2004, 2006 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-11-05

#ifndef __NLP_HPP__
#define __NLP_HPP__

#include "IpTNLP.hpp"
#include <string>
#include <array>
#include <memory>
#include <span>
#include "GoldsteinPrice.hpp"


struct GoldsteinPriceModel;

class GoldsteinPrice : public Ipopt::TNLP
{
private:
    std::vector<GoldsteinPriceModel> previousCalls;
public:
    std::array<Ipopt::Number, 2> x0{};
    std::string exit_status{};
    HessianMode mode{HessianMode::auto_diff};
    double damping_threshold{0.2};

    GoldsteinPrice(Ipopt::Number x0, Ipopt::Number y0, HessianMode mode = HessianMode::auto_diff);

    /** default destructor */
    ~GoldsteinPrice() = default;

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the nlp */
    bool get_nlp_info(
        Ipopt::Index &n,
        Ipopt::Index &m,
        Ipopt::Index &nnz_jac_g,
        Ipopt::Index &nnz_h_lag,
        IndexStyleEnum &index_style) override;

    /** Method to return the bounds for my problem */
    bool get_bounds_info(
        Ipopt::Index n,
        Ipopt::Number *x_l,
        Ipopt::Number *x_u,
        Ipopt::Index m,
        Ipopt::Number *g_l,
        Ipopt::Number *g_u) override;

    /** Method to return the starting point for the algorithm */
    bool get_starting_point(
        Ipopt::Index n,
        bool init_x,
        Ipopt::Number *x,
        bool init_z,
        Ipopt::Number *z_L,
        Ipopt::Number *z_U,
        Ipopt::Index m,
        bool init_lambda,
        Ipopt::Number *lambda) override;

    /** Method to return the objective value */
    bool eval_f(
        Ipopt::Index n,
        const Ipopt::Number *x,
        bool new_x,
        Ipopt::Number &obj_value) override;

    /** Method to return the gradient of the objective */
    bool eval_grad_f(
        Ipopt::Index n,
        const Ipopt::Number *x,
        bool new_x,
        Ipopt::Number *grad_f) override;

    /** Method to return the constraint residuals */
    bool eval_g(
        Ipopt::Index n,
        const Ipopt::Number *x,
        bool new_x,
        Ipopt::Index m,
        Ipopt::Number *g) override;

    /** Method to return:
     *   1) The structure of the Jacobian (if "values" is NULL)
     *   2) The values of the Jacobian (if "values" is not NULL)
     */
    bool eval_jac_g(
        Ipopt::Index n,
        const Ipopt::Number *x,
        bool new_x,
        Ipopt::Index m,
        Ipopt::Index nele_jac,
        Ipopt::Index *iRow,
        Ipopt::Index *jCol,
        Ipopt::Number *values) override;

    /** Method to return:
     *   1) The structure of the Hessian of the Lagrangian (if "values" is NULL)
     *   2) The values of the Hessian of the Lagrangian (if "values" is not NULL)
     */
    bool eval_h(
        Ipopt::Index n,
        const Ipopt::Number *x,
        bool new_x,
        Ipopt::Number obj_factor,
        Ipopt::Index m,
        const Ipopt::Number *lambda,
        bool new_lambda,
        Ipopt::Index nele_hess,
        Ipopt::Index *iRow,
        Ipopt::Index *jCol,
        Ipopt::Number *values) override;

    bool intermediate_callback(Ipopt::AlgorithmMode mode,
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
                               [[maybe_unused]] Ipopt::IpoptCalculatedQuantities *ip_cq) override;

    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    void finalize_solution(
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
        Ipopt::IpoptCalculatedQuantities *ip_cq) override;
    //@}

private:
    /**@name Methods to block default compiler methods.
     *
     * The compiler automatically generates the following three methods.
     *  Since the default compiler implementation is generally not what
     *  you want (for all but the most simple classes), we usually
     *  put the declarations of these methods in the private section
     *  and never implement them. This prevents the compiler from
     *  implementing an incorrect "default" behavior without us
     *  knowing. (See Scott Meyers book, "Effective C++")
     */
    //@{
    GoldsteinPrice(
        const GoldsteinPrice &);

    GoldsteinPrice &operator=(
        const GoldsteinPrice &);
    //@}

    void update(std::span<const Ipopt::Number> x);
};

#endif // __NLP_HPP__