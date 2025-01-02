#ifndef __GOLDSTEINPRICE_HPP__
#define __GOLDSTEINPRICE_HPP__
#include <Eigen/Dense>
#include <autodiff/forward/dual.hpp>

enum HessianMode
{
    auto_diff,
    dfp,
    ipopt_lbfgs,
    bfgs,
    sr1,
    lsr1,
    ldfp,
    lbfgs
};

template <typename T>
T model(T x, T y)
{
    return (1. + pow(x + y + 1., 2) * (19. - 14. * x + 3. * pow(x, 2) - 14. * y + 6. * x * y + 3. * pow(y, 2))) * (30. + pow(2. * x - 3. * y, 2) * (18. - 32. * x + 12. * pow(x, 2) + 48. * y - 36. * x * y + 27. * pow(y, 2)));
}

struct GoldsteinPriceModel
{
    Eigen::Vector2d x;
    Eigen::Vector2d g;
    double f;
    Eigen::Matrix2d B;
    GoldsteinPriceModel(std::span<const Ipopt::Number> xSolver)
    {
        x << xSolver[0], xSolver[1];
        autodiff::dual _x = x[0];
        autodiff::dual _y = x[1];
        f = model<double>(xSolver[0], xSolver[1]);
        g[0] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y));
        g[1] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));
        B = Eigen::Matrix2d::Identity();
    }
    bool isNewX(std::span<const Ipopt::Number> xSolver) const
    {
        Eigen::Vector2d xTmp;
        xTmp << xSolver[0], xSolver[1];
        return (xTmp - x).norm();
    }

    Eigen::MatrixXd hessian() {
        autodiff::dual2nd _x = x[0];
        autodiff::dual2nd _y = x[1];
        Eigen::Matrix2d H;
        {
            auto [u0, u1, der] = autodiff::derivatives(model<autodiff::dual2nd>, autodiff::wrt(_x, _x), autodiff::at(_x, _y));
            H(0,0) = der;
        }
        {
            auto [u0, u1, der] = autodiff::derivatives(model<autodiff::dual2nd>, autodiff::wrt(_x, _y), autodiff::at(_x, _y));
            H(1,0) = H(0,1) = der;
        }
        {
            auto [u0, u1, der] = autodiff::derivatives(model<autodiff::dual2nd>, autodiff::wrt(_y, _y), autodiff::at(_x, _y));
            H(1,1) = der;
        }
        return H;
    }
};

inline Eigen::MatrixXd sr1_update(const GoldsteinPriceModel &current, const GoldsteinPriceModel &previous) {
    Eigen::Vector2d s = current.x - previous.x;
    Eigen::Vector2d y = current.g - previous.g;
    Eigen::Vector2d r = y - previous.B * s;
    return previous.B + (r * r.transpose()) / r.dot(s);
}

inline Eigen::MatrixXd damped_bfgs(const GoldsteinPriceModel &current, const GoldsteinPriceModel &previous, double damping = 0.2)
{
    Eigen::Vector2d s = current.x - previous.x;
    Eigen::Vector2d y = current.g - previous.g;
    
    Eigen::Vector2d Bs = previous.B * s;
    double sTBs = s.dot(Bs);
    double yTs = y.dot(s);
    double theta = yTs >= damping * sTBs ? 1. : (1.-damping)* sTBs / (sTBs - yTs);
    Eigen::Vector2d r = theta * y + (1. - theta) * Bs;
    return previous.B - (Bs * Bs.transpose()) / sTBs + (r * r.transpose()) / r.dot(s);
}

inline Eigen::MatrixXd damped_dfp(const GoldsteinPriceModel &current, const GoldsteinPriceModel &previous, double damping = 0.2)
{
    Eigen::Vector2d s = current.x - previous.x;
    Eigen::Vector2d y = current.g - previous.g;
    
    Eigen::Vector2d Bs = previous.B * s;
    double sTBs = s.dot(Bs);
    double yTs = y.dot(s);
    double theta = yTs >= damping * sTBs ? 1. : (1.-damping)* sTBs / (sTBs - yTs);
    Eigen::Vector2d r = theta * y + (1. - theta) * Bs;
    return (Eigen::Matrix2d::Identity() - r*s.transpose()/s.dot(r)) * previous.B *(Eigen::Matrix2d::Identity() - s*r.transpose()/s.dot(r)) + r * r.transpose() / r.dot(s);
}


#endif // __GOLDSTEINPRICE_HPP__