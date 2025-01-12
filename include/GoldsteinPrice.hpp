#ifndef __GOLDSTEINPRICE_HPP__
#define __GOLDSTEINPRICE_HPP__
#include <Eigen/Dense>
#include <concepts>
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
template <std::floating_point T>
struct GoldsteinPriceModel
{
    Eigen::Matrix<T, 2, 1> x;
    Eigen::Matrix<T, 2, 1> g;
    T f;
    Eigen::Matrix<T, 2, 2> B;
    GoldsteinPriceModel(std::span<const T> xSolver)
    {
        x << xSolver[0], xSolver[1];
        autodiff::dual _x = x[0];
        autodiff::dual _y = x[1];
        f = model<T>(xSolver[0], xSolver[1]);
        g[0] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_x), autodiff::at(_x, _y));
        g[1] = autodiff::derivative(model<autodiff::dual>, autodiff::wrt(_y), autodiff::at(_x, _y));
        B = Eigen::Matrix<T, 2, 2>::Identity();
    }
    bool isNewX(std::span<const T> xSolver) const
    {
        Eigen::Matrix<T, 2, 1> xTmp;
        xTmp << xSolver[0], xSolver[1];
        return (xTmp - x).norm();
    }

    Eigen::Matrix<T, 2, 2> hessian() const {
        autodiff::dual2nd _x = x[0];
        autodiff::dual2nd _y = x[1];
        Eigen::Matrix<T, 2, 2> H;
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
template <std::floating_point T>
Eigen::Matrix<T, 2, 2> sr1_update(const GoldsteinPriceModel<T> &current, const GoldsteinPriceModel<T> &previous) {
    Eigen::Matrix<T, 2, 1> s = current.x - previous.x;
    Eigen::Matrix<T, 2, 1> y = current.g - previous.g;
    Eigen::Matrix<T, 2, 1> r = y - previous.B * s;
    return previous.B + (r * r.transpose()) / r.dot(s);
}

template<std::floating_point T>
Eigen::Matrix<T, 2, 2> damped_bfgs(const GoldsteinPriceModel<T> &current, const GoldsteinPriceModel<T> &previous, T damping = 0.2)
{
    Eigen::Matrix<T, 2, 1> s = current.x - previous.x;
    Eigen::Matrix<T, 2, 1> y = current.g - previous.g;
    
    Eigen::Matrix<T, 2, 1> Bs = previous.B * s;
    T sTBs = s.dot(Bs);
    T yTs = y.dot(s);
    T theta = yTs >= damping * sTBs ? 1. : (1.-damping)* sTBs / (sTBs - yTs);
    Eigen::Matrix<T, 2, 1> r = theta * y + (1. - theta) * Bs;
    return previous.B - (Bs * Bs.transpose()) / sTBs + (r * r.transpose()) / r.dot(s);
}

template<std::floating_point T>
Eigen::Matrix<T, 2, 2> damped_dfp(const GoldsteinPriceModel<T> &current, const GoldsteinPriceModel<T> &previous, T damping = 0.2)
{
    Eigen::Matrix<T, 2, 1> s = current.x - previous.x;
    Eigen::Matrix<T, 2, 1> y = current.g - previous.g;
    
    Eigen::Matrix<T, 2, 1> Bs = previous.B * s;
    T sTBs = s.dot(Bs);
    T yTs = y.dot(s);
    T theta = yTs >= damping * sTBs ? 1. : (1.-damping)* sTBs / (sTBs - yTs);
    Eigen::Matrix<T, 2, 1> r = theta * y + (1. - theta) * Bs;
    return (Eigen::Matrix<T, 2, 2>::Identity() - r*s.transpose()/s.dot(r)) * previous.B *(Eigen::Matrix<T, 2, 2>::Identity() - s*r.transpose()/s.dot(r)) + r * r.transpose() / r.dot(s);
}


#endif // __GOLDSTEINPRICE_HPP__