#ifndef UNO_MODEL_HPP
#define UNO_MODEL_HPP

#include <vector>
#include <limits>
#include <span>
#include "GoldsteinPrice.hpp"
#include "model/Model.hpp"
#include "linear_algebra/SparseVector.hpp"
#include "linear_algebra/Vector.hpp"
#include "linear_algebra/RectangularMatrix.hpp"
#include "linear_algebra/SymmetricMatrix.hpp"
#include "optimization/Iterate.hpp"
#include "symbolic/CollectionAdapter.hpp"
#include "tools/UserCallbacks.hpp"
#include "tools/Timer.hpp"


class DataModel 
        : public uno::Model
    {
    protected:
        std::vector<double> _variable_lower_bounds;
        std::vector<double> _variable_upper_bounds;
        std::vector<double> _constraint_lower_bounds;
        std::vector<double> _constraint_upper_bounds;

        std::vector<uno::BoundType> _variable_status;    /*!< Status of the variables (EQUALITY, BOUNDED_LOWER, BOUNDED_UPPER, BOUNDED_BOTH_SIDES) */
        std::vector<uno::FunctionType> _constraint_type; /*!< Types of the constraints (LINEAR, QUADRATIC, NONLINEAR) */
        std::vector<uno::BoundType> _constraint_status;  /*!< Status of the constraints (EQUAL_BOUNDS, BOUNDED_LOWER, BOUNDED_UPPER, BOUNDED_BOTH_SIDES,UNBOUNDED) */
        std::vector<size_t> _linear_constraints;
        

        uno::SparseVector<size_t> _slacks{};

        uno::Timer _timer;
    private:
        // lists of variables and constraints + corresponding collection objects
        std::vector<size_t> _equality_constraints;
        std::vector<size_t> _inequality_constraints;
        uno::CollectionAdapter<std::vector<size_t> &> _equality_constraints_collection;
        uno::CollectionAdapter<std::vector<size_t> &> _inequality_constraints_collection;
        std::vector<size_t> _lower_bounded_variables;
        uno::CollectionAdapter<std::vector<size_t> &> _lower_bounded_variables_collection;
        std::vector<size_t> _upper_bounded_variables;
        uno::CollectionAdapter<std::vector<size_t> &> _upper_bounded_variables_collection;
        std::vector<size_t> _single_lower_bounded_variables; // indices of the single lower-bounded variables
        uno::CollectionAdapter<std::vector<size_t> &> _single_lower_bounded_variables_collection;
        std::vector<size_t> _single_upper_bounded_variables; // indices of the single upper-bounded variables
        uno::Vector<size_t> _fixed_variables;
        uno::CollectionAdapter<std::vector<size_t> &> _single_upper_bounded_variables_collection;
        uno::CollectionAdapter<std::vector<size_t> &> _linear_constraints_collection;

    public:
        DataModel(size_t number_variables, size_t number_constraints, const std::string &name = "DataModel")
            : uno::Model(name, number_variables, number_constraints, 1.),
              _variable_lower_bounds(number_variables),
              _variable_upper_bounds(number_variables),
              _constraint_lower_bounds(number_constraints),
              _constraint_upper_bounds(number_constraints),
              _variable_status(number_variables),
              _constraint_type(number_constraints),
              _constraint_status(number_constraints),
              _linear_constraints(0),
              _slacks(0),
              _timer(),
              _equality_constraints(0),
              _inequality_constraints(0),
              _equality_constraints_collection(_equality_constraints),
              _inequality_constraints_collection(_inequality_constraints),
              _lower_bounded_variables(0),
              _lower_bounded_variables_collection(_lower_bounded_variables),
              _upper_bounded_variables(0),
              _upper_bounded_variables_collection(_upper_bounded_variables),
              _single_lower_bounded_variables(0),
              _single_lower_bounded_variables_collection(_single_lower_bounded_variables),
              _single_upper_bounded_variables(0),
              _fixed_variables(0),
              _single_upper_bounded_variables_collection(_single_upper_bounded_variables),
              _linear_constraints_collection(_linear_constraints)
            {}

        virtual ~DataModel() override = default;

        const uno::Collection<size_t> &get_equality_constraints() const override
        {
            return _equality_constraints_collection;
        }

        const uno::Collection<size_t> &get_inequality_constraints() const override
        {
            return _inequality_constraints_collection;
        }

        const uno::Collection<size_t> &get_linear_constraints() const override
        {
            return _linear_constraints_collection;
        }

        const uno::SparseVector<size_t> &get_slacks() const override
        {
            return _slacks;
        }

        const uno::Collection<size_t> &get_single_lower_bounded_variables() const override
        {
            return _single_lower_bounded_variables_collection;
        }

        const uno::Collection<size_t> &get_single_upper_bounded_variables() const override
        {
            return _single_upper_bounded_variables_collection;
        }

        const uno::Collection<size_t> &get_lower_bounded_variables() const override
        {
            return _lower_bounded_variables_collection;
        }

        const uno::Collection<size_t> &get_upper_bounded_variables() const override
        {
            return _upper_bounded_variables_collection;
        }

        const uno::Vector<size_t>& get_fixed_variables() const override {
            return _fixed_variables;
        }

        double variable_lower_bound(size_t variable_index) const override { 
            return _variable_lower_bounds[variable_index]; 
            }

        double variable_upper_bound(size_t variable_index) const override { 
            return _variable_upper_bounds[variable_index]; 
        }

        uno::BoundType get_variable_bound_type(size_t variable_index) const override {
            return _variable_status[variable_index];
        }

        double constraint_lower_bound(size_t constraint_index) const override {
            return _constraint_lower_bounds[constraint_index];
        }

        double constraint_upper_bound(size_t constraint_index) const override {
            return _constraint_upper_bounds[constraint_index];
        }
        
        uno::BoundType get_constraint_bound_type(size_t constraint_index) const override {
            return _constraint_status[constraint_index];
        }
        uno::FunctionType get_constraint_type(size_t constraint_index) const override {
            return _constraint_type[constraint_index];
        }

        void postprocess_solution(uno::Iterate &iterate, uno::IterateStatus termination_status) const override {
            
        }

        void initialise_from_data()
        {
            for (std::size_t i = 0; i < number_variables; ++i)
            {
                if (_variable_lower_bounds[i] == _variable_upper_bounds[i])
                {
                    _variable_status[i] = uno::BoundType::EQUAL_BOUNDS;
                    _lower_bounded_variables.emplace_back(i);
                    _upper_bounded_variables.emplace_back(i);
                    _fixed_variables.emplace_back(i);
                }
                else if (std::isfinite(_variable_lower_bounds[i]) && std::isfinite(_variable_upper_bounds[i]))
                {
                    _variable_status[i] = uno::BoundType::BOUNDED_BOTH_SIDES;
                    _lower_bounded_variables.emplace_back(i);
                    _upper_bounded_variables.emplace_back(i);
                }
                else if (std::isfinite(_variable_lower_bounds[i]))
                {
                    _variable_status[i] = uno::BoundType::BOUNDED_LOWER;
                    _lower_bounded_variables.emplace_back(i);
                    _single_lower_bounded_variables.emplace_back(i);
                }
                else if (std::isfinite(_variable_upper_bounds[i]))
                {
                    _variable_status[i] = uno::BoundType::BOUNDED_UPPER;
                    _upper_bounded_variables.emplace_back(i);
                    _single_upper_bounded_variables.emplace_back(i);
                }
                else
                {
                    _variable_status[i] = uno::BoundType::UNBOUNDED;
                }
            }

            for (std::size_t i = 0; i<number_constraints; ++i) {
                if (_constraint_lower_bounds[i] == _constraint_upper_bounds[i])
                {
                    _constraint_status[i] = uno::BoundType::EQUAL_BOUNDS;
                    _equality_constraints.emplace_back(i);
                }
                else if (std::isfinite(_constraint_lower_bounds[i]) && std::isfinite(_constraint_upper_bounds[i]))
                {
                    _constraint_status[i] = uno::BoundType::BOUNDED_BOTH_SIDES;
                    _inequality_constraints.emplace_back(i);
                }
                else if (std::isfinite(_constraint_lower_bounds[i]))
                {
                    _constraint_status[i] = uno::BoundType::BOUNDED_LOWER;
                    _inequality_constraints.emplace_back(i);
                }
                else if (std::isfinite(_constraint_upper_bounds[i]))
                {
                    _constraint_status[i] = uno::BoundType::BOUNDED_UPPER;
                    _inequality_constraints.emplace_back(i);
                }
                else
                {
                    _constraint_status[i] = uno::BoundType::UNBOUNDED;
                }
            }
        }

    };



class GoldSteinPriceUserCallbacks
    : public uno::UserCallbacks
{
    std::vector<GoldsteinPriceModel<double>> previousCalls;
    HessianMode mode;
    double damping_threshold = 0.2;

    void notify_acceptable_iterate(const uno::Vector<double>& primals, const uno::Multipliers& multipliers, double objective_multiplier) override {

        if (previousCalls.size()>1 && previousCalls[0].B.isApprox(Eigen::Matrix2d::Identity()))
        {
            Eigen::Vector2d s = previousCalls[1].x - previousCalls[0].x;
            Eigen::Vector2d y = previousCalls[1].g - previousCalls[0].g;
            previousCalls[0].B = Eigen::Matrix2d::Identity()*y.dot(y)/s.dot(y);
        }
        if (previousCalls.size()>1)
            previousCalls.erase(previousCalls.begin());
    }

    void notify_new_primals(const uno::Vector<double>& primals) override {
        GoldsteinPriceModel<double> currentCall(primals);
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
    
    void notify_new_multipliers(const uno::Multipliers& multipliers) override {

    }

    public:
    GoldSteinPriceUserCallbacks(double _x0, double _y0, HessianMode _mode) : mode(_mode) {
        std::array<double, 2> x0{_x0, _y0};
        previousCalls.emplace_back(GoldsteinPriceModel<double>(x0));
    }

    const GoldsteinPriceModel<double>& get_previous_call() const {
        return previousCalls.back();
    }

    const HessianMode& get_mode() const {
        return mode;
    }
};

class GoldSteinPriceUnoModel
        : public DataModel
    {
    private:
        GoldSteinPriceUserCallbacks* callbacks;
    public:
        

        GoldSteinPriceUnoModel(GoldSteinPriceUserCallbacks* _callbacks) : DataModel(2, 0, "GoldsteinPrice"), callbacks(_callbacks) {
            _variable_lower_bounds = {-2., -2.};
            _variable_upper_bounds = {2., 2.};
            initialise_from_data();
        }
        
        double evaluate_objective(const uno::Vector<double> &x) const override { 
            return callbacks->get_previous_call().f; 
        }

        void evaluate_objective_gradient(const uno::Vector<double> &x, uno::SparseVector<double> &gradient) const override
        {
            gradient.insert(0, callbacks->get_previous_call().g[0]);
            gradient.insert(1, callbacks->get_previous_call().g[1]);
        }

        void evaluate_constraints(const uno::Vector<double> &x, std::vector<double> &constraints) const override
        {
        }

        void evaluate_constraint_gradient(const uno::Vector<double> &x, size_t constraint_index, uno::SparseVector<double> &gradient) const override
        {
            
        }
        void evaluate_constraint_jacobian(const uno::Vector<double> &x, uno::RectangularMatrix<double> &constraint_jacobian) const override
        {
            
        }
        void evaluate_lagrangian_hessian(const uno::Vector<double> &x, double objective_multiplier, const uno::Vector<double> &multipliers,
                                         uno::SymmetricMatrix<size_t, double> &hessian) const override
        {
            // return the values
            Eigen::Matrix2d _hessian;
            if (HessianMode::auto_diff == callbacks->get_mode())
            {
                _hessian = callbacks->get_previous_call().hessian();
            }
            else {
                _hessian = callbacks->get_previous_call().B;
            }
            
            hessian.reset();
            hessian.insert(objective_multiplier * _hessian(0,0), 0, 0);
            hessian.insert(objective_multiplier * _hessian(1,0), 1, 0);
            hessian.finalize_column(0);
            hessian.insert(objective_multiplier * _hessian(1,1), 1, 1);
            hessian.finalize_column(1);
        }

        void initial_primal_point(uno::Vector<double> &x) const override {
            std::copy(callbacks->get_previous_call().x.cbegin(), callbacks->get_previous_call().x.cend(), x.begin());
        }

        void initial_dual_point(uno::Vector<double> &multipliers) const override {
            std::fill(multipliers.begin(), multipliers.end(), 0.0);
        }

        size_t number_objective_gradient_nonzeros() const override { return number_variables; }
        size_t number_jacobian_nonzeros() const override { return 2 * number_variables; }
        size_t number_hessian_nonzeros() const override { return number_variables * number_variables; }
        
    };



#endif