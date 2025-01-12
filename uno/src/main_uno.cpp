
#include <iostream>
#include <string>
#include <stdexcept>
#include "ingredients/globalization_mechanisms/GlobalizationMechanism.hpp"
#include "ingredients/globalization_mechanisms/GlobalizationMechanismFactory.hpp"
#include "ingredients/constraint_relaxation_strategies/ConstraintRelaxationStrategy.hpp"
#include "ingredients/constraint_relaxation_strategies/ConstraintRelaxationStrategyFactory.hpp"
#include "uno_model.hpp"
#include "Uno.hpp"
#include "model/ModelFactory.hpp"
#include "options/Options.hpp"
#include "options/Presets.hpp"
#include "options/DefaultOptions.hpp"
#include "tools/Logger.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <arrow/util/type_fwd.h>

struct Result
{
    double x{};
    double y{};
    double fVal{};
    std::size_t iter{};
    std::string status{};
    std::string exit_message{};
    double cpu_time{};
};

Result singlePointSolve(double x0, double y0, HessianMode mode = HessianMode::auto_diff, const std::string &print_level = "SILENT", double damping_threshold = 0.2, const std::string& preset_ = "filtersqp") {
   
   uno::Options options = uno::DefaultOptions::load();
   uno::Options solvers_options = uno::DefaultOptions::determine_solvers();
   options.overwrite_with(solvers_options);
   uno::Options preset = uno::Presets::get_preset_options(preset_);
   options.overwrite_with(preset);
   uno::Logger::set_logger(print_level);
   
   GoldSteinPriceUserCallbacks user_callbacks(x0, y0, mode);
   std::unique_ptr<uno::Model> nlp_model = std::make_unique<GoldSteinPriceUnoModel>(&user_callbacks);
   std::unique_ptr<uno::Model> model = uno::ModelFactory::reformulate(std::move(nlp_model), options);
   
   uno::Iterate initial_iterate(model->number_variables, model->number_constraints);
   model->initial_primal_point(initial_iterate.primals);
   model->project_onto_variable_bounds(initial_iterate.primals);
   model->initial_dual_point(initial_iterate.multipliers.constraints);
   initial_iterate.feasibility_multipliers.reset();

   auto constraint_relaxation_strategy = uno::ConstraintRelaxationStrategyFactory::create(*model, options);
   auto globalization_mechanism = uno::GlobalizationMechanismFactory::create(*constraint_relaxation_strategy, options);
   uno::Uno uno = uno::Uno(*globalization_mechanism, options);

   
   // solve the instance
   uno::Result result = uno.solve(*model, initial_iterate, options, user_callbacks);
   Result retval;
   retval.x = result.solution.primals[0];
   retval.y = result.solution.primals[1];
   retval.fVal = result.solution.evaluations.objective;
   retval.iter = result.iteration;
   retval.cpu_time = result.cpu_time;
   retval.exit_message = optimization_status_to_message(result.optimization_status);
   retval.status = iterate_status_to_message(result.solution.status);
   return retval;

}

arrow::Status writeToParquet(std::string filename,
                             const std::vector<double> &x,
                             const std::vector<double> &y,
                             const std::vector<double> &fVal,
                             const std::vector<std::size_t> &iter,
                             const std::vector<std::string> &status,
                             const std::vector<std::string> &exit_message,
                             const std::vector<double> &cpu_time,
                             const std::vector<double> &x0,
                             const std::vector<double> &y0)
{
    arrow::DoubleBuilder doubleBuilder;
    ARROW_RETURN_NOT_OK(doubleBuilder.AppendValues(x));
    std::shared_ptr<arrow::Array> arr_x;
    ARROW_ASSIGN_OR_RAISE(arr_x, doubleBuilder.Finish());
    ARROW_RETURN_NOT_OK(doubleBuilder.AppendValues(y));
    std::shared_ptr<arrow::Array> arr_y;
    ARROW_ASSIGN_OR_RAISE(arr_y, doubleBuilder.Finish());
    ARROW_RETURN_NOT_OK(doubleBuilder.AppendValues(fVal));
    std::shared_ptr<arrow::Array> arr_fVal;
    ARROW_ASSIGN_OR_RAISE(arr_fVal, doubleBuilder.Finish());
    ARROW_RETURN_NOT_OK(doubleBuilder.AppendValues(cpu_time));
    std::shared_ptr<arrow::Array> arr_time;
    ARROW_ASSIGN_OR_RAISE(arr_time, doubleBuilder.Finish());
    std::shared_ptr<arrow::Array> arr_x0, arr_y0;
    ARROW_RETURN_NOT_OK(doubleBuilder.AppendValues(x0));
    ARROW_ASSIGN_OR_RAISE(arr_x0, doubleBuilder.Finish());
    ARROW_RETURN_NOT_OK(doubleBuilder.AppendValues(y0));
    ARROW_ASSIGN_OR_RAISE(arr_y0, doubleBuilder.Finish());

    arrow::UInt64Builder uint64Builder;
    ARROW_RETURN_NOT_OK(uint64Builder.AppendValues(iter));
    std::shared_ptr<arrow::Array> arr_iter;
    ARROW_ASSIGN_OR_RAISE(arr_iter, uint64Builder.Finish());

    arrow::StringBuilder stringbuilder;
    std::shared_ptr<arrow::Array> arr_status;
    ARROW_RETURN_NOT_OK(stringbuilder.AppendValues(status));
    ARROW_ASSIGN_OR_RAISE(arr_status, stringbuilder.Finish());

    ARROW_RETURN_NOT_OK(stringbuilder.AppendValues(exit_message));
    std::shared_ptr<arrow::Array> arr_exit_message;
    ARROW_ASSIGN_OR_RAISE(arr_exit_message, stringbuilder.Finish());

    std::shared_ptr<arrow::Field> field_x = arrow::field("x", arrow::float64()),
                                  field_y = arrow::field("y", arrow::float64()),
                                  field_fVal = arrow::field("fVal", arrow::float64()),
                                  field_iter = arrow::field("iter", arrow::uint64()),
                                  field_status = arrow::field("status", arrow::utf8()),
                                  field_exit_message = arrow::field("exit_message", arrow::utf8()),
                                  field_time = arrow::field("cpu_time", arrow::float64()),
                                  field_x0 = arrow::field("x0", arrow::float64()),
                                  field_y0 = arrow::field("y0", arrow::float64());
    std::shared_ptr<arrow::Schema> schema = arrow::schema({field_x, field_y, field_fVal, field_iter, field_status, field_exit_message, field_time, field_x0, field_y0});

    std::shared_ptr<arrow::RecordBatch> rbatch = arrow::RecordBatch::Make(schema, arr_x->length(), {arr_x, arr_y, arr_fVal, arr_iter, arr_status, arr_exit_message, arr_time, arr_x0, arr_y0});

    std::shared_ptr<arrow::Table> tbl = arrow::Table::Make(schema, {arr_x, arr_y, arr_fVal, arr_iter, arr_status, arr_exit_message, arr_time, arr_x0, arr_y0}, arr_x->length());

    std::shared_ptr<arrow::io::FileOutputStream> outfile;

    ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(filename));

    std::shared_ptr<parquet::WriterProperties> props = parquet::WriterProperties::Builder()
                                                           .max_row_group_length(64 * 1024)
                                                           ->created_by("UnoScanner")
                                                           ->version(parquet::ParquetVersion::PARQUET_2_LATEST)
                                                           ->data_page_version(parquet::ParquetDataPageVersion::V2)
                                                           ->compression(arrow::Compression::SNAPPY)
                                                           ->build();

    std::shared_ptr<parquet::ArrowWriterProperties> arrow_props =
        parquet::ArrowWriterProperties::Builder()
            .store_schema()
            ->build();

    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*tbl,
                                                   arrow::default_memory_pool(), outfile,
                                                   /*chunk_size=*/3, props, arrow_props));

    return arrow::Status::OK();
}

HessianMode mapMode(const std::string &approximation){
    HessianMode mode = HessianMode::auto_diff;
    if ("autodiff" == approximation)
    {
        mode = HessianMode::auto_diff;
    }
    else if ("dfp" == approximation)
    {
        mode = HessianMode::dfp;
    }
    else if ("bfgs" == approximation)
    {
        mode = HessianMode::bfgs;
    }
    else if ("sr1" == approximation)
    {
        mode = HessianMode::sr1;
    }
    else if ("lsr1" == approximation)
    {
        mode = HessianMode::lsr1;
    }
    else if ("ldfp" == approximation)
    {
        mode = HessianMode::ldfp;
    }
    else if ("lbfgs" == approximation)
    {
        mode = HessianMode::lbfgs;
    }
    else
    {
        std::cerr << "Unknown approximation method: " << approximation << std::endl;
        return HessianMode::auto_diff;
    }
    return mode;
}



void doScan(const std::string &approximation, double damping_threshold)
{
    HessianMode mode = mapMode(approximation);
    double xLb = -2., yLb = -2.;
    double xUb = 2., yUb = 2.;
    std::size_t nX = 173, nY = 181;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> fVal;
    std::vector<std::size_t> iter;
    std::vector<std::string> status;
    std::vector<std::string> exit_message;
    std::vector<double> cpu_time;
    std::vector<double> x0;
    std::vector<double> y0;
#pragma omp parallel for shared(x, y, fVal, iter, status, exit_message, cpu_time, x0, y0)
    for (std::size_t iX = 0; iX < nX; ++iX)
    {
        std::vector<double> x_inner;
        std::vector<double> y_inner;
        std::vector<double> fVal_inner;
        std::vector<std::size_t> iter_inner;
        std::vector<std::string> status_inner;
        std::vector<std::string> exit_message_inner;
        std::vector<double> cpu_time_inner;
        std::vector<double> x0_inner;
        std::vector<double> y0_inner;
        for (std::size_t iY = 0; iY < nY; ++iY)
        {
            double _x0 = xLb + iX * (xUb - xLb) / (nX - 1);
            double _y0 = yLb + iY * (yUb - yLb) / (nY - 1);
            auto res = singlePointSolve(_x0, _y0, mode, "SILENT", damping_threshold);
            x_inner.emplace_back(res.x);
            y_inner.emplace_back(res.y);
            fVal_inner.emplace_back(res.fVal);
            iter_inner.emplace_back(res.iter);
            status_inner.emplace_back(res.status);
            exit_message_inner.emplace_back(res.exit_message);
            cpu_time_inner.emplace_back(res.cpu_time);
            x0_inner.emplace_back(_x0);
            y0_inner.emplace_back(_y0);
        }
#pragma omp critical
        {
            x.insert(x.end(), x_inner.begin(), x_inner.end());
            y.insert(y.end(), y_inner.begin(), y_inner.end());
            fVal.insert(fVal.end(), fVal_inner.begin(), fVal_inner.end());
            iter.insert(iter.end(), iter_inner.begin(), iter_inner.end());
            status.insert(status.end(), status_inner.begin(), status_inner.end());
            exit_message.insert(exit_message.end(), exit_message_inner.begin(), exit_message_inner.end());
            cpu_time.insert(cpu_time.end(), cpu_time_inner.begin(), cpu_time_inner.end());
            x0.insert(x0.end(), x0_inner.begin(), x0_inner.end());
            y0.insert(y0.end(), y0_inner.begin(), y0_inner.end());
        }
        double tTotal = std::accumulate(cpu_time_inner.begin(), cpu_time_inner.end(), 0.);
        std::cout << "Total CPU time: " << tTotal << "s to process " << cpu_time_inner.size() << " problems " << fVal.size() / nY << "/" << nX << "." << std::endl;
    }

    double tTotal = std::accumulate(cpu_time.begin(), cpu_time.end(), 0.);
    std::cout << "Total CPU time: " << tTotal << "s for " << nX * nY << " problems." << std::endl;

    auto st = writeToParquet(approximation + ".parquet", x, y, fVal, iter, status, exit_message, cpu_time, x0, y0);
    if (!st.ok())
    {
        std::cerr << st << std::endl;
        return;
    }
    else
    {
        std::cout << "Wrote to " << approximation << ".parquet" << std::endl;
    }
}

void singleShot(HessianMode mode, double damping_threshold)
{
    auto res = singlePointSolve(0., 0., mode, "SILENT", damping_threshold);
}

int main(
    int argc,
    char **argv)
{

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " --scan|--single <approximation>" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string approximation = argv[2];
    double damping_threshold = 0.2;
    if (argc > 3)
    {
        damping_threshold = std::stod(argv[3]);
    }

    if (mode == "--scan")
    {
        doScan(approximation, damping_threshold);
    }
    else if (mode == "--single")
    {
        singleShot(mapMode(approximation), damping_threshold);
    }
    else
    {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}