// Copyright (C) 2004, 2009 International Business Machines and others.
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-11-05

#include "IpIpoptApplication.hpp"
#include "IpSolveStatistics.hpp"
#include "nlp.hpp"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <arrow/util/type_fwd.h>

#include <iostream>
#include <vector>
#include <numeric>

struct Result
{
    double x{};
    double y{};
    double fVal{};
    std::size_t iter{};
    int status{};
    std::string exit_message{};
    double cpu_time{};
};

Result singlePointSolve(double x0, double y0, HessianMode mode = HessianMode::auto_diff)
{
    Result retval{};
    // Create an instance of your nlp...
    Ipopt::SmartPtr<Ipopt::TNLP> nlp = new GoldsteinPrice(x0, y0);
    // Create an instance of the IpoptApplication
    //
    // We are using the factory, since this allows us to compile this
    // example with an Ipopt Windows DLL
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app = IpoptApplicationFactory();
    app->Options()->SetIntegerValue("print_level", 0);
    if (mode == HessianMode::ipopt_lbfgs)
    {
        app->Options()->SetStringValue("hessian_approximation", "limited-memory");
    }
    // Initialize the IpoptApplication and process the options
    Ipopt::ApplicationReturnStatus status;
    status = app->Initialize();
    if (status != Ipopt::Solve_Succeeded)
        return retval;

    status = app->OptimizeTNLP(nlp);

    retval.iter = app->Statistics()->IterationCount();
    retval.fVal = app->Statistics()->FinalObjective();
    retval.x = reinterpret_cast<GoldsteinPrice *>(GetRawPtr(nlp))->x0[0];
    retval.y = reinterpret_cast<GoldsteinPrice *>(GetRawPtr(nlp))->x0[1];
    retval.status = static_cast<int>(status);
    retval.exit_message = reinterpret_cast<GoldsteinPrice *>(GetRawPtr(nlp))->exit_status;
    retval.cpu_time = app->Statistics()->TotalCpuTime();
    return retval;
}

arrow::Status writeToParquet(std::string filename,
                             const std::vector<double> &x,
                             const std::vector<double> &y,
                             const std::vector<double> &fVal,
                             const std::vector<std::size_t> &iter,
                             const std::vector<int> &status,
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

    arrow::Int32Builder int32Builder;
    ARROW_RETURN_NOT_OK(int32Builder.AppendValues(status));
    std::shared_ptr<arrow::Array> arr_status;
    ARROW_ASSIGN_OR_RAISE(arr_status, int32Builder.Finish());

    arrow::StringBuilder stringbuilder;
    ARROW_RETURN_NOT_OK(stringbuilder.AppendValues(exit_message));
    std::shared_ptr<arrow::Array> arr_exit_message;
    ARROW_ASSIGN_OR_RAISE(arr_exit_message, stringbuilder.Finish());

    std::shared_ptr<arrow::Field>   field_x = arrow::field("x", arrow::float64()), 
                                    field_y = arrow::field("y", arrow::float64()), 
                                    field_fVal = arrow::field("fVal", arrow::float64()), 
                                    field_iter = arrow::field("iter", arrow::uint64()), 
                                    field_status = arrow::field("status", arrow::int32()), 
                                    field_exit_message = arrow::field("exit_message", arrow::utf8()), 
                                    field_time = arrow::field("cpu_time", arrow::float64()),
                                    field_x0 = arrow::field("x0", arrow::float64()),
                                    field_y0 = arrow::field("y0", arrow::float64());
    std::shared_ptr<arrow::Schema> schema = arrow::schema({field_x,field_y,field_fVal,field_iter,field_status,field_exit_message,field_time,field_x0,field_y0});

    std::shared_ptr<arrow::RecordBatch> rbatch = arrow::RecordBatch::Make(schema, arr_x->length(), {arr_x,arr_y,arr_fVal,arr_iter,arr_status,arr_exit_message,arr_time,arr_x0,arr_y0});

    std::shared_ptr<arrow::Table> tbl = arrow::Table::Make(schema, {arr_x,arr_y,arr_fVal,arr_iter,arr_status,arr_exit_message,arr_time,arr_x0,arr_y0}, arr_x->length());

    std::shared_ptr<arrow::io::FileOutputStream> outfile;

    ARROW_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(filename));
  
    std::shared_ptr<parquet::WriterProperties> props = parquet::WriterProperties::Builder()
        .max_row_group_length(64 * 1024)
        ->created_by("IpoptScanner")
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

int main(
    int,
    char **)
{

    double xLb = -2., yLb = -3.;
    double xUb = 2., yUb = 1.;
    std::size_t nX = 173, nY = 181;

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> fVal;
    std::vector<std::size_t> iter;
    std::vector<int> status;
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
        std::vector<int> status_inner;
        std::vector<std::string> exit_message_inner;
        std::vector<double> cpu_time_inner;
        std::vector<double> x0_inner;
        std::vector<double> y0_inner;
        for (std::size_t iY = 0; iY < nY; ++iY)
        {
            double _x0 = xLb + iX * (xUb - xLb) / (nX - 1);
            double _y0 = yLb + iY * (yUb - yLb) / (nY - 1);
            auto res = singlePointSolve(_x0, _y0, HessianMode::bfgs);
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
        std::cout << "Total CPU time: " << tTotal << "s to process " << cpu_time_inner.size() << " problems " << fVal.size()/nY << "/" << nX << "." << std::endl;
    }

    double tTotal = std::accumulate(cpu_time.begin(), cpu_time.end(), 0.);
    std::cout << "Total CPU time: " << tTotal << "s for " << nX * nY << " problems." << std::endl;

    std::string modename = "bfgs";

    auto st = writeToParquet(modename + ".parquet", x, y, fVal, iter, status, exit_message, cpu_time, x0, y0);
    if (!st.ok())
    {
        std::cerr << st << std::endl;
        return 1;
    } else {
        std::cout << "Wrote to " << modename << ".parquet" << std::endl;
    }

    return 0;
}