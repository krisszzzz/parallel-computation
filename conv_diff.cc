
#include <boost/hana.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/type_index.hpp>
#include <cmath>
#include <matplot/matplot.h>

namespace conv_eq {
struct Ranges {
  int x_steps;
  int t_steps;
  double x_max;
  double t_max;
};

class SolverBase {
public:
  using InitFunc = std::function<double(double)>;
  // getters
  template <typename Self> auto &&getRanges(this Self &&self) {
    return std::forward<Self>(self).ranges_;
  }

  template <typename Self> auto &&getXCond(this Self &&self) {
    return std::forward<Self>(self).x_cond_;
  }

  template <typename Self> auto &&getTCond(this Self &&self) {
    return std::forward<Self>(self).t_cond_;
  }

  template <typename Self> matplot::vector_2d solveEquation(this Self &&self) {
    return self.solveEquationImpl();
  }

  template <typename Self> auto meshGrid(this Self &&self) {
    return matplot::meshgrid(matplot::linspace(0, self.getRanges().x_max,
                                               self.getRanges().x_steps + 1),
                             matplot::linspace(0, self.getRanges().t_max,
                                               self.getRanges().t_steps + 1));
  }

protected:
  void setInitConds(const InitFunc &x_func, const InitFunc &t_func) {
    double x_step = getRanges().x_max / getRanges().x_steps;
    double t_step = getRanges().t_max / getRanges().t_steps;
    for (int i = 0; i < getRanges().x_steps + 1; i++) {
      getXCond()[i] = x_func(x_step * i);
    }
    for (int i = 0; i < getRanges().t_steps + 1; i++) {
      getTCond()[i] = t_func(t_step * i);
    }
  }

private:
  Ranges ranges_;
  matplot::vector_1d x_cond_;
  matplot::vector_1d t_cond_;
};

class SolverSeq : public SolverBase {
public:
  friend class SolverBase;

  SolverSeq(Ranges &ranges, const InitFunc &x_func, const InitFunc &t_func) {
    getRanges() = ranges;
    getXCond().resize(ranges.x_steps + 1);
    getTCond().resize(ranges.t_steps + 1);
    setInitConds(std::forward<const InitFunc>(x_func),
                 std::forward<const InitFunc>(t_func));
  }

private:
  matplot::vector_2d solveEquationImpl() const {
    matplot::vector_2d U(getRanges().t_steps + 1,
                         matplot::vector_1d(getRanges().x_steps + 1));
    double x_step = getRanges().x_max / getRanges().x_steps;
    double t_step = getRanges().t_max / getRanges().t_steps;

    // set init conditions
    for (int i = 0; i < getRanges().x_steps + 1; i++) {
      U[0][i] = getXCond()[i];
    }
    for (int i = 0; i < getRanges().t_steps + 1; i++) {
      U[i][0] = getTCond()[i];
    }

    for (int k = 0; k < getRanges().t_steps; k++) {
      int m = 1;
      for (; m < getRanges().x_steps; m++) {
        // 4-point difference scheme
        U[k + 1][m] = U[k][m] -
                      (t_step / (2 * x_step)) * (U[k][m + 1] - U[k][m - 1]) +
                      (0.5 * t_step * t_step / (x_step * x_step)) *
                          (U[k][m + 1] - 2 * U[k][m] + U[k][m - 1]);
      }
      // 3-point difference scheme for boundary
      U[k + 1][m] = U[k][m] - (t_step / x_step) * (U[k][m] - U[k][m - 1]);
    }

    return U;
  }
};

class SolverParallel : public SolverBase {
public:
  friend class SolverBase;

  template <typename Self> auto &&getCommunicator(this Self &&self) {
    return self.comm_ref_;
  }

  SolverParallel(boost::mpi::communicator &comm_ref, Ranges &ranges,
                 const InitFunc &x_func, const InitFunc &t_func)
      : comm_ref_(comm_ref) {
    getRanges() = ranges;
    getXCond().resize(ranges.x_steps + 1);
    getTCond().resize(ranges.t_steps + 1);
    setInitConds(std::forward<const InitFunc>(x_func),
                 std::forward<const InitFunc>(t_func));
  }

private:
  matplot::vector_2d solveEquationImpl() const {
    constexpr int tag = 0;
    int nodes_per_proc =
        (getRanges().x_steps - 1) / (getCommunicator().size() - 1) + 1;

    int curr_rank = getCommunicator().rank();

    if (curr_rank == 0) {
      matplot::vector_2d U(getRanges().t_steps + 1,
                           matplot::vector_1d(getRanges().x_steps + 1));
      for (int i = 0; i < getRanges().x_steps + 1; i++) {
        U[0][i] = getXCond()[i];
      }
      for (int i = 0; i < getRanges().t_steps + 1; i++) {
        U[i][0] = getTCond()[i];
      }

      for (int i = 1; i < getCommunicator().size(); i++) {
        int data_begin = (i - 1) * nodes_per_proc + 1;
        int data_end =
            std::min(i * nodes_per_proc + 1, getRanges().x_steps + 1);
        int data_length = data_end - data_begin;

        matplot::vector_2d U_per_proc(getRanges().t_steps,
                                      matplot::vector_1d(data_length));
        getCommunicator().recv(i, tag, U_per_proc);

        for (int k = 1; k < getRanges().t_steps + 1; k++) {
          for (int m = data_begin; m < data_end; m++) {
            U[k][m] = U_per_proc[k - 1][m - data_begin];
          }
        }
      }
      return U;
    } else {
      int data_begin = (getCommunicator().rank() - 1) * nodes_per_proc;
      int data_end = std::min(getCommunicator().rank() * nodes_per_proc,
                              getRanges().x_steps + 1);
      int data_length = data_end - data_begin;
      bool is_last = data_end == getRanges().x_steps + 1;
      // bool is_last = curr_rank - 1 == getCommunicator().size();
      matplot::vector_2d U(getRanges().t_steps,
                           matplot::vector_1d(data_length));
      // it is required 2 additional points to calculate next grid
      // using 4-point scheme
      matplot::vector_1d prev_grid(data_length + 2);

      // set initial conditions
      for (int i = data_begin;
           i < std::min(data_begin + data_length + 2, getRanges().x_steps + 1);
           i++) {
        prev_grid[i - data_begin] = getXCond()[i];
      }

      for (int k = 0; k < getRanges().t_steps; k++) {
        getNextGrid(U[k], prev_grid, data_length, is_last);

        if (curr_rank % 2 == 0) {
          if (curr_rank > 1) {
            getCommunicator().send(curr_rank - 1, tag, U[k][0]);
            getCommunicator().recv(curr_rank - 1, tag, prev_grid[0]);
          }
          if (curr_rank < getCommunicator().size() - 1) {
            getCommunicator().recv(curr_rank + 1, tag,
                                   prev_grid[prev_grid.size() - 1]);
            getCommunicator().send(curr_rank + 1, tag, U[k][data_length - 1]);
          }
        } else {
          if (curr_rank < getCommunicator().size() - 1) {
            getCommunicator().recv(curr_rank + 1, tag,
                                   prev_grid[prev_grid.size() - 1]);
            getCommunicator().send(curr_rank + 1, tag, U[k][data_length - 1]);
          }
          if (curr_rank > 1) {
            getCommunicator().send(curr_rank - 1, tag, U[k][0]);
            getCommunicator().recv(curr_rank - 1, tag, prev_grid[0]);
          }
        }

        if (curr_rank == 1) {
          prev_grid[0] = getTCond()[k];
        }

        std::copy(U[k].begin(), U[k].end(), std::next(prev_grid.begin()));
      }

      getCommunicator().send(0, tag, U);

      return {};
    }
  }

  void getNextGrid(matplot::vector_1d &next_grid,
                   const matplot::vector_1d &prev_grid, int grid_len,
                   bool is_last) const {
    double x_step = getRanges().x_max / getRanges().x_steps;
    double t_step = getRanges().t_max / getRanges().t_steps;

    // process last point using 3-point scheme
    if (is_last) {
      grid_len--;
    }

    int m = 1;
    for (; m < grid_len + 1; m++) {
      // 4-point difference scheme
      next_grid[m - 1] =
          prev_grid[m] -
          (t_step / (2 * x_step)) * (prev_grid[m + 1] - prev_grid[m - 1]) +
          (0.5 * t_step * t_step / (x_step * x_step)) *
              (prev_grid[m + 1] - 2 * prev_grid[m] + prev_grid[m - 1]);
    }

    if (is_last) {
      // 3-point difference scheme for boundary
      next_grid[m - 1] =
          prev_grid[m] - (t_step / x_step) * (prev_grid[m] - prev_grid[m - 1]);
    }
  }

  boost::mpi::communicator &comm_ref_;
};
}; // namespace conv_eq

int main(int argc, char *argv[]) {
  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator world;

  using namespace conv_eq;

  Ranges ranges{
      .x_steps = 40,
      .t_steps = 80,

      .x_max = 3,
      .t_max = 3,
  };
  SolverSeq solver_seq{
      ranges, [](double x) { return std::exp(-x) * std::sin(x) * std::sin(x); },
      [](double t) { return 0.0; }};

  SolverParallel solver_par{
      world, ranges,
      [](double x) { return std::exp(-x) * std::sin(x) * std::sin(x); },
      [](double t) { return 0.0; }};

  int curr_rank = world.rank();
  auto tp = boost::hana::make_tuple(solver_seq, solver_par);

  boost::hana::for_each(tp, [curr_rank](const auto &impl) {
    constexpr int test_count = 10'000;

    boost::mpi::timer time;
    double begin = time.elapsed();

    auto [X, T] = impl.meshGrid();
    matplot::vector_2d U{};

    for (int i = 0; i < test_count; i++) {
      if constexpr (std::is_same_v<std::decay_t<decltype(impl)>, SolverSeq>) {
        if (curr_rank == 0) {
          U = impl.solveEquation();
        }
      } else {
        U = impl.solveEquation();
      }
    }

    double passed_time = time.elapsed() - begin;

    // the counted result in the process with rank == 0
    if (curr_rank == 0) {
      std::cout << std::format("Implemenation: {:15}, time: {:.5}",
                               boost::typeindex::type_id<decltype(impl)>()
                                   .pretty_name(),
                               passed_time)
                << "\n";
      matplot::surf(X, T, U);
      matplot::show();
    }
  });
}
