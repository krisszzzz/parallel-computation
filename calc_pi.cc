// Calculate pi using Leibnitz series
// Implement sequential calcuation and parrallel one using boost MPI API

#include <boost/hana.hpp>
#include <boost/mpi.hpp>
#include <boost/type_index.hpp>
#include <format>
#include <iostream>

// No CRTP needed now, we can use deducing this now
class PiCalcIface {
public:
  PiCalcIface(boost::mpi::communicator &comm_ref) : comm_ref_(comm_ref){};
  template <typename Self> auto &&getCommunicator(this Self &&self) {
    return self.comm_ref_;
  }

  template <typename Self>
  double calculatePi(this Self &&self, int num_of_terms) {
    return self.calculatePi(num_of_terms);
  }

protected:
  double calculatePiPartial(int term_begin, int term_end) const {
    double sum = 0.0;
    for (int i = term_begin; i < term_end; i++) {
      double sign = i % 2 == 0 ? 1.0 : -1.0;
      sum += (sign * 1) / (2.0 * i + 1.0);
    }
    return 4 * sum;
  }

private:
  boost::mpi::communicator &comm_ref_;
};

class PiCalcSeq : public PiCalcIface {
public:
  double calculatePi(int num_of_terms) const {
    double sum = 0.0;
    if (getCommunicator().rank() == 0) {
      sum = calculatePiPartial(0, num_of_terms);
    }
    // wait for all processes
    getCommunicator().barrier();
    return sum;
  }
};

class PiCalcParallel : public PiCalcIface {
public:
  double calculatePi(int num_of_terms) const {
    constexpr int tag = 0;
    int terms_per_proc = num_of_terms / getCommunicator().size();
    double sum = 0.0;

    if (getCommunicator().rank() == 0) {
      // collect result from ranks [1, size()]
      for (int i = 1; i < getCommunicator().size(); i++) {
        double partial_sum = 0.0;
        getCommunicator().recv(i, tag, partial_sum);
        sum += partial_sum;
      }
    } else {
      int terms_begin = terms_per_proc * (getCommunicator().rank() - 1);
      int terms_end = terms_begin + terms_per_proc;
      double res = calculatePiPartial(terms_begin, terms_end);

      getCommunicator().send(0, tag, res);
    }

    return sum;
  }

private:
};

int main(int argc, char *argv[]) {
  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator world;

  PiCalcSeq pi_seq{world};
  PiCalcParallel pi_par{world};

  int curr_rank = world.rank();
  auto tp = boost::hana::make_tuple(pi_seq, pi_par);

  boost::hana::for_each(tp, [curr_rank](const auto &impl) {
    constexpr int num_of_terms = 1'000'000;
    constexpr int test_count = 1'000;
    double result = 0.0;
    boost::mpi::timer time;
    double begin = time.elapsed();
    for (int i = 0; i < test_count; i++) {
      result = impl.calculatePi(num_of_terms);
    }

    double passed_time = time.elapsed() - begin;

    // output only once
    if (curr_rank == 0) {
      std::cout
          << std::format(
                 "Implemenation: {:15}, result: {:.10}, time: {:.5}",
                 boost::typeindex::type_id<decltype(impl)>().pretty_name(),
                 result, passed_time)
          << "\n";
    }
  });
}
