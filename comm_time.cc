// Calculate overhead for communacation using MPI send() and recv() functions

#include <boost/mpi.hpp>
#include <format>
#include <iostream>

int main(int argc, char *argv[]) {
  boost::mpi::environment env{argc, argv};
  boost::mpi::communicator world;
  boost::mpi::timer time;

  constexpr int n_test = 100'000;
  constexpr int dest = 1;
  constexpr int tag = 0;
  assert(world.size() == 2 && "Expect 2 mpi processes\n");

  if (world.rank() == 0) {
    world.barrier();

    double send_begin = time.elapsed();
    for (int i = 0; i < n_test; i++) {
      double any_data = 1.0;
      world.send(dest, tag, any_data);
    }
    double send_end = time.elapsed() - send_begin;
    std::cout << std::format("Send time (for n_test = {}): {:.5f}", n_test,
                             send_end)
              << "\n";
    world.barrier();
    double recv_begin = time.elapsed();
    for (int i = 0; i < n_test; i++) {
      double recv_data = 0.0;
      world.recv(boost::mpi::any_source, tag, recv_data);
    }
    double recv_end = time.elapsed() - recv_begin;

    std::cout << std::format("Recv time (for n_test = {}): {:.5f}", n_test,
                             recv_end)
              << "\n";

  } else {
    world.barrier();

    double recv_data = 0.0;
    for (int i = 0; i < n_test; i++) {
      world.recv(boost::mpi::any_source, tag, recv_data);
    }

    world.barrier();
    for (int i = 0; i < n_test; i++) {
      double any_data = 1.0;
      world.send(0, tag, any_data);
    }
  }
}