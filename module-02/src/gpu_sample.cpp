#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue;

  std::cout << "Device: "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';

  return 0;
}