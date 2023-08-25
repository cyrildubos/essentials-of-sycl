#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue;

  std::cout << "info::device::name          : "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';
  std::cout << "info::device::local_mem_size: "
            << queue.get_device().get_info<sycl::info::device::local_mem_size>()
            << '\n';

  auto type = queue.get_device().get_info<sycl::info::device::local_mem_type>();

  if (type == sycl::info::local_mem_type::local)
    std::cout << "info::device::local_mem_type: info::local_mem_type::local"
              << "\n";
  else if (type == sycl::info::local_mem_type::global)
    std::cout << "info::device::local_mem_type: info::local_mem_type::global"
              << "\n";
  else if (type == sycl::info::local_mem_type::none)
    std::cout << "info::device::local_mem_type: info::local_mem_type::none"
              << "\n";

  return 0;
}