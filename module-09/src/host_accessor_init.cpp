#include <sycl/sycl.hpp>

constexpr std::size_t size = 1'024;

int main() {
  sycl::queue queue;

  sycl::buffer<int> buffer_in(size);
  sycl::buffer<int> buffer_out(size);

  {
    sycl::host_accessor accessor_in(buffer_in);
    sycl::host_accessor accessor_out(buffer_out);

    for (auto index = 0; index < size; ++index) {
      accessor_in[index] = index;
      accessor_out[index] = 0;
    }
  }

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_in(buffer_in, handler);
    sycl::accessor accessor_out(buffer_out, handler);

    handler.parallel_for(sycl::range<1>(size), [=](auto identifier) {
      accessor_out[identifier] = accessor_in[identifier];
    });
  });

  sycl::host_accessor accessor_out(buffer_out);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_out[index] << '\n';

  return 0;
}