#include <sycl/sycl.hpp>

constexpr std::size_t size = 1'024;

int main() {
  sycl::queue queue;

  std::vector<int> vector_in(size);
  std::vector<int> vector_out(size);

  for (auto index = 0; index < size; ++index)
    vector_in[index] = index;

  std::fill(vector_out.begin(), vector_out.end(), 0);

  sycl::buffer buffer_in(vector_in);
  sycl::buffer buffer_out(vector_out);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_in(buffer_in, handler);
    sycl::accessor accessor_out(buffer_out, handler);

    handler.parallel_for(sycl::range<1>(size), [=](auto identifier) {
      accessor_out[identifier] = accessor_in[identifier] * 2;
    });
  });

  sycl::host_accessor accessor_out(buffer_out);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_out[index] << '\n';

  return 0;
}