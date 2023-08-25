#include <iomanip>

#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

int main() {
  std::vector<float> matrix_a(size * size);
  std::vector<float> matrix_b(size * size);
  std::vector<float> matrix_c(size * size);
  std::vector<float> matrix_d(size * size);

  auto value_0 = 2.0f;
  auto value_1 = 3.0f;

  for (auto index_i = 0; index_i < size; ++index_i)
    for (auto index_j = 0; index_j < size; ++index_j) {
      matrix_a[index_i * size + index_j] = value_0++;
      matrix_b[index_i * size + index_j] = value_1++;
      matrix_c[index_i * size + index_j] = 0.0f;
      matrix_d[index_i * size + index_j] = 0.0f;
    }

  sycl::queue queue;

  sycl::buffer buffer_a(matrix_a);
  sycl::buffer buffer_b(matrix_b);
  sycl::buffer buffer_c(matrix_c);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::read_only);
    sycl::accessor accessor_b(buffer_b, handler, sycl::read_only);
    sycl::accessor accessor_c(buffer_c, handler, sycl::write_only);

    sycl::range<2> global_size(size, size);
    sycl::range<2> group_size(size, size);

    sycl::local_accessor<float, 2> local_a(sycl::range<2>(size, size), handler);
    sycl::local_accessor<float, 2> local_b(sycl::range<2>(size, size), handler);

    handler.parallel_for(
        sycl::nd_range<2>(global_size, group_size), [=](auto item) {
          const auto index_i = item.get_global_id(0);
          const auto index_j = item.get_global_id(1);

          const auto index_x = item.get_local_id(0);
          const auto index_y = item.get_local_id(1);

          local_a[index_x][index_y] = accessor_a[index_i * size + index_j];
          local_b[index_x][index_y] = accessor_b[index_i * size + index_j];

          sycl::group_barrier(item.get_group());

          auto value = 0.0f;

          for (auto index_k = 0; index_k < size; ++index_k)
            value += local_a[index_x][index_k] * local_b[index_k][index_y];

          accessor_c[index_i * size + index_j] = value;
        });
  });

  sycl::host_accessor accessor_c(buffer_c, sycl::read_only);

  auto has_failed = false;

  for (auto index_i = 0; index_i < size; ++index_i) {
    for (auto index_j = 0; index_j < size; ++index_j) {
      for (auto index_k = 0; index_k < size; ++index_k)
        matrix_d[index_i * size + index_j] +=
            matrix_a[index_i * size + index_k] *
            matrix_b[index_k * size + index_j];

      if (matrix_d[index_i * size + index_j] !=
          matrix_c[index_i * size + index_j])
        has_failed = true;

      std::cout << std::setw(6) << matrix_c[index_i * size + index_j] << ' ';
    }

    std::cout << '\n';
  }

  std::cout << (has_failed ? "FAIL" : "PASS") << '\n';

  return 0;
}