#include <sycl/sycl.hpp>

int main() {
  std::vector<int> vector;

  sycl::buffer buffer_1(vector);
  sycl::buffer buffer_2(vector.begin(), vector.end());

  std::array<int, 42> array;

  sycl::buffer buffer_3(array);

  double data[4] = {1.1, 2.2, 3.3, 4.4};

  sycl::buffer buffer_4(data, sycl::range(4));

  return 0;
}