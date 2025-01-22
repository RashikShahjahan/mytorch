#include "tensor.h"

int main() {
    // Suppose we have x = 2.0, y = 3.0
    auto x = Tensor::create(xt::xarray<double>({2.0}));
    auto y = Tensor::create(xt::xarray<double>({3.0}));

    // z = x + y
    auto z = (*x) + y;  // Dereference x
    // w = z * x
    auto w = (*z) * x;  // Dereference z

    // Let's call backward on w
    w->backward();

    std::cout << "x: " << *x << std::endl;
    std::cout << "y: " << *y << std::endl;
    std::cout << "z: " << *z << std::endl;
    std::cout << "w: " << *w << std::endl;

    return 0;
}
