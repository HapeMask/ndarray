#include <iostream>
#include <tuple>
#include <typeinfo>

#include "ndarray.hpp"
#include "ndmath.hpp"
using nda::range;
using nda::ndarray;

static constexpr int N = 3;
static constexpr int M = 3;

int main(int argc, char** args) {
    ndarray<float, N, M> x = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1 };

    std::cerr << x << std::endl;
    std::cerr << x.shape() << std::endl;
    return 0;
}
