#pragma once

#include <type_traits>
#include <initializer_list>
#include <array>

#include "cxxabi.h"
#include <string>

/*
 * Template metaprogramming tools for manipulating variadic non-type template
 * parameters (possibly encapsulated in a struct called a shape pack).
 *
 * TODO: Documenting complex TMP code is hard. Make this better.
 */
constexpr size_t DYNAMIC_SHAPE = 0;

template <typename T, typename... Ts>
struct count_integral {
    static constexpr size_t value = count_integral<Ts...>::value + (std::is_integral<T>::value ? 1 : 0);
};
template <typename T>
struct count_integral<T> {
    static constexpr size_t value = std::is_integral<T>::value ? 1 : 0;
};

template <size_t N, size_t I, size_t... Is>
struct nth_in_size_pack {
    static constexpr size_t value = nth_in_size_pack<N-1, Is...>::value;
};
template <size_t I, size_t... Is>
struct nth_in_size_pack<0, I, Is...> {
    static constexpr size_t value = I;
};

template <size_t N, typename T, typename... Ts>
struct nth_in_type_pack {
    static_assert((N-1) < sizeof...(Ts), "Invalid pack index.");
    using type = typename nth_in_type_pack<N-1, Ts...>::type;
};
template <typename T, typename... Ts>
struct nth_in_type_pack<0, T, Ts...> {
    using type = T;
};

template <size_t N, size_t... Is>
inline constexpr size_t _at() {
    static_assert((N >= 0) && (N < sizeof...(Is)), "Invalid pack index.");
    return nth_in_size_pack<N, Is...>::value;
}

template <typename T, typename... Ts>
struct all_integral {
    static constexpr bool value = std::is_integral<T>::value && all_integral<Ts...>::value;
};
template <typename T>
struct all_integral<T> {
    static constexpr bool value = std::is_integral<T>::value;
};


template  <size_t Q, typename ShapePack, size_t I = ShapePack::len-1>
struct count_in {
    static constexpr size_t value = count_in<Q, ShapePack, I-1>::value + ( (Q==ShapePack::template at<I>) ? 1 : 0 );
};

template <size_t Q, typename ShapePack>
struct count_in<Q, ShapePack, 0> {
    static constexpr size_t value = ( (Q==ShapePack::template at<0>) ? 1 : 0 );
};

template <size_t Q, typename ShapePack>
struct contains {
    static constexpr bool value = (count_in<Q, ShapePack>::value > 0);
};

/*
 * Find the Nth occurrence of Q in ShapePack. Requires that ShapePack actually
 * contain at least N Qs.
 */
template <size_t N, size_t Q, typename ShapePack, size_t match_count = 0, size_t cur_ind = 0>
struct nth_match_index_in_pack {
    static_assert(N < count_in<Q, ShapePack>::value, "Pack does not contain sufficient matches.");
    static_assert((N >= 0) && (N < ShapePack::len), "Invalid pack index.");

    static constexpr size_t value =
        std::conditional<cur_ind == ShapePack::len-1,
            std::integral_constant<size_t, cur_ind>,
            typename std::conditional<ShapePack::template at<cur_ind> == Q,
                typename std::conditional<match_count == N,
                    std::integral_constant<size_t, cur_ind>,
                    nth_match_index_in_pack<N, Q, ShapePack, match_count+1, cur_ind+1>
                >::type,
            nth_match_index_in_pack<N, Q, ShapePack, match_count, cur_ind+1>
        >::type>::type::value;
};

/*
 * Checks each dimension in a pair of shape packs to make sure they match for
 * elementwise operations (including copy construction and assignment).
 *
 * Dynamic dimensions match with any other dimension, they are checked at
 * runtime.
 */
template <typename Pack1, typename Pack2, size_t I = Pack1::len-1>
struct elementwise_compatible {
    static constexpr bool value = std::conditional<Pack1::len == Pack2::len,
                                   std::integral_constant<bool,
                                   ((Pack1::template at<I> == Pack2::template at<I>) ||
                                   (Pack1::template at<I> == DYNAMIC_SHAPE) ||
                                   (Pack2::template at<I> == DYNAMIC_SHAPE)) &&
                                   elementwise_compatible<Pack1, Pack2, I-1>::value>,
                                   std::false_type>::type::value;

};
template <typename Pack1, typename Pack2>
struct elementwise_compatible<Pack1, Pack2, 0> {
    static constexpr bool value = Pack1::len == Pack2::len &&
                                  ((Pack1::template at<0> == Pack2::template at<0>) ||
                                   (Pack1::template at<0> == DYNAMIC_SHAPE) ||
                                   (Pack2::template at<0> == DYNAMIC_SHAPE));
};

/*
 * A basic shape pack.
 */
template <size_t... Shape>
struct shape_pack {
    static constexpr size_t len = sizeof...(Shape);

    template <size_t I>
    static constexpr size_t at = _at<I, Shape...>();
};

template <size_t A, size_t B>
constexpr size_t tmax() { return ((A>B) ? A : B); }

/*
 * A shape pack representing the elementwise maximum between two shape packs.
 * This is the shape that results from an elementwise operation.
 */
template <typename Pack1, typename Pack2>
struct max_shape_pack {
    static_assert(Pack1::len == Pack2::len, "Cannot take max of packs of different length.");

    static constexpr size_t len = Pack1::len;

    template <size_t I>
    static constexpr size_t at = tmax<Pack1::template at<I>, Pack2::template at<I>>();
};

// Placeholder for a full axis as a slice.
struct all{};
template <typename T>
static constexpr bool is_all(const T& t) { return std::is_same<typename std::decay<T>::type, all>::value; }

template <typename Pack, typename... Slices>
struct sliced_shape_pack {
    static constexpr size_t len = Pack::len;

    template <size_t I>
    static constexpr size_t __at(size_t i=I, typename std::enable_if<I < sizeof...(Slices), void*>::type = 0) {
        return std::conditional<std::is_integral<typename nth_in_type_pack<I, Slices...>::type>::value,
                                                  std::integral_constant<size_t, 1>,
                                                  typename std::conditional<
                                                      std::is_same<typename std::decay<typename nth_in_type_pack<I, Slices...>::type>::type, all>::value,
                                                      std::integral_constant<size_t, Pack::template at<I>>,
                                                      std::integral_constant<size_t, 0>
                                                  >::type
                                                 >::type::value;
    }
    template <size_t I>
    static constexpr size_t __at(size_t i=I, typename std::enable_if<I >= sizeof...(Slices), void*>::type = 0) {
        return Pack::template at<I>;
    }

    template <size_t I>
    static constexpr size_t at = __at<I>();
};

/*
 * Fill an std::array with values from a shape pack.
 */
template <typename ShapePack, size_t I>
struct _array_filler {
    static void fill(std::array<size_t, ShapePack::len>& arr) {
        arr[I] = ShapePack::template at<I>;
        _array_filler<ShapePack, I-1>::fill(arr);
    }
};
template <typename ShapePack>
struct _array_filler<ShapePack, 0> {
    static void fill(std::array<size_t, ShapePack::len>& arr) {
        arr[0] = ShapePack::template at<0>;
    }
};
template <typename ShapePack>
void fill_array(std::array<size_t, ShapePack::len>& arr) {
    _array_filler<ShapePack, ShapePack::len-1>::fill(arr);

}

#define make_trait_tester(name) static constexpr bool _is_ ## name(...) { return false; } \
template <typename T> \
static constexpr auto _is_ ## name(T* t) -> decltype(T::_is_ ## name) \
{ return T::_is_ ## name; } \
template <typename T> \
struct is_ ## name { \
    static constexpr bool value = _is_ ## name((T*)0); \
}

make_trait_tester(expression);
make_trait_tester(array);

/*
 * Macros for expression and scalar type traits.
 */
#define enable_if_array(type) typename = std::enable_if_t<is_array<type>::value>
#define enable_if_expression(type) typename = std::enable_if_t<is_expression<type>::value>
#define is_array_or_expr(type) (is_array<type>::value || is_expression<type>::value)
#define enable_if_array_or_expr(type) std::enable_if_t<is_array_or_expr(type)>
#define enable_if_scalar(type) typename = std::enable_if_t<std::is_arithmetic<type>::value>
#define enable_if_compatible(type1, type2) typename = std::enable_if_t<elementwise_compatible<type1, type2>::value>

/*
 * Demangle the name of a type. Useful for debugging.
 */
std::string demangle(const char* name) {
    char* buf = new char[1024];
    size_t len = 1024;
    int status;
    abi::__cxa_demangle(name, buf, &len, &status);
    std::string s(buf);
    delete[] buf;
    return s;
}
