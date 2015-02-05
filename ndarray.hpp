#pragma once

#include <iostream>
#include <cstring>
#include <string>
#include <array>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <initializer_list>
#include "cxxabi.h"

#define is_slicing_pack(Expr, Slices) (contains_slice<Slices...>::value || (sizeof...(Slices) < Expr::ndim && all_integral<Slices...>::value))
#define is_non_slicing_pack(Expr, Slices) (sizeof...(Slices) == Expr::ndim && all_integral<Slices...>::value)

#define make_trait_tester(trait) constexpr bool _is_ ## trait(...) { return false; } \
template <typename T> \
constexpr auto _is_ ## trait(T* t) -> decltype(T::_is_ ## trait) \
{ return T::_is_ ## trait; } \
template <typename T> \
struct is_ ## trait { \
    static constexpr bool value = _is_ ## trait((typename std::decay<T>::type*)0); \
}

namespace nda {
    make_trait_tester(expression);
    make_trait_tester(constant);
    make_trait_tester(array);
    make_trait_tester(slice);

    template <typename T, typename... Ts>
    struct contains_slice {
        static constexpr bool value = is_slice<T>::value || contains_slice<Ts...>::value;
    };
    template <typename T>
    struct contains_slice<T> {
        static constexpr bool value = is_slice<T>::value;
    };
}

/*
 * Macros for expression and scalar type traits.
 */
#define enable_if_array(type) typename = std::enable_if_t<is_array<type>::value>
#define enable_if_expression(type) typename = std::enable_if_t<is_expression<type>::value>
#define is_array_or_expr(type) (is_array<type>::value || is_expression<type>::value)
#define enable_if_array_or_expr(type) typename = std::enable_if_t<is_array_or_expr(type)>
#define enable_if_scalar(type) typename = std::enable_if_t<std::is_arithmetic<type>::value>
#define enable_if_compatible(type1, type2) typename = std::enable_if_t<elementwise_compatible<type1, type2>::value>

/*
 * Template metaprogramming tools for manipulating variadic non-type template
 * parameters (possibly encapsulated in a struct called a shape pack).
 */
constexpr size_t DYNAMIC_SHAPE = 0;

template <typename T, typename... Ts>
struct count_integral {
    static constexpr size_t value = count_integral<Ts...>::value + (std::is_integral<typename std::decay<T>::type>::value ? 1 : 0);
};
template <typename T>
struct count_integral<T> {
    static constexpr size_t value = std::is_integral<typename std::decay<T>::type>::value ? 1 : 0;
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
    static constexpr bool value = std::is_integral<typename std::decay<T>::type>::value && all_integral<Ts...>::value;
};
template <typename T>
struct all_integral<T> {
    static constexpr bool value = std::is_integral<typename std::decay<T>::type>::value;
};


template  <size_t Q, typename ShapePack, size_t I = ShapePack::len-1>
struct count_in {
    static constexpr size_t value = count_in<Q, ShapePack, I-1>::value + ( (Q==ShapePack::template at<I>) ? 1 : 0 );
};

template <size_t Q, typename ShapePack>
struct count_in<Q, ShapePack, 0> {
    static constexpr size_t value = ( (Q==ShapePack::template at<0>) ? 1 : 0 );
};

/*
 * Find the index of the Nth occurrence of Q in ShapePack. Requires that
 * ShapePack actually contain at least N Qs.
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
 * Find the index of the Nth non-integral type in a type pack. Requires that
 * the pack actually contain at least N non-integral types.
 */
template <size_t N, size_t match_count, size_t cur_ind, typename... Types>
struct _nth_non_integral_index {
    static_assert(N < sizeof...(Types) - count_integral<Types...>::value, "Pack does not contain sufficient non-integral types.");
    static_assert((N >= 0) && (N < sizeof...(Types)), "Invalid pack index.");

    static constexpr size_t value =
        std::conditional<cur_ind == sizeof...(Types)-1,
            std::integral_constant<size_t, cur_ind>,
            typename std::conditional<!std::is_integral<typename nth_in_type_pack<cur_ind, Types...>::type>::value,
                typename std::conditional<match_count == N,
                    std::integral_constant<size_t, cur_ind>,
                    _nth_non_integral_index<N, match_count+1, cur_ind+1, Types...>
                >::type,
            _nth_non_integral_index<N, match_count, cur_ind+1, Types...>
        >::type>::type::value;
};

template <size_t N, typename... Types>
struct nth_non_integral_index {
    static constexpr size_t value = _nth_non_integral_index<N, 0, 0, Types...>::value;
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
struct all { static constexpr bool _is_slice = true; };
template <typename T>
static constexpr bool is_all(const T& t) { return std::is_same<typename std::decay<T>::type, all>::value; }

template <typename Pack, typename... Slices>
struct sliced_shape_pack {
    static constexpr size_t len = Pack::len - count_integral<Slices...>::value;

    template <size_t I>
    static constexpr size_t __at(typename std::enable_if<I < sizeof...(Slices), void*>::type = 0) {
        static_assert(I < Pack::len, "Invalid pack index.");
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
    static constexpr size_t __at(typename std::enable_if<I >= sizeof...(Slices), void*>::type = 0) {
        static_assert(I < Pack::len, "Invalid pack index.");
        return Pack::template at<I>;
    }

    template <size_t I>
    static constexpr size_t at = __at<
    std::conditional<I < sizeof...(Slices) - count_integral<Slices...>::value,
        nth_non_integral_index<I, Slices...>,
        std::integral_constant<size_t, sizeof...(Slices)+I>
    >::type::value>();
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

namespace nda {

/*
 * A constant expression template containing a constant value and a matching
 * expression (to define the shape of the constant).
 */
template <typename T, typename Expr>
struct constant {
    static_assert(!is_constant<Expr>::value,
            "Cannot create a constant expression attached to another constant.");

    using value_type = T;
    using shape_type = typename Expr::shape_type;

    T value;
    const Expr& expr;

    static constexpr size_t ndim = Expr::ndim;
    size_t size() const { return expr.size(); }
    std::array<size_t, ndim> shape() const { return expr.shape(); }

    explicit constant(const T& t, const Expr& e) : value(t), expr(e)
    {}

    struct const_iterator {
        T ivalue;
        typename Expr::const_iterator iter;

        const_iterator(const constant& p, bool end = false) : ivalue(p.value),
        iter(end ? p.expr.end() : p.expr.begin()) {}

        const value_type& operator*() const { return ivalue; }
        bool operator==(const const_iterator& other) const { return (ivalue == other.ivalue) && (iter == other.iter); }
        bool operator!=(const const_iterator& other) const { return !(this->operator==(other)); }

        const_iterator& operator++() { ++iter; return *this; }
        const_iterator& operator--() { --iter; return *this; }
        const_iterator operator++(int) { const_iterator copy(*this); ++iter; return *this; }
        const_iterator operator--(int) { const_iterator copy(*this); --iter; return *this; }
    };

    const_iterator begin() const { return const_iterator(*this); }
    const_iterator end() const { return const_iterator(*this, true); }
};

/*
 * Expression template class for binary expressions operating on individual
 * matching elements from a pair of arrays (elementwise operators).
 */
template <typename Expr1, typename Expr2, template<typename, typename> class BinaryOp>
struct elemwise_binary_expr {
    using Op = BinaryOp<Expr1, Expr2>;
    using value_type = typename Op::value_type;
    using shape_type = typename Op::shape_type;
    static constexpr bool _is_expression = true;

    using expr1_type = typename std::conditional<is_array<Expr1>::value, const Expr1&, const Expr1>::type;
    using expr2_type = typename std::conditional<is_array<Expr2>::value, const Expr2&, const Expr2>::type;

    expr1_type lhs;
    expr2_type rhs;

    static constexpr size_t ndim = Op::ndim;
    size_t size() const { return lhs.size(); }
    std::array<size_t, ndim> shape() const { return lhs.shape(); }

    elemwise_binary_expr(const Expr1& l, const Expr2& r) : lhs(l), rhs(r) {
        assert(l.size() == r.size());
        assert(shape_match(l.shape(), r.shape()));
    }

    /*
     * Iterator access is provided to evaluate the expression for use in array
     * construction.
     */
    struct const_iterator {
        typename Expr1::const_iterator iter1;
        typename Expr2::const_iterator iter2;

        const_iterator(const elemwise_binary_expr& p, bool end=false) :
            iter1(end ? p.lhs.end() : p.lhs.begin()),
            iter2(end ? p.rhs.end() : p.rhs.begin()) {}

        value_type operator*() const { return Op::eval(*iter1, *iter2); }

        bool operator==(const const_iterator& other) const { return (iter1 == other.iter1) && (iter2 == other.iter2); }
        bool operator!=(const const_iterator& other) const { return !(this->operator==(other)); }

        const_iterator& operator++() { ++iter1; ++iter2; return *this; }
        const_iterator& operator--() { --iter1; --iter2; return *this; }
        const_iterator operator++(int) { const_iterator copy(*this); ++iter1; ++iter2; return copy; }
        const_iterator operator--(int) { const_iterator copy(*this); --iter1; --iter2; return copy; }
    };

    const_iterator begin() const { return const_iterator(*this); }
    const_iterator end() const { return const_iterator(*this, true); }
};

/*
 * See above, but for unary operations on individual array elements.
 */
template <typename Expr, template<typename> class UnaryOp>
struct elemwise_unary_expr {
    using Op = UnaryOp<Expr>;
    using value_type = typename Op::value_type;
    using shape_type = typename Expr::shape_type;
    using expr_type = typename std::conditional<is_array<Expr>::value, const Expr&, const Expr>::type;
    static constexpr bool _is_expression = true;

    expr_type expr;

    static constexpr size_t ndim = Op::ndim;
    size_t size() const { return expr.size(); }
    std::array<size_t, ndim> shape() const { return expr.shape(); }

    elemwise_unary_expr(const Expr& e) : expr(e) { }

    value_type operator()(const std::array<size_t, ndim>& inds) const {
        return Op::eval(expr(inds));
    }

    struct const_iterator {
        typename Expr::const_iterator iter;

        explicit const_iterator(const elemwise_unary_expr& p, bool end=false) :
            iter(end ? p.expr.end() : p.expr.begin()) {}

        value_type operator*() const { return Op::eval(*iter); }
        bool operator==(const const_iterator& other) const { return iter == other.iter; }
        bool operator!=(const const_iterator& other) const { return !(this->operator==(other)); }

        const_iterator& operator++() { ++iter; return *this; }
        const_iterator& operator--() { --iter; return *this; }
        const_iterator operator++(int) { const_iterator copy(*this); ++iter; return copy; }
        const_iterator operator--(int) { const_iterator copy(*this); --iter; return copy; }
    };

    const_iterator begin() const { return const_iterator(*this); }
    const_iterator end() const { return const_iterator(*this, true); }
};

/*
 * Macros to define the required functions/operators and expression template
 * operator classes for basic operators (+,-,*,/), unary, and binary functions.
 *
 * Each requires 3 overloaded forms: one for a pair of arrays/expressions, and
 * one for a single array/expression with a broadcasted scalar on either side.
 */
#define make_basic_elemwise_binary_expr_op(op, name) \
template <typename Expr1, typename Expr2> \
struct name ## _op { \
    using value_type = decltype(typename Expr1::value_type() op typename Expr2::value_type()); \
    using shape_type = max_shape_pack<typename Expr1::shape_type, typename Expr2::shape_type>; \
    static constexpr value_type eval(const value_type& l, const value_type& r) { return l op r; } \
    static constexpr size_t ndim = Expr1::ndim; \
}; \
template <typename Expr1, typename Expr2> \
typename std::enable_if<is_expression<Expr1>::value && is_expression<Expr2>::value, \
elemwise_binary_expr<Expr1, Expr2, name ## _op>>::type \
operator op(const Expr1& lhs, const Expr2& rhs) { \
    return {lhs, rhs}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_expression<Expr>::value && std::is_arithmetic<T>::value, \
elemwise_binary_expr<Expr, constant<T, Expr>, name ## _op>>::type \
operator op(const Expr& lhs, const T& rhs) { \
    return {lhs, constant<T, Expr>{rhs, lhs}}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_expression<Expr>::value && std::is_arithmetic<T>::value, \
elemwise_binary_expr<constant<T, Expr>, Expr, name ## _op>>::type \
operator op(const T& lhs, const Expr& rhs) { \
    return {constant<T, Expr>{lhs, rhs}, rhs}; \
}

#define make_func_elemwise_unary_expr_op(func, name) \
template <typename Expr> \
struct name ## _op { \
    using value_type = decltype(func(typename Expr::value_type())); \
    using shape_type = typename Expr::shape_type; \
    static constexpr value_type eval(const value_type& e) { return func(e); } \
    static constexpr size_t ndim = Expr::ndim; \
}; \
template <typename Expr> \
typename std::enable_if<is_expression<Expr>::value, \
elemwise_unary_expr<Expr, name ## _op>>::type \
name(const Expr& expr) { return expr; }

#define make_func_elemwise_binary_expr_op(func, name) \
template <typename Expr1, typename Expr2> \
struct name ## _op { \
    using value_type = decltype(func(typename Expr1::value_type(), typename Expr2::value_type())); \
    using shape_type = max_shape_pack<typename Expr1::shape_type, typename Expr2::shape_type>; \
    static constexpr value_type eval(const value_type& l, const value_type& r) { return func(l, r); } \
    static constexpr size_t ndim = Expr1::ndim; \
}; \
template <typename Expr1, typename Expr2> \
typename std::enable_if<is_expression<Expr1>::value && is_expression<Expr2>::value, \
elemwise_binary_expr<Expr1, Expr2, name ## _op>>::type \
func(const Expr1& lhs, const Expr2& rhs) { \
    return {lhs, rhs}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_expression<Expr>::value && std::is_arithmetic<T>::value, \
elemwise_binary_expr<Expr, constant<T, Expr>, name ## _op>>::type \
func(const Expr& lhs, const T& rhs) { \
    return {lhs, constant<T, Expr>{rhs, lhs}}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_expression<Expr>::value && std::is_arithmetic<T>::value, \
elemwise_binary_expr<constant<T, Expr>, Expr, name ## _op>>::type \
func(const T& lhs, const Expr& rhs) { \
    return {constant<T, Expr>{lhs, rhs}, rhs}; \
}

/*
 * Macro-based definitions of expression-template-generating operators for
 * basic operations, as well as unary and binary cmath functions.
 */
make_basic_elemwise_binary_expr_op(+,           add);
make_basic_elemwise_binary_expr_op(-,           sub);
make_basic_elemwise_binary_expr_op(*,           mul);
make_basic_elemwise_binary_expr_op(/,           div);

make_func_elemwise_binary_expr_op(pow,          pow);
make_func_elemwise_binary_expr_op(atan2,        atan2);

make_func_elemwise_unary_expr_op(std::exp,      exp);
make_func_elemwise_unary_expr_op(std::log,      log);
make_func_elemwise_unary_expr_op(std::log2,     log2);
make_func_elemwise_unary_expr_op(std::log10,    log10);
make_func_elemwise_unary_expr_op(std::sin,      sin);
make_func_elemwise_unary_expr_op(std::cos,      cos);
make_func_elemwise_unary_expr_op(std::tan,      tan);
make_func_elemwise_unary_expr_op(std::asin,     asin);
make_func_elemwise_unary_expr_op(std::acos,     acos);
make_func_elemwise_unary_expr_op(std::atan,     atan);

// The only unary basic operator supported currently is negation, no need for
// another macro type.
template <typename Expr>
struct neg_op {
    using value_type = decltype(-(typename Expr::value_type()));
    using shape_type = typename Expr::shape_type;

    static constexpr value_type eval(const value_type& v) { return -v; }
    static constexpr size_t ndim = Expr::ndim;
};
template <typename Expr>
typename std::enable_if<is_expression<Expr>::value,
elemwise_unary_expr<Expr, neg_op>>::type
operator-(const Expr& expr) { return expr; }

struct range {
    static constexpr bool _is_slice = true;
    int start, stop, step;

    range() : start(-1), stop(-1), step(-1) {}
    range(const all& a) : start(-1), stop(-1), step(-1) {}
    explicit range(int sto) : start(0), stop(sto), step(1) {}
    range(int sta, int sto, int ste=1) : start(sta), stop(sto), step(ste) {}

    struct rng_iter {
        const range& rng;
        int cur;

        rng_iter(const range& r) : rng(r), cur(r.start) {}
        rng_iter(const range& r, int init) : rng(r), cur(init) {}

        const int& operator*() const { return cur; }

        rng_iter& operator++() { cur += rng.step; return *this; }
        rng_iter operator++(int) { rng_iter tmp(*this); cur += rng.step; return tmp; }

        rng_iter& operator+=(int x) { cur += x*rng.step; return *this; }
        rng_iter operator+(int x) const { return rng_iter(*this) += x; }
        rng_iter& operator-=(int x) { cur -= x*rng.step; return *this; }
        rng_iter operator-(int x) const { return rng_iter(*this) -= x; }

        bool operator==(const rng_iter& other) const { return other.cur == cur && other.rng == rng; }
        bool operator!=(const rng_iter& other) const { return !(this->operator==(other)); }
    };

    rng_iter begin() const { return rng_iter(*this); }
    rng_iter end() const { return rng_iter(*this, start + len()*step); }

    int len() const { return (stop - start) / step + (((abs(stop-start)%abs(step)) == 0) ? 0 : 1); }
    bool operator==(const range& other) const { return other.start == start && other.stop == stop && other.step == step; }
    bool operator!=(const range& other) const { return !(this->operator==(other)); }
};

using slice = range;

/*
 * Convert a list of parameters, which may be indices or slices, to a
 * list of slices.
 */
template<typename I0>
slice _index_to_slice(const I0& i0, typename std::enable_if<std::is_integral<I0>::value, void*>::type = 0) {
    return slice(i0, i0+1);
}
template<typename I0>
slice _index_to_slice(const I0& i0, typename std::enable_if<is_slice<I0>::value, void*>::type = 0) {
    return i0;
}
template <size_t S, typename I0, typename... Indices>
void _convert_indices(std::array<slice, S>& slices, std::array<bool, S>& ind_types, int n, const I0& i0, const Indices&... inds) {
    slices[n] = _index_to_slice(i0);
    ind_types[n] = is_slice<I0>::value;
    _convert_indices(slices, ind_types, n+1, inds...);
}
template <size_t S, typename I0>
void _convert_indices(std::array<slice, S>& slices, std::array<bool, S>& ind_types, int n, const I0& i0) {
    slices[n] = _index_to_slice(i0);
    ind_types[n] = is_slice<I0>::value;
}

template <typename Expr, typename... Slices>
struct slice_expr {
    // Sliced arrays are still arrays, other expressions are not.
    static constexpr bool _is_array = is_array<Expr>::value;
    static constexpr bool _is_expression = true;

    using value_type = typename std::decay<typename Expr::value_type>::type;
    using access_type = typename std::conditional<is_array<Expr>::value, value_type&, value_type>::type;
    using const_access_type = typename std::conditional<is_array<Expr>::value, const value_type&, const value_type>::type;
    using shape_type = sliced_shape_pack<typename Expr::shape_type, Slices...>;
    using expr_ref = typename std::conditional<std::is_const<Expr>::value, const Expr&, Expr&>::type;

    static constexpr size_t ndim = Expr::ndim - count_integral<Slices...>::value;
    static constexpr size_t fulldim = Expr::ndim;

    expr_ref expr;
    std::array<slice, fulldim> _slices = {};
    std::array<bool, fulldim> dim_preserved = {};
    std::array<size_t, ndim> sliced_dims = {};
    std::array<size_t, ndim> _shape = {};
    size_t _size;

    const size_t& size() const { return _size; }
    const std::array<size_t, ndim>& shape() const { return _shape; }

    template<enable_if_array_or_expr(Expr)>
    slice_expr(expr_ref e, const Slices&... slices) : expr(e) {
        const auto& a_shp = expr.shape();
        std::array<slice, fulldim> converted_slices;
        for(int i=0; i<fulldim; ++i) { dim_preserved[i] = true; }
        _convert_indices(converted_slices, dim_preserved, 0, slices...);

        _size = 1;
        int si = 0;
        for(int i=0; i<fulldim; ++i) {
            const auto& sl = converted_slices[i];
            if( dim_preserved[i] ) {
                _slices[i] = (sl == slice()) ? slice(a_shp[i]) : sl;
                _shape[si] = _slices[i].len();
                _size *= _shape[si];
                sliced_dims[si] = i;
                ++si;
            } else {
                _slices[i] = sl;
            }
        }
    }

    template <typename... Is, typename = std::enable_if_t<sizeof...(Is) == ndim && all_integral<Is...>::value>>
    const_access_type operator()(Is... is) const {
        return (*this)({{static_cast<size_t>(is)}...});
    }
    template <typename... Is, typename = std::enable_if_t<sizeof...(Is) == ndim && all_integral<Is...>::value>>
    access_type operator()(Is... is) {
        return (*this)({{static_cast<size_t>(is)}...});
    }
    const_access_type operator()(const std::array<size_t, ndim>& slice_inds) const {
        std::array<size_t, fulldim> inds;
        int si = 0;
        for(int i=0; i < fulldim; ++i) {
            if( dim_preserved[i] ) {
                assert(slice_inds[si] < _shape[i] && "Invalid index.");
                inds[i] = *(_slices[i].begin() + slice_inds[si]);
                ++si;
            } else {
                inds[i] = _slices[i].start;
            }
        }
        return expr(inds);
    }
    access_type operator()(const std::array<size_t, ndim>& slice_inds) {
        std::array<size_t, fulldim> inds;
        int si = 0;
        for(int i=0; i < fulldim; ++i) {
            if( dim_preserved[i] ) {
                assert(slice_inds[si] < _shape[i] && "Invalid index.");
                inds[i] = *(_slices[i].begin() + slice_inds[si]);
                ++si;
            } else {
                inds[i] = _slices[i].start;
            }
        }
        return expr(inds);
    }

    /*
     * Slice access operator (returns another slice expression).
     */
    template <typename... Slices2, typename = std::enable_if_t<is_slicing_pack(slice_expr, Slices2)>>
    const slice_expr<const slice_expr<const Expr, Slices...>, Slices2...> operator()(const Slices2&... slices2) const {
        return {*this, slices2...};
    }
    template <typename... Slices2, typename = std::enable_if_t<is_slicing_pack(slice_expr, Slices2)>>
    slice_expr<slice_expr<Expr, Slices...>, Slices2...> operator()(const Slices2&... slices2) {
        return {*this, slices2...};
    }

    template<typename E = Expr>
    typename std::enable_if<is_array<E>::value, slice_expr&>::type
    operator=(const value_type v) {
        for(auto it = begin(); it != end(); ++it) { (*it) = v; }
        return *this;
    }

    template <typename E, bool Mutable>
    struct iterator {
        E expr;

        // The {} is required to value-initialize the array with zeroes.
        std::array<size_t, ndim> cur_pos{};

        iterator(E e) : expr(e) { }
        iterator(E e, const std::array<size_t, ndim>& p) : expr(e), cur_pos(p) {}

        const_access_type operator*() const { return expr(cur_pos); }

        template<bool M = Mutable>
        typename std::enable_if<M, access_type>::type
        operator*() { return expr(this->cur_pos); }

        bool operator==(const iterator& other) const {
            bool equal = true;
            for(int i=0; i<ndim; ++i) {
                equal = equal && (cur_pos[i] == other.cur_pos[i]);
            }
            return equal;
        }
        bool operator!=(const iterator& other) const { return !(this->operator==(other)); }

        iterator& operator++() {
            int carries = 0;
            for(int i=ndim-1; i>=0; --i) {
                if(cur_pos[i] < expr.shape()[i]-1) {
                    ++cur_pos[i];
                    break;
                } else {
                    cur_pos[i] = 0;
                    ++carries;
                }
            }

            if (carries == ndim) {
                cur_pos = expr.shape();
            }
            return *this;
        }

        iterator& operator--() {
            int zeros = 0;
            for(int i=ndim-1; i>=0; --i) {
                if(cur_pos[i] > 0) {
                    --cur_pos[i];
                    break;
                } else {
                    cur_pos[i] = expr.shape()[i]-1;
                    ++zeros;
                }
            }

            if (zeros == ndim) {
                cur_pos = expr.shape();
            }
            return *this;
        }

        iterator operator++(int) {
            iterator copy(*this);
            ++(*this);
            return copy;
        }
        iterator operator--(int) {
            iterator copy(*this);
            --(*this);
            return copy;
        }
    };

    using const_iterator = iterator<const slice_expr&, false>;
    using mutable_iterator = iterator<slice_expr&, true>;

    const_iterator begin() const { return const_iterator(*this); }
    const_iterator end() const { return const_iterator(*this, _shape); }
    mutable_iterator begin() { return mutable_iterator(*this); }
    mutable_iterator end() { return mutable_iterator(*this, _shape); }
};

/*
 * Ensures instances of any two types with shape() methods have matching
 * shapes.
 */
template <typename Arr1, typename Arr2>
bool shape_match(const Arr1& s1, const Arr2& s2) {
    assert(s1.size() == s2.size());
    bool match = true;
    for(size_t i=0; i < s1.size(); ++i) {
        match = match && (s1[i] == s2[i]);
    }
    return match;
}

//////////////////////
// NDARRAY
//////////////////////

/*
 * Base n-dimensional array class. Shape is defined by a static ShapePack-like
 * struct encapsulating the variadic shapes.
 */
template <typename T, typename ShapePack>
class nda_impl {
    public:
        using value_type = T;
        using shape_type = ShapePack;
        static constexpr bool _is_array = true;
        static constexpr bool _is_expression = true;

        static constexpr size_t ndim = ShapePack::len;
        size_t _size;
        std::array<size_t, ndim> _strides;
        std::array<size_t, ndim> _shape;

        const std::array<size_t, ndim>& shape() const { return _shape; }
        const std::array<size_t, ndim>& strides() const { return _strides; }
        const size_t& size() const { return _size; }

        T* data = nullptr;

        ////////////////////////
        //  Default Constructor
        ////////////////////////
        nda_impl() {
            static_assert(n_dynamic_dims == 0, "Must specify all dynamic dimensions for construction.");

            fill_array<ShapePack>(_shape);
            _compute_size();
            _compute_basic_strides();
            _alloc_data();
        }

        //////////////////////////////////
        //  Initializer-list Constructor
        //////////////////////////////////
        nda_impl(const std::initializer_list<T>& items) {
            static_assert(n_dynamic_dims < 2,
                    "Cannot construct arrays with multiple dynamic dimensions from an initializer list (yet).");

            fill_array<ShapePack>(_shape);

            size_t sz = 1;
            size_t dyn_dim = 0;
            for(size_t i=0; i<ndim; ++i) {
                if(_shape[i] == 0) { dyn_dim = i; } else { sz *= _shape[i]; }
            }

            if (n_dynamic_dims == 1) {
                _shape[dyn_dim] = items.size() / sz;
            }

            _compute_size();
            assert(_size == items.size());
            _compute_basic_strides();
            _alloc_data(items);
        }

        //////////////////////
        //  Copy Constructors
        //////////////////////
        /*
         * Two copy constructors are needed, one "true" copy constructor for
         * other instances of nda_impl with the same (possibly dynamic) shape,
         * and one for instances with other (compatible) shapes. The same holds
         * for move constructors and assignment operators.
         */
        nda_impl(const nda_impl& other) : _size(other._size), _strides(other._strides), _shape(other._shape)
        {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            _alloc_data();
            memcpy(data, other.data, _size*sizeof(T));
        }
        template <typename OShape>
        nda_impl(const nda_impl<T, OShape>& other) : _size(other._size), _strides(other._strides), _shape(other._shape)
        {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all fixed dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            _alloc_data();
            memcpy(data, other.data, _size*sizeof(T));
        }

        //////////////////////
        //  Move Constructors
        //////////////////////
        nda_impl(nda_impl&& other) : _size(other._size),
                                     _strides(other._strides),
                                     _shape(other._shape),
                                     data(other.data)
        {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            other.data = nullptr;
        }

        template <typename OShape>
        nda_impl(nda_impl<T, OShape>&& other) : _size(other._size),
                                                _strides(other._strides),
                                                _shape(other._shape),
                                                data(other.data)
        {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all known dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            other.data = nullptr;
        }

        ///////////////////////////////
        //  Dynamic Shape Constructor
        ///////////////////////////////
        template <typename... DynShapes, typename = std::enable_if_t<all_integral<DynShapes...>::value>>
        nda_impl(const DynShapes&... dynamic_shapes) {
            static_assert(n_dynamic_dims > 0, "Tried to call dynamic shape constructor with no dynamic dims.");
            static_assert(sizeof...(DynShapes) == n_dynamic_dims, "Must specify all dynamic dimensions for construction.");

            fill_array<ShapePack>(_shape);
            _fill_dynamic_shapes(dynamic_shapes...);
            _compute_size();
            _compute_basic_strides();
            _alloc_data();
        }

        ////////////////////////////////////
        //  Expression Template Constructor
        ////////////////////////////////////
        template <typename Expr,
                 enable_if_expression(Expr),
                 enable_if_compatible(ShapePack, typename Expr::shape_type)>
        nda_impl(const Expr& ex) {
            fill_array<ShapePack>(_shape);
            for(int i=0; i<ndim; ++i) {
                if (_shape[i] == DYNAMIC_SHAPE) { _shape[i] = ex.shape()[i]; }
            }
            _compute_size();

            assert(shape_match(_shape, ex.shape()));
            assert(_size == ex.size());

            _compute_basic_strides();
            _alloc_data();

            auto it = begin();
            for(const auto x : ex){
                (*it) = x;
                ++it;
            }

        }

        ~nda_impl() {
            if (data) {
                delete[] data;
            }
        }

        ///////////////////////////////////
        // Copy/Move Assignment Operators
        ///////////////////////////////////
        nda_impl& operator=(const nda_impl& other) {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            memcpy(data, other.data, _size*sizeof(T));
            return *this;
        }
        template <typename OShape>
        nda_impl& operator=(const nda_impl<T, OShape>& other) {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all known dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            memcpy(data, other.data, _size*sizeof(T));
            return *this;
        }

        nda_impl& operator=(nda_impl&& other) {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            if(this != &other) {
                data = other.data;
                other.data = nullptr;
            }
            return *this;
        }
        template <typename OShape>
        nda_impl& operator=(nda_impl<T, OShape>&& other) {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all known dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            if(this != &other) {
                data = other.data;
                other.data = nullptr;
            }
            return *this;
        }

        ////////////////////////////////////
        //  Expression Template Assignment
        ////////////////////////////////////
        template <typename Expr, enable_if_expression(Expr)>
        nda_impl& operator=(const Expr& ex) {
            fill_array<ShapePack>(_shape);
            for(int i=0; i<ndim; ++i) {
                if (_shape[i] == DYNAMIC_SHAPE) { _shape[i] = ex.shape()[i]; }
            }
            _compute_size();
            assert(shape_match(_shape, ex.shape()));
            assert(_size == ex.size());

            _compute_basic_strides();
            _alloc_data();

            for(size_t i=0; i < _size; ++i) {
                data[i] = ex(i);
            }
            return *this;
        }

        ///////////////////////
        //  Scalar Assignment
        ///////////////////////
        nda_impl& operator=(const T& s) {
            for(int i=0; i<_size; ++i) {
                data[i] = s;
            }
            return *this;
        }

        /////////////////////
        //  Access Operators
        /////////////////////
        template <typename... Indices>
        typename std::enable_if<is_non_slicing_pack(nda_impl, Indices), const T&>::type
        operator()(const Indices&... indices) const {
            return data[_compute_index(indices...)];
        }
        template <typename... Indices>
        typename std::enable_if<is_non_slicing_pack(nda_impl, Indices), T&>::type
        operator()(const Indices&... indices) {
            return data[_compute_index(indices...)];
        }
        const T& operator()(const std::array<size_t, ndim>& ind_arr) const {
            return data[_compute_index(ind_arr)];
        }
        T& operator()(const std::array<size_t, ndim>& ind_arr) {
            return data[_compute_index(ind_arr)];
        }

        /*
         * Slice access operator (returns a slice expression).
         */
        template <typename... Slices>
        typename std::enable_if<is_slicing_pack(nda_impl, Slices), 
        const slice_expr<const nda_impl<T, ShapePack>, Slices...>>::type
            operator()(const Slices&... slices) const {
            return {*this, slices...};
        }
        template <typename... Slices>
        typename std::enable_if<is_slicing_pack(nda_impl, Slices), 
        slice_expr<nda_impl<T, ShapePack>, Slices...>>::type
            operator()(const Slices&... slices) {
            return {*this, slices...};
        }

        //////////////////
        //  Iterators
        //////////////////
        using const_iterator = const T*;

        /*
         * The raw data pointer works fine as an iterator as long as the array
         * has not been sliced.
         */
        const_iterator begin() const { return data; }
        const_iterator end() const { return data + _size; }

        T* begin() { return data; }
        T* end() { return data + _size; }

        template <typename OShape>
        nda_impl& operator+=(const nda_impl<T, OShape>& rhs) {
            auto it = begin();
            auto r_it = rhs.begin();
            for(; it != end() && r_it != rhs.end(); ++it, ++r_it) {
                (*it) += (*r_it);
            }
            return (*this);
        }

    private:
        static constexpr size_t n_dynamic_dims = count_in<DYNAMIC_SHAPE, ShapePack>::value;
        static constexpr size_t n_static_dims = ndim - n_dynamic_dims;
        static constexpr bool has_fixed_shape = (n_dynamic_dims == 0);

        /*
         * Given a list of runtime shapes, fill in the missing dynamic values
         * in the _shapes array.
         */
        template <typename... DynShapes>
        void _fill_dynamic_shapes(const DynShapes&... dynamic_shapes) {
            static_assert(sizeof...(dynamic_shapes) == n_dynamic_dims, "Invalid number of dynamic dimensions.");
            __fill_dynamic_shapes<0>(dynamic_shapes...);
        }
        template <size_t si, typename... DynShapes>
        void __fill_dynamic_shapes(const size_t& s0, const DynShapes&... dynamic_shapes) {
            __fill_dynamic_shapes<si>(s0);
            __fill_dynamic_shapes<si+1>(dynamic_shapes...);
        }
        template <size_t si>
        void __fill_dynamic_shapes(const size_t& s0) {
            _shape[nth_match_index_in_pack<si, DYNAMIC_SHAPE, shape_type>::value] = s0;
        }

        void _alloc_data() {
            if(data) { delete[] data; }
            data = new T[_size];
        }

        void _alloc_data(const std::initializer_list<T>& items) {
            if(data) { delete[] data; }
            data = new T[items.size()];
            std::copy(items.begin(), items.end(), data);
        }

        void _compute_size() {
            _size = 1;
            for (const auto& si : _shape) { _size *= si; }
        }

        /*
         * Compute strides assuming the array has not been sliced.
         */
        void _compute_basic_strides() {
            _strides[ndim-1] = 1;
            for(size_t i=2; i<=ndim; ++i) {
                _strides[ndim-i] = _strides[ndim-i+1] * _shape[ndim-i+1];
            }
        }

        /*
         * Compute an offset into the data array given a list of indices.
         */
        size_t _compute_index(const std::array<size_t, ndim>& indices) const {
            int ind = 0;
            for(int i=0; i < ndim; ++i) {
                ind += _strides[i] * indices[i];
            }
            return ind;
        }
        template <typename... Indices>
        size_t _compute_index(const size_t& i0, const Indices&... indices) const {
            const size_t ind = i0 * _strides[ndim - sizeof...(Indices) - 1] + _compute_index(indices...);
            assert(ind < _size && "Index out of range.");
            return ind;
        }

        size_t _compute_index() const { return 0; }
};

template <typename T, size_t... Shape>
using ndarray = nda_impl<T, shape_pack<Shape...>>;

template <typename ArrExpr>
using array_like = nda_impl<typename ArrExpr::value_type, typename ArrExpr::shape_type>;
} // END namespace nda

template <typename Expr>
typename std::enable_if<nda::is_expression<Expr>::value && Expr::ndim == 1,
std::ostream&>::type
operator<<(std::ostream& out, const Expr& expr) {
    out << "[";
    for(size_t i=0; i < expr.shape()[0]; ++i) {
        out << expr(i);
        if(i < expr.shape()[0]-1) { out << "    "; }
    }
    out << "]";
    return out;
}

template <typename Expr>
typename std::enable_if<nda::is_expression<Expr>::value && Expr::ndim == 2,
std::ostream&>::type
operator<<(std::ostream& out, const Expr& expr) {
    out << "[";
    for(size_t i=0; i < expr.shape()[0]; ++i) {
        if(i > 0) {out << " ";}
        out << expr(i);
        if(i < expr.shape()[0]-1) { out << std::endl; }
    }
    out << "]";
    return out;
}

template <typename Expr>
typename std::enable_if<nda::is_expression<Expr>::value && (Expr::ndim > 2),
std::ostream&>::type
operator<<(std::ostream& out, const Expr& expr) {
    for(size_t i=0; i < expr.shape()[0]; ++i) {
        out << "[";
        out << expr(i);
        out << "]";
        if(i < expr.shape()[0]-1) { out << std::endl; }
    }
    return out;
}
