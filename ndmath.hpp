#pragma once

#include "ndarray.hpp"
#include <cmath>
#include <typeinfo>
#include <memory>

namespace nda {

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

    /*
     * These *must* be values (copies) since evaluating an expression might
     * depend on values that would have otherwise been out of scope at the time
     * of the expression's evaluation.
     *
     * Example: exp(pow(2, x)) needs to refer to the constant 2 which would
     * normally have been destroyed after the call to pow().
     */
    const Expr1 lhs;
    const Expr2 rhs;

    static constexpr size_t ndim = Expr1::ndim;
    size_t size() const { return lhs.size(); }
    std::array<size_t, ndim> shape() const { return lhs.shape(); }

    elemwise_binary_expr(const Expr1& l, const Expr2& r) : lhs(l), rhs(r) {
        assert(l.size() == r.size());
        assert(shape_match(l.shape(), r.shape()));
    }

    /*
     * Direct index access (like ndarrays) is provided for convenience in case
     * someone uses 'auto' and winds up with an un-evaulated expression
     * template.
     */
    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    value_type operator()(Is... i) const { return Op::eval(lhs(i...), rhs(i...)); }

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
    static constexpr bool _is_expression = true;

    const Expr expr;

    static constexpr size_t ndim = Expr::ndim;
    size_t size() const { return expr.size(); }
    std::array<size_t, ndim> shape() const { return expr.shape(); }

    elemwise_unary_expr(const Expr& e) : expr(e) { }

    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    value_type operator()(Is... i) const { return Op::eval(expr(i...)); }

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
 * Although the expression templates have interfaces like nda_impl, an nda_impl
 * can't simply be treated as an expression itself, otherwise it would get
 * copied everywhere (see comment in elemwise_binary_expr).
 */
template <typename Arr>
struct variable {
    using value_type = typename Arr::value_type;
    using shape_type = typename Arr::shape_type;
    static constexpr bool _is_expression = true;

    const Arr& arr;

    static constexpr size_t ndim = Arr::ndim;
    size_t size() const { return arr.size(); }
    std::array<size_t, ndim> shape() const { return arr.shape(); }

    variable(const Arr& a) : arr(a) {}

    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    const value_type& operator()(Is... i) const { return arr(i...); }

    struct const_iterator {
        typename Arr::const_iterator iter;

        explicit const_iterator(const variable& p, bool end=false) :
            iter(end ? p.arr.end() : p.arr.begin()) {}

        const value_type& operator*() const { return *iter; }
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
 * A constant expression template containing a constant value and a matching
 * expression (to define the shape of the constant).
 */
template <typename T, typename Expr>
struct constant {
    using value_type = T;
    using shape_type = typename Expr::shape_type;
    static constexpr bool _is_expression = true;

    T value;
    const Expr expr;

    static constexpr size_t ndim = Expr::ndim;
    size_t size() const { return expr.size(); }
    std::array<size_t, ndim> shape() const { return expr.shape(); }

    explicit constant(const T& t, const Expr& e) : value(t), expr(e)
    {}

    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    const value_type& operator()(Is... i) const { return value; }

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
 * Macros to define the required functions/operators and expression template
 * operator classes for basic operators (+,-,*,/), unary, and binary functions.
 *
 * Each requires 3 overloaded forms: one for a pair of arrays/expressions, and
 * one for a single array/expression with a broadcasted scalar on either side.
 */
#define nda_to_variable(T) typename std::conditional<is_expression<T>::value, T, variable<T>>::type
#define as_expr_or_var(T, x) static_cast<nda_to_variable(T)>((x))

#define make_basic_elemwise_binary_expr_op(op, name) \
template <typename Expr1, typename Expr2> \
struct name ## _op { \
    using value_type = decltype(typename Expr1::value_type() op typename Expr2::value_type()); \
    using shape_type = max_shape_pack<typename Expr1::shape_type, typename Expr2::shape_type>; \
    static constexpr value_type eval(const value_type& l, const value_type& r) { return l op r; } \
}; \
template <typename Expr1, typename Expr2> \
typename std::enable_if<is_array_or_expr(Expr1) && is_array_or_expr(Expr2), \
elemwise_binary_expr<nda_to_variable(Expr1), nda_to_variable(Expr2), name ## _op>>::type \
operator op(const Expr1& lhs, const Expr2& rhs) { \
    return {as_expr_or_var(Expr1, lhs), \
            as_expr_or_var(Expr2, rhs)}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_array_or_expr(Expr) && std::is_arithmetic<T>::value, \
elemwise_binary_expr<nda_to_variable(Expr), constant<T, nda_to_variable(Expr)>, name ## _op>>::type \
operator op(const Expr& lhs, const T& rhs) { \
    return {as_expr_or_var(Expr, lhs), \
            constant<T, nda_to_variable(Expr)>{rhs, as_expr_or_var(Expr, lhs)}}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_array_or_expr(Expr) && std::is_arithmetic<T>::value, \
elemwise_binary_expr<constant<T, nda_to_variable(Expr)>, nda_to_variable(Expr), name ## _op>>::type \
operator op(const T& lhs, const Expr& rhs) { \
    return {constant<T, nda_to_variable(Expr)>{lhs, as_expr_or_var(Expr, rhs)}, \
            as_expr_or_var(Expr, rhs)}; \
}

#define make_func_elemwise_unary_expr_op(func, name) \
template <typename Expr> \
struct name ## _op { \
    using value_type = decltype(func(typename Expr::value_type())); \
    using shape_type = typename Expr::shape_type; \
    static constexpr value_type eval(const value_type& e) { return func(e); } \
}; \
template <typename Expr> \
typename std::enable_if<is_array_or_expr(Expr), \
elemwise_unary_expr<nda_to_variable(Expr), name ## _op>>::type \
name(const Expr& expr) { return as_expr_or_var(Expr, expr); }

#define make_func_elemwise_binary_expr_op(func, name) \
template <typename Expr1, typename Expr2> \
struct name ## _op { \
    using value_type = decltype(func(typename Expr1::value_type(), typename Expr2::value_type())); \
    using shape_type = max_shape_pack<typename Expr1::shape_type, typename Expr2::shape_type>; \
    static constexpr value_type eval(const value_type& l, const value_type& r) { return func(l, r); } \
}; \
template <typename Expr1, typename Expr2> \
typename std::enable_if<is_array_or_expr(Expr1) && is_array_or_expr(Expr2), \
elemwise_binary_expr<nda_to_variable(Expr1), nda_to_variable(Expr2), name ## _op>>::type \
func(const Expr1& lhs, const Expr2& rhs) { \
    return {as_expr_or_var(Expr1, lhs), \
            as_expr_or_var(Expr2, rhs)}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_array_or_expr(Expr) && std::is_arithmetic<T>::value, \
elemwise_binary_expr<nda_to_variable(Expr), constant<T, nda_to_variable(Expr)>, name ## _op>>::type \
func(const Expr& lhs, const T& rhs) { \
    return {as_expr_or_var(Expr, lhs), \
            constant<T, nda_to_variable(Expr)>{rhs, as_expr_or_var(Expr, lhs)}}; \
} \
template <typename Expr, typename T> \
typename std::enable_if<is_array_or_expr(Expr) && std::is_arithmetic<T>::value, \
elemwise_binary_expr<constant<T, nda_to_variable(Expr)>, nda_to_variable(Expr), name ## _op>>::type \
func(const T& lhs, const Expr& rhs) { \
    return {constant<T, nda_to_variable(Expr)>{lhs, as_expr_or_var(Expr, rhs)}, \
            as_expr_or_var(Expr, rhs)}; \
}

/*
 * Macro-based definitions of expression-template-generating operators for
 * basic operations, as well as unary and binary cmath functions.
 */
make_basic_elemwise_binary_expr_op(+, add);
make_basic_elemwise_binary_expr_op(-, sub);
make_basic_elemwise_binary_expr_op(*, mul);
make_basic_elemwise_binary_expr_op(/, div);

make_func_elemwise_binary_expr_op(pow, pow);
make_func_elemwise_binary_expr_op(atan2, atan2);

make_func_elemwise_unary_expr_op(exp, exp);
make_func_elemwise_unary_expr_op(log, log);
make_func_elemwise_unary_expr_op(log2, log2);
make_func_elemwise_unary_expr_op(log10, log10);
make_func_elemwise_unary_expr_op(sin, sin);
make_func_elemwise_unary_expr_op(cos, cos);
make_func_elemwise_unary_expr_op(tan, tan);
make_func_elemwise_unary_expr_op(asin, asin);
make_func_elemwise_unary_expr_op(acos, acos);
make_func_elemwise_unary_expr_op(atan, atan);

// The only unary basic operator supported currently is negation, no need for
// another macro type.
template <typename Expr>
struct neg_op {
    using value_type = decltype(-(typename Expr::value_type()));
    using shape_type = typename Expr::shape_type;

    static constexpr value_type eval(const value_type& v) { return -v; }
};
template <typename Expr>
typename std::enable_if<is_array_or_expr(Expr),
elemwise_unary_expr<nda_to_variable(Expr), neg_op>>::type
operator-(const Expr& expr) { return as_expr_or_var(Expr, expr); }

} // END namespace nda
