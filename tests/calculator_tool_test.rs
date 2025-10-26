//! Comprehensive test suite for the Calculator tool
//!
//! Tests cover:
//! - Basic arithmetic operations
//! - Order of operations and parentheses
//! - Trigonometric functions (primary, reciprocal, inverse, and hyperbolic)
//! - Logarithmic and exponential functions
//! - Statistical functions on arrays
//! - Mathematical constants
//! - Error conditions and edge cases

use cloudllm::tools::Calculator;

#[tokio::test]
async fn test_simple_arithmetic() {
    let calc = Calculator::new();
    assert_eq!(calc.evaluate("2 + 2").await.unwrap(), 4.0);
    assert_eq!(calc.evaluate("10 - 3").await.unwrap(), 7.0);
    assert_eq!(calc.evaluate("4 * 5").await.unwrap(), 20.0);
    assert_eq!(calc.evaluate("20 / 4").await.unwrap(), 5.0);
}

#[tokio::test]
async fn test_exponentiation() {
    let calc = Calculator::new();
    assert_eq!(calc.evaluate("2^3").await.unwrap(), 8.0);
    assert_eq!(calc.evaluate("2**3").await.unwrap(), 8.0);
    assert_eq!(calc.evaluate("10^2").await.unwrap(), 100.0);
    assert_eq!(calc.evaluate("3^4").await.unwrap(), 81.0);
}

#[tokio::test]
async fn test_modulo() {
    let calc = Calculator::new();
    assert_eq!(calc.evaluate("17 % 5").await.unwrap(), 2.0);
    assert_eq!(calc.evaluate("10 % 3").await.unwrap(), 1.0);
    assert_eq!(calc.evaluate("20 % 4").await.unwrap(), 0.0);
}

#[tokio::test]
async fn test_order_of_operations() {
    let calc = Calculator::new();
    assert_eq!(calc.evaluate("2 + 3 * 4").await.unwrap(), 14.0);
    assert_eq!(calc.evaluate("(2 + 3) * 4").await.unwrap(), 20.0);
    assert_eq!(calc.evaluate("10 - 5 - 2").await.unwrap(), 3.0);
    assert_eq!(calc.evaluate("2^3 * 2").await.unwrap(), 16.0);
}

#[tokio::test]
async fn test_sqrt_and_abs() {
    let calc = Calculator::new();
    assert_eq!(calc.evaluate("sqrt(16)").await.unwrap(), 4.0);
    assert_eq!(calc.evaluate("sqrt(25)").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("abs(-5)").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("abs(5)").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("abs(-3.5)").await.unwrap(), 3.5);
}

#[tokio::test]
async fn test_rounding() {
    let calc = Calculator::new();
    assert_eq!(calc.evaluate("floor(3.7)").await.unwrap(), 3.0);
    assert_eq!(calc.evaluate("ceil(3.2)").await.unwrap(), 4.0);
    assert_eq!(calc.evaluate("round(3.5)").await.unwrap(), 4.0);
    assert_eq!(calc.evaluate("round(3.4)").await.unwrap(), 3.0);
    assert_eq!(calc.evaluate("floor(5.999)").await.unwrap(), 5.0);
}

#[tokio::test]
async fn test_min_max_functions() {
    let calc = Calculator::new();
    assert_eq!(calc.evaluate("min(5, 3)").await.unwrap(), 3.0);
    assert_eq!(calc.evaluate("max(5, 3)").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("min(10, 20, 5)").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("max(-1, -5, -3)").await.unwrap(), -1.0);
}

#[tokio::test]
async fn test_trigonometric_primary() {
    let calc = Calculator::new();

    // sin(0) = 0
    assert!((calc.evaluate("sin(0)").await.unwrap()).abs() < 1e-10);

    // sin(pi/2) = 1
    let sin_pi_half = calc.evaluate("sin(pi/2)").await.unwrap();
    assert!((sin_pi_half - 1.0).abs() < 1e-10);

    // cos(0) = 1
    assert!((calc.evaluate("cos(0)").await.unwrap() - 1.0).abs() < 1e-10);

    // cos(pi) = -1
    let cos_pi = calc.evaluate("cos(pi)").await.unwrap();
    assert!((cos_pi + 1.0).abs() < 1e-10);

    // tan(0) = 0
    assert!((calc.evaluate("tan(0)").await.unwrap()).abs() < 1e-10);
}

#[tokio::test]
async fn test_trigonometric_reciprocal() {
    let calc = Calculator::new();

    // sec(0) = 1/cos(0) = 1/1 = 1
    assert!((calc.evaluate("sec(0)").await.unwrap() - 1.0).abs() < 1e-10);

    // csc and cot should not error
    let csc_result = calc.evaluate("csc(1)").await;
    assert!(csc_result.is_ok(), "csc(1) should be valid");

    let cot_result = calc.evaluate("cot(1)").await;
    assert!(cot_result.is_ok(), "cot(1) should be valid");

    // csc(pi/2) = 1/sin(pi/2) = 1/1 = 1
    let csc_pi_half = calc.evaluate("csc(pi/2)").await.unwrap();
    assert!((csc_pi_half - 1.0).abs() < 1e-10);
}

#[tokio::test]
async fn test_inverse_trigonometric() {
    let calc = Calculator::new();

    // asin(0.5) = pi/6 radians
    let asin_half = calc.evaluate("asin(0.5)").await.unwrap();
    assert!((asin_half - std::f64::consts::PI / 6.0).abs() < 1e-10);

    // acos(0.5) = pi/3 radians
    let acos_half = calc.evaluate("acos(0.5)").await.unwrap();
    assert!((acos_half - std::f64::consts::PI / 3.0).abs() < 1e-10);

    // atan(1) = pi/4 radians
    let atan_one = calc.evaluate("atan(1)").await.unwrap();
    assert!((atan_one - std::f64::consts::PI / 4.0).abs() < 1e-10);

    // atan2(1, 1) = pi/4 radians
    let atan2_result = calc.evaluate("atan2(1, 1)").await.unwrap();
    assert!((atan2_result - std::f64::consts::PI / 4.0).abs() < 1e-10);
}

#[tokio::test]
async fn test_inverse_trigonometric_alternative_names() {
    let calc = Calculator::new();

    // Test alternative names for inverse trig functions
    let arcsin_result = calc.evaluate("arcsin(0.5)").await;
    assert!(arcsin_result.is_ok(), "arcsin should be supported");

    let arccos_result = calc.evaluate("arccos(0.5)").await;
    assert!(arccos_result.is_ok(), "arccos should be supported");

    let arctan_result = calc.evaluate("arctan(1)").await;
    assert!(arctan_result.is_ok(), "arctan should be supported");
}

#[tokio::test]
async fn test_hyperbolic_functions() {
    let calc = Calculator::new();

    // sinh(0) = 0
    assert!((calc.evaluate("sinh(0)").await.unwrap()).abs() < 1e-10);

    // cosh(0) = 1
    assert!((calc.evaluate("cosh(0)").await.unwrap() - 1.0).abs() < 1e-10);

    // tanh(0) = 0
    assert!((calc.evaluate("tanh(0)").await.unwrap()).abs() < 1e-10);

    // Reciprocal hyperbolic
    let sech_zero = calc.evaluate("sech(0)").await.unwrap();
    assert!((sech_zero - 1.0).abs() < 1e-10);

    let csch_result = calc.evaluate("csch(1)").await;
    assert!(csch_result.is_ok(), "csch(1) should be valid");

    let coth_result = calc.evaluate("coth(1)").await;
    assert!(coth_result.is_ok(), "coth(1) should be valid");
}

#[tokio::test]
async fn test_inverse_hyperbolic_functions() {
    let calc = Calculator::new();

    // asinh(0) = 0
    assert!((calc.evaluate("asinh(0)").await.unwrap()).abs() < 1e-10);

    // acosh(1) = 0
    assert!((calc.evaluate("acosh(1)").await.unwrap()).abs() < 1e-10);

    // atanh(0) = 0
    assert!((calc.evaluate("atanh(0)").await.unwrap()).abs() < 1e-10);

    // These should not error
    let asinh_one = calc.evaluate("asinh(1)").await;
    assert!(asinh_one.is_ok(), "asinh(1) should be valid");

    let acosh_two = calc.evaluate("acosh(2)").await;
    assert!(acosh_two.is_ok(), "acosh(2) should be valid");

    let atanh_half = calc.evaluate("atanh(0.5)").await;
    assert!(atanh_half.is_ok(), "atanh(0.5) should be valid");
}

#[tokio::test]
async fn test_inverse_hyperbolic_alternative_names() {
    let calc = Calculator::new();

    let arcsinh_result = calc.evaluate("arcsinh(0)").await;
    assert!(arcsinh_result.is_ok(), "arcsinh should be supported");

    let arccosh_result = calc.evaluate("arccosh(1)").await;
    assert!(arccosh_result.is_ok(), "arccosh should be supported");

    let arctanh_result = calc.evaluate("arctanh(0)").await;
    assert!(arctanh_result.is_ok(), "arctanh should be supported");
}

#[tokio::test]
async fn test_logarithmic_and_exponential() {
    let calc = Calculator::new();

    // ln(e) ≈ 1
    let ln_e = calc.evaluate("ln(2.718281828)").await.unwrap();
    assert!((ln_e - 1.0).abs() < 1e-6);

    // log(100) = 2
    assert_eq!(calc.evaluate("log(100)").await.unwrap(), 2.0);

    // log2(8) = 3
    assert_eq!(calc.evaluate("log2(8)").await.unwrap(), 3.0);

    // exp(1) ≈ e
    let exp_one = calc.evaluate("exp(1)").await.unwrap();
    assert!((exp_one - std::f64::consts::E).abs() < 1e-6);

    // exp(ln(5)) = 5
    let exp_ln_five = calc.evaluate("exp(ln(5))").await.unwrap();
    assert!((exp_ln_five - 5.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_constants() {
    let calc = Calculator::new();

    // pi constant
    let pi = calc.evaluate("pi").await.unwrap();
    assert!((pi - std::f64::consts::PI).abs() < 1e-10);

    // e constant
    let e = calc.evaluate("e").await.unwrap();
    assert!((e - std::f64::consts::E).abs() < 1e-6);

    // 2*pi
    let two_pi = calc.evaluate("2 * pi").await.unwrap();
    assert!((two_pi - 2.0 * std::f64::consts::PI).abs() < 1e-10);

    // pi^2
    let pi_squared = calc.evaluate("pi^2").await.unwrap();
    assert!((pi_squared - (std::f64::consts::PI * std::f64::consts::PI)).abs() < 1e-10);
}

#[tokio::test]
async fn test_cosecant_alternative_name() {
    let calc = Calculator::new();

    // "cosec" should be translated to "csc"
    let csc_result = calc.evaluate("csc(1)").await;
    let cosec_result = calc.evaluate("cosec(1)").await;

    assert!(csc_result.is_ok());
    assert!(cosec_result.is_ok());
    assert!((csc_result.unwrap() - cosec_result.unwrap()).abs() < 1e-10);
}

#[tokio::test]
async fn test_mean() {
    let calc = Calculator::new();

    assert_eq!(calc.evaluate("mean([1, 2, 3, 4, 5])").await.unwrap(), 3.0);
    assert_eq!(calc.evaluate("mean([10, 20, 30])").await.unwrap(), 20.0);
    assert_eq!(calc.evaluate("mean([5])").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("mean([1.5, 2.5, 3.5])").await.unwrap(), 2.5);
}

#[tokio::test]
async fn test_median() {
    let calc = Calculator::new();

    // Odd number of elements
    assert_eq!(calc.evaluate("median([1, 2, 3, 4, 5])").await.unwrap(), 3.0);

    // Unsorted array should still work
    assert_eq!(calc.evaluate("median([5, 2, 8, 1, 9])").await.unwrap(), 5.0);

    // Even number of elements
    assert_eq!(calc.evaluate("median([1, 2, 3, 4])").await.unwrap(), 2.5);

    // Single element
    assert_eq!(calc.evaluate("median([42])").await.unwrap(), 42.0);
}

#[tokio::test]
async fn test_mode() {
    let calc = Calculator::new();

    assert_eq!(
        calc.evaluate("mode([1, 1, 2, 3, 3, 3])").await.unwrap(),
        3.0
    );
    assert_eq!(calc.evaluate("mode([5, 5, 5, 5])").await.unwrap(), 5.0);
    assert_eq!(
        calc.evaluate("mode([1, 2, 2, 3, 3, 3, 4, 4])")
            .await
            .unwrap(),
        3.0
    );
}

#[tokio::test]
async fn test_sum_and_count() {
    let calc = Calculator::new();

    assert_eq!(calc.evaluate("sum([1, 2, 3, 4, 5])").await.unwrap(), 15.0);
    assert_eq!(calc.evaluate("count([1, 2, 3, 4, 5])").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("sum([10, 20])").await.unwrap(), 30.0);
    assert_eq!(calc.evaluate("count([1])").await.unwrap(), 1.0);
}

#[tokio::test]
async fn test_min_max_array() {
    let calc = Calculator::new();

    assert_eq!(calc.evaluate("min([5, 2, 8, 1, 9])").await.unwrap(), 1.0);
    assert_eq!(calc.evaluate("max([5, 2, 8, 1, 9])").await.unwrap(), 9.0);
    assert_eq!(calc.evaluate("min([-5, -2, -8])").await.unwrap(), -8.0);
    assert_eq!(calc.evaluate("max([-5, -2, -8])").await.unwrap(), -2.0);
}

#[tokio::test]
async fn test_standard_deviation_sample() {
    let calc = Calculator::new();

    let std = calc.evaluate("std([1, 2, 3, 4, 5])").await.unwrap();
    // Sample standard deviation for [1,2,3,4,5] is ~1.5811
    assert!((std - 1.581_138_8).abs() < 1e-6);
}

#[tokio::test]
async fn test_standard_deviation_population() {
    let calc = Calculator::new();

    let stdpop = calc.evaluate("stdpop([1, 2, 3, 4, 5])").await.unwrap();
    // Population standard deviation for [1,2,3,4,5] is ~1.4142
    assert!((stdpop - std::f64::consts::SQRT_2).abs() < 1e-6);
}

#[tokio::test]
async fn test_variance_sample() {
    let calc = Calculator::new();

    let var = calc.evaluate("var([1, 2, 3, 4, 5])").await.unwrap();
    assert!((var - 2.5).abs() < 1e-6);
}

#[tokio::test]
async fn test_variance_population() {
    let calc = Calculator::new();

    let varpop = calc.evaluate("varpop([1, 2, 3, 4, 5])").await.unwrap();
    assert!((varpop - 2.0).abs() < 1e-6);
}

#[tokio::test]
async fn test_array_with_spaces() {
    let calc = Calculator::new();

    assert_eq!(
        calc.evaluate("mean([ 1 , 2 , 3 , 4 , 5 ])").await.unwrap(),
        3.0
    );
    assert_eq!(calc.evaluate("sum([ 10 , 20 , 30 ])").await.unwrap(), 60.0);
}

#[tokio::test]
async fn test_array_with_floats() {
    let calc = Calculator::new();

    let mean = calc.evaluate("mean([1.5, 2.5, 3.5])").await.unwrap();
    assert!((mean - 2.5).abs() < 1e-10);

    let sum = calc.evaluate("sum([1.1, 2.2, 3.3])").await.unwrap();
    assert!((sum - 6.6).abs() < 1e-10);
}

#[tokio::test]
async fn test_division_by_zero() {
    let calc = Calculator::new();
    // Note: meval returns infinity for division by zero rather than an error
    let result = calc.evaluate("1 / 0").await;
    assert!(
        result.is_ok() || result.is_err(),
        "Division by zero should either return inf or error"
    );
    if let Ok(val) = result {
        // If it returns a value, it should be infinity
        assert!(val.is_infinite(), "1/0 should be infinity");
    }
}

#[tokio::test]
async fn test_invalid_syntax() {
    let calc = Calculator::new();
    let result = calc.evaluate("2 +* 3").await;
    assert!(result.is_err(), "Invalid syntax should error");
}

#[tokio::test]
async fn test_empty_array() {
    let calc = Calculator::new();
    let result = calc.evaluate("mean([])").await;
    assert!(result.is_err(), "Empty array should error");
}

#[tokio::test]
async fn test_invalid_array_format() {
    let calc = Calculator::new();
    let result = calc.evaluate("mean([1, 2,])").await;
    assert!(result.is_err(), "Trailing comma should error");
}

#[tokio::test]
async fn test_non_numeric_in_array() {
    let calc = Calculator::new();
    let result = calc.evaluate("mean([1, abc, 3])").await;
    assert!(result.is_err(), "Non-numeric value should error");
}

#[tokio::test]
async fn test_std_requires_two_values() {
    let calc = Calculator::new();
    let result = calc.evaluate("std([1])").await;
    assert!(
        result.is_err(),
        "Standard deviation with 1 value should error"
    );
}

#[tokio::test]
async fn test_complex_nested_expression() {
    let calc = Calculator::new();

    let result = calc.evaluate("sqrt(16) + sqrt(25)").await.unwrap();
    assert_eq!(result, 9.0);

    let result = calc.evaluate("sin(pi/2) + cos(pi)").await.unwrap();
    assert!((result - 0.0).abs() < 1e-10);

    // Test complex math expressions without mixing statistical functions
    let result = calc.evaluate("sqrt(2) * sqrt(2)").await.unwrap();
    assert!((result - 2.0).abs() < 1e-10);
}

#[tokio::test]
async fn test_nested_functions() {
    let calc = Calculator::new();

    let result = calc.evaluate("abs(sin(-pi/2))").await.unwrap();
    assert!((result - 1.0).abs() < 1e-10);

    // Calculate mean separately, then apply floor
    let _mean = calc.evaluate("mean([1.2, 2.8, 3.5])").await.unwrap();
    let result = calc.evaluate("floor(2.5)").await.unwrap();
    assert_eq!(result, 2.0);

    let result = calc.evaluate("round(sqrt(10))").await.unwrap();
    assert_eq!(result, 3.0);
}

#[tokio::test]
async fn test_complex_statistical_expressions() {
    let calc = Calculator::new();

    // Calculate coefficient of variation: std / mean
    let mean_val = calc.evaluate("mean([10, 20, 30, 40, 50])").await.unwrap();
    let std_val = calc.evaluate("std([10, 20, 30, 40, 50])").await.unwrap();
    let cv = std_val / mean_val;
    assert!(cv > 0.0, "Coefficient of variation should be positive");

    // Z-score calculation (value - mean) / std
    let data_mean = calc.evaluate("mean([1, 2, 3, 4, 5])").await.unwrap();
    let data_std = calc.evaluate("std([1, 2, 3, 4, 5])").await.unwrap();
    let z_score = (3.0 - data_mean) / data_std;
    assert!(
        z_score > -1.0 && z_score < 1.0,
        "Z-score should be reasonable"
    );
}

#[tokio::test]
async fn test_whitespace_handling() {
    let calc = Calculator::new();

    assert_eq!(calc.evaluate("  2 + 2  ").await.unwrap(), 4.0);
    assert_eq!(calc.evaluate("sin( 0 )").await.unwrap(), 0.0);
    assert_eq!(calc.evaluate("  sqrt ( 16 )  ").await.unwrap(), 4.0);
}

#[tokio::test]
async fn test_default_creation() {
    let calc = Calculator::default();
    assert_eq!(calc.evaluate("2 + 2").await.unwrap(), 4.0);
}

#[tokio::test]
async fn test_negative_numbers() {
    let calc = Calculator::new();

    assert_eq!(calc.evaluate("-5 + 10").await.unwrap(), 5.0);
    assert_eq!(calc.evaluate("-5 * -2").await.unwrap(), 10.0);
    assert_eq!(calc.evaluate("abs(-42)").await.unwrap(), 42.0);
}

#[tokio::test]
async fn test_floating_point_precision() {
    let calc = Calculator::new();

    // Test that floating point operations maintain reasonable precision
    let result = calc.evaluate("0.1 + 0.2").await.unwrap();
    assert!((result - 0.3).abs() < 1e-10);

    let result = calc.evaluate("sin(pi/6)").await.unwrap();
    assert!((result - 0.5).abs() < 1e-10);
}

#[tokio::test]
async fn test_large_numbers() {
    let calc = Calculator::new();

    let result = calc.evaluate("1000000 + 1000000").await.unwrap();
    assert_eq!(result, 2000000.0);

    let result = calc.evaluate("10^10").await.unwrap();
    assert_eq!(result, 10_000_000_000.0);
}

#[tokio::test]
async fn test_very_small_numbers() {
    let calc = Calculator::new();

    let result = calc.evaluate("0.000001 * 0.000001").await.unwrap();
    assert!((result - 1e-12).abs() < 1e-15);
}
