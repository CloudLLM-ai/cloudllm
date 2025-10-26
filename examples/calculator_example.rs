//! Example demonstrating the Calculator tool with comprehensive usage patterns
//!
//! This example shows how to use the Calculator for various mathematical operations:
//! - Basic arithmetic
//! - Trigonometric functions
//! - Statistical operations
//! - Complex expressions

use cloudllm::tools::Calculator;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let calc = Calculator::new();

    println!("=== CloudLLM Calculator Tool Example ===\n");

    // ===== ARITHMETIC OPERATIONS =====
    println!("--- Arithmetic Operations ---");
    demo_arithmetic(&calc).await?;

    // ===== TRIGONOMETRIC FUNCTIONS =====
    println!("\n--- Trigonometric Functions (in radians) ---");
    demo_trigonometric(&calc).await?;

    // ===== LOGARITHMIC & EXPONENTIAL =====
    println!("\n--- Logarithmic & Exponential Functions ---");
    demo_logarithmic(&calc).await?;

    // ===== STATISTICAL FUNCTIONS =====
    println!("\n--- Statistical Functions ---");
    demo_statistical(&calc).await?;

    // ===== COMPLEX EXPRESSIONS =====
    println!("\n--- Complex Expressions ---");
    demo_complex(&calc).await?;

    // ===== ERROR HANDLING =====
    println!("\n--- Error Handling ---");
    demo_error_handling(&calc).await?;

    println!("\n✓ All examples completed successfully!");
    Ok(())
}

async fn demo_arithmetic(calc: &Calculator) -> Result<(), Box<dyn std::error::Error>> {
    let examples = vec![
        ("2 + 2", "Simple addition"),
        ("10 - 3", "Subtraction"),
        ("4 * 5", "Multiplication"),
        ("20 / 4", "Division"),
        ("2^3", "Exponentiation"),
        ("17 % 5", "Modulo operation"),
        ("(2 + 3) * 4", "Order of operations with parentheses"),
    ];

    for (expr, desc) in examples {
        match calc.evaluate(expr).await {
            Ok(result) => println!("  {} = {} ({})", expr, result, desc),
            Err(e) => println!("  {} ERROR: {} ({})", expr, e, desc),
        }
    }
    Ok(())
}

async fn demo_trigonometric(calc: &Calculator) -> Result<(), Box<dyn std::error::Error>> {
    let examples = vec![
        ("sin(0)", "Sine of 0"),
        ("sin(pi/2)", "Sine of π/2"),
        ("cos(0)", "Cosine of 0"),
        ("cos(pi)", "Cosine of π"),
        ("tan(0)", "Tangent of 0"),
        ("asin(0.5)", "Inverse sine of 0.5"),
        ("acos(0.5)", "Inverse cosine of 0.5"),
        ("atan(1)", "Inverse tangent of 1"),
        ("csc(1)", "Cosecant of 1"),
        ("sec(0)", "Secant of 0"),
        ("cot(1)", "Cotangent of 1"),
    ];

    for (expr, desc) in examples {
        match calc.evaluate(expr).await {
            Ok(result) => println!("  {} = {:.6} ({})", expr, result, desc),
            Err(e) => println!("  {} ERROR: {} ({})", expr, e, desc),
        }
    }
    Ok(())
}

async fn demo_logarithmic(calc: &Calculator) -> Result<(), Box<dyn std::error::Error>> {
    let examples = vec![
        ("ln(2.718281828)", "Natural log of e"),
        ("log(100)", "Log base 10 of 100"),
        ("log2(8)", "Log base 2 of 8"),
        ("exp(1)", "e raised to power 1"),
        ("exp(ln(5))", "e^(ln(5)) should equal 5"),
    ];

    for (expr, desc) in examples {
        match calc.evaluate(expr).await {
            Ok(result) => println!("  {} = {:.6} ({})", expr, result, desc),
            Err(e) => println!("  {} ERROR: {} ({})", expr, e, desc),
        }
    }
    Ok(())
}

async fn demo_statistical(calc: &Calculator) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Array: [1, 2, 3, 4, 5]");

    let examples = vec![
        ("mean([1, 2, 3, 4, 5])", "Arithmetic mean"),
        ("median([1, 2, 3, 4, 5])", "Median value"),
        ("mode([1, 1, 2, 3, 3, 3])", "Most frequent value"),
        ("sum([1, 2, 3, 4, 5])", "Sum of all values"),
        ("count([1, 2, 3, 4, 5])", "Number of elements"),
        ("min([5, 2, 8, 1, 9])", "Minimum value"),
        ("max([5, 2, 8, 1, 9])", "Maximum value"),
        ("std([1, 2, 3, 4, 5])", "Sample standard deviation"),
        ("stdpop([1, 2, 3, 4, 5])", "Population standard deviation"),
        ("var([1, 2, 3, 4, 5])", "Sample variance"),
        ("varpop([1, 2, 3, 4, 5])", "Population variance"),
    ];

    for (expr, desc) in examples {
        match calc.evaluate(expr).await {
            Ok(result) => println!("  {} = {:.6} ({})", expr, result, desc),
            Err(e) => println!("  {} ERROR: {} ({})", expr, e, desc),
        }
    }
    Ok(())
}

async fn demo_complex(calc: &Calculator) -> Result<(), Box<dyn std::error::Error>> {
    let examples = vec![
        ("sqrt(16) + sqrt(25)", "Square root operations combined"),
        ("sin(pi/2) + cos(pi)", "Trigonometric combo"),
        ("abs(sin(-pi/2))", "Nested functions"),
        ("floor(3.7) + ceil(3.2)", "Rounding functions"),
        ("2 * pi", "Circle diameter multiplier"),
        ("e^2", "e squared"),
    ];

    for (expr, desc) in examples {
        match calc.evaluate(expr).await {
            Ok(result) => println!("  {} = {:.6} ({})", expr, result, desc),
            Err(e) => println!("  {} ERROR: {} ({})", expr, e, desc),
        }
    }
    Ok(())
}

async fn demo_error_handling(calc: &Calculator) -> Result<(), Box<dyn std::error::Error>> {
    let error_examples = vec![
        ("1 / 0", "Division by zero (returns infinity)"),
        ("2 +* 3", "Invalid syntax"),
        ("mean([])", "Empty array"),
        ("mean([1, 2,])", "Malformed array"),
    ];

    for (expr, desc) in error_examples {
        match calc.evaluate(expr).await {
            Ok(result) => {
                if result.is_infinite() {
                    println!("  {} = infinity ({})", expr, desc);
                } else {
                    println!("  {} = {} ({})", expr, result, desc);
                }
            }
            Err(e) => println!("  {} → ERROR ({}): {}", expr, desc, e),
        }
    }
    Ok(())
}
