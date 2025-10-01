use cloudllm::client_wrapper::TokenUsage;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_async_mutex_usage_tracking() {
    // Test that we can lock and update the mutex in an async context
    let usage_mutex = Mutex::new(Some(TokenUsage {
        input_tokens: 10,
        output_tokens: 20,
        total_tokens: 30,
    }));

    // Lock and update the value
    {
        let mut guard = usage_mutex.lock().await;
        *guard = Some(TokenUsage {
            input_tokens: 100,
            output_tokens: 200,
            total_tokens: 300,
        });
    }

    // Read the value back
    let guard = usage_mutex.lock().await;
    let usage = guard.as_ref().unwrap();
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 200);
    assert_eq!(usage.total_tokens, 300);
}

#[tokio::test]
async fn test_concurrent_mutex_access() {
    use std::sync::Arc;

    // Test that multiple async tasks can access the mutex concurrently
    let usage_mutex = Arc::new(Mutex::new(Some(TokenUsage {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
    })));

    let mut handles = vec![];

    // Spawn 10 tasks that all update the mutex
    for i in 0..10 {
        let mutex_clone = Arc::clone(&usage_mutex);
        let handle = tokio::spawn(async move {
            let mut guard = mutex_clone.lock().await;
            if let Some(ref mut usage) = *guard {
                usage.input_tokens += i;
                usage.total_tokens += i;
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify the final value
    let guard = usage_mutex.lock().await;
    let usage = guard.as_ref().unwrap();
    // Sum of 0..10 is 45
    assert_eq!(usage.input_tokens, 45);
    assert_eq!(usage.total_tokens, 45);
}
