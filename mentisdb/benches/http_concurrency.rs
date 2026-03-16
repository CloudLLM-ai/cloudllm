//! HTTP concurrency benchmark for the MentisDB REST server.
//!
//! This harness-free benchmark starts a live `mentisdbd` HTTP server **in-process**
//! on a dynamically assigned port (no external daemon required) and measures how
//! many concurrent tokio tasks the server can serve for both write and read paths.
//!
//! Two benchmark suites are run back-to-back at **100 / 1 000 / 10 000** concurrent
//! tokio tasks:
//!
//! - **Write wave** — each task appends one thought via `POST /v1/thoughts`.
//! - **Read wave** — each task reads the chain head via `POST /v1/head`.
//!
//! Per-suite, the following metrics are reported:
//!
//! | metric        | description                                     |
//! |---------------|-------------------------------------------------|
//! | `wall_ms`     | total elapsed wall-clock time in milliseconds   |
//! | `req/s`       | throughput (N tasks / wall_ms × 1000)           |
//! | `p50_ms`      | median per-task round-trip latency              |
//! | `p95_ms`      | 95th-percentile per-task round-trip latency     |
//! | `p99_ms`      | 99th-percentile per-task round-trip latency     |
//! | `errors`      | number of tasks that received a non-2xx status  |
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench http_concurrency
//! ```
//!
//! The binary prints two Markdown tables to stdout and exits with code 0 on
//! success, or 1 if the server failed to start.

use mentisdb::server::{start_servers, MentisDbServerConfig, MentisDbServiceConfig};
use mentisdb::StorageAdapterKind;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::task::JoinSet;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Chain key used for all benchmark operations.
const CHAIN_KEY: &str = "bench";

/// Number of sequential warm-up appends performed before measurement begins.
const WARMUP_COUNT: usize = 10;

/// Concurrency levels (number of parallel tokio tasks) exercised in each wave.
const CONCURRENCY_LEVELS: &[usize] = &[100, 1_000, 10_000];

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Programme entry point.
///
/// Starts the server, runs warm-up, then runs write and read waves at each
/// configured concurrency level, and finally prints the result tables.
#[tokio::main]
async fn main() {
    // Keep TempDir alive for the entire benchmark so the chain files on disk
    // are not cleaned up before the server finishes.
    let temp_dir = TempDir::new().expect("failed to create temporary benchmark directory");

    // Build a server config that binds both MCP and REST to ephemeral OS-chosen
    // ports (port 0), eliminating any risk of collisions with running services.
    let config = MentisDbServerConfig {
        service: MentisDbServiceConfig::new(
            temp_dir.path().to_path_buf(),
            CHAIN_KEY,
            StorageAdapterKind::Binary,
        ),
        mcp_addr: "127.0.0.1:0"
            .parse()
            .expect("static address literal must parse"),
        rest_addr: "127.0.0.1:0"
            .parse()
            .expect("static address literal must parse"),
    };

    let handles = start_servers(config)
        .await
        .expect("in-process mentisdbd failed to start — cannot run HTTP concurrency benchmark");

    let rest_base = Arc::new(format!("http://{}", handles.rest.local_addr()));
    eprintln!("mentisdbd REST listening at {rest_base}");

    let client = Arc::new(
        Client::builder()
            .pool_max_idle_per_host(512)
            .build()
            .expect("failed to build reqwest client"),
    );

    // Warm up: prime the chain so reads find at least some content.
    warmup(&client, &rest_base).await;

    // Write wave ---------------------------------------------------------------
    let mut write_rows: Vec<(usize, BenchRow)> = Vec::new();
    for &n in CONCURRENCY_LEVELS {
        eprintln!("write wave  n={n}…");
        let row = run_write_wave(Arc::clone(&client), Arc::clone(&rest_base), n).await;
        write_rows.push((n, row));
    }

    // Read wave ----------------------------------------------------------------
    let mut read_rows: Vec<(usize, BenchRow)> = Vec::new();
    for &n in CONCURRENCY_LEVELS {
        eprintln!("read wave   n={n}…");
        let row = run_read_wave(Arc::clone(&client), Arc::clone(&rest_base), n).await;
        read_rows.push((n, row));
    }

    // Output ------------------------------------------------------------------
    print_table("Write  —  POST /v1/thoughts", &write_rows);
    println!();
    print_table("Read   —  POST /v1/head", &read_rows);
}

// ---------------------------------------------------------------------------
// Benchmark rows
// ---------------------------------------------------------------------------

/// Aggregated benchmark result for one concurrency level.
#[derive(Debug)]
struct BenchRow {
    /// Total elapsed wall-clock time for all N tasks to complete.
    wall_time: Duration,
    /// Requests per second: `n / wall_time_secs`.
    throughput_rps: f64,
    /// Median per-task round-trip latency.
    p50: Duration,
    /// 95th-percentile per-task round-trip latency.
    p95: Duration,
    /// 99th-percentile per-task round-trip latency.
    p99: Duration,
    /// Number of tasks that received a non-2xx HTTP response or encountered a
    /// transport error.
    errors: usize,
}

// ---------------------------------------------------------------------------
// Warm-up
// ---------------------------------------------------------------------------

/// Append [`WARMUP_COUNT`] sequential thoughts so the chain is not cold when
/// measurement begins.
///
/// Failures during warm-up are non-fatal — they are printed as warnings.
async fn warmup(client: &Client, base_url: &str) {
    eprintln!("warming up with {WARMUP_COUNT} sequential appends…");
    for i in 0..WARMUP_COUNT {
        let body = build_append_body(i);
        match client
            .post(format!("{base_url}/v1/thoughts"))
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {}
            Ok(resp) => eprintln!("  warmup[{i}] non-2xx: {}", resp.status()),
            Err(err) => eprintln!("  warmup[{i}] error: {err}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Wave runners
// ---------------------------------------------------------------------------

/// Spawn `n` tasks concurrently, each appending one thought, and return
/// aggregated latency statistics.
async fn run_write_wave(client: Arc<Client>, base_url: Arc<String>, n: usize) -> BenchRow {
    let wall_start = Instant::now();
    let mut set: JoinSet<(Duration, bool)> = JoinSet::new();

    for i in 0..n {
        let c = Arc::clone(&client);
        let url = Arc::clone(&base_url);
        set.spawn(async move {
            let body = build_append_body(i);
            let t0 = Instant::now();
            let ok = c
                .post(format!("{url}/v1/thoughts"))
                .json(&body)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            (t0.elapsed(), ok)
        });
    }

    collect_wave_results(set, wall_start, n).await
}

/// Spawn `n` tasks concurrently, each reading the chain head, and return
/// aggregated latency statistics.
async fn run_read_wave(client: Arc<Client>, base_url: Arc<String>, n: usize) -> BenchRow {
    let wall_start = Instant::now();
    let mut set: JoinSet<(Duration, bool)> = JoinSet::new();

    for _ in 0..n {
        let c = Arc::clone(&client);
        let url = Arc::clone(&base_url);
        set.spawn(async move {
            let body = json!({ "chain_key": CHAIN_KEY });
            let t0 = Instant::now();
            let ok = c
                .post(format!("{url}/v1/head"))
                .json(&body)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            (t0.elapsed(), ok)
        });
    }

    collect_wave_results(set, wall_start, n).await
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build the JSON body for an append request for task index `i`.
///
/// Uses a distinct `agent_id` per task to exercise the agent registry path.
fn build_append_body(i: usize) -> serde_json::Value {
    json!({
        "chain_key":    CHAIN_KEY,
        "agent_id":     format!("agent-{i}"),
        "thought_type": "Summary",
        "content":      format!("bench thought {i}"),
    })
}

/// Drain a [`JoinSet`] whose tasks each return `(per_task_duration, success)`,
/// compute percentiles, and return a [`BenchRow`].
///
/// # Arguments
///
/// * `set`         — running task set to drain.
/// * `wall_start`  — `Instant` captured before any tasks were spawned.
/// * `n`           — expected number of tasks (used only for throughput calc).
async fn collect_wave_results(
    mut set: JoinSet<(Duration, bool)>,
    wall_start: Instant,
    n: usize,
) -> BenchRow {
    let mut durations: Vec<Duration> = Vec::with_capacity(n);
    let mut errors: usize = 0;

    while let Some(result) = set.join_next().await {
        match result {
            Ok((dur, true)) => durations.push(dur),
            Ok((dur, false)) => {
                durations.push(dur);
                errors += 1;
            }
            Err(join_err) => {
                // Task panicked — count as error, push a zero duration so the
                // length stays consistent with `n`.
                eprintln!("task panicked: {join_err}");
                durations.push(Duration::ZERO);
                errors += 1;
            }
        }
    }

    let wall_time = wall_start.elapsed();
    let throughput_rps = n as f64 / wall_time.as_secs_f64();

    // Sort to allow index-based percentile extraction.
    durations.sort_unstable();

    let p50 = percentile(&durations, 0.50);
    let p95 = percentile(&durations, 0.95);
    let p99 = percentile(&durations, 0.99);

    BenchRow {
        wall_time,
        throughput_rps,
        p50,
        p95,
        p99,
        errors,
    }
}

/// Return the duration at the given fractional percentile of a **sorted** slice.
///
/// Returns [`Duration::ZERO`] when the slice is empty.
///
/// # Arguments
///
/// * `sorted` — slice sorted in ascending order.
/// * `pct`    — fractional percentile in `[0.0, 1.0]`.
fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    // Nearest-rank formula: index = ceil(pct * len) - 1, clamped to valid range.
    let idx = ((pct * sorted.len() as f64).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Print a Markdown table with results for all concurrency levels.
///
/// # Arguments
///
/// * `title` — table title printed as a Markdown heading above the table.
/// * `rows`  — slice of `(concurrency, BenchRow)` pairs.
fn print_table(title: &str, rows: &[(usize, BenchRow)]) {
    println!("## {title}");
    println!();
    println!(
        "| {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} |",
        "concurrent", "wall_ms", "req/s", "p50_ms", "p95_ms", "p99_ms", "errors"
    );
    println!(
        "|{:->12}|{:->12}|{:->12}|{:->12}|{:->12}|{:->12}|{:->10}|",
        "", "", "", "", "", "", ""
    );
    for (n, row) in rows {
        println!(
            "| {:>10} | {:>10.1} | {:>10.1} | {:>10.3} | {:>10.3} | {:>10.3} | {:>8} |",
            n,
            row.wall_time.as_secs_f64() * 1000.0,
            row.throughput_rps,
            row.p50.as_secs_f64() * 1000.0,
            row.p95.as_secs_f64() * 1000.0,
            row.p99.as_secs_f64() * 1000.0,
            row.errors,
        );
    }
}
