//! Binary-level end-to-end test: `train` -> `quantize` -> `serve` over HTTP.
//!
//! Unlike `train_lora_dummy` / `quantize_artifacts` (which drive the library
//! API), this test executes the **real binaries** through `std::process::Command`
//! and exercises the published Jev contract over a live socket:
//!
//! 1. a tiny deterministic Llama checkpoint and a Jev-native dataset are written
//!    into a temp dir (no network, no real weights);
//! 2. `typed-lm-trainer train` runs LoRA and writes `adapter.safetensors`;
//! 3. `typed-lm-trainer quantize` merges the adapter and writes an FP8 artifact;
//! 4. `typed-lm-serve` loads the merged artifact on an ephemeral port and the
//!    test drives `GET /health`, `GET /health/live`, `GET /v1/models` and
//!    `POST /v1/systemone` with all three question types (`noul`/`choice`/`score`).
//!
//! The binaries are located through `CARGO_BIN_EXE_*`, which Cargo injects for
//! integration tests of a crate that itself defines binaries.

mod support;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Locates the `typed-lm-trainer` binary built by Cargo.
///
/// The binary belongs to a sibling crate, so Cargo does not inject a
/// `CARGO_BIN_EXE_*` compile-time variable here; the run-time variable
/// `CARGO_BIN_EXE_typed-lm-serve` anchors `target/<profile>/typed-lm-trainer`.
fn trainer_binary() -> anyhow::Result<PathBuf> {
    let sibling = std::env::var("CARGO_BIN_EXE_typed-lm-serve")
        .map_err(|_| anyhow::anyhow!("CARGO_BIN_EXE_typed-lm-serve is not set"))?;
    let directory = PathBuf::from(sibling)
        .parent()
        .ok_or_else(|| anyhow::anyhow!("serve binary has no parent directory"))?
        .to_path_buf();
    let trainer = directory.join(format!("typed-lm-trainer{}", std::env::consts::EXE_SUFFIX));
    if !trainer.exists() {
        return Err(anyhow::anyhow!(
            "trainer binary not found at '{}'; run `cargo test --workspace` so both \
             binaries are built first",
            trainer.display()
        ));
    }
    Ok(trainer)
}

/// Locates the `typed-lm-serve` binary built by Cargo.
fn serve_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_typed-lm-serve"))
}

/// Picks a free TCP port by binding to port 0 and reading the assignment back.
fn pick_free_port() -> anyhow::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

/// Minimal HTTP/1.1 exchange: writes the request, reads the whole response.
fn http_request(
    host: &str,
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> anyhow::Result<(u16, String)> {
    let mut stream = TcpStream::connect((host, port))?;
    stream.set_read_timeout(Some(Duration::from_secs(180)))?;
    stream.set_write_timeout(Some(Duration::from_secs(180)))?;
    let body = body.unwrap_or("");
    let mut request =
        format!("{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n");
    if !body.is_empty() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");
    request.push_str(body);
    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw).to_string();
    let status = text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP status line: {:?}", text.lines().next()))?;
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, value)| value.to_string())
        .unwrap_or_default();
    Ok((status, body))
}

/// Waits until `GET /health/live` answers `200`, or fails after `timeout`.
fn wait_for_server(host: &str, port: u16, timeout: Duration) -> anyhow::Result<()> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok((status, _)) = http_request(host, port, "GET", "/health/live", None) {
            if status == 200 {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    Err(anyhow::anyhow!(
        "server on {host}:{port} did not become live within {timeout:?}"
    ))
}

/// RAII guard that kills the server process on drop, even on a test panic.
struct ServerProcess {
    child: Child,
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Spawns `typed-lm-serve` bound to `127.0.0.1:port` over `model_directory`.
fn spawn_server(model_directory: &Path, port: u16) -> anyhow::Result<ServerProcess> {
    let child = Command::new(serve_binary())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--model-id")
        .arg(model_directory)
        .arg("--served-model-name")
        .arg("jev-latest")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    Ok(ServerProcess { child })
}

/// Reads any available stderr lines for diagnostics on failure.
fn drain_stderr(server: &mut ServerProcess) -> String {
    let mut collected = String::new();
    if let Some(stderr) = server.child.stderr.take() {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok).take(40) {
            collected.push_str(&line);
            collected.push('\n');
        }
    }
    collected
}

#[test]
fn train_quantize_serve_round_trip_over_http() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let checkpoint = directory.path().join("checkpoint");
    support::write_tiny_checkpoint(&checkpoint)?;
    let dataset = directory.path().join("dataset.jsonl");
    support::write_jev_dataset(&dataset)?;

    let adapter_directory = directory.path().join("adapter");
    let quantized_directory = directory.path().join("quantized");

    // 1. Fine-tune a LoRA adapter through the real binary. The dummy checkpoint
    //    has no `resources/memory.md` in scope (temp CWD), so the training
    //    context is empty — the same context the server uses below.
    let train_output = Command::new(trainer_binary()?)
        .current_dir(directory.path())
        .arg("train")
        .arg("--model-id")
        .arg(&checkpoint)
        .arg("--dataset")
        .arg(&dataset)
        .arg("--output-directory")
        .arg(&adapter_directory)
        .arg("--method")
        .arg("lora")
        .arg("--lora-rank")
        .arg("4")
        .arg("--lora-alpha")
        .arg("8")
        .arg("--epochs")
        .arg("2")
        .arg("--batch-size")
        .arg("2")
        .arg("--learning-rate")
        .arg("1e-2")
        .arg("--max-sequence-length")
        .arg("32")
        .output()?;
    assert!(
        train_output.status.success(),
        "train failed: {}",
        String::from_utf8_lossy(&train_output.stderr)
    );
    assert!(adapter_directory.join("adapter.safetensors").exists());
    assert!(adapter_directory.join("adapter_config.json").exists());

    // 2. Merge the adapter into the base and quantize to FP8 through the binary.
    let quantize_output = Command::new(trainer_binary()?)
        .current_dir(directory.path())
        .arg("quantize")
        .arg("--model-id")
        .arg(&checkpoint)
        .arg("--adapter-directory")
        .arg(&adapter_directory)
        .arg("--quantization")
        .arg("fp8")
        .arg("--output-directory")
        .arg(&quantized_directory)
        .output()?;
    assert!(
        quantize_output.status.success(),
        "quantize failed: {}",
        String::from_utf8_lossy(&quantize_output.stderr)
    );
    assert!(quantized_directory.join("model.safetensors").exists());
    assert!(quantized_directory
        .join("quantization_config.json")
        .exists());

    // The quantized directory holds only the weights; the server still needs the
    // tokenizer and config, which are copied next to the merged weights so the
    // checkpoint resolves as a complete directory.
    std::fs::copy(
        checkpoint.join("config.json"),
        quantized_directory.join("config.json"),
    )?;
    std::fs::copy(
        checkpoint.join("tokenizer.json"),
        quantized_directory.join("tokenizer.json"),
    )?;

    // 3. Serve the quantized artifact and exercise the Jev contract.
    let host = "127.0.0.1";
    let port = pick_free_port()?;
    let mut server = spawn_server(&quantized_directory, port)?;
    if let Err(error) = wait_for_server(host, port, Duration::from_secs(120)) {
        let diagnostics = drain_stderr(&mut server);
        return Err(anyhow::anyhow!("{error}\nserver stderr:\n{diagnostics}"));
    }

    // 3a. Liveness.
    let (status, body) = http_request(host, port, "GET", "/health/live", None)?;
    assert_eq!(status, 200, "health/live body: {body}");
    assert!(body.contains("\"ok\""), "health/live body: {body}");

    // 3b. Readiness (startup seconds reported).
    let (status, body) = http_request(host, port, "GET", "/health", None)?;
    assert_eq!(status, 200, "health body: {body}");
    assert!(body.contains("startup_seconds"), "health body: {body}");

    // 3c. Model listing announces the served model and the `jev-` alias.
    let (status, body) = http_request(host, port, "GET", "/v1/models", None)?;
    assert_eq!(status, 200, "models body: {body}");
    assert!(body.contains("jev-latest"), "models body: {body}");

    // 3d. Typed evaluation covering all three question types in one request.
    let request = serde_json::json!({
        "model": "jev-latest",
        "state": "charged twice",
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": "Refund?"
            },
            "dept": {
                "type": "choice",
                "instructions": "Dept?",
                "criteria": {"billing": "Payments", "technical": "Bugs"}
            },
            "urgency": {
                "type": "score",
                "instructions": "Urgent?",
                "criteria": ["Routine", "Urgent", "Emergency"]
            }
        }
    })
    .to_string();
    let (status, body) = http_request(host, port, "POST", "/v1/systemone", Some(&request))?;
    if status != 200 {
        let diagnostics = drain_stderr(&mut server);
        return Err(anyhow::anyhow!(
            "systemone returned {status}: {body}\nserver stderr:\n{diagnostics}"
        ));
    }
    let response: serde_json::Value = serde_json::from_str(&body)?;
    assert_eq!(response["model"], "jev-latest");
    assert!(response["usage"]["input_tokens"].as_u64().is_some());
    let answers = response["answers"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("answers must be an object: {body}"))?;

    let noul = answers
        .get("refund")
        .ok_or_else(|| anyhow::anyhow!("missing noul answer"))?;
    assert_eq!(noul["type"], "noul");
    let noul_value = noul["noul"]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("noul must be numeric"))?;
    assert!((0.0..=1.0).contains(&noul_value), "noul out of range");

    let choice = answers
        .get("dept")
        .ok_or_else(|| anyhow::anyhow!("missing choice answer"))?;
    assert_eq!(choice["type"], "choice");
    assert!(choice["choice"].is_string(), "choice must be a label");
    assert!(choice["probabilities"].is_object());
    assert!(choice["confidence"].is_number());

    let score = answers
        .get("urgency")
        .ok_or_else(|| anyhow::anyhow!("missing score answer"))?;
    assert_eq!(score["type"], "score");
    assert!(score["score"].is_number());
    assert!(score["legend"].is_object());
    assert!(score["probabilities"].is_object());
    assert!(score["confidence"].is_number());

    Ok(())
}

#[test]
fn an_unknown_model_is_rejected_with_404() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let checkpoint = directory.path().join("checkpoint");
    support::write_tiny_checkpoint(&checkpoint)?;

    let host = "127.0.0.1";
    let port = pick_free_port()?;
    let mut server = spawn_server(&checkpoint, port)?;
    if let Err(error) = wait_for_server(host, port, Duration::from_secs(120)) {
        let diagnostics = drain_stderr(&mut server);
        return Err(anyhow::anyhow!("{error}\nserver stderr:\n{diagnostics}"));
    }

    let request = serde_json::json!({
        "model": "some-other-model",
        "state": "charged twice",
        "questions": {
            "refund": {"type": "noul", "instructions": "Refund?"}
        }
    })
    .to_string();
    let (status, body) = http_request(host, port, "POST", "/v1/systemone", Some(&request))?;
    assert_eq!(status, 404, "unknown model body: {body}");
    assert!(body.contains("error"), "error envelope expected: {body}");
    Ok(())
}
