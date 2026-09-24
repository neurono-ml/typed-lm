//! Shared helpers for the trainer integration tests.
//!
//! These build a tiny Llama checkpoint on disk (config + tokenizer + weights)
//! and a small Jev-native dataset, so an end-to-end train/quantize run needs no
//! network access and no real model.
//!
//! Each integration test compiles this module separately and uses a subset of
//! the helpers, so unused-function warnings are expected and suppressed.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};
use tokenizers::models::wordlevel::WordLevelBuilder;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

/// Tiny Llama configuration used by every integration test.
pub fn tiny_configuration_json() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": false
    })
}

/// A deterministic pseudo-random tensor (no external RNG dependency).
fn deterministic_tensor(shape: (usize, usize), scale: f32) -> anyhow::Result<Tensor> {
    let device = Device::Cpu;
    let count = shape.0 * shape.1;
    let mut seed = 0x1234_5678_u64;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        values.push((((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale);
    }
    Ok(Tensor::from_vec(values, shape, &device)?)
}

/// A deterministic pseudo-random positive vector (for RMSNorm weights).
fn deterministic_vector(size: usize) -> anyhow::Result<Tensor> {
    let device = Device::Cpu;
    let mut seed = 0x9abc_def0_u64;
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        values.push((((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * 0.1 + 1.0);
    }
    Ok(Tensor::from_vec(values, size, &device)?)
}

/// Builds the weight tensors for the tiny Llama configuration.
pub fn tiny_weights() -> anyhow::Result<HashMap<String, Tensor>> {
    let hidden_size = 16_usize;
    let intermediate_size = 32_usize;
    let vocab_size = 64_usize;
    let head_dimension = hidden_size / 4;
    let query_size = head_dimension * 4;
    let key_value_size = head_dimension * 2;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        deterministic_tensor((vocab_size, hidden_size), 0.1)?,
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        deterministic_tensor((vocab_size, hidden_size), 0.1)?,
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        deterministic_vector(hidden_size)?,
    );
    tensors.insert(
        "model.layers.0.input_layernorm.weight".to_string(),
        deterministic_vector(hidden_size)?,
    );
    tensors.insert(
        "model.layers.0.post_attention_layernorm.weight".to_string(),
        deterministic_vector(hidden_size)?,
    );
    for (projection, output, input) in [
        ("self_attn.q_proj", query_size, hidden_size),
        ("self_attn.k_proj", key_value_size, hidden_size),
        ("self_attn.v_proj", key_value_size, hidden_size),
        ("self_attn.o_proj", hidden_size, query_size),
        ("mlp.gate_proj", intermediate_size, hidden_size),
        ("mlp.up_proj", intermediate_size, hidden_size),
        ("mlp.down_proj", hidden_size, intermediate_size),
    ] {
        tensors.insert(
            format!("model.layers.0.{projection}.weight"),
            deterministic_tensor((output, input), 0.1)?,
        );
    }
    Ok(tensors)
}

/// Writes a tiny word-level tokenizer covering the template markers and labels.
pub fn write_tokenizer(path: &Path) -> anyhow::Result<()> {
    let vocabulary: Vec<(&str, u32)> = vec![
        ("<|system|>", 0),
        ("<|user|>", 1),
        ("<|model|>", 2),
        ("<|im_start|>system", 3),
        ("<|im_start|>user", 4),
        ("<|im_start|>assistant", 5),
        ("<|im_end|>", 6),
        ("State:", 7),
        ("Question:", 8),
        ("A", 9),
        ("B", 10),
        ("C", 11),
        ("charged", 12),
        ("twice", 13),
        ("Refund?", 14),
        ("Dept?", 15),
        ("billing", 16),
        ("technical", 17),
        ("[UNK]", 18),
    ];
    let words: HashMap<String, u32> = vocabulary
        .iter()
        .map(|(token, identifier)| (token.to_string(), *identifier))
        .collect();
    let word_level = WordLevelBuilder::default()
        .vocab(words)
        .unk_token("[UNK]".to_string())
        .build()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let mut tokenizer = Tokenizer::new(word_level);
    tokenizer.with_pre_tokenizer(Whitespace);
    tokenizer
        .save(path, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}

/// Writes a tiny Llama checkpoint (config, tokenizer and weights) into `root`.
pub fn write_tiny_checkpoint(root: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(root)?;
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&tiny_configuration_json())?,
    )?;
    write_tokenizer(&root.join("tokenizer.json"))?;
    candle_core::safetensors::save(&tiny_weights()?, root.join("model.safetensors"))?;
    Ok(())
}

/// Writes a small Jev-native JSONL dataset covering all three question types.
pub fn write_jev_dataset(path: &Path) -> anyhow::Result<()> {
    let records = [
        serde_json::json!({
            "state": "charged twice",
            "questions": {
                "refund": {"type": "noul", "instructions": "Refund?", "answer": "yes"},
                "dept": {
                    "type": "choice", "instructions": "Dept?",
                    "criteria": {"billing": "Payments", "technical": "Bugs"},
                    "answer": "billing"
                }
            }
        }),
        serde_json::json!({
            "state": "charged twice",
            "questions": {
                "refund": {"type": "noul", "instructions": "Refund?", "answer": "no"},
                "dept": {
                    "type": "choice", "instructions": "Dept?",
                    "criteria": {"billing": "Payments", "technical": "Bugs"},
                    "answer": "technical"
                }
            }
        }),
    ];
    let mut body = String::new();
    for record in records {
        body.push_str(&record.to_string());
        body.push('\n');
    }
    std::fs::write(path, body)?;
    Ok(())
}
