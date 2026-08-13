//! Config-file-defined providers (settings.json `providers` array).
//!
//! A provider is an OpenAI-compatible endpoint: id + base URL + model + key
//! source. The key itself is never stored here — `apiKeyEnv` names the env
//! var (or auth.json entry) that holds it, resolved by `keys.rs`. The built-in
//! `deepseek` remains the zero-config fallback.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// One configured OpenAI-compatible provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    /// Stable id, `[a-z0-9-_]` (used by `--provider`, env var `<ID>_API_KEY`).
    pub id: String,
    /// Human-friendly name shown in the TUI list (defaults to `id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Base URL without the trailing `/chat/completions`, e.g.
    /// `https://api.openai.com/v1` or `http://localhost:11434/v1`.
    pub base_url: String,
    /// Default model id, e.g. `gpt-4o-mini`.
    pub model: String,
    /// Whether the model emits reasoning (thinking) deltas.
    #[serde(default)]
    pub reasoning: bool,
    /// Whether the model accepts image attachments (vision).
    #[serde(default)]
    pub supports_images: bool,
    /// Context window in tokens (planning budget for the gauge + compaction).
    #[serde(default = "default_context_window")]
    pub context_window: u64,
    /// USD per 1M tokens: cache miss input, cache hit input, output.
    #[serde(default)]
    pub pricing: Pricing,
    /// Env var name holding the key (default `<ID>_API_KEY`); the value is
    /// resolved through `keys.rs` (env → auth.json → settings.json).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Provider wire protocol: `"openai"` (default) or `"anthropic"`.
    /// Anthropic providers speak the native Messages API with prefix caching
    /// and extended thinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Extra model ids to offer in `/model` beyond `model` (e.g. a reasoning
    /// sibling). When absent, well-known ids get a built-in suggestion list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
}

/// USD per 1M tokens.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pricing {
    #[serde(default)]
    pub in_miss: f64,
    #[serde(default)]
    pub in_hit: f64,
    #[serde(default)]
    pub out: f64,
}

pub const fn default_context_window() -> u64 {
    128_000
}

impl ProviderConfig {
    /// The env var (or auth.json key) that holds this provider's key.
    pub fn key_env(&self) -> String {
        self.api_key_env
            .clone()
            .unwrap_or_else(|| format!("{}_API_KEY", self.id.to_uppercase()))
    }

    /// A provider id is valid when it only contains `[a-z0-9-_]`.
    pub fn is_valid_id(id: &str) -> bool {
        !id.is_empty()
            && id
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-' || ch == '_')
    }
}

/// Complete settings configuration from `<agent_dir>/settings.json`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub tasks: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
}

/// Load full settings config from `<agent_dir>/settings.json`. Missing file
/// or missing key yields default config.
pub fn load_settings_config(agent_dir: &Path) -> SettingsConfig {
    let path = agent_dir.join("settings.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return SettingsConfig::default();
    };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&text) else {
        return SettingsConfig::default();
    };
    let default_provider = root.get("defaultProvider").and_then(|v| v.as_str()).map(ToString::to_string);
    let default_model = root.get("defaultModel").and_then(|v| v.as_str()).map(ToString::to_string);
    let mut tasks = std::collections::BTreeMap::new();
    if let Some(t_obj) = root.get("tasks").and_then(|v| v.as_object()) {
        for (k, v) in t_obj {
            if let Some(s) = v.as_str() {
                tasks.insert(k.clone(), s.to_string());
            }
        }
    }
    let mut providers = Vec::new();
    if let Some(p_arr) = root.get("providers").and_then(|v| v.as_array()) {
        for entry in p_arr {
            if let Ok(config) = serde_json::from_value::<ProviderConfig>(entry.clone()) {
                if ProviderConfig::is_valid_id(&config.id) && !config.base_url.is_empty() && !config.model.is_empty() {
                    providers.push(config);
                }
            }
        }
    }
    SettingsConfig {
        default_provider,
        default_model,
        tasks,
        providers,
    }
}

/// Create a starter settings.json template when launching agent-m for the first time.
pub fn ensure_default_settings(agent_dir: &Path) -> std::io::Result<()> {
    let _ = std::fs::create_dir_all(agent_dir);
    let _ = ensure_default_auth(agent_dir);
    let path = agent_dir.join("settings.json");
    if path.exists() {
        return Ok(());
    }
    let template = serde_json::json!({
        "defaultProvider": "deepseek",
        "defaultModel": "deepseek-chat",
        "tasks": {
            "plan": "deepseek/deepseek-reasoner",
            "build": "deepseek/deepseek-chat",
            "compact": "deepseek/deepseek-chat",
            "subagent": "deepseek/deepseek-chat",
            "refine": "deepseek/deepseek-reasoner"
        },
        "providers": [
            {
                "id": "deepseek",
                "name": "DeepSeek API",
                "baseUrl": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "reasoning": false,
                "supportsImages": false,
                "contextWindow": 1000000,
                "pricing": {
                    "inMiss": 0.27,
                    "inHit": 0.07,
                    "out": 1.10
                },
                "models": [
                    "deepseek-chat",
                    "deepseek-reasoner"
                ]
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "baseUrl": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "reasoning": false,
                "supportsImages": true,
                "contextWindow": 128000,
                "pricing": {
                    "inMiss": 0.15,
                    "inHit": 0.075,
                    "out": 0.60
                },
                "models": [
                    "gpt-4o-mini",
                    "gpt-4o",
                    "o3-mini"
                ]
            },
            {
                "id": "anthropic",
                "name": "Anthropic Claude",
                "baseUrl": "https://api.anthropic.com/v1",
                "model": "claude-3-5-sonnet-20241022",
                "type": "anthropic",
                "reasoning": true,
                "supportsImages": true,
                "contextWindow": 200000,
                "pricing": {
                    "inMiss": 3.00,
                    "inHit": 0.30,
                    "out": 15.00
                },
                "models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-7-sonnet-20250219",
                    "claude-3-5-haiku-20241022"
                ]
            },
            {
                "id": "nvidia",
                "name": "NVIDIA NIM",
                "baseUrl": "https://integrate.api.nvidia.com/v1",
                "model": "meta/llama-3.3-70b-instruct",
                "reasoning": false,
                "supportsImages": false,
                "contextWindow": 128000,
                "models": [
                    "meta/llama-3.3-70b-instruct",
                    "deepseek-ai/deepseek-r1"
                ]
            },
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "baseUrl": "https://openrouter.ai/api/v1",
                "model": "anthropic/claude-3.5-sonnet",
                "reasoning": false,
                "supportsImages": true,
                "contextWindow": 200000,
                "models": [
                    "anthropic/claude-3.5-sonnet",
                    "deepseek/deepseek-r1",
                    "google/gemini-2.0-flash-001"
                ]
            }
        ]
    });
    let pretty = serde_json::to_string_pretty(&template)?;
    write_atomic(&path, &pretty)
}

/// Create a starter auth.json template when launching agent-m for the first time.
pub fn ensure_default_auth(agent_dir: &Path) -> std::io::Result<()> {
    let _ = std::fs::create_dir_all(agent_dir);
    let path = agent_dir.join("auth.json");
    if path.exists() {
        return Ok(());
    }
    let template = serde_json::json!({
        "deepseek": "YOUR_DEEPSEEK_API_KEY_HERE"
    });
    let pretty = serde_json::to_string_pretty(&template)?;
    write_atomic(&path, &pretty)
}

/// Parse a task route entry string (e.g. `"deepseek/deepseek-reasoner"` or `"deepseek-reasoner"`).
/// Returns `(provider_id_or_model, optional_model)`.
pub fn parse_task_route(route: &str) -> (String, Option<String>) {
    let trimmed = route.trim();
    if let Some((p, m)) = trimmed.split_once('/') {
        (p.to_string(), Some(m.to_string()))
    } else {
        (trimmed.to_string(), None)
    }
}

/// Resolve provider and model for a given task role (e.g. `"plan"`, `"build"`, `"compact"`, `"subagent"`, `"refine"`).
/// Checks `tasks.<role>`, falling back to `default_provider` / `default_model`.
pub fn resolve_task_route(config: &SettingsConfig, task: &str) -> Option<(String, Option<String>)> {
    if let Some(route) = config.tasks.get(task) {
        let (p, m) = parse_task_route(route);
        return Some((p, m));
    }
    if let Some(ref dp) = config.default_provider {
        return Some((dp.clone(), config.default_model.clone()));
    }
    None
}

/// Set or remove a task route in `<agent_dir>/settings.json`.
pub fn set_task_route(agent_dir: &Path, task: &str, route: Option<&str>) -> std::io::Result<()> {
    let path = agent_dir.join("settings.json");
    let mut root = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    
    if let Some(r) = route {
        if root.get("tasks").is_none() || !root["tasks"].is_object() {
            root["tasks"] = serde_json::json!({});
        }
        root["tasks"][task] = serde_json::Value::String(r.to_string());
    } else if let Some(tasks) = root.get_mut("tasks").and_then(|t| t.as_object_mut()) {
        tasks.remove(task);
    }
    
    let pretty = serde_json::to_string_pretty(&root)?;
    write_atomic(&path, &pretty)
}

/// Load the `providers` array from `<agent_dir>/settings.json`. Missing file
/// or missing key → empty. Invalid entries are skipped (logged via `None`),
/// never fatal — a bad entry must not brick the whole config.
pub fn load_provider_configs(agent_dir: &Path) -> Vec<ProviderConfig> {
    load_settings_config(agent_dir).providers
}

/// Persist the providers array, merging into the existing settings.json and
/// preserving every other key.
pub fn save_provider_configs(
    agent_dir: &Path,
    providers: &[ProviderConfig],
) -> std::io::Result<()> {
    let path = agent_dir.join("settings.json");
    let mut root = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    root["providers"] = serde_json::to_value(providers).unwrap_or(serde_json::json!([]));
    let pretty = serde_json::to_string_pretty(&root)?;
    write_atomic(&path, &pretty)
}

fn write_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp = path.with_extension(format!("tmp-{nanos}"));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    {
        use std::io::Write;
        let mut file = options.open(&tmp)?;
        file.write_all(format!("{content}\n").as_bytes())?;
        file.sync_all()?;
    }
    std::fs::rename(tmp, path)
}

#[cfg(test)]
mod tests {
    #[test]
    fn config_models_primary_first_then_extra_ids() {
        let config = ProviderConfig {
            models: Some(vec!["gpt-5".to_string(), "o3-mini".to_string()]),
            id: "openai".to_string(),
            name: None,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o-mini".to_string(),
            reasoning: true,
            supports_images: true,
            context_window: 128_000,
            pricing: Pricing::default(),
            api_key_env: None,
            r#type: None,
        };
        let specs = config_models(&config);
        let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["gpt-4o-mini", "gpt-5", "o3-mini"]);
        // Reasoning models on OpenAI-compatible endpoints get the effort
        // variants; vision is inherited.
        let first = &specs[0];
        assert!(first.supports_effort);
        assert_eq!(first.variants, vec!["default", "low", "high", "max"]);
        assert!(first.supports_images);
    }

    #[test]
    fn config_models_falls_back_to_known_suggestions() {
        let config = ProviderConfig {
            models: None,
            id: "openai".to_string(),
            name: None,
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            reasoning: false,
            supports_images: false,
            context_window: 128_000,
            pricing: Pricing::default(),
            api_key_env: None,
            r#type: None,
        };
        let specs = config_models(&config);
        let ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids[0], "gpt-4o");
        assert!(ids.len() > 1, "known suggestions should fill the list");
        assert!(ids.contains(&"gpt-4o-mini"));
    }

    #[test]
    fn config_models_deepseek_gets_no_effort_variants() {
        let config = ProviderConfig {
            models: None,
            id: "deepseek".to_string(),
            name: None,
            base_url: "https://api.deepseek.com".to_string(),
            model: "deepseek-reasoner".to_string(),
            reasoning: true,
            supports_images: false,
            context_window: 1_000_000,
            pricing: Pricing::default(),
            api_key_env: None,
            r#type: None,
        };
        let specs = config_models(&config);
        let first = &specs[0];
        assert!(first.reasoning);
        assert!(!first.supports_effort, "DeepSeek has no reasoning_effort");
    }

    use super::*;
    use tempfile::tempdir;

    #[test]
    fn loads_valid_providers_with_defaults() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{
                "theme": "dark",
                "providers": [
                    {"id": "openai", "baseUrl": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
                    {"id": "local", "name": "Ollama", "baseUrl": "http://localhost:11434/v1",
                     "model": "llama3.2", "contextWindow": 131072, "reasoning": false}
                ]
            }"#,
        )
        .unwrap();
        let providers = load_provider_configs(dir.path());
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].id, "openai");
        assert_eq!(providers[0].context_window, 128_000, "default window");
        assert_eq!(providers[0].key_env(), "OPENAI_API_KEY", "derived env name");
        assert_eq!(providers[1].name.as_deref(), Some("Ollama"));
        assert_eq!(providers[1].context_window, 131_072);
        assert!(!providers[1].reasoning);
    }

    #[test]
    fn invalid_entries_are_skipped_not_fatal() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"providers": [
                {"id": "ok", "baseUrl": "https://x.example/v1", "model": "m"},
                {"id": "Bad ID!", "baseUrl": "https://x.example/v1", "model": "m"},
                {"id": "nourl", "baseUrl": "", "model": "m"},
                {"id": "nomodel", "baseUrl": "https://x.example/v1", "model": ""},
                "not an object"
            ]}"#,
        )
        .unwrap();
        let providers = load_provider_configs(dir.path());
        assert_eq!(providers.len(), 1, "only the valid entry survives");
        assert_eq!(providers[0].id, "ok");
    }

    #[test]
    fn save_merges_and_preserves_other_keys() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"theme": "light", "allowedPaths": ["/tmp"]}"#,
        )
        .unwrap();
        let configs = vec![ProviderConfig {
            models: None,
            id: "openai".into(),
            name: None,
            base_url: "https://api.openai.com/v1".into(),
            model: "gpt-4o-mini".into(),
            reasoning: false,
            supports_images: true,
            context_window: 128_000,
            pricing: Pricing::default(),
            api_key_env: None,
            r#type: None,
        }];
        save_provider_configs(dir.path(), &configs).unwrap();
        let saved: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("settings.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(saved["theme"], "light", "unrelated key preserved");
        assert_eq!(saved["allowedPaths"][0], "/tmp");
        assert_eq!(saved["providers"][0]["id"], "openai");
        assert_eq!(
            saved["providers"][0]["supportsImages"], true,
            "camelCase on the wire"
        );
    }

    #[test]
    fn missing_config_yields_empty() {
        let dir = tempdir().unwrap();
        assert!(load_provider_configs(dir.path()).is_empty());
    }
}

/// Build a `Provider` from config. The key is resolved through `keys.rs`
/// (env → auth.json → settings.json) via the config's `apiKeyEnv`; pass
/// `api_key_override` (the CLI `--api-key`) to force a specific key.
/// Model suggestions for well-known provider ids, used when the config does
/// not declare a `models` list. Conservative, current-as-of-2026 ids; the
/// configured `model` always stays first in the picker.
const KNOWN_MODELS: &[(&str, &[&str])] = &[
    (
        "openai",
        &[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "o3-mini",
            "gpt-5",
        ],
    ),
    (
        "anthropic",
        &["claude-sonnet-4-5", "claude-opus-4-1", "claude-3-7-sonnet"],
    ),
    ("groq", &["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]),
    (
        "openrouter",
        &["anthropic/claude-sonnet-4-5", "openai/gpt-5"],
    ),
];

/// Build the ModelSpec list a config provider should advertise: the primary
/// `model` first, then the config's `models` list, then (fallback) the
/// well-known suggestions for the provider id.
fn config_models(config: &ProviderConfig) -> Vec<crate::ModelSpec> {
    let mut ids = vec![config.model.clone()];
    if let Some(models) = &config.models {
        for id in models {
            if id != &config.model {
                ids.push(id.clone());
            }
        }
    } else {
        if let Some((_, suggestions)) = KNOWN_MODELS
            .iter()
            .find(|(known_id, _)| *known_id == config.id)
        {
            for id in *suggestions {
                if id != &config.model {
                    ids.push((*id).to_string());
                }
            }
        }
    }
    ids.into_iter()
        .map(|id| {
            let mut spec = crate::ModelSpec::new(id.clone())
                .reasoning(config.reasoning || id.contains("reason") || id.starts_with("o"))
                .context_window(config.context_window)
                .pricing(
                    config.pricing.in_miss,
                    config.pricing.in_hit,
                    config.pricing.out,
                );
            if id == config.model {
                spec = spec.name(config.name.clone().unwrap_or_else(|| config.id.clone()));
            }
            if config.supports_images {
                spec = spec.supports_images(true);
            }
            // OpenAI-compatible endpoints that offer reasoning expose a
            // `reasoning_effort` parameter → the Default/low/high/max variant
            // selector. DeepSeek's own API does not (its built-in specs
            // leave supports_effort off).
            if spec.reasoning && config.id != "deepseek" {
                spec = spec.effort(&["default", "low", "high", "max"]);
            }
            spec
        })
        .collect()
}

pub fn provider_from_config(
    config: &ProviderConfig,
    api_key_override: Option<String>,
    agent_dir: &Path,
) -> Box<dyn crate::Provider> {
    let api_key = api_key_override
        .or_else(|| crate::resolve_api_key(&config.key_env(), &config.id, agent_dir));
    let specs = config_models(config);
    if config.r#type.as_deref() == Some("anthropic") {
        return Box::new(crate::AnthropicProvider::new(
            config.id.clone(),
            config.name.clone().unwrap_or_else(|| config.id.clone()),
            config.base_url.clone(),
            api_key,
            specs,
        ));
    }
    // The primary model spec keeps its name; mark it explicitly.
    Box::new(crate::OpenAiCompatibleProvider::new(
        config.id.clone(),
        config.name.clone().unwrap_or_else(|| config.id.clone()),
        config.base_url.clone(),
        api_key,
        specs,
    ))
}

#[cfg(test)]
mod factory_tests {
    use super::*;
    use tempfile::tempdir;

    fn config(id: &str) -> ProviderConfig {
        ProviderConfig {
            models: None,
            id: id.into(),
            name: None,
            base_url: "https://api.example.com/v1".into(),
            model: "model-x".into(),
            reasoning: true,
            supports_images: false,
            context_window: 300_000,
            pricing: Pricing {
                in_miss: 1.0,
                in_hit: 0.2,
                out: 2.0,
            },
            api_key_env: None,
            r#type: None,
        }
    }

    #[test]
    fn factory_builds_spec_from_config() {
        let dir = tempdir().unwrap();
        let provider = provider_from_config(&config("acme"), None, dir.path());
        let models = provider.models();
        assert_eq!(models.len(), 1);
        let spec = &models[0];
        assert_eq!(spec.id, "model-x");
        assert_eq!(spec.context_window, Some(300_000), "window from config");
        assert!(spec.reasoning, "reasoning from config");
        assert!((spec.price_in_miss - 1.0).abs() < f64::EPSILON);
        assert_eq!(provider.id(), "acme");
        assert_eq!(provider.display_name(), "acme");
        // Default env name derived from the id.
        assert_eq!(config("acme").key_env(), "ACME_API_KEY");
    }

    #[test]
    fn factory_resolves_key_from_agent_dir_auth() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("auth.json"), r#"{"acme": "sk-acme-key"}"#).unwrap();
        let provider = provider_from_config(&config("acme"), None, dir.path());
        assert_eq!(provider.api_key(), Some("sk-acme-key"));
        // CLI override wins.
        let provider =
            provider_from_config(&config("acme"), Some("sk-override".into()), dir.path());
        assert_eq!(provider.api_key(), Some("sk-override"));
    }
}
