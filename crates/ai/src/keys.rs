//! API key resolution: environment variable, then `auth.json`, then settings.

use serde_json::Value;
use std::path::Path;

/// Resolve an API key for `provider_id`, checking in order:
///
/// 1. `<agent_dir>/auth.json` — both the nested shape
///    `{"providers": {"deepseek": {"apiKey": "..."}}}` and the flat shape
///    `{"deepseek": "..."}`,
/// 2. `<agent_dir>/settings.json`, same two shapes.
///
/// Environment variables are intentionally not checked — use auth.json.
pub fn resolve_api_key(_env_var: &str, provider_id: &str, agent_dir: &Path) -> Option<String> {
    if let Some(key) = read_key_from_file(&agent_dir.join("auth.json"), provider_id) {
        return Some(key);
    }

    if let Some(key) = read_key_from_file(&agent_dir.join("settings.json"), provider_id) {
        tracing::warn!(
            "API key for {} found in settings.json! This file is world-readable (0644). Please move it to auth.json immediately.",
            provider_id
        );
        return Some(key);
    }

    None
}

fn read_key_from_file(path: &Path, provider_id: &str) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;

    // Preferred shape: JSON (nested or flat).
    if let Ok(root) = serde_json::from_str::<Value>(&contents) {
        let nested = root
            .get("providers")
            .and_then(|providers| providers.get(provider_id))
            .and_then(|provider| provider.get("apiKey"))
            .and_then(Value::as_str);
        if let Some(key) = nested
            && !key.is_empty()
        {
            return Some(key.to_string());
        }

        let flat = root.get(provider_id).and_then(Value::as_str);
        if let Some(key) = flat
            && !key.is_empty()
        {
            return Some(key.to_string());
        }
    }

    // Tolerated shape: dotenv-style lines, which users commonly write by
    // mistake — e.g. `DEEPSEEK_API_KEY=sk-…` or `deepseek=sk-…`. Both map to
    // the provider id.
    read_dotenv_line(&contents, provider_id)
}

/// Parse `KEY=value` lines (dotenv-style, no quoting rules beyond trimming).
/// Accepts `DEEPSEEK_API_KEY=…` (env-var name) and `deepseek=…` (provider id).
fn read_dotenv_line(contents: &str, provider_id: &str) -> Option<String> {
    let env_provider = provider_id.to_uppercase();
    for line in contents.lines() {
        let line = line.trim().strip_prefix("export ").unwrap_or(line.trim());
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        if key.eq_ignore_ascii_case(provider_id) {
            return Some(value.to_string());
        }
        let upper = key.to_uppercase();
        let bare = upper.strip_suffix("_API_KEY").unwrap_or(&upper);
        if bare == env_provider {
            return Some(value.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn reads_nested_auth_shape() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            r#"{"providers": {"deepseek": {"apiKey": "nested-key"}}}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            Some("nested-key".to_string())
        );
    }

    #[test]
    fn reads_flat_auth_shape() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("auth.json"), r#"{"deepseek": "flat-key"}"#).unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            Some("flat-key".to_string())
        );
    }

    #[test]
    fn auth_json_wins_over_env_var() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            r#"{"providers": {"deepseek": {"apiKey": "file-key"}}}"#,
        )
        .unwrap();
        unsafe { std::env::set_var("AGENT_M_TEST_ENV_KEY", "env-key") };
        // env var is ignored; auth.json is the source of truth
        assert_eq!(
            resolve_api_key("AGENT_M_TEST_ENV_KEY", "deepseek", dir.path()),
            Some("file-key".to_string())
        );
    }

    #[test]
    fn missing_key_returns_none() {
        let dir = tempdir().unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            None
        );
    }
}

#[cfg(test)]
mod dotenv_tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn reads_dotenv_style_auth_line() {
        // This is exactly what users write by mistake: a KEY=value line in
        // auth.json instead of JSON.
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            "DEEPSEEK_API_KEY=sk-dotenv-line\n",
        )
        .unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            Some("sk-dotenv-line".to_string())
        );
    }

    #[test]
    fn reads_bare_provider_line_and_export_prefix() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            "export deepseek = sk-bare-key\n",
        )
        .unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            Some("sk-bare-key".to_string())
        );
    }

    #[test]
    fn mixed_file_is_not_valid_json_and_falls_back_to_dotenv() {
        // A JSON object followed by a KEY=value line is not valid JSON, so the
        // tolerant path reads the dotenv line. Pure-JSON files (covered by the
        // other tests) still use the JSON shapes.
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            "{\"deepseek\": \"json-key\"}\nDEEPSEEK_API_KEY=sk-dotenv\n",
        )
        .unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            Some("sk-dotenv".to_string())
        );
    }

    #[test]
    fn empty_and_unrelated_lines_are_ignored() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            "# comment\n\nOPENAI_API_KEY=sk-other\nDEEPSEEK_API_KEY=\n",
        )
        .unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            None
        );
    }
}
