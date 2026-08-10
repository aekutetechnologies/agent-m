//! API key resolution: environment variable, then `auth.json`, then settings.

use serde_json::Value;
use std::path::Path;

/// Resolve an API key for `provider_id`, checking in order:
///
/// 1. the environment variable `env_var` (e.g. `DEEPSEEK_API_KEY`),
/// 2. `<agent_dir>/auth.json` — both the nested shape
///    `{"providers": {"deepseek": {"apiKey": "..."}}}` and the flat shape
///    `{"deepseek": "..."}`,
/// 3. `<agent_dir>/settings.json`, same two shapes.
pub fn resolve_api_key(env_var: &str, provider_id: &str, agent_dir: &Path) -> Option<String> {
    if let Ok(key) = std::env::var(env_var)
        && !key.is_empty()
    {
        return Some(key);
    }

    for file in ["auth.json", "settings.json"] {
        if let Some(key) = read_key_from_file(&agent_dir.join(file), provider_id) {
            return Some(key);
        }
    }

    None
}

fn read_key_from_file(path: &Path, provider_id: &str) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    let root: Value = serde_json::from_str(&contents).ok()?;

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
    fn falls_back_to_settings_json() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("settings.json"),
            r#"{"providers": {"deepseek": {"apiKey": "settings-key"}}}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_api_key("UNSET_VAR_XYZ", "deepseek", dir.path()),
            Some("settings-key".to_string())
        );
    }

    #[test]
    fn env_var_wins() {
        let dir = tempdir().unwrap();
        fs::write(
            dir.path().join("auth.json"),
            r#"{"providers": {"deepseek": {"apiKey": "file-key"}}}"#,
        )
        .unwrap();
        unsafe { std::env::set_var("AGENT_M_TEST_ENV_KEY", "env-key") };
        assert_eq!(
            resolve_api_key("AGENT_M_TEST_ENV_KEY", "deepseek", dir.path()),
            Some("env-key".to_string())
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
