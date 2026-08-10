//! `agent-m plugins ...`: install, list, remove, and update out-of-tree
//! cdylib plugins (pi-style extensions). Each plugin is a separate repo with
//! a `plugin.json` manifest; install clones/builds it and copies the artifact
//! into `~/.agent-m/plugins/<name>/`.

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The manifest file every plugin repo ships.
pub const MANIFEST: &str = "plugin.json";

/// `~/.agent-m/plugins`
pub fn plugins_dir(agent_dir: &Path) -> PathBuf {
    agent_dir.join("plugins")
}

pub fn run_install(agent_dir: &Path, source: &str, rev: Option<&str>) -> Result<()> {
    let dir = plugins_dir(agent_dir);
    std::fs::create_dir_all(&dir)?;

    // Stage the plugin source: clone git URLs; build local dirs in place (so
    // relative path dependencies, e.g. on the plugin SDK, keep resolving).
    let is_git = source.starts_with("http://")
        || source.starts_with("https://")
        || source.starts_with("git@");
    let staging = std::env::temp_dir().join(format!("agent-m-plugin-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    let source_dir: PathBuf = if is_git {
        let mut command = Command::new("git");
        command.args(["clone", "--depth", "1"]);
        if let Some(rev) = rev {
            command.args(["--branch", rev]);
        }
        command.arg(source).arg(&staging);
        let status = command
            .status()
            .with_context(|| format!("cannot run git clone for {source}"))?;
        if !status.success() {
            bail!("git clone failed for {source}");
        }
        staging.clone()
    } else {
        let source_path = PathBuf::from(source);
        if !source_path.is_dir() {
            bail!("plugin source `{source}` is not a directory");
        }
        source_path
    };

    // Read or generate the manifest to learn the plugin name + entry.
    let manifest_path = source_dir.join(MANIFEST);
    let (name, entry): (String, String) = if manifest_path.is_file() {
        let text = std::fs::read_to_string(&manifest_path)?;
        let manifest: Value = serde_json::from_str(&text)
            .with_context(|| format!("invalid {MANIFEST} in {source}"))?;
        let name = manifest
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("plugin.json lacks `name`"))?
            .to_string();
        let entry = manifest
            .get("entry")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| default_entry_name(&source_dir, &name));
        (name, entry)
    } else {
        let name = source_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("plugin")
            .to_string();
        let entry = default_entry_name(&source_dir, &name);
        (name, entry)
    };

    // Build the plugin in release mode.
    let build = Command::new("cargo")
        .args(["build", "--release", "--manifest-path"])
        .arg(source_dir.join("Cargo.toml"))
        .status()
        .with_context(|| format!("cannot build plugin {name} (cargo required at install time)"))?;
    if !build.success() {
        bail!("plugin build failed for {name}");
    }

    // Find and copy the built artifact (workspace members build into an
    // ancestor workspace's target dir).
    let release = find_release_dir(&source_dir)
        .ok_or_else(|| anyhow!("no target/release found under {}", source_dir.display()))?;
    let artifact = release.join(&entry);
    let artifact = if artifact.is_file() {
        artifact
    } else {
        // Fall back to any cdylib in the release dir.
        let candidates = std::fs::read_dir(&release)
            .ok()
            .map(|entries| {
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        name.ends_with(".dylib") || name.ends_with(".so") || name.ends_with(".dll")
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        candidates
            .first()
            .ok_or_else(|| anyhow!("no cdylib artifact found in {}", release.display()))?
            .clone()
    };

    // Install into ~/.agent-m/plugins/<name>/.
    let install_dir = dir.join(&name);
    if install_dir.exists() {
        bail!(
            "plugin `{name}` already installed (remove it first: `agent-m plugins remove {name}`)"
        );
    }
    std::fs::create_dir_all(&install_dir)?;
    std::fs::copy(&artifact, install_dir.join(&entry))?;
    // Write the manifest so the registry scan knows the entry name.
    let manifest = json!({ "name": name, "version": "0.1.0", "entry": entry });
    std::fs::write(
        install_dir.join(MANIFEST),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    let _ = std::fs::remove_dir_all(&staging);
    println!("installed plugin `{name}` -> {}", install_dir.display());
    Ok(())
}

pub fn run_list(agent_dir: &Path) -> Result<()> {
    let dir = plugins_dir(agent_dir);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        println!("no plugins installed ({} does not exist)", dir.display());
        return Ok(());
    };
    let mut found = false;
    for entry in entries.flatten() {
        let plugin_dir = entry.path();
        let manifest_path = plugin_dir.join(MANIFEST);
        if !manifest_path.is_file() {
            continue;
        }
        found = true;
        let text = std::fs::read_to_string(&manifest_path)?;
        let manifest: Value = serde_json::from_str(&text).unwrap_or_default();
        let name = manifest.get("name").and_then(Value::as_str).unwrap_or("?");
        let entry_name = manifest.get("entry").and_then(Value::as_str).unwrap_or("?");
        let ok = plugin_dir.join(entry_name).is_file();
        println!("{name}\t{}", if ok { "ready" } else { "MISSING ENTRY" });
    }
    if !found {
        println!("no plugins installed");
    }
    Ok(())
}

pub fn run_remove(agent_dir: &Path, name: &str) -> Result<()> {
    let dir = plugins_dir(agent_dir).join(name);
    if !dir.exists() {
        bail!("plugin `{name}` is not installed");
    }
    std::fs::remove_dir_all(&dir)?;
    println!("removed plugin `{name}`");
    Ok(())
}

pub fn run_update(agent_dir: &Path, name: Option<&str>) -> Result<()> {
    // Rebuild the installed plugin in place (re-run cargo build --release on
    // the source is not possible without the source; for git-sourced plugins
    // we re-clone. MVP: re-run install from the plugin's manifest `source`
    // field when present.)
    let dir = plugins_dir(agent_dir);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        bail!("no plugins installed");
    };
    let mut updated = false;
    for entry in entries.flatten() {
        let plugin_dir = entry.path();
        let manifest_path = plugin_dir.join(MANIFEST);
        if !manifest_path.is_file() {
            continue;
        }
        let plugin_name = plugin_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if name.is_some() && name != Some(plugin_name) {
            continue;
        }
        let text = std::fs::read_to_string(&manifest_path)?;
        let manifest: Value = serde_json::from_str(&text).unwrap_or_default();
        if let Some(source) = manifest.get("source").and_then(Value::as_str) {
            std::fs::remove_dir_all(&plugin_dir)?;
            run_install(agent_dir, source, None)?;
            updated = true;
        } else {
            println!(
                "plugin `{plugin_name}` has no `source` in its manifest; reinstall from its repo"
            );
        }
    }
    if !updated && let Some(name) = name {
        bail!("plugin `{name}` is not installed");
    }
    if !updated && name.is_none() {
        println!("nothing to update");
    }
    Ok(())
}

/// Find the `target/release` dir for a build: the crate's own, or the
/// nearest ancestor workspace's.
fn find_release_dir(start: &Path) -> Option<PathBuf> {
    for dir in start.ancestors() {
        let candidate = dir.join("target/release");
        if candidate.is_dir() {
            return Some(candidate);
        }
    }
    None
}

/// Default artifact name: `lib<name>.dylib|.so` / `<name>.dll`.
fn default_entry_name(staging: &Path, name: &str) -> String {
    // Try to find the crate name from the root Cargo.toml, else use `name`.
    let crate_name = std::fs::read_to_string(staging.join("Cargo.toml"))
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.trim_start().starts_with("name"))
                .and_then(|line| line.split('=').nth(1))
                .map(|value| value.trim().trim_matches('"').to_string())
        })
        .unwrap_or_else(|| name.replace('-', "_"));
    if cfg!(target_os = "macos") {
        format!("lib{crate_name}.dylib")
    } else if cfg!(target_os = "windows") {
        format!("{crate_name}.dll")
    } else {
        format!("lib{crate_name}.so")
    }
}
