//! A small WGSL preprocessor.
//!
//! WGSL has no `#include` and no conditional compilation, which is painful when
//! the same lattice tables, index arithmetic and colour maps are needed by the
//! solver, the metrics reductions and three different render passes. This adds
//! the two directives we actually need and nothing else:
//!
//! ```text
//! #include "lbm/common.wgsl"     // resolved against the shader root, once per unit
//! #if SOME_DEFINE ... #else ... #endif
//! ```
//!
//! Values from [`ShaderDefines::value`] are also substituted for bare
//! `#DEFINE_NAME` tokens, which is how grid dimensions and workgroup sizes get
//! baked in as constants rather than read from a uniform on every invocation.

use anyhow::{bail, Context as _, Result};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct ShaderDefines {
    flags: HashSet<String>,
    values: HashMap<String, String>,
}

impl ShaderDefines {
    pub fn new() -> Self {
        Self::default()
    }

    /// Define a name for `#if` tests.
    pub fn flag(mut self, name: impl Into<String>) -> Self {
        self.flags.insert(name.into());
        self
    }

    /// Define a name that also substitutes textually for `#NAME`.
    pub fn value(mut self, name: impl Into<String>, value: impl ToString) -> Self {
        let name = name.into();
        self.flags.insert(name.clone());
        self.values.insert(name, value.to_string());
        self
    }

    pub fn is_defined(&self, name: &str) -> bool {
        self.flags.contains(name)
    }

    /// A stable key for pipeline caching. Sorted so it does not depend on the
    /// iteration order of the underlying maps.
    pub fn cache_key(&self) -> String {
        let mut parts: Vec<String> = self
            .flags
            .iter()
            .map(|f| match self.values.get(f) {
                Some(v) => format!("{f}={v}"),
                None => f.clone(),
            })
            .collect();
        parts.sort();
        parts.join(",")
    }
}

/// Loads and preprocesses WGSL from a directory tree.
pub struct ShaderLoader {
    root: PathBuf,
    /// Sources injected by name rather than read from disk. Used for generated
    /// code such as the lattice tables, so they cannot drift from the Rust side.
    virtual_files: HashMap<String, String>,
}

impl ShaderLoader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            virtual_files: HashMap::new(),
        }
    }

    /// Register generated source under a name that `#include` can resolve.
    pub fn add_virtual(&mut self, name: impl Into<String>, source: impl Into<String>) {
        self.virtual_files.insert(name.into(), source.into());
    }

    pub fn load(&self, entry: &str, defines: &ShaderDefines) -> Result<String> {
        let mut seen = HashSet::new();
        let mut out = String::new();
        self.expand(entry, defines, &mut seen, &mut out, 0)?;
        Ok(out)
    }

    /// Load, preprocess and hand straight to wgpu.
    pub fn create_module(
        &self,
        device: &wgpu::Device,
        entry: &str,
        defines: &ShaderDefines,
    ) -> Result<wgpu::ShaderModule> {
        let source = self.load(entry, defines)?;
        Ok(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(entry),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        }))
    }

    fn read(&self, name: &str) -> Result<String> {
        if let Some(v) = self.virtual_files.get(name) {
            return Ok(v.clone());
        }
        let path: PathBuf = self.root.join(name);
        std::fs::read_to_string(&path).with_context(|| format!("reading shader {}", path.display()))
    }

    fn expand(
        &self,
        name: &str,
        defines: &ShaderDefines,
        seen: &mut HashSet<String>,
        out: &mut String,
        depth: usize,
    ) -> Result<()> {
        if depth > 32 {
            bail!("shader include depth exceeded 32 at {name}; likely a cycle");
        }
        // Include-once. Cheaper and less surprising than requiring include guards.
        if !seen.insert(name.to_string()) {
            return Ok(());
        }

        let source = self.read(name)?;

        // Stack of (currently emitting, this chain has already taken a branch).
        let mut stack: Vec<(bool, bool)> = Vec::new();
        let emitting = |s: &Vec<(bool, bool)>| s.iter().all(|(e, _)| *e);

        for (lineno, line) in source.lines().enumerate() {
            let t = line.trim_start();

            if let Some(cond) = t.strip_prefix("#if ") {
                let taken = eval_condition(cond.trim(), defines);
                stack.push((taken, taken));
                continue;
            }
            if t == "#else" {
                let (active, taken) = stack
                    .last_mut()
                    .with_context(|| format!("{name}:{}: #else without #if", lineno + 1))?;
                *active = !*taken;
                *taken = true;
                continue;
            }
            if let Some(cond) = t.strip_prefix("#elif ") {
                let (active, taken) = stack
                    .last_mut()
                    .with_context(|| format!("{name}:{}: #elif without #if", lineno + 1))?;
                if *taken {
                    *active = false;
                } else {
                    *active = eval_condition(cond.trim(), defines);
                    *taken = *active;
                }
                continue;
            }
            if t == "#endif" {
                stack
                    .pop()
                    .with_context(|| format!("{name}:{}: #endif without #if", lineno + 1))?;
                continue;
            }

            if !emitting(&stack) {
                continue;
            }

            if let Some(rest) = t.strip_prefix("#include ") {
                let inc = rest.trim().trim_matches('"');
                self.expand(inc, defines, seen, out, depth + 1)?;
                continue;
            }

            out.push_str(&substitute(line, defines));
            out.push('\n');
        }

        if !stack.is_empty() {
            bail!("{name}: {} unterminated #if block(s)", stack.len());
        }
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Supports `NAME`, `!NAME`, and `A && B` / `A || B` without precedence games.
/// Deliberately minimal: anything more complex belongs in Rust, not in a shader.
fn eval_condition(cond: &str, defines: &ShaderDefines) -> bool {
    if let Some((a, b)) = cond.split_once("&&") {
        return eval_condition(a.trim(), defines) && eval_condition(b.trim(), defines);
    }
    if let Some((a, b)) = cond.split_once("||") {
        return eval_condition(a.trim(), defines) || eval_condition(b.trim(), defines);
    }
    match cond.strip_prefix('!') {
        Some(rest) => !defines.is_defined(rest.trim()),
        None => defines.is_defined(cond),
    }
}

/// Replace `#NAME` with its defined value. Longest names first, so `#NX_TOTAL`
/// is not clobbered by a shorter `#NX`.
fn substitute(line: &str, defines: &ShaderDefines) -> String {
    if !line.contains('#') || defines.values.is_empty() {
        return line.to_string();
    }
    let mut names: Vec<&String> = defines.values.keys().collect();
    names.sort_by_key(|n| std::cmp::Reverse(n.len()));

    let mut s = line.to_string();
    for name in names {
        let token = format!("#{name}");
        if s.contains(&token) {
            s = s.replace(&token, &defines.values[name]);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loader_with(files: &[(&str, &str)]) -> ShaderLoader {
        let mut l = ShaderLoader::new(".");
        for (n, s) in files {
            l.add_virtual(*n, *s);
        }
        l
    }

    #[test]
    fn includes_are_expanded_once() {
        let l = loader_with(&[
            (
                "main.wgsl",
                "#include \"a.wgsl\"\n#include \"a.wgsl\"\nmain();",
            ),
            ("a.wgsl", "fn a() {}"),
        ]);
        let out = l.load("main.wgsl", &ShaderDefines::new()).unwrap();
        assert_eq!(
            out.matches("fn a() {}").count(),
            1,
            "include-once was not honoured"
        );
        assert!(out.contains("main();"));
    }

    #[test]
    fn conditionals_select_the_right_branch() {
        let src = "#if FP16\nfp16\n#else\nfp32\n#endif";
        let l = loader_with(&[("m.wgsl", src)]);

        let a = l
            .load("m.wgsl", &ShaderDefines::new().flag("FP16"))
            .unwrap();
        assert!(a.contains("fp16") && !a.contains("fp32"));

        let b = l.load("m.wgsl", &ShaderDefines::new()).unwrap();
        assert!(b.contains("fp32") && !b.contains("fp16"));
    }

    #[test]
    fn elif_chain_takes_only_one_branch() {
        let src = "#if A\na\n#elif B\nb\n#elif C\nc\n#else\nd\n#endif";
        let l = loader_with(&[("m.wgsl", src)]);
        let out = l
            .load("m.wgsl", &ShaderDefines::new().flag("B").flag("C"))
            .unwrap();
        assert!(out.contains('b'));
        assert!(!out.contains('c') && !out.contains('a') && !out.contains('d'));
    }

    #[test]
    fn nested_conditionals_respect_the_outer_branch() {
        let src = "#if OUTER\n#if INNER\nboth\n#endif\nouter\n#endif\nalways";
        let l = loader_with(&[("m.wgsl", src)]);
        let out = l
            .load("m.wgsl", &ShaderDefines::new().flag("INNER"))
            .unwrap();
        // OUTER is not defined, so nothing inside it may appear.
        assert!(!out.contains("both") && !out.contains("outer"));
        assert!(out.contains("always"));
    }

    #[test]
    fn value_substitution_prefers_the_longer_name() {
        let l = loader_with(&[("m.wgsl", "let a = #NX; let b = #NX_TOTAL;")]);
        let d = ShaderDefines::new()
            .value("NX", 4u32)
            .value("NX_TOTAL", 64u32);
        let out = l.load("m.wgsl", &d).unwrap();
        assert!(out.contains("let a = 4;"), "got {out}");
        assert!(out.contains("let b = 64;"), "got {out}");
    }

    #[test]
    fn unterminated_conditional_is_an_error() {
        let l = loader_with(&[("m.wgsl", "#if A\nx")]);
        assert!(l.load("m.wgsl", &ShaderDefines::new().flag("A")).is_err());
    }

    #[test]
    fn cache_key_is_order_independent() {
        let a = ShaderDefines::new().flag("B").value("A", 1);
        let b = ShaderDefines::new().value("A", 1).flag("B");
        assert_eq!(a.cache_key(), b.cache_key());
    }
}
