use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub root: PathBuf,
    pub python: PathBuf,
    pub device: String,
    pub opaque: bool,
    pub view: String,
    pub image: Option<PathBuf>,
    pub classify: bool,
    pub smoke_output: Option<PathBuf>,
}

impl Config {
    pub fn from_args() -> Result<Self, String> {
        let args: Vec<_> = std::env::args().skip(1).collect();
        let option = |name: &str| -> Option<String> {
            args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
        };
        let explicit = option("--resources").or_else(|| std::env::var("MATERIAL_STUDIO_ROOT").ok());
        let root = if let Some(root) = explicit {
            PathBuf::from(root)
        } else {
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
            find_root(&[exe.parent().unwrap_or(&cwd), &cwd]).ok_or(
                "Cannot locate study resources. Start Material Studio with --resources ROOT.",
            )?
        };
        let root = root
            .canonicalize()
            .map_err(|e| format!("Resource folder is unavailable: {e}"))?;
        // Windows canonicalize adds a verbatim-device prefix. The Python protocol
        // deliberately accepts ordinary absolute local drive paths only.
        let root = ordinary_local_path(root);
        let python = option("--python")
            .or_else(|| std::env::var("MATERIAL_STUDIO_PYTHON").ok())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                [
                    "runtime/python.exe",
                    "python/python.exe",
                    ".venv/Scripts/python.exe",
                    ".venv/bin/python",
                ]
                .into_iter()
                .map(|p| root.join(p))
                .find(|p| p.is_file())
                .unwrap_or_else(|| PathBuf::from("python"))
            });
        let device = option("--device").unwrap_or_else(|| "auto".into());
        if !["auto", "cpu", "cuda"].contains(&device.as_str()) {
            return Err("Device must be auto, cpu or cuda.".into());
        }
        Ok(Self {
            root,
            python,
            device,
            opaque: args.iter().any(|a| a == "--opaque"),
            view: option("--view").unwrap_or_else(|| "classify".into()),
            image: option("--image").map(|image| {
                let path = PathBuf::from(image);
                if path.is_absolute() {
                    path
                } else {
                    std::env::current_dir().unwrap_or_default().join(path)
                }
            }),
            classify: args.iter().any(|a| a == "--classify"),
            smoke_output: option("--smoke-worker").map(PathBuf::from),
        })
    }
}

fn ordinary_local_path(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    if let Some(rest) = path.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        if rest.as_bytes().get(1) == Some(&b':') {
            return PathBuf::from(rest);
        }
    }
    path
}

fn find_root(starts: &[&Path]) -> Option<PathBuf> {
    for start in starts {
        for path in start.ancestors() {
            if path.join("app/assets/study.json").is_file()
                || path.join("scripts/demo/worker.py").is_file()
            {
                return Some(path.to_owned());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(windows)]
    fn canonical_drive_path_is_compatible_with_local_only_worker() {
        assert_eq!(
            ordinary_local_path(PathBuf::from(r"\\?\C:\Demo\Materials")),
            PathBuf::from(r"C:\Demo\Materials")
        );
    }
    #[test]
    fn root_discovery_uses_resources_not_working_directory_names() {
        let base =
            std::env::temp_dir().join(format!("material-studio-root-{}", std::process::id()));
        std::fs::create_dir_all(base.join("app/assets")).unwrap();
        std::fs::write(base.join("app/assets/study.json"), "{}").unwrap();
        assert_eq!(find_root(&[&base.join("bin/nested")]), Some(base.clone()));
        std::fs::remove_dir_all(base).unwrap();
    }
}
