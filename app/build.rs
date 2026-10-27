use std::{env, fs, path::PathBuf, process::Command};

fn resource_compiler() -> PathBuf {
    if let Some(path) = env::var_os("RC") {
        return path.into();
    }
    if let Some(paths) = env::var_os("PATH") {
        for dir in env::split_paths(&paths) {
            let path = dir.join("rc.exe");
            if path.is_file() {
                return path;
            }
        }
    }
    let sdk = env::var_os("WindowsSdkDir").map(PathBuf::from).or_else(|| {
        env::var_os("ProgramFiles(x86)").map(|root| PathBuf::from(root).join("Windows Kits/10"))
    });
    if let Some(sdk) = sdk {
        let arch = match env::consts::ARCH {
            "aarch64" => "arm64",
            "x86" => "x86",
            _ => "x64",
        };
        let mut versions = fs::read_dir(sdk.join("bin"))
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name();
                let version = name
                    .to_str()?
                    .split('.')
                    .map(str::parse::<u32>)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?;
                Some((version, entry.path().join(arch).join("rc.exe")))
            })
            .collect::<Vec<_>>();
        versions.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, path) in versions {
            if path.is_file() {
                return path;
            }
        }
    }
    panic!("Install the Windows 10/11 SDK resource compiler, or set RC to rc.exe's full path");
}

fn main() {
    for file in ["build.rs", "resources/windows/app.rc", "assets/icon.ico"] {
        println!("cargo:rerun-if-changed={file}");
    }
    for name in ["RC", "WindowsSdkDir", "ProgramFiles(x86)", "PATH"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    // Host-independent parsing and metadata work; non-Windows builds need no SDK.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ENV").as_deref(),
        Ok("msvc"),
        "Material Studio's Windows resources require the MSVC target"
    );
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let version = env::var("CARGO_PKG_VERSION").unwrap();
    let numbers = ["MAJOR", "MINOR", "PATCH"].map(|part| {
        env::var(format!("CARGO_PKG_VERSION_{part}"))
            .unwrap()
            .parse::<u16>()
            .expect("Windows version component must fit in u16")
            .to_string()
    });
    let icon = root
        .join("assets/icon.ico")
        .to_string_lossy()
        .replace('\\', "/");
    let source = format!(
        "#define MATERIAL_STUDIO_ICON \"{icon}\"\n\
         #define MATERIAL_STUDIO_VERSION_NUM {},0\n\
         #define MATERIAL_STUDIO_VERSION_STR \"{version}\\0\"\n{}",
        numbers.join(","),
        fs::read_to_string(root.join("resources/windows/app.rc")).unwrap()
    );
    let rc = out.join("material-studio.rc");
    let res = out.join("material-studio.res");
    fs::write(&rc, source).unwrap();
    let status = Command::new(resource_compiler())
        .arg("/nologo")
        .arg("/c65001")
        .arg("/fo")
        .arg(&res)
        .arg(&rc)
        .status()
        .expect("Could not launch Windows SDK resource compiler");
    assert!(status.success(), "Windows resource compilation failed");
    // Link only to the app binary. The GPUI dependency still owns its manifest.
    println!("cargo:rustc-link-arg-bin=material-studio={}", res.display());
}
