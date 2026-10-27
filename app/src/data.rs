use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const CLASSES: [&str; 7] = [
    "wood", "metal", "stone", "concrete", "ceramic", "fabric", "leather",
];

#[derive(Clone, Debug, Deserialize)]
pub struct Study {
    pub schema_version: u32,
    pub class_names: Vec<String>,
    pub primary_model: String,
    pub default_model: String,
    pub models: Vec<Model>,
    pub dataset: Dataset,
    pub samples: Vec<Sample>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub id: String,
    pub display_name: String,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub accuracy_ci: Interval,
    pub macro_f1_ci: Interval,
    pub confusion_matrix: Vec<Vec<u64>>,
    pub latency: Latency,
    pub size: ModelSize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Interval {
    pub low: f64,
    pub high: f64,
    pub level: f64,
    pub unit: String,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Latency {
    pub median_ms: f64,
    pub p95_ms: f64,
    pub device: String,
    pub batch_size: u64,
    pub repetitions: u64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct ModelSize {
    pub artifact_bytes: u64,
    pub total_deployment_bytes: u64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Dataset {
    pub development: Split,
    pub holdout: Split,
    pub classes: Vec<Class>,
    pub scope: String,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Split {
    pub images: u64,
    pub materials: u64,
    pub source_groups: u64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Class {
    pub name: String,
    pub definition: String,
    pub development_images: u64,
    pub holdout_images: u64,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Sample {
    pub id: String,
    pub label: String,
    pub class_name: String,
    pub path: PathBuf,
    pub attribution: String,
    pub license: String,
}

impl Study {
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join("app/assets/study.json");
        let content = std::fs::read_to_string(&path)
            .map_err(|e| format!("Study resources could not be opened: {e}"))?;
        let study: Self = serde_json::from_str(&content)
            .map_err(|e| format!("Study resources could not be read: {e}"))?;
        study.validate()?;
        Ok(study)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.class_names != CLASSES {
            return Err("The study resource format is unsupported.".into());
        }
        for model in &self.models {
            if model.confusion_matrix.len() != 7
                || model.confusion_matrix.iter().any(|r| r.len() != 7)
                || !(0.0..=1.0).contains(&model.accuracy)
                || !(0.0..=1.0).contains(&model.macro_f1)
            {
                return Err("Study metrics have invalid dimensions or values.".into());
            }
            for interval in [&model.accuracy_ci, &model.macro_f1_ci] {
                if interval.unit != "source_group"
                    || interval.low > interval.high
                    || !(0.0..=1.0).contains(&interval.low)
                    || !(0.0..=1.0).contains(&interval.high)
                    || !(0.0..1.0).contains(&interval.level)
                {
                    return Err("Study confidence intervals are invalid.".into());
                }
            }
        }
        if self.models.len() != 4
            || !self.models.iter().any(|m| m.id == self.default_model)
            || !self.models.iter().any(|m| m.id == self.primary_model)
        {
            return Err("The four study models are not available in these resources.".into());
        }
        let mut sample_ids = std::collections::HashSet::new();
        for sample in &self.samples {
            if !sample_ids.insert(&sample.id)
                || !CLASSES.contains(&sample.class_name.as_str())
                || sample.path.is_absolute()
                || sample
                    .path
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err("A sample has an invalid resource path.".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn study() -> Study {
        serde_json::from_str(include_str!("../assets/study.json")).unwrap()
    }
    #[test]
    fn actual_study_contract_is_valid() {
        study().validate().unwrap();
    }
    #[test]
    fn malformed_matrix_is_rejected() {
        let mut s = study();
        s.models[0].confusion_matrix.pop();
        assert!(s.validate().is_err());
    }
    #[test]
    fn sample_path_cannot_escape_resource_folder() {
        let mut s = study();
        s.samples[0].path = "../private.png".into();
        assert!(s.validate().is_err());
    }
}

pub fn title(value: &str) -> String {
    let mut chars = value.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
        .unwrap_or_default()
}
