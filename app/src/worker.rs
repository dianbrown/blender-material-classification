//! A bounded, persistent JSONL worker. No process or pipe I/O runs on the UI thread.
use crate::{config::Config, data::CLASSES};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Deserialize)]
pub struct Prediction {
    pub id: String,
    pub model: String,
    pub class_names: Vec<String>,
    pub probabilities: Vec<f64>,
    pub predicted_class: String,
    pub inference_ms: f64,
    pub device: String,
    #[serde(default)]
    pub load_ms: f64,
    #[serde(default)]
    pub model_cached: bool,
}

impl Prediction {
    fn validate(&self) -> Result<(), String> {
        if self.class_names != CLASSES
            || self.probabilities.len() != 7
            || self
                .probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            || (self.probabilities.iter().sum::<f64>() - 1.0).abs() > 0.01
            || !CLASSES.contains(&self.predicted_class.as_str())
            || !self.inference_ms.is_finite()
            || self.inference_ms < 0.0
        {
            return Err("The model returned an invalid prediction.".into());
        }
        let maximum = self.probabilities.iter().copied().fold(0.0_f64, f64::max);
        let predicted_index = CLASSES
            .iter()
            .position(|name| *name == self.predicted_class)
            .unwrap();
        if self.probabilities[predicted_index] < maximum {
            return Err("The predicted class does not match the model scores.".into());
        }
        Ok(())
    }
}

pub enum Event {
    Ready,
    Failed(String),
    Finished {
        id: String,
        result: Result<Prediction, WorkerError>,
    },
}

/// A request rejection is a valid reply from a healthy worker. Only a broken
/// transport or protocol requires discarding the process and its cached models.
#[derive(Debug)]
pub enum WorkerError {
    Rejected { code: String, message: String },
    Connection(String),
}

impl WorkerError {
    pub fn requires_restart(&self) -> bool {
        matches!(self, Self::Connection(_))
    }
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected { message, .. } | Self::Connection(message) => f.write_str(message),
        }
    }
}
enum Request {
    Predict {
        id: String,
        model: String,
        path: PathBuf,
    },
    Health,
}

pub struct Worker {
    tx: SyncSender<Request>,
    pub events: Receiver<Event>,
    busy: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl Worker {
    pub fn start(config: Config) -> Self {
        let (tx, rx) = mpsc::sync_channel(1);
        let (events_tx, events) = mpsc::channel();
        let busy = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_busy = busy.clone();
        let worker_stop = stop.clone();
        let join = thread::spawn(move || {
            let mut process = match Process::start(&config, &worker_stop) {
                Ok(p) => {
                    let _ = events_tx.send(Event::Ready);
                    Some(p)
                }
                Err(e) => {
                    let _ = events_tx.send(Event::Failed(e));
                    None
                }
            };
            while !worker_stop.load(Ordering::Acquire) {
                let request = match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(r) => r,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if let Some(p) = process.as_mut() {
                            if matches!(p.child.try_wait(), Ok(Some(_))) {
                                process = None;
                                let _ = events_tx.send(Event::Failed(
                                    "The inference process stopped. Retry to restart it.".into(),
                                ));
                            }
                        }
                        continue;
                    }
                    Err(_) => break,
                };
                if process.is_none() {
                    match Process::start(&config, &worker_stop) {
                        Ok(p) => {
                            process = Some(p);
                            let _ = events_tx.send(Event::Ready);
                        }
                        Err(e) => {
                            worker_busy.store(false, Ordering::Release);
                            match request {
                                Request::Predict { id, .. } => {
                                    let _ = events_tx.send(Event::Finished {
                                        id,
                                        result: Err(WorkerError::Connection(e)),
                                    });
                                }
                                Request::Health => {
                                    let _ = events_tx.send(Event::Failed(e));
                                }
                            }
                            continue;
                        }
                    }
                }
                let p = process.as_mut().unwrap();
                match request {
                    Request::Predict { id, model, path } => {
                        let result = p.exchange(json!({"id": id, "op": "predict", "model": model, "image_path": path}), &worker_stop, Duration::from_secs(180))
                            .and_then(|v| parse_prediction(v, &id, &model).map_err(WorkerError::Connection));
                        // A protocol failure or process crash must never poison later requests.
                        if result.as_ref().is_err_and(|error| error.requires_restart()) {
                            process = None;
                        }
                        worker_busy.store(false, Ordering::Release);
                        let _ = events_tx.send(Event::Finished { id, result });
                    }
                    Request::Health => {
                        let result = p.exchange(
                            json!({"id":"health", "op":"health"}),
                            &worker_stop,
                            Duration::from_secs(10),
                        );
                        worker_busy.store(false, Ordering::Release);
                        match result {
                            Ok(_) => {
                                let _ = events_tx.send(Event::Ready);
                            }
                            Err(e) => {
                                if e.requires_restart() {
                                    process = None;
                                }
                                let _ = events_tx.send(Event::Failed(e.to_string()));
                            }
                        }
                    }
                }
            }
            drop(process);
        });
        Self {
            tx,
            events,
            busy,
            stop,
            join: Some(join),
        }
    }

    pub fn busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }
    pub fn predict(&self, id: String, model: String, path: PathBuf) -> Result<(), String> {
        self.submit(Request::Predict { id, model, path })
    }
    pub fn retry(&self) -> Result<(), String> {
        self.submit(Request::Health)
    }
    fn submit(&self, request: Request) -> Result<(), String> {
        if self
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("An inference is already running.".into());
        }
        if self.tx.try_send(request).is_err() {
            self.busy.store(false, Ordering::Release);
            return Err("The inference queue is unavailable.".into());
        }
        Ok(())
    }
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn parse_prediction(value: Value, id: &str, model: &str) -> Result<Prediction, String> {
    let prediction: Prediction = serde_json::from_value(value)
        .map_err(|e| format!("The inference response could not be read: {e}"))?;
    if prediction.id != id || prediction.model != model {
        return Err("The inference response did not match this request.".into());
    }
    prediction.validate()?;
    Ok(prediction)
}

struct Process {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<Result<String, String>>,
}
impl Process {
    fn start(config: &Config, stop: &AtomicBool) -> Result<Self, String> {
        let mut command = Command::new(&config.python);
        command
            .arg("-u")
            .arg(config.root.join("scripts/demo/worker.py"))
            .arg("--root")
            .arg(&config.root)
            .arg("--device")
            .arg(&config.device)
            .current_dir(&config.root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        // Diagnostics are retained without creating a visible helper console.
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(std::env::temp_dir().join("material-studio-worker.log"));
        command.stderr(log.map(Stdio::from).unwrap_or_else(|_| Stdio::null()));
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn().map_err(|e| format!("Could not start local inference: {e}. Check the Python runtime in your resource folder."))?;
        let stdin = child
            .stdin
            .take()
            .ok_or("The inference input pipe is unavailable.")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("The inference output pipe is unavailable.")?;
        let (tx, lines) = mpsc::sync_channel(4);
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if tx.send(line.map_err(|e| e.to_string())).is_err() {
                    return;
                }
            }
        });
        let mut p = Self {
            child,
            stdin,
            lines,
        };
        let line = p.read(stop, Duration::from_secs(30))?;
        let ready: Value = serde_json::from_str(&line)
            .map_err(|_| "Local inference sent an invalid startup response.")?;
        if ready["event"] != "ready" || ready["protocol_version"] != 1 {
            return Err("Local inference uses an unsupported protocol.".into());
        }
        Ok(p)
    }
    fn read(&mut self, stop: &AtomicBool, timeout: Duration) -> Result<String, String> {
        let start = Instant::now();
        loop {
            if stop.load(Ordering::Acquire) {
                return Err("Inference was cancelled.".into());
            }
            if start.elapsed() >= timeout {
                return Err("Inference took too long. Retry to restart the local worker.".into());
            }
            match self.lines.recv_timeout(Duration::from_millis(50)) {
                Ok(line) => return line,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("The inference process stopped. Retry to restart it.".into());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
    fn exchange(
        &mut self,
        request: Value,
        stop: &AtomicBool,
        timeout: Duration,
    ) -> Result<Value, WorkerError> {
        writeln!(self.stdin, "{request}")
            .and_then(|_| self.stdin.flush())
            .map_err(|e| WorkerError::Connection(format!("Cannot contact local inference: {e}")))?;
        let line = self.read(stop, timeout).map_err(WorkerError::Connection)?;
        let value: Value = serde_json::from_str(&line).map_err(|_| {
            WorkerError::Connection("Local inference returned malformed data.".into())
        })?;
        decode_response(value, &request["id"])
    }
}

fn decode_response(value: Value, id: &Value) -> Result<Value, WorkerError> {
    if &value["id"] != id {
        return Err(WorkerError::Connection(
            "Local inference returned an unexpected request identifier.".into(),
        ));
    }
    match value["ok"].as_bool() {
        Some(true) => Ok(value),
        Some(false) => match (
            value["error"]["code"].as_str(),
            value["error"]["message"].as_str(),
        ) {
            (Some(code), Some(message)) if !code.is_empty() && !message.is_empty() => {
                Err(WorkerError::Rejected {
                    code: code.into(),
                    message: message.into(),
                })
            }
            _ => Err(WorkerError::Connection(
                "Local inference returned an invalid error response.".into(),
            )),
        },
        None => Err(WorkerError::Connection(
            "Local inference returned an invalid response status.".into(),
        )),
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "{}", json!({"id":"shutdown", "op":"shutdown"}));
        let _ = self.stdin.flush();
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(2) {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response() -> Value {
        json!({"id":"7", "model":"dinov2_finetuned", "class_names":CLASSES, "probabilities":[0.4,0.1,0.1,0.1,0.1,0.1,0.1], "predicted_class":"wood", "inference_ms":12.5, "device":"cpu"})
    }
    #[test]
    fn valid_response_preserves_class_order() {
        assert_eq!(
            parse_prediction(response(), "7", "dinov2_finetuned")
                .unwrap()
                .probabilities
                .len(),
            7
        );
    }
    #[test]
    fn stale_response_is_rejected() {
        assert!(parse_prediction(response(), "8", "dinov2_finetuned").is_err());
    }
    #[test]
    fn incorrect_model_is_rejected() {
        assert!(parse_prediction(response(), "7", "resnet_fixed").is_err());
    }
    #[test]
    fn invalid_distribution_is_rejected() {
        let mut v = response();
        v["probabilities"][0] = json!(0.9);
        assert!(parse_prediction(v, "7", "dinov2_finetuned").is_err());
    }
    #[test]
    fn wrong_class_order_is_rejected() {
        let mut v = response();
        v["class_names"][0] = json!("metal");
        assert!(parse_prediction(v, "7", "dinov2_finetuned").is_err());
    }

    #[test]
    fn structured_rejection_preserves_code_and_does_not_require_restart() {
        let error = decode_response(json!({"id":"7", "ok":false, "error":{"code":"invalid_image", "message":"Cannot decode image."}}), &json!("7")).unwrap_err();
        assert!(!error.requires_restart());
        assert!(matches!(error, WorkerError::Rejected { code, .. } if code == "invalid_image"));
    }

    #[test]
    fn malformed_or_mismatched_error_requires_restart() {
        for response in [
            json!({"id":"other", "ok":false, "error":{"code":"invalid_image", "message":"Bad image"}}),
            json!({"id":"7", "ok":false, "error":{"message":"Bad image"}}),
            json!({"id":"7", "error":{"code":"invalid_image", "message":"Bad image"}}),
        ] {
            assert!(
                decode_response(response, &json!("7"))
                    .unwrap_err()
                    .requires_restart()
            );
        }
    }

    #[test]
    fn rejected_image_preserves_worker_pid_and_cached_model() {
        let root = std::env::temp_dir().join(format!(
            "material-studio-rejection-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(root.join("scripts/demo")).unwrap();
        std::fs::write(root.join("scripts/demo/worker.py"), r#"
import json, os, pathlib, sys
root = pathlib.Path(__file__).resolve().parents[2]
with (root / 'processes.txt').open('a') as log:
    log.write(str(os.getpid()) + '\n')
loaded = False
print(json.dumps({'event':'ready','protocol_version':1}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if r['op'] == 'shutdown':
        break
    if pathlib.Path(r['image_path']).name == 'invalid.png':
        print(json.dumps({'id':r['id'],'ok':False,'error':{'code':'invalid_image','message':'Cannot decode image.'}}), flush=True)
        continue
    print(json.dumps({'id':r['id'],'ok':True,'model':r['model'],'class_names':['wood','metal','stone','concrete','ceramic','fabric','leather'],'probabilities':[.4,.1,.1,.1,.1,.1,.1],'predicted_class':'wood','inference_ms':12.5,'device':'cpu','model_cached':loaded,'load_ms':0 if loaded else 10}), flush=True)
    loaded = True
"#).unwrap();
        let config = Config {
            root: root.clone(),
            python: std::env::var_os("MATERIAL_STUDIO_TEST_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| "python".into()),
            device: "cpu".into(),
            opaque: true,
            view: "classify".into(),
            image: None,
            classify: false,
            smoke_output: None,
        };
        let mut worker = Worker::start(config);
        assert!(matches!(
            worker.events.recv_timeout(Duration::from_secs(10)).unwrap(),
            Event::Ready
        ));
        for (id, filename) in [
            ("cold", "valid.png"),
            ("invalid", "invalid.png"),
            ("warm", "valid.png"),
        ] {
            worker
                .predict(id.into(), "dinov2_finetuned".into(), root.join(filename))
                .unwrap();
            // A restarted process emits Ready here; accepting only Finished catches cache loss.
            match worker.events.recv_timeout(Duration::from_secs(10)).unwrap() {
                Event::Finished {
                    id: reply_id,
                    result,
                } => {
                    assert_eq!(reply_id, id);
                    if id == "invalid" {
                        assert!(!result.unwrap_err().requires_restart());
                    } else {
                        let prediction = result.unwrap();
                        assert_eq!(prediction.model_cached, id == "warm");
                        if id == "warm" {
                            assert_eq!(prediction.load_ms, 0.0);
                        }
                    }
                }
                _ => panic!("Worker restarted after a recoverable request rejection"),
            }
        }
        worker.shutdown();
        assert_eq!(
            std::fs::read_to_string(root.join("processes.txt"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "Requires MATERIAL_STUDIO_INTEGRATION_ROOT with saved models and a configured Python runtime"]
    fn real_worker_rejection_preserves_cached_model() {
        let root = PathBuf::from(
            std::env::var_os("MATERIAL_STUDIO_INTEGRATION_ROOT")
                .expect("Set MATERIAL_STUDIO_INTEGRATION_ROOT to the complete study resources"),
        );
        let python = PathBuf::from(
            std::env::var_os("MATERIAL_STUDIO_TEST_PYTHON")
                .expect("Set MATERIAL_STUDIO_TEST_PYTHON to the study Python runtime"),
        );
        let invalid = std::env::temp_dir().join(format!(
            "material-studio-invalid-{}.png",
            std::process::id()
        ));
        std::fs::write(&invalid, b"This is not a PNG image").unwrap();
        let valid = root.join("app/assets/samples/wood_01.png");
        let config = Config {
            root,
            python,
            device: "cpu".into(),
            opaque: true,
            view: "classify".into(),
            image: None,
            classify: false,
            smoke_output: None,
        };
        let mut worker = Worker::start(config);
        assert!(matches!(
            worker.events.recv_timeout(Duration::from_secs(30)).unwrap(),
            Event::Ready
        ));
        for (id, path) in [
            ("cold", valid.clone()),
            ("invalid", invalid.clone()),
            ("warm", valid),
        ] {
            worker
                .predict(id.into(), "dinov2_finetuned".into(), path)
                .unwrap();
            match worker
                .events
                .recv_timeout(Duration::from_secs(180))
                .unwrap()
            {
                Event::Finished {
                    id: reply_id,
                    result,
                } => {
                    assert_eq!(reply_id, id);
                    if id == "invalid" {
                        assert!(!result.unwrap_err().requires_restart());
                    } else {
                        let prediction = result.unwrap();
                        assert_eq!(prediction.model_cached, id == "warm");
                        // The real worker times its cache lookup, so a cached
                        // model can legitimately report a small nonzero load_ms.
                        println!(
                            "{id}: device={} cached={} load_ms={:.2} inference_ms={:.2}",
                            prediction.device,
                            prediction.model_cached,
                            prediction.load_ms,
                            prediction.inference_ms
                        );
                    }
                }
                _ => panic!("Real worker restarted after an invalid image"),
            }
        }
        worker.shutdown();
        std::fs::remove_file(invalid).unwrap();
    }

    #[test]
    fn worker_bounds_requests_recovers_from_crash_and_shuts_down() {
        let root = std::env::temp_dir().join(format!(
            "material-studio-worker-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(root.join("scripts/demo")).unwrap();
        // This process fixture deliberately crashes once, then supplies a real JSONL pipe response.
        std::fs::write(root.join("scripts/demo/worker.py"), r#"
import json, os, pathlib, sys, time
root = pathlib.Path(__file__).resolve().parents[2]
print(json.dumps({'event':'ready','protocol_version':1}), flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if r['op'] == 'shutdown':
        break
    if r['op'] == 'health':
        print(json.dumps({'id':r['id'],'ok':True}), flush=True)
        continue
    marker = root / 'crashed'
    if not marker.exists():
        marker.write_text('yes')
        os._exit(3)
    if r['model'] == 'hang':
        time.sleep(60)
    time.sleep(.15)
    print(json.dumps({'id':r['id'],'ok':True,'model':r['model'],'class_names':['wood','metal','stone','concrete','ceramic','fabric','leather'],'probabilities':[.4,.1,.1,.1,.1,.1,.1],'predicted_class':'wood','inference_ms':12.5,'device':'cpu'}), flush=True)
"#).unwrap();
        let config = Config {
            root: root.clone(),
            python: std::env::var_os("MATERIAL_STUDIO_TEST_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| "python".into()),
            device: "cpu".into(),
            opaque: true,
            view: "classify".into(),
            image: None,
            classify: false,
            smoke_output: None,
        };
        let mut worker = Worker::start(config);
        assert!(matches!(
            worker.events.recv_timeout(Duration::from_secs(10)).unwrap(),
            Event::Ready
        ));
        worker
            .predict(
                "first".into(),
                "dinov2_finetuned".into(),
                root.join("image.png"),
            )
            .unwrap();
        assert!(
            worker
                .predict(
                    "overflow".into(),
                    "dinov2_finetuned".into(),
                    root.join("image.png")
                )
                .is_err()
        );
        assert!(matches!(
            worker.events.recv_timeout(Duration::from_secs(10)).unwrap(),
            Event::Finished { result: Err(_), .. }
        ));
        assert!(!worker.busy());
        worker
            .predict(
                "second".into(),
                "dinov2_finetuned".into(),
                root.join("image.png"),
            )
            .unwrap();
        assert!(matches!(
            worker.events.recv_timeout(Duration::from_secs(10)).unwrap(),
            Event::Ready
        ));
        match worker.events.recv_timeout(Duration::from_secs(10)).unwrap() {
            Event::Finished {
                id,
                result: Ok(prediction),
            } => {
                assert_eq!(id, "second");
                assert_eq!(prediction.predicted_class, "wood");
            }
            _ => panic!("Worker failed to recover after the fixture crash"),
        }
        worker
            .predict("closing".into(), "hang".into(), root.join("image.png"))
            .unwrap();
        thread::sleep(Duration::from_millis(50));
        let start = Instant::now();
        worker.shutdown();
        assert!(start.elapsed() < Duration::from_secs(4));
        std::fs::remove_dir_all(root).unwrap();
    }
}
