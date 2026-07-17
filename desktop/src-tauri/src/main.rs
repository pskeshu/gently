// Gently desktop shell.
//
// A thin Tauri (WebView2) window that OWNS the Python backend. On launch it
// spawns `launch_gently.py --no-browser`, shows a splash while uvicorn boots,
// then navigates the window to the live UI (http://localhost:<port>). The whole
// UI is served by Python — this shell holds no application logic.
//
// Process ownership: the Python child spawns its own device-layer grandchild
// (via DeviceLayerSupervisor). To guarantee no orphans, on Windows we put the
// spawned Python into a Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
// children inherit the job, so when this shell exits or crashes the OS reaps the
// entire tree. See docs/superpowers/specs/2026-07-02-unified-launcher-design.md.
//
// Graceful shutdown (issue #85): on window-close the shell first ASKS the
// backend to stop (POST /api/shutdown — drains replay ingest, stops the device
// layer via its clean SIGTERM path), waits briefly for the port to go down,
// and only then exits — where the kill + Job-close floor still applies as the
// unchanged crash-safe fallback.
//
// Deferred (documented): bundling the Python environment for a redistributable
// installer (embeddable Python / PyInstaller sidecar).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent};

/// True once a graceful shutdown handshake is in flight (or done). A second
/// close request while true is allowed straight through to the hard path.
static SHUTTING: AtomicBool = AtomicBool::new(false);

/// Backend process + (Windows) the job that owns its tree, held for the app's
/// whole lifetime as Tauri managed state.
struct Backend {
    child: Mutex<Option<Child>>,
    #[cfg(windows)]
    job: jobkill::Job,
}

fn main() {
    #[cfg(windows)]
    let job = jobkill::create_kill_on_close().expect("failed to create Job Object");

    let backend = Backend {
        child: Mutex::new(None),
        #[cfg(windows)]
        job,
    };

    tauri::Builder::default()
        .manage(backend)
        .setup(|app| {
            let handle = app.handle().clone();
            // Boot the backend off the UI thread so the splash paints immediately.
            std::thread::spawn(move || boot_backend(handle));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error building the Gently desktop app")
        .run(|app_handle, event| match event {
            // Window close → graceful backend handshake first (issue #85).
            // Intercept the close, ask the backend to stop over HTTP, and only
            // exit once it's down (or the deadline passes). If a handshake is
            // already in flight (or the user insists with a second close), let
            // the close proceed to the hard path below.
            RunEvent::WindowEvent {
                event: tauri::WindowEvent::CloseRequested { api, .. },
                ..
            } => {
                if SHUTTING.load(Ordering::SeqCst) {
                    return;
                }
                SHUTTING.store(true, Ordering::SeqCst);
                api.prevent_close();
                let handle = app_handle.clone();
                std::thread::spawn(move || graceful_shutdown(handle));
            }
            RunEvent::Exit => {
                let b = app_handle.state::<Backend>();
                // Best-effort direct kill of the child we spawned...
                // (trailing `;` so the MutexGuard temporary drops before `b` —
                // without it this is the block's tail expression on non-Windows,
                // where the cfg'd jobkill line below is compiled out: E0597)
                if let Some(mut child) = b.child.lock().unwrap().take() {
                    let _ = child.kill();
                };
                // ...and close the job handle, which kill-on-close uses to reap
                // the whole tree (python + device-layer grandchild). Correct even
                // if the child kill above missed a grandchild.
                #[cfg(windows)]
                jobkill::close(&b.job);
            }
            _ => {}
        });
}

/// The graceful shutdown handshake (issue #85), run off the UI thread.
///
/// POST /api/shutdown → 200: backend is draining (device layer stopped via its
/// SIGTERM path, replay batches flushed) — wait for its port to close, then
/// exit. 409: an acquisition is running — ask the operator in the webview; on
/// confirm the PAGE resends with `{"confirm": true}` (same-origin fetch, no
/// second Rust HTTP path). Any exit still runs the RunEvent::Exit arm, so the
/// kill + Job-close floor is unchanged.
fn graceful_shutdown(app: tauri::AppHandle) {
    let port = viz_port();
    match post_shutdown(port, false) {
        Some(200) => {
            // Backend acknowledged — give it a bounded window to drain + exit.
            wait_for_port_closed("127.0.0.1", port, 6);
            app.exit(0);
        }
        Some(409) => {
            // Mid-run guard tripped. Confirm in the webview; the page itself
            // resends the shutdown with {"confirm": true} if the user agrees.
            if let Some(win) = app.get_webview_window("main") {
                let _ = win.eval(
                    "if(confirm('An acquisition is running — quit anyway?'))\
                     {fetch('/api/shutdown',{method:'POST',\
                     headers:{'Content-Type':'application/json'},\
                     body:JSON.stringify({confirm:true})});}",
                );
            }
            if wait_for_port_closed("127.0.0.1", port, 10) {
                app.exit(0);
            } else {
                // Backend still up — the operator cancelled. Keep the window
                // open and re-arm the handshake for the next close.
                SHUTTING.store(false, Ordering::SeqCst);
            }
        }
        // Anything else (backend dead, hung, or an unexpected status): nothing
        // to hand-shake with — exit now; the Exit arm reaps the tree.
        _ => {
            app.exit(0);
        }
    }
}

/// POST /api/shutdown to the local backend over a raw TCP socket — no HTTP
/// client dependency (style precedent: `wait_for_port` below). Returns the
/// response status code, or None if the request failed outright.
fn post_shutdown(port: u16, confirm: bool) -> Option<u16> {
    use std::io::{Read, Write};
    use std::net::TcpStream;

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().ok()?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let body = if confirm { r#"{"confirm":true}"# } else { "{}" };
    let req = format!(
        "POST /api/shutdown HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut buf = [0u8; 512];
    let n = stream.read(&mut buf).ok()?;
    // Status line: "HTTP/1.1 200 OK" — the second token is the code.
    String::from_utf8_lossy(&buf[..n])
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// Poll a TCP port until nothing accepts (backend gone), or `secs` elapse.
/// Returns true if the port closed within the deadline.
fn wait_for_port_closed(host: &str, port: u16, secs: u64) -> bool {
    use std::net::TcpStream;
    let addr: std::net::SocketAddr = match format!("{host}:{port}").parse() {
        Ok(a) => a,
        Err(_) => return true,
    };
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_err() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    false
}

/// Spawn the Python backend, wait for its server, then navigate the window to it.
fn boot_backend(app: tauri::AppHandle) {
    let repo = repo_root();
    let python = python_exe(&repo);

    let mut args: Vec<String> = vec!["launch_gently.py".into(), "--no-browser".into()];
    if let Ok(extra) = std::env::var("GENTLY_LAUNCH_ARGS") {
        args.extend(extra.split_whitespace().map(str::to_string));
    }

    set_status(&app, "Starting the backend…");
    eprintln!("[gently-desktop] spawning: {} {:?} (cwd={})", python.display(), args, repo.display());

    let mut cmd = Command::new(&python);
    cmd.args(&args).current_dir(&repo);
    // Release: run the console-subsystem Python backend WITHOUT a console window
    // (a GUI parent would otherwise pop one). Debug (`tauri dev`) inherits the dev
    // console so backend logs stay visible while developing.
    #[cfg(all(windows, not(debug_assertions)))]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            set_status(&app, &format!("Failed to start backend: {e}"));
            eprintln!("[gently-desktop] spawn failed: {e}");
            return;
        }
    };

    // Put the child in our kill-on-close job so its tree can't outlive us.
    #[cfg(windows)]
    {
        let b = app.state::<Backend>();
        if let Err(e) = jobkill::assign(&b.job, &child) {
            eprintln!("[gently-desktop] job assign failed (continuing): {e:?}");
        }
    }
    *app.state::<Backend>().child.lock().unwrap() = Some(child);

    let port = viz_port();
    set_status(&app, "Waiting for the server…");
    if wait_for_port("127.0.0.1", port, 120) {
        let url = format!("http://localhost:{port}");
        eprintln!("[gently-desktop] backend up — navigating to {url}");
        if let Some(win) = app.get_webview_window("main") {
            match url.parse() {
                Ok(u) => { let _ = win.navigate(u); }
                Err(e) => set_status(&app, &format!("Bad backend URL: {e}")),
            }
        }
    } else {
        set_status(&app, "Backend did not become ready in time. Check the terminal log.");
        eprintln!("[gently-desktop] timed out waiting for 127.0.0.1:{port}");
    }
}

/// Repo root: `GENTLY_HOME` override, else the compile-time project root
/// (desktop/src-tauri/../..). The compile-time path is correct for `tauri dev`
/// and for a build run on this machine; a redistributable build should set
/// `GENTLY_HOME` (or bundle Python — see the module header).
fn repo_root() -> PathBuf {
    if let Ok(p) = std::env::var("GENTLY_HOME") {
        return PathBuf::from(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("..").join("..");
    std::fs::canonicalize(&root).unwrap_or(root)
}

/// Python interpreter: `GENTLY_PYTHON` override, else the repo venv, else PATH.
fn python_exe(repo: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("GENTLY_PYTHON") {
        return PathBuf::from(p);
    }
    let venv = if cfg!(windows) {
        repo.join(".venv").join("Scripts").join("python.exe")
    } else {
        repo.join(".venv").join("bin").join("python")
    };
    if venv.exists() {
        venv
    } else {
        PathBuf::from("python")
    }
}

/// Viz port — mirrors the backend's `VIZ_PORT` env (default 8080).
fn viz_port() -> u16 {
    std::env::var("VIZ_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080)
}

/// Poll a TCP port until something accepts, or `secs` elapse.
fn wait_for_port(host: &str, port: u16, secs: u64) -> bool {
    use std::net::TcpStream;
    let addr: std::net::SocketAddr = match format!("{host}:{port}").parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(800)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(700));
    }
    false
}

/// Update the splash status line via the JS hook it exposes.
fn set_status(app: &tauri::AppHandle, msg: &str) {
    if let Some(win) = app.get_webview_window("main") {
        let safe = msg.replace('\\', "\\\\").replace('\'', "\\'");
        let js = format!("window.__gentlyStatus && window.__gentlyStatus('{safe}')");
        let _ = win.eval(js.as_str());
    }
}

/// Windows Job Object helper — kill-on-close ownership of the backend tree.
#[cfg(windows)]
mod jobkill {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// Owns a job HANDLE. `Send`/`Sync` so it can live in Tauri managed state;
    /// the handle is only touched from create/assign/close.
    pub struct Job(pub HANDLE);
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    pub fn create_kill_on_close() -> windows::core::Result<Job> {
        unsafe {
            let job = CreateJobObjectW(None, None)?;
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )?;
            Ok(Job(job))
        }
    }

    /// Assign a freshly-spawned child to the job. Its own children inherit the
    /// job (Windows default), so grandchildren are covered too.
    pub fn assign(job: &Job, child: &Child) -> windows::core::Result<()> {
        unsafe { AssignProcessToJobObject(job.0, HANDLE(child.as_raw_handle())) }
    }

    /// Close the job handle. With kill-on-close, this terminates every process
    /// still in the job.
    pub fn close(job: &Job) {
        unsafe {
            let _ = CloseHandle(job.0);
        }
    }
}
