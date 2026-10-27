//! Finding paired devices on the network and talking to them: a server that
//! accepts only paired devices, a client, discovery (mDNS, so addresses can
//! change), and a background check of which peers are reachable.
use std::collections::HashMap;
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

use super::wire::{Channel, Msg};
use super::{AskLevel, Identity, Peer, Peers};

const SERVICE: &str = "_ojp._tcp.local.";
const MAX_CONNECTIONS: usize = 8;
/// Devices listen here when they can, so a peer's stored address stays valid across restarts.
pub const DEFAULT_PORT: u16 = 47600;
/// A reply may take this long to start or continue (a big model on a slow CPU).
const REPLY_TIMEOUT: Duration = Duration::from_secs(300);

/// Announce a service on the local network until the returned daemon is dropped.
pub(crate) fn advertise(service: &str, instance: &str, port: u16, props: &[(&str, &str)]) -> Option<ServiceDaemon> {
    let daemon = ServiceDaemon::new().ok()?;
    let info = ServiceInfo::new(service, instance, &format!("{instance}.local."), "", port, props).ok()?.enable_addr_auto();
    daemon.register(info).ok()?;
    Some(daemon)
}

pub(crate) struct Found {
    pub name: String,
    pub instance: String,
    pub addr: SocketAddr,
}

/// Services of this type seen on the network within `timeout`.
pub(crate) fn browse(service: &str, timeout: Duration) -> Vec<Found> {
    let Ok(daemon) = ServiceDaemon::new() else { return Vec::new() };
    let Ok(rx) = daemon.browse(service) else { return Vec::new() };
    let deadline = Instant::now() + timeout;
    let mut found: Vec<Found> = Vec::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(ServiceEvent::ServiceResolved(info)) => {
                let name = info.get_property_val_str("name").unwrap_or("").to_string();
                let instance = info.get_fullname().split('.').next().unwrap_or("").to_string();
                // Prefer IPv4: it works on every network we run on.
                let addr = info.get_addresses().iter().map(|a| a.to_ip_addr()).find(|a| a.is_ipv4()).map(|ip| SocketAddr::new(ip, info.get_port()));
                if let Some(addr) = addr
                    && !found.iter().any(|f| f.instance == instance)
                {
                    found.push(Found { name, instance, addr });
                }
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let _ = daemon.shutdown();
    found
}

/// What this device can do right now, shared with the connections.
#[derive(Default)]
pub struct Caps {
    /// A model is loaded that can answer chat.
    pub chat: AtomicBool,
    /// Requests from other devices can be run here.
    pub ask: AtomicBool,
    /// The model is a main (full-size) model, not just the small router.
    pub main: AtomicBool,
}

pub enum Request {
    Chat(Vec<(String, String)>),
    Ask(String),
    Notify { title: String, text: String },
    Sync(crate::memory::SyncState),
}

/// Where the handler puts its answer: tokens and notes while it works, then one ending.
pub struct Reply {
    tx: Sender<Msg>,
}

impl Reply {
    pub fn tok(&self, text: &str) {
        let _ = self.tx.send(Msg::Tok { text: text.into() });
    }
    pub fn note(&self, text: &str) {
        let _ = self.tx.send(Msg::Note { text: text.into() });
    }
    pub fn end(self, text: &str) {
        let _ = self.tx.send(Msg::End { text: text.into() });
    }
    pub fn err(self, msg: &str) {
        let _ = self.tx.send(Msg::Err { msg: msg.into() });
    }
    pub fn ok(self) {
        let _ = self.tx.send(Msg::Ok);
    }
    pub fn sync(self, state: crate::memory::SyncState) {
        let _ = self.tx.send(Msg::SyncReply { state });
    }
}

pub struct Incoming {
    pub peer: Peer,
    pub request: Request,
    pub reply: Reply,
}

pub struct Server {
    pub port: u16,
    _mdns: Option<ServiceDaemon>,
}

impl Server {
    /// Listen for paired devices. Their requests arrive on `incoming`; whoever owns the
    /// models answers through each request's `reply`.
    pub fn start(me: Identity, caps: Arc<Caps>, incoming: Sender<Incoming>) -> Result<Server> {
        let listener = (DEFAULT_PORT..DEFAULT_PORT + 10)
            .find_map(|p| TcpListener::bind(("0.0.0.0", p)).ok())
            .map_or_else(|| TcpListener::bind("0.0.0.0:0").context("opening the device link port"), Ok)?;
        let port = listener.local_addr()?.port();
        let mdns = advertise(SERVICE, &me.fingerprint(), port, &[("name", me.name.as_str())]);
        let live = Arc::new(AtomicUsize::new(0));
        std::thread::Builder::new().name("link-accept".into()).spawn(move || {
            for stream in listener.incoming().flatten() {
                if live.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
                    live.fetch_sub(1, Ordering::SeqCst);
                    continue;
                }
                let (me, caps, incoming, live) = (me.clone(), caps.clone(), incoming.clone(), live.clone());
                std::thread::spawn(move || {
                    let _ = serve(stream, &me, &caps, &incoming);
                    live.fetch_sub(1, Ordering::SeqCst);
                });
            }
        })?;
        Ok(Server { port, _mdns: mdns })
    }
}

fn serve(stream: TcpStream, me: &Identity, caps: &Caps, incoming: &Sender<Incoming>) -> Result<()> {
    let mut ch = Channel::respond(stream, &me.private_key())?;
    // Only paired devices get past this point; the key was proven by the handshake.
    let Some(peer) = Peers::load().by_key(&ch.remote).cloned() else { bail!("unknown device") };
    ch.set_timeout(Some(Duration::from_secs(120)))?;
    while let Some(msg) = ch.recv()? {
        let denied = |what: &str| Msg::Err { msg: format!("{} doesn't allow {what} from this device", me.name) };
        let request = match msg {
            Msg::Ping => {
                ch.send(&Msg::Pong {
                    name: me.name.clone(),
                    chat: caps.chat.load(Ordering::Relaxed) && peer.perms.chat,
                    ask: caps.ask.load(Ordering::Relaxed) && peer.perms.ask != AskLevel::Off,
                    main: caps.main.load(Ordering::Relaxed),
                })?;
                continue;
            }
            Msg::Chat { messages } if peer.perms.chat => Request::Chat(messages),
            Msg::Chat { .. } => {
                ch.send(&denied("using its model"))?;
                continue;
            }
            Msg::Ask { text } if peer.perms.ask != AskLevel::Off => Request::Ask(text),
            Msg::Ask { .. } => {
                ch.send(&denied("requests"))?;
                continue;
            }
            Msg::Notify { title, text } if peer.perms.notify => Request::Notify { title, text },
            Msg::Notify { .. } => {
                ch.send(&denied("messages"))?;
                continue;
            }
            Msg::Sync { state } if peer.perms.sync => Request::Sync(state),
            Msg::Sync { .. } => {
                ch.send(&denied("memory sharing"))?;
                continue;
            }
            _ => {
                ch.send(&Msg::Err { msg: "unexpected message".into() })?;
                continue;
            }
        };
        let (tx, rx) = mpsc::channel();
        if incoming.send(Incoming { peer: peer.clone(), request, reply: Reply { tx } }).is_err() {
            ch.send(&Msg::Err { msg: "this device isn't accepting requests right now".into() })?;
            continue;
        }
        loop {
            match rx.recv_timeout(REPLY_TIMEOUT) {
                Ok(m) => {
                    let last = matches!(m, Msg::End { .. } | Msg::Err { .. } | Msg::Ok | Msg::SyncReply { .. });
                    ch.send(&m)?;
                    if last {
                        break;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    ch.send(&Msg::Err { msg: "took too long".into() })?;
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    ch.send(&Msg::Err { msg: "stopped before finishing".into() })?;
                    break;
                }
            }
        }
    }
    Ok(())
}

/// Calls other devices.
#[derive(Clone)]
pub struct Client {
    pub me: Identity,
}

impl Client {
    pub fn new(me: Identity) -> Client {
        Client { me }
    }

    /// Open a channel to a paired device: its last known addresses first, then a look around the network.
    pub fn connect(&self, peer: &Peer) -> Result<Channel> {
        let want = super::unhex(&peer.public).unwrap_or_default();
        let try_addr = |addr: SocketAddr| -> Result<Channel> {
            let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(800))?;
            let ch = Channel::initiate(stream, &self.me.private_key())?;
            if ch.remote != want {
                bail!("{addr} answered with a different key than {}'s", peer.name);
            }
            Ok(ch)
        };
        for a in &peer.addrs {
            if let Some(addr) = a.to_socket_addrs().ok().and_then(|mut i| i.next())
                && let Ok(ch) = try_addr(addr)
            {
                return Ok(ch);
            }
        }
        let fp = peer.fingerprint();
        for found in browse(SERVICE, Duration::from_millis(1500)) {
            if found.instance == fp
                && let Ok(ch) = try_addr(found.addr)
            {
                // Remember where it is now.
                let mut peers = Peers::load();
                if let Some(p) = peers.peers.iter_mut().find(|p| p.public == peer.public) {
                    p.addrs = vec![found.addr.to_string()];
                    let _ = peers.save();
                }
                return Ok(ch);
            }
        }
        bail!("{} isn't reachable right now", peer.name)
    }

    /// Send a request and follow the answer: tokens and notes go to `on`, the final text is returned.
    pub fn request(&self, peer: &Peer, msg: Msg, on: &mut dyn FnMut(&Msg)) -> Result<String> {
        let mut ch = self.connect(peer)?;
        ch.set_timeout(Some(REPLY_TIMEOUT))?;
        ch.send(&msg)?;
        loop {
            match ch.recv()? {
                Some(Msg::End { text }) => return Ok(text),
                Some(Msg::Ok) => return Ok(String::new()),
                Some(Msg::Err { msg }) => bail!("{msg}"),
                Some(other @ (Msg::Tok { .. } | Msg::Note { .. })) => on(&other),
                Some(other @ Msg::SyncReply { .. }) => {
                    on(&other);
                    return Ok(String::new());
                }
                Some(_) => {}
                None => bail!("{} hung up", peer.name),
            }
        }
    }

    /// Is the peer there, and what can it do for us?
    pub fn ping(&self, peer: &Peer) -> Result<Msg> {
        let mut ch = self.connect(peer)?;
        ch.set_timeout(Some(Duration::from_secs(5)))?;
        ch.send(&Msg::Ping)?;
        match ch.recv()? {
            Some(pong @ Msg::Pong { .. }) => Ok(pong),
            _ => bail!("{} didn't answer", peer.name),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeerState {
    pub name: String,
    pub chat: bool,
    pub ask: bool,
    pub main: bool,
    pub checked: Instant,
}

/// Which paired devices are reachable, refreshed in the background.
pub struct Presence {
    states: Mutex<HashMap<String, PeerState>>,
}

impl Presence {
    /// Start checking every `every`; stops once the returned handle is dropped.
    pub fn start(me: Identity, every: Duration) -> Arc<Presence> {
        let presence = Arc::new(Presence { states: Mutex::new(HashMap::new()) });
        let weak: Weak<Presence> = Arc::downgrade(&presence);
        let client = Client::new(me);
        std::thread::Builder::new()
            .name("link-presence".into())
            .spawn(move || {
                while let Some(p) = weak.upgrade() {
                    p.refresh(&client);
                    drop(p);
                    // Sleep in short steps so a dropped handle ends the thread promptly.
                    let until = Instant::now() + every;
                    while Instant::now() < until && weak.strong_count() > 0 {
                        std::thread::sleep(Duration::from_millis(500));
                    }
                }
            })
            .ok();
        presence
    }

    /// Check every paired device now.
    pub fn refresh(&self, client: &Client) {
        let peers = Peers::load();
        let mut next = HashMap::new();
        for peer in &peers.peers {
            if let Ok(Msg::Pong { name, chat, ask, main }) = client.ping(peer) {
                next.insert(peer.public.clone(), PeerState { name, chat, ask, main, checked: Instant::now() });
            }
        }
        *self.states.lock().unwrap_or_else(|e| e.into_inner()) = next;
    }

    /// Present only if the peer answered the last check.
    pub fn state(&self, peer: &Peer) -> Option<PeerState> {
        self.states.lock().unwrap_or_else(|e| e.into_inner()).get(&peer.public).cloned()
    }

    pub fn online(&self) -> Vec<String> {
        self.states.lock().unwrap_or_else(|e| e.into_inner()).values().map(|s| s.name.clone()).collect()
    }
}
