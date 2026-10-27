//! Finding paired devices on the network and talking to them: a server that
//! accepts only paired devices, a client, discovery (mDNS, so addresses can
//! change), and a background check of which peers are reachable.
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

use super::wire::{Channel, Msg};
use super::{AskLevel, Identity, Peer, Peers};
#[cfg(test)]
use super::hex;

const SERVICE: &str = "_ojp._tcp.local.";
const MAX_CONNECTIONS: usize = 8;
/// Devices listen here when they can, so a peer's stored address stays valid across restarts.
pub const DEFAULT_PORT: u16 = 47600;
/// A reply may take this long to start or continue (a big model on a slow CPU).
const REPLY_TIMEOUT: Duration = Duration::from_secs(300);

/// Pairing waits on a port from here, so another device can find it by looking for that port.
pub const PAIR_PORT: u16 = 47620;

fn real_interface(name: &str) -> bool {
    !(name == "lo" || ["docker", "br-", "veth", "virbr", "lxc", "vmnet", "podman"].iter().any(|p| name.starts_with(p)))
}

/// This device's own IPv4 addresses worth sharing with a peer: LAN and VPN (Tailscale, Netbird),
/// not loopback, link-local or container bridges.
pub fn local_ips() -> Vec<Ipv4Addr> {
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .filter(|i| real_interface(&i.name) && !i.is_loopback())
        .filter_map(|i| match i.addr {
            if_addrs::IfAddr::V4(a) if !a.ip.is_link_local() => Some(a.ip),
            _ => None,
        })
        .collect()
}

/// Devices on the local network that accept connections on one of `ports`. This works where
/// multicast discovery doesn't (another program holding the mDNS port, a locked-down Wi-Fi).
/// It only looks at this device's own /24 on each real network interface, and only at the given ports.
pub(crate) fn scan(ports: &[u16]) -> Vec<SocketAddr> {
    let mut targets: Vec<SocketAddr> = Vec::new();
    for iface in if_addrs::get_if_addrs().unwrap_or_default() {
        if !real_interface(&iface.name) || iface.is_loopback() {
            continue;
        }
        let if_addrs::IfAddr::V4(a) = iface.addr else { continue };
        if a.ip.is_link_local() || a.ip.octets()[0] == 100 {
            // 100.x is a VPN: its peers are reached by stored address, not by scanning.
            continue;
        }
        let (ip, mut mask) = (u32::from(a.ip), u32::from(a.netmask));
        // Big networks are not scanned whole: the 254 neighbours in this device's own /24.
        if mask < 0xFFFF_FF00 {
            mask = 0xFFFF_FF00;
        }
        let net = ip & mask;
        for host in 1..(!mask) {
            let candidate = net | host;
            // (Normally a device doesn't look for itself; OJP_LINK_SCAN_SELF=1 allows it, for testing on one machine.)
            if candidate != ip || std::env::var_os("OJP_LINK_SCAN_SELF").is_some() {
                for &p in ports {
                    let t = SocketAddr::new(IpAddr::V4(Ipv4Addr::from(candidate)), p);
                    if !targets.contains(&t) {
                        targets.push(t);
                    }
                }
            }
        }
    }
    let queue = Arc::new(Mutex::new(targets));
    let found = Arc::new(Mutex::new(Vec::new()));
    let workers: Vec<_> = (0..128)
        .map(|_| {
            let (queue, found) = (queue.clone(), found.clone());
            std::thread::spawn(move || {
                loop {
                    let Some(t) = queue.lock().unwrap_or_else(|e| e.into_inner()).pop() else { break };
                    if TcpStream::connect_timeout(&t, Duration::from_millis(250)).is_ok() {
                        found.lock().unwrap_or_else(|e| e.into_inner()).push(t);
                    }
                }
            })
        })
        .collect();
    for w in workers {
        let _ = w.join();
    }
    let mut out = found.lock().unwrap_or_else(|e| e.into_inner()).clone();
    out.sort();
    out
}

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

/// Paired-device services visible on the network, as (name, address) pairs (for diagnosis).
pub fn discover(timeout: Duration) -> Vec<(String, SocketAddr)> {
    browse(SERVICE, timeout).into_iter().map(|f| (f.name, f.addr)).collect()
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

    /// Try each address at once; the first that answers as the right device wins.
    fn race(&self, peer: &Peer, addrs: &[SocketAddr]) -> Option<(Channel, SocketAddr)> {
        if addrs.is_empty() {
            return None;
        }
        let want = super::unhex(&peer.public).unwrap_or_default();
        let (tx, rx) = mpsc::channel();
        for &addr in addrs {
            let (tx, private, want) = (tx.clone(), self.me.private_key(), want.clone());
            std::thread::spawn(move || {
                let result = TcpStream::connect_timeout(&addr, Duration::from_millis(1200))
                    .map_err(anyhow::Error::from)
                    .and_then(|s| Channel::initiate(s, &private))
                    .and_then(|ch| if ch.remote == want { Ok(ch) } else { bail!("a different device answered") });
                if let Ok(ch) = result {
                    let _ = tx.send((ch, addr));
                }
            });
        }
        drop(tx);
        rx.recv_timeout(Duration::from_millis(2500)).ok()
    }

    fn remember(peer: &Peer, addr: SocketAddr) {
        let mut peers = Peers::load();
        if let Some(p) = peers.peers.iter_mut().find(|p| p.public == peer.public) {
            let a = addr.to_string();
            p.addrs.retain(|x| *x != a);
            p.addrs.insert(0, a);
            p.addrs.truncate(8);
            let _ = peers.save();
        }
    }

    /// Open a channel to a paired device: its known addresses (LAN and VPN) all at once, then a look
    /// around this network for it, then multicast discovery.
    pub fn connect(&self, peer: &Peer) -> Result<Channel> {
        let known: Vec<SocketAddr> = peer.addrs.iter().filter_map(|a| a.to_socket_addrs().ok()?.next()).collect();
        if let Some((ch, _)) = self.race(peer, &known) {
            return Ok(ch);
        }
        let nearby: Vec<SocketAddr> = scan(&[DEFAULT_PORT]).into_iter().filter(|a| !known.contains(a)).collect();
        if let Some((ch, addr)) = self.race(peer, &nearby) {
            Self::remember(peer, addr);
            return Ok(ch);
        }
        let fp = peer.fingerprint();
        let announced: Vec<SocketAddr> = browse(SERVICE, Duration::from_millis(1500)).into_iter().filter(|f| f.instance == fp).map(|f| f.addr).collect();
        if let Some((ch, addr)) = self.race(peer, &announced) {
            Self::remember(peer, addr);
            return Ok(ch);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::Perms;

    #[test]
    fn paired_devices_talk_unpaired_are_refused_and_permissions_hold() {
        let _g = crate::TEST_ENV.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("ojp-net-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("OJP_LINK_DIR", &dir) };
        let (a, b, stranger) = (Identity::generate("a", "desktop").unwrap(), Identity::generate("b", "phone").unwrap(), Identity::generate("c", "phone").unwrap());
        let (tx, rx) = mpsc::channel::<Incoming>();
        let caps = Arc::new(Caps::default());
        caps.chat.store(true, Ordering::Relaxed);
        caps.ask.store(true, Ordering::Relaxed);
        let server = Server::start(a.clone(), caps, tx).unwrap();
        std::thread::spawn(move || {
            for inc in rx {
                match inc.request {
                    Request::Chat(m) => {
                        inc.reply.tok("he");
                        inc.reply.tok("llo");
                        inc.reply.end(&format!("hello {}", m.len()));
                    }
                    _ => inc.reply.err("nope"),
                }
            }
        });
        let addr = format!("127.0.0.1:{}", server.port);
        let peer_a = Peer { name: "a".into(), kind: "desktop".into(), public: hex(&a.public_key()), addrs: vec![addr.clone()], perms: Perms::default() };
        let peer_b = Peer { name: "b".into(), kind: "phone".into(), public: hex(&b.public_key()), addrs: vec![], perms: Perms::default() };
        let mut peers = Peers::default();
        peers.upsert(peer_a.clone());
        peers.upsert(peer_b.clone());
        peers.save().unwrap();

        let chat = || Msg::Chat { messages: vec![("user".into(), "hi".into())] };
        let mut toks = Vec::new();
        let text = Client::new(b.clone()).request(&peer_a, chat(), &mut |m| if let Msg::Tok { text } = m { toks.push(text.clone()) }).unwrap();
        assert_eq!((text.as_str(), toks.join("").as_str()), ("hello 1", "hello"));
        assert!(matches!(Client::new(b.clone()).ping(&peer_a).unwrap(), Msg::Pong { chat: true, ask: true, .. }));

        // A device that was never paired gets nothing, even though it can reach the port.
        assert!(Client::new(stranger).request(&peer_a, chat(), &mut |_| {}).is_err());

        // Pointing at the right address with the wrong expected key is caught by the client too.
        let mut wrong_key = peer_a.clone();
        wrong_key.public = hex(&stranger_key());
        assert!(Client::new(b.clone()).request(&wrong_key, chat(), &mut |_| {}).is_err());

        // Switching a permission off takes effect on the next request.
        peers.peers.iter_mut().find(|p| p.name == "b").unwrap().perms.chat = false;
        peers.save().unwrap();
        let err = Client::new(b).request(&peer_a, chat(), &mut |_| {}).unwrap_err();
        assert!(format!("{err}").contains("doesn't allow"), "{err}");
        std::fs::remove_dir_all(dir).ok();
    }

    fn stranger_key() -> Vec<u8> {
        Identity::generate("x", "phone").unwrap().public_key()
    }
}
