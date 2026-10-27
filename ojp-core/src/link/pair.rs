//! Pairing two devices by hand. One shows a one-time 8-digit code, the other
//! types it in. The code is a password for SPAKE2 (a key exchange that leaks
//! nothing a listener could test guesses against), so someone on the same
//! network can try only one guess per code, and the host gives up after three
//! failed connections. The key that comes out protects the exchange of each
//! device's own public key; from then on each recognises the other by that key.
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use spake2::{Ed25519Group, Identity as SpakeId, Password, Spake2};

use super::wire::{Channel, Msg, PARAMS_PAIR, read_frame, write_frame};
use super::{Identity, Peer, Peers, Perms, hex};

const PAIR_SERVICE: &str = "_ojppair._tcp.local.";
const MAX_FAILED: usize = 3;

pub struct Offer {
    pub code: String,
    pub port: u16,
}

fn new_code() -> String {
    let mut b = [0u8; 8];
    getrandom::fill(&mut b).expect("system randomness");
    let n = u64::from_le_bytes(b) % 100_000_000;
    format!("{:04}-{:04}", n / 10_000, n % 10_000)
}

fn digits(code: &str) -> String {
    code.chars().filter(char::is_ascii_digit).collect()
}

/// Run SPAKE2 on the code and wrap the stream in a Noise session keyed by the result.
fn secure(mut stream: TcpStream, code: &str, initiator: bool) -> Result<Channel> {
    stream.set_read_timeout(Some(Duration::from_secs(8)))?;
    stream.set_write_timeout(Some(Duration::from_secs(8)))?;
    let (spake, out) = Spake2::<Ed25519Group>::start_symmetric(&Password::new(digits(code).as_bytes()), &SpakeId::new(b"ojp-pairing-v1"));
    write_frame(&mut stream, &out)?;
    let theirs = read_frame(&mut stream)?;
    let key = spake.finish(&theirs).map_err(|e| anyhow::anyhow!("pairing key exchange failed: {e:?}"))?;
    let psk: [u8; 32] = Sha256::digest(&key).into();
    let builder = snow::Builder::new(PARAMS_PAIR.parse()?).psk(0, &psk)?;
    let mut buf = vec![0u8; 1024];
    let noise = if initiator {
        let mut hs = builder.build_initiator()?;
        let n = hs.write_message(&[], &mut buf)?;
        write_frame(&mut stream, &buf[..n])?;
        hs.read_message(&read_frame(&mut stream)?, &mut buf).map_err(|_| anyhow::anyhow!("wrong code"))?;
        hs.into_transport_mode()?
    } else {
        let mut hs = builder.build_responder()?;
        hs.read_message(&read_frame(&mut stream)?, &mut buf).map_err(|_| anyhow::anyhow!("wrong code"))?;
        let n = hs.write_message(&[], &mut buf)?;
        write_frame(&mut stream, &buf[..n])?;
        hs.into_transport_mode()?
    };
    Ok(Channel::from_transport(stream, noise))
}

/// Swap identities over the secured session and trust the other device.
fn exchange(mut ch: Channel, me: &Identity, my_port: u16) -> Result<Peer> {
    ch.send(&Msg::Pair { name: me.name.clone(), kind: me.kind.clone(), public: hex(&me.public_key()), port: my_port })?;
    let Some(Msg::Pair { name, kind, public, port }) = ch.recv()? else { bail!("the other device didn't introduce itself") };
    if public == hex(&me.public_key()) {
        bail!("that is this device");
    }
    if super::unhex(&public).is_none_or(|k| k.len() != 32) {
        bail!("the other device sent a broken key");
    }
    super::ensure_name(&name).context("the other device has an unusable name")?;
    let addrs = match (ch.peer_addr(), port) {
        (Some(a), p) if p != 0 => vec![SocketAddr::new(a.ip(), p).to_string()],
        _ => Vec::new(),
    };
    let peer = Peer { name, kind, public, addrs, perms: Perms::default() };
    let mut peers = Peers::load();
    peers.upsert(peer.clone());
    peers.save()?;
    Ok(peers.by_key(&super::unhex(&peer.public).unwrap_or_default()).cloned().unwrap_or(peer))
}

/// Show a code and wait for another device to enter it. `announce` is called
/// once the code and port are ready (print or display them there).
pub fn host(me: &Identity, my_port: u16, timeout: Duration, announce: &mut dyn FnMut(&Offer)) -> Result<Peer> {
    let listener = TcpListener::bind("0.0.0.0:0")?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();
    let code = new_code();
    let _ad = super::net::advertise(PAIR_SERVICE, &me.fingerprint(), port, &[("name", me.name.as_str())]);
    announce(&Offer { code: code.clone(), port });
    let deadline = Instant::now() + timeout;
    let mut failed = 0;
    while Instant::now() < deadline && failed < MAX_FAILED {
        let stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        stream.set_nonblocking(false)?;
        match secure(stream, &code, false).and_then(|ch| exchange(ch, me, my_port)) {
            Ok(peer) => return Ok(peer),
            Err(_) => failed += 1,
        }
    }
    if failed >= MAX_FAILED {
        bail!("too many wrong tries; start pairing again for a new code")
    }
    bail!("nobody entered the code in time")
}

/// Enter the code shown on the other device. Finds it on the network unless `addr` is given.
pub fn join(me: &Identity, code: &str, my_port: u16, addr: Option<SocketAddr>) -> Result<Peer> {
    if digits(code).len() != 8 {
        bail!("the code is 8 digits, like 1234-5678");
    }
    let addr = match addr {
        Some(a) => a,
        None => {
            let found = super::net::browse(PAIR_SERVICE, Duration::from_secs(4));
            match found.as_slice() {
                [] => bail!("didn't find a device waiting to pair on this network (is pairing open on the other one? you can also give its address)"),
                [one] => one.addr,
                many => bail!(
                    "several devices are waiting to pair ({}); give the address of the right one",
                    many.iter().map(|f| format!("{} at {}", f.name, f.addr)).collect::<Vec<_>>().join(", ")
                ),
            }
        }
    };
    let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5)).with_context(|| format!("connecting to {addr}"))?;
    let ch = secure(stream, code, true)?;
    exchange(ch, me, my_port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_code_pairs_wrong_code_does_not() {
        let _g = crate::TEST_ENV.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("ojp-pair-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Both devices share this process, so they share one peers file; that is enough to check the exchange.
        unsafe { std::env::set_var("OJP_LINK_DIR", &dir) };
        let a = Identity::generate("alpha", "desktop").unwrap();
        let b = Identity::generate("beta", "phone").unwrap();
        let offer = std::sync::Arc::new(std::sync::Mutex::new(None::<(String, u16)>));
        let seen = offer.clone();
        let host_thread = std::thread::spawn(move || host(&a, 1111, Duration::from_secs(20), &mut |o| *seen.lock().unwrap() = Some((o.code.clone(), o.port))));
        let (code, port) = loop {
            if let Some(o) = offer.lock().unwrap().take() {
                break o;
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let wrong = if code == "0000-0000" { "1111-1111" } else { "0000-0000" };
        assert!(join(&b, wrong, 2222, Some(addr)).is_err(), "a wrong code must not pair");
        let joined = join(&b, &code, 2222, Some(addr)).unwrap();
        let hosted = host_thread.join().unwrap().unwrap();
        assert_eq!((joined.name.as_str(), joined.kind.as_str()), ("alpha", "desktop"));
        assert_eq!((hosted.name.as_str(), hosted.kind.as_str()), ("beta", "phone"));
        assert_eq!(joined.addrs, vec!["127.0.0.1:1111".to_string()]);
        assert_eq!(hosted.addrs, vec!["127.0.0.1:2222".to_string()]);
        std::fs::remove_dir_all(dir).ok();
    }
}
