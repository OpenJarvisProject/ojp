//! The encrypted channel between two paired devices: Noise XX with each
//! device's own key (both sides learn and check the other's key), then
//! length-prefixed JSON messages in ChaCha20-Poly1305 frames.
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const PARAMS_XX: &str = "Noise_XX_25519_ChaChaPoly_BLAKE2s";
pub const PARAMS_PAIR: &str = "Noise_NNpsk0_25519_ChaChaPoly_BLAKE2s";
/// The most one message may hold (a memory sync is the biggest thing sent).
const MAX_MESSAGE: usize = 8 << 20;
const CHUNK: usize = 60_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Msg {
    Ping,
    /// What this device can do for the asker.
    Pong { name: String, chat: bool, ask: bool, main: bool },
    /// Answer these (role, text) messages with your model.
    Chat { messages: Vec<(String, String)> },
    /// Do what this request says, with your own assistant.
    Ask { text: String },
    /// Part of a reply.
    Tok { text: String },
    /// The other device wants this device's user to approve something before it is done:
    /// answer with `Answer`. Ends the reply like `End`.
    Question { text: String },
    /// The user's yes or no to the last `Question`.
    Answer { yes: bool },
    /// A progress note ("tool: remind(...)").
    Note { text: String },
    /// The reply is complete.
    End { text: String },
    Err { msg: String },
    Notify { title: String, text: String },
    Ok,
    /// Sent by both sides while pairing: who I am and where my service listens.
    Pair {
        name: String,
        kind: String,
        public: String,
        port: u16,
        /// Every address this device can be reached at (LAN, VPN), so a later connection can use whichever works.
        #[serde(default)]
        ips: Vec<String>,
    },
    /// Merge these memories and send yours back.
    Sync { state: crate::memory::SyncState },
    SyncReply { state: crate::memory::SyncState },
}

pub fn write_frame(s: &mut impl Write, data: &[u8]) -> std::io::Result<()> {
    s.write_all(&(data.len() as u16).to_be_bytes())?;
    s.write_all(data)?;
    s.flush()
}

pub fn read_frame(s: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 2];
    s.read_exact(&mut len)?;
    let mut buf = vec![0u8; u16::from_be_bytes(len) as usize];
    s.read_exact(&mut buf)?;
    Ok(buf)
}

pub struct Channel {
    stream: TcpStream,
    noise: snow::TransportState,
    /// The other device's public key, proven by the handshake.
    pub remote: Vec<u8>,
}

impl Channel {
    /// The connecting side.
    pub fn initiate(mut stream: TcpStream, private: &[u8]) -> Result<Channel> {
        stream.set_read_timeout(Some(Duration::from_secs(8)))?;
        stream.set_write_timeout(Some(Duration::from_secs(8)))?;
        let mut hs = snow::Builder::new(PARAMS_XX.parse()?).local_private_key(private)?.build_initiator()?;
        let mut buf = vec![0u8; 65535];
        let n = hs.write_message(&[], &mut buf)?;
        write_frame(&mut stream, &buf[..n])?;
        hs.read_message(&read_frame(&mut stream)?, &mut buf).context("the other device didn't answer correctly")?;
        let n = hs.write_message(&[], &mut buf)?;
        write_frame(&mut stream, &buf[..n])?;
        Self::finish(stream, hs)
    }

    /// The accepting side.
    pub fn respond(mut stream: TcpStream, private: &[u8]) -> Result<Channel> {
        stream.set_read_timeout(Some(Duration::from_secs(8)))?;
        stream.set_write_timeout(Some(Duration::from_secs(8)))?;
        let mut hs = snow::Builder::new(PARAMS_XX.parse()?).local_private_key(private)?.build_responder()?;
        let mut buf = vec![0u8; 65535];
        hs.read_message(&read_frame(&mut stream)?, &mut buf).context("bad handshake")?;
        let n = hs.write_message(&[], &mut buf)?;
        write_frame(&mut stream, &buf[..n])?;
        hs.read_message(&read_frame(&mut stream)?, &mut buf).context("bad handshake")?;
        Self::finish(stream, hs)
    }

    fn finish(stream: TcpStream, hs: snow::HandshakeState) -> Result<Channel> {
        let remote = hs.get_remote_static().context("no key from the other device")?.to_vec();
        let noise = hs.into_transport_mode()?;
        Ok(Channel { stream, noise, remote })
    }

    /// Wrap an already-keyed pairing session (see `pair`).
    pub(crate) fn from_transport(stream: TcpStream, noise: snow::TransportState) -> Channel {
        Channel { stream, noise, remote: Vec::new() }
    }

    pub fn set_timeout(&self, t: Option<Duration>) -> Result<()> {
        self.stream.set_read_timeout(t)?;
        Ok(())
    }

    pub fn peer_addr(&self) -> Option<std::net::SocketAddr> {
        self.stream.peer_addr().ok()
    }

    pub fn send(&mut self, msg: &Msg) -> Result<()> {
        let body = serde_json::to_vec(msg)?;
        if body.len() > MAX_MESSAGE {
            bail!("message too large");
        }
        let mut buf = vec![0u8; CHUNK + 64];
        let mut chunks = body.chunks(CHUNK).peekable();
        loop {
            let chunk = chunks.next().unwrap_or(&[]);
            let more = chunks.peek().is_some();
            let mut plain = Vec::with_capacity(chunk.len() + 1);
            plain.push(u8::from(more));
            plain.extend_from_slice(chunk);
            let n = self.noise.write_message(&plain, &mut buf)?;
            write_frame(&mut self.stream, &buf[..n])?;
            if !more {
                return Ok(());
            }
        }
    }

    /// The next message; `None` when the other side closed the connection.
    pub fn recv(&mut self) -> Result<Option<Msg>> {
        let mut body = Vec::new();
        let mut buf = vec![0u8; 65535];
        loop {
            let frame = match read_frame(&mut self.stream) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && body.is_empty() => return Ok(None),
                Err(e) => return Err(e.into()),
            };
            let n = self.noise.read_message(&frame, &mut buf).context("a message failed to decrypt")?;
            let (flag, chunk) = buf[..n].split_first().context("empty frame")?;
            body.extend_from_slice(chunk);
            if body.len() > MAX_MESSAGE {
                bail!("message too large");
            }
            if *flag == 0 {
                return Ok(Some(serde_json::from_slice(&body).context("unreadable message")?));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn two_devices_handshake_and_exchange_big_messages() {
        let a = snow::Builder::new(PARAMS_XX.parse().unwrap()).generate_keypair().unwrap();
        let b = snow::Builder::new(PARAMS_XX.parse().unwrap()).generate_keypair().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let bp = b.private.clone();
        let server = std::thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            let mut ch = Channel::respond(s, &bp).unwrap();
            let Some(Msg::Chat { messages }) = ch.recv().unwrap() else { panic!("expected chat") };
            ch.send(&Msg::End { text: messages[0].1.chars().rev().collect() }).unwrap();
            (ch.remote.clone(), ch.recv().unwrap())
        });
        let mut ch = Channel::initiate(TcpStream::connect(addr).unwrap(), &a.private).unwrap();
        assert_eq!(ch.remote, b.public, "the connecting side learned the server's key");
        let big = "abc".repeat(100_000);
        ch.send(&Msg::Chat { messages: vec![("user".into(), big.clone())] }).unwrap();
        let Some(Msg::End { text }) = ch.recv().unwrap() else { panic!("expected end") };
        assert_eq!(text, big.chars().rev().collect::<String>());
        drop(ch);
        let (seen, last) = server.join().unwrap();
        assert_eq!(seen, a.public, "the server learned the client's key");
        assert!(last.is_none(), "closing ends the conversation cleanly");
    }
}
