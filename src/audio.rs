//! Turn arriving audio bytes into `Vec<f32>` at 16 kHz mono — the shape
//! sherpa-onnx's Parakeet `OfflineRecognizer` wants (README "stdin",
//! `docs/engine.md`). This module knows nothing about sherpa-onnx; it only
//! produces the sample buffer the recognizer is later handed.
//!
//! Two entry points, both taking `impl Read`, matching the two ways audio
//! reaches auris (README "Flags", "stdin"):
//!
//! - [`decode`] sniffs the container from the stream itself — the stdin/file
//!   path, where auris does not know in advance what it is being handed.
//! - [`decode_raw_pcm`] takes an already-known layout — the
//!   `arecord -f S16_LE -r 16000 | auris` case from README's synopsis, where
//!   there is no container to sniff, only a rate/channel/format triple the
//!   caller supplies.
//!
//! WAV is the only container decoded. Every codec accepted here is a decoder
//! linked into the binary; the frontend that talks to auris produces WAV, so
//! nothing else is needed, and [`decode`] names the common wrong-container
//! magics in its error rather than accepting them.

use std::io::{Cursor, ErrorKind, Read};

/// Bytes read from the source are capped here — reading a stream unbounded
/// into memory is a machine-filling bug, not a hardening nicety. 256 MiB is
/// generous for any single utterance and small next to the ~1.5 GB the
/// loaded recognizer already holds (README "The daemon").
pub const MAX_INPUT_BYTES: u64 = 256 * 1024 * 1024;

/// Decoded audio is separately capped at this many seconds of 16 kHz mono —
/// the byte cap above bounds the *encoded* size, but a long enough recording
/// at a low bitrate can still decode into more samples than any one
/// utterance auris is meant to hold in memory at once.
pub const MAX_DECODED_SECONDS: u32 = 10 * 60;

/// The sample rate every path in this module converges on — what
/// `OfflineRecognizer` was built for (`docs/engine.md`).
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Below this maximum-RMS-over-a-window threshold, [`is_silent`] calls the
/// audio silent. −60 dBFS (1e-3), chosen ~100x below the quietest speech
/// fixture measured in this repo (`tests/fixtures/stereo-44100.wav`, max
/// 30 ms window RMS 1.1377e-1) and far above pure digital zero
/// (`tests/fixtures/silence.wav`, exactly 0.0). The real Parakeet model
/// hallucinates "Okay." on 1 s of digital silence rather than returning
/// nothing (`tests/fixtures.rs::silence_yields_no_transcript`), which broke
/// README "Exit codes"'s exit-1-on-silence contract undetected; this gate
/// exists to catch that before the recognizer is ever reached. It is
/// calibrated to catch "no signal arrived," not "quiet room" — there is no
/// ambient-noise fixture in this repo to calibrate a room-tone threshold
/// against, so don't read a pass here as a claim this handles real-world
/// background noise. This is not VAD: `docs/streaming.md`'s Silero VAD
/// segmentation is future work, and supersedes this gate once built.
const SILENCE_RMS_THRESHOLD: f32 = 1e-3;

/// Window size [`is_silent`] computes the max RMS over, in samples at
/// [`TARGET_SAMPLE_RATE`] — 30 ms, short enough that a brief burst of real
/// speech inside an otherwise-silent buffer isn't averaged away by a
/// longer window.
const SILENCE_WINDOW_SAMPLES: usize = (TARGET_SAMPLE_RATE as usize * 30) / 1000;

/// True when the maximum RMS over any [`SILENCE_WINDOW_SAMPLES`]-sample
/// window of `samples` is below [`SILENCE_RMS_THRESHOLD`] — see that
/// constant's doc comment for the threshold, the margin it was chosen
/// against, and why this is not VAD. `samples` shorter than one window (or
/// empty) is treated as a single window covering the whole slice.
pub fn is_silent(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true;
    }
    let window = SILENCE_WINDOW_SAMPLES.min(samples.len()).max(1);

    // Running sum of squares over a sliding window, O(n): add the entering
    // sample's square, drop the leaving one's, each step.
    let mut sum_sq: f64 = samples[..window]
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum();
    let mut max_sum_sq = sum_sq;
    for i in window..samples.len() {
        let entering = samples[i] as f64;
        let leaving = samples[i - window] as f64;
        sum_sq += entering * entering - leaving * leaving;
        if sum_sq > max_sum_sq {
            max_sum_sq = sum_sq;
        }
    }
    let max_rms = (max_sum_sq / window as f64).sqrt() as f32;
    max_rms < SILENCE_RMS_THRESHOLD
}

/// The range of source sample rates this module will resample from —
/// generously wide (8 kHz telephony through 384 kHz pro audio), but not
/// unbounded. rubato's FFT resampler sizes its buffers by
/// `rate_in / gcd(rate_in, TARGET_SAMPLE_RATE)`, so an unvalidated rate
/// coprime with 16 kHz can allocate gigabytes; nothing may reach the
/// resampler without first passing through this range.
const MIN_SAMPLE_RATE: u32 = 1_000;
const MAX_SAMPLE_RATE: u32 = 384_000;

/// Raw PCM sample layouts [`decode_raw_pcm`] understands. Only what
/// `arecord`-style callers plausibly hand auris: no 24-bit, no big-endian —
/// add a variant when a real caller needs one, not speculatively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawSampleFormat {
    /// Signed 16-bit little-endian PCM.
    S16Le,
    /// 32-bit IEEE float little-endian.
    F32Le,
}

impl RawSampleFormat {
    fn bytes_per_sample(self) -> usize {
        match self {
            RawSampleFormat::S16Le => 2,
            RawSampleFormat::F32Le => 4,
        }
    }
}

/// Every way turning audio into samples can fail. `exit_code` gives the
/// README "Exit codes" mapping directly: NOTHING_TRANSCRIBED (1) for
/// anything about the audio itself being unusable, USAGE (2) only for a
/// caller-supplied parameter that was never valid regardless of the bytes
/// behind it.
#[derive(Debug)]
pub enum AudioError {
    /// The underlying reader failed (a real I/O error, not a size-cap trip).
    Io(std::io::Error),
    /// The stream yielded zero bytes — distinct from silence (a WAV full of
    /// zero samples) and from garbage (bytes that just aren't WAV).
    NoInput,
    /// The first bytes of the stream did not look like WAV. Names the
    /// container it did look like, when recognisable, so the caller isn't
    /// left guessing.
    NotWav(String),
    /// `hound` rejected the stream: malformed, truncated, or an
    /// unsupported WAV variant.
    Wav(hound::Error),
    /// Resampling to 16 kHz failed.
    Resample(String),
    /// The source produced more than [`MAX_INPUT_BYTES`].
    TooLarge,
    /// The decoded audio is longer than [`MAX_DECODED_SECONDS`].
    TooLong,
    /// Decoding produced zero samples — a run with no transcript exits 1,
    /// so this counts as failure, not an empty success.
    Empty,
    /// Raw PCM bytes did not divide evenly into whole frames.
    TruncatedFrame,
    /// Caller asked for zero channels.
    InvalidChannels,
    /// Caller asked for a sample rate outside `MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE`.
    InvalidRate,
    /// A WAV header declared a sample rate outside
    /// `MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE` — a property of the bytes, not
    /// of the caller, so unlike `InvalidRate` this is not a usage error.
    UnsupportedRate(u32),
}

impl AudioError {
    /// Maps to the README "Exit codes" contract: 1 (NOTHING_TRANSCRIBED)
    /// for unusable audio, 2 (USAGE) only for a parameter that was invalid
    /// on its own terms, independent of what bytes followed it.
    pub fn exit_code(&self) -> i32 {
        match self {
            AudioError::InvalidChannels | AudioError::InvalidRate => 2,
            _ => 1,
        }
    }
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::Io(e) => write!(f, "failed to read audio: {e}"),
            AudioError::NoInput => write!(f, "input was empty; expected a wav stream"),
            AudioError::NotWav(detected) => write!(
                f,
                "not a wav file ({detected}); expected a wav stream (RIFF/WAVE)"
            ),
            AudioError::Wav(e) => write!(f, "failed to decode wav: {e}"),
            AudioError::Resample(msg) => write!(f, "failed to resample audio: {msg}"),
            AudioError::TooLarge => write!(
                f,
                "audio exceeds the {} MiB input size cap",
                MAX_INPUT_BYTES / 1024 / 1024
            ),
            AudioError::TooLong => write!(
                f,
                "decoded audio exceeds the {}-minute duration cap",
                MAX_DECODED_SECONDS / 60
            ),
            AudioError::Empty => write!(f, "no audio samples decoded"),
            AudioError::TruncatedFrame => {
                write!(f, "raw pcm data length is not a whole number of frames")
            }
            AudioError::InvalidChannels => write!(f, "channel count must be nonzero"),
            AudioError::InvalidRate => write!(
                f,
                "sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} hz"
            ),
            AudioError::UnsupportedRate(rate) => write!(
                f,
                "wav declares an unsupported sample rate ({rate} hz, must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE})"
            ),
        }
    }
}

impl std::error::Error for AudioError {}

/// A `Read` wrapper that fails once more than `cap` bytes have come through
/// it, instead of letting the caller (hound, or a raw `read_to_end`) pull an
/// unbounded amount of data into memory. The failure is tagged with
/// `ErrorKind::OutOfMemory` so callers can distinguish "source is too big"
/// from an ordinary I/O failure without string-matching an error message.
struct LimitedReader<R> {
    inner: R,
    remaining: u64,
}

impl<R: Read> LimitedReader<R> {
    fn new(inner: R, cap: u64) -> Self {
        LimitedReader {
            inner,
            remaining: cap,
        }
    }
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 {
            // Already at the cap: read one probe byte to tell "the source
            // ends exactly here" (fine) from "there is more" (too large),
            // without pulling in a full extra buffer's worth of data.
            let mut probe = [0u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0),
                _ => Err(std::io::Error::new(
                    ErrorKind::OutOfMemory,
                    "input exceeds size cap",
                )),
            };
        }
        let n = self.inner.read(buf)?;
        self.remaining = self.remaining.saturating_sub(n as u64);
        Ok(n)
    }
}

fn is_size_cap_error(e: &std::io::Error) -> bool {
    e.kind() == ErrorKind::OutOfMemory
}

fn map_hound_error(e: hound::Error) -> AudioError {
    if let hound::Error::IoError(io_err) = &e
        && is_size_cap_error(io_err)
    {
        return AudioError::TooLarge;
    }
    AudioError::Wav(e)
}

fn map_io_error(e: std::io::Error) -> AudioError {
    if is_size_cap_error(&e) {
        AudioError::TooLarge
    } else {
        AudioError::Io(e)
    }
}

/// Averages interleaved multi-channel samples down to mono. A no-op (cheap
/// clone) when `channels == 1`, which is the common case — most audio
/// handed to auris is already mono.
fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let channels = channels as usize;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Rejects audio that is empty or longer than [`MAX_DECODED_SECONDS`],
/// checked against the *native* sample rate before resampling so a long
/// low-rate file is rejected without first paying for a resample of it.
fn check_duration(mono: &[f32], sample_rate: u32) -> Result<(), AudioError> {
    if mono.is_empty() {
        return Err(AudioError::Empty);
    }
    let seconds = mono.len() as f64 / sample_rate as f64;
    if seconds > MAX_DECODED_SECONDS as f64 {
        return Err(AudioError::TooLong);
    }
    Ok(())
}

/// Resamples mono `f32` audio to [`TARGET_SAMPLE_RATE`]. Skips the
/// resampler entirely when `input_rate` already matches — the no-op path
/// design decision 3 requires, since a source already at 16 kHz mono must
/// not pay for a resample it doesn't need.
fn resample_to_target(input: &[f32], input_rate: u32) -> Result<Vec<f32>, AudioError> {
    if input_rate == TARGET_SAMPLE_RATE || input.is_empty() {
        return Ok(input.to_vec());
    }

    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};

    const CHANNELS: usize = 1;
    const CHUNK_SIZE: usize = 1024;

    let mut resampler = Fft::<f32>::new(
        input_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        CHUNK_SIZE,
        CHANNELS,
        FixedSync::Both,
    )
    .map_err(|e| AudioError::Resample(e.to_string()))?;

    let input_adapter = InterleavedSlice::new(input, CHANNELS, input.len())
        .map_err(|e| AudioError::Resample(e.to_string()))?;

    let output = resampler
        .process_all(&input_adapter, input.len(), None)
        .map_err(|e| AudioError::Resample(e.to_string()))?;

    Ok(output.take_data())
}

/// Decodes a WAV stream (already confirmed to start `RIFF`) into mono `f32`
/// samples at [`TARGET_SAMPLE_RATE`]. Supports 16-bit PCM and 32-bit IEEE
/// float, mono or multi-channel, at any source sample rate (design
/// decision 1) — nothing else, since those are the only WAV variants a
/// frontend producing WAV for auris needs.
fn decode_wav(reader: impl Read) -> Result<Vec<f32>, AudioError> {
    let mut wav = hound::WavReader::new(reader).map_err(map_hound_error)?;
    let spec = wav.spec();
    if spec.channels == 0 {
        return Err(AudioError::Wav(hound::Error::FormatError(
            "wav header declares zero channels",
        )));
    }
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&spec.sample_rate) {
        return Err(AudioError::UnsupportedRate(spec.sample_rate));
    }

    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => wav
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0).map_err(map_hound_error))
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => wav
            .samples::<f32>()
            .map(|s| s.map_err(map_hound_error))
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(AudioError::Wav(hound::Error::Unsupported));
        }
    };

    let mono = downmix_to_mono(&interleaved, spec.channels);
    check_duration(&mono, spec.sample_rate)?;
    resample_to_target(&mono, spec.sample_rate)
}

/// Recognises a wrong-container magic in `head` and names it, or falls back
/// to a generic "not a wav" message. `head` is whatever could be read
/// before EOF, which may be shorter than the magics being compared.
fn describe_non_wav(head: &[u8]) -> String {
    const EBML: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
    if head.starts_with(&EBML) {
        return "looks like webm/matroska".to_string();
    }
    if head.starts_with(b"OggS") {
        return "looks like ogg".to_string();
    }
    if head.starts_with(b"fLaC") {
        return "looks like flac".to_string();
    }
    if head.starts_with(b"ID3") || (head.len() >= 2 && head[0] == 0xFF && (head[1] & 0xE0) == 0xE0)
    {
        return "looks like mp3".to_string();
    }
    if head.len() >= 8 && &head[4..8] == b"ftyp" {
        return "looks like mp4/m4a".to_string();
    }
    "unrecognised header".to_string()
}

/// Reads up to `buf.len()` bytes, stopping early only at EOF. Used to peek
/// the container-sniffing magic without assuming the reader hands back a
/// full buffer in one call.
fn fill_as_much_as_possible(reader: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// Decodes audio of unknown container from `reader`, sniffing the first few
/// bytes to tell WAV from everything else. This is the stdin/file path
/// (README "stdin"): auris does not know in advance what it has been
/// handed, so it looks.
pub fn decode(reader: impl Read) -> Result<Vec<f32>, AudioError> {
    let mut limited = LimitedReader::new(reader, MAX_INPUT_BYTES);
    let mut magic = [0u8; 8];
    let n = fill_as_much_as_possible(&mut limited, &mut magic).map_err(map_io_error)?;

    if n == 0 {
        return Err(AudioError::NoInput);
    }
    if n >= 4 && &magic[0..4] == b"RIFF" {
        let chained = Cursor::new(magic[..n].to_vec()).chain(limited);
        decode_wav(chained)
    } else {
        Err(AudioError::NotWav(describe_non_wav(&magic[..n])))
    }
}

/// Decodes raw PCM of an already-known layout — no container, no sniffing.
/// This is the `arecord -f S16_LE -r 16000 | auris` case from README's
/// synopsis, where the caller supplies `--rate`/`--channels` because there
/// is nothing in the stream itself to read them from.
pub fn decode_raw_pcm(
    reader: impl Read,
    sample_rate: u32,
    channels: u16,
    format: RawSampleFormat,
) -> Result<Vec<f32>, AudioError> {
    if channels == 0 {
        return Err(AudioError::InvalidChannels);
    }
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(AudioError::InvalidRate);
    }

    let mut limited = LimitedReader::new(reader, MAX_INPUT_BYTES);
    let mut bytes = Vec::new();
    limited.read_to_end(&mut bytes).map_err(map_io_error)?;

    let bytes_per_sample = format.bytes_per_sample();
    let frame_bytes = bytes_per_sample * channels as usize;
    if bytes.len() % frame_bytes != 0 {
        return Err(AudioError::TruncatedFrame);
    }

    let interleaved: Vec<f32> = match format {
        RawSampleFormat::S16Le => bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        RawSampleFormat::F32Le => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    };

    let mono = downmix_to_mono(&interleaved, channels);
    check_duration(&mono, sample_rate)?;
    resample_to_target(&mono, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    /// Builds an in-memory WAV file with `hound`, matching the fixture the
    /// tests need — no fixture files on disk.
    fn write_wav_i16(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    fn write_wav_f32(sample_rate: u32, channels: u16, samples: &[f32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for &s in samples {
                writer.write_sample(s).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn mono_16k_16bit_is_a_no_op() {
        let samples: Vec<i16> = vec![0, 1000, -1000, 32767, -32768, 5];
        let wav = write_wav_i16(16_000, 1, &samples);
        let out = decode(Cursor::new(wav)).unwrap();
        assert_eq!(out.len(), samples.len());
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1000.0 / 32768.0).abs() < 1e-6);
        assert!((out[3] - 32767.0 / 32768.0).abs() < 1e-6);
        assert!((out[4] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn stereo_44100_resamples_and_downmixes() {
        let in_rate = 44_100;
        let n = 4410; // 0.1s
        let mut samples = Vec::with_capacity(n * 2);
        for i in 0..n {
            let t = i as f32 / in_rate as f32;
            let v = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let s = (v * 16000.0) as i16;
            samples.push(s);
            samples.push(s / 2); // distinct but non-cancelling right channel
        }
        let wav = write_wav_i16(in_rate as u32, 2, &samples);
        let out = decode(Cursor::new(wav)).unwrap();

        let expected_len = n * TARGET_SAMPLE_RATE as usize / in_rate;
        let tolerance = expected_len / 10 + 8;
        assert!(
            (out.len() as i64 - expected_len as i64).unsigned_abs() as usize <= tolerance,
            "out.len()={} expected~={}",
            out.len(),
            expected_len
        );

        assert!(out.iter().any(|&v| v.abs() > 0.01), "output looks silent");
        assert!(out.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn float32_round_trips() {
        let samples = vec![0.0f32, 0.5, -0.5, 0.999, -1.0];
        let wav = write_wav_f32(16_000, 1, &samples);
        let out = decode(Cursor::new(wav)).unwrap();
        assert_eq!(out.len(), samples.len());
        for (a, b) in out.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn truncated_wav_errors_not_panics() {
        let wav = write_wav_i16(16_000, 1, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let cut = &wav[..wav.len() - 4]; // chop off part of the data chunk
        let result = decode(Cursor::new(cut.to_vec()));
        assert!(result.is_err());
    }

    #[test]
    fn empty_input_is_no_input_not_not_wav() {
        let err = decode(Cursor::new(Vec::<u8>::new())).unwrap_err();
        assert!(matches!(err, AudioError::NoInput));
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn plain_text_is_rejected() {
        let err = decode(Cursor::new(b"this is not audio at all".to_vec())).unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert!(matches!(err, AudioError::NotWav(_)));
    }

    #[test]
    fn webm_magic_is_named_in_the_error() {
        let mut blob = vec![0x1A, 0x45, 0xDF, 0xA3];
        blob.extend_from_slice(&[0u8; 16]);
        let err = decode(Cursor::new(blob)).unwrap_err();
        match err {
            AudioError::NotWav(msg) => assert!(msg.contains("webm")),
            other => panic!("expected NotWav, got {other:?}"),
        }
    }

    #[test]
    fn raw_pcm_s16le_stereo_round_trips() {
        let mut bytes = Vec::new();
        for &(l, r) in &[(0i16, 0i16), (1000, -1000), (-32768, 32767)] {
            bytes.write_all(&l.to_le_bytes()).unwrap();
            bytes.write_all(&r.to_le_bytes()).unwrap();
        }
        let out = decode_raw_pcm(Cursor::new(bytes), 16_000, 2, RawSampleFormat::S16Le).unwrap();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn raw_pcm_odd_trailing_byte_errors() {
        let bytes = vec![0u8, 1, 2]; // 3 bytes: not a whole number of s16 frames
        let err =
            decode_raw_pcm(Cursor::new(bytes), 16_000, 1, RawSampleFormat::S16Le).unwrap_err();
        assert!(matches!(err, AudioError::TruncatedFrame));
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn raw_pcm_zero_channels_is_a_usage_error() {
        let err = decode_raw_pcm(
            Cursor::new(Vec::<u8>::new()),
            16_000,
            0,
            RawSampleFormat::S16Le,
        )
        .unwrap_err();
        assert!(matches!(err, AudioError::InvalidChannels));
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn raw_pcm_zero_rate_is_a_usage_error() {
        let err = decode_raw_pcm(Cursor::new(Vec::<u8>::new()), 0, 1, RawSampleFormat::S16Le)
            .unwrap_err();
        assert!(matches!(err, AudioError::InvalidRate));
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn raw_pcm_absurd_rate_is_a_usage_error() {
        // Rejected before any resampler is built, so this must return fast —
        // if it ever hangs, the bounds check moved to the wrong place.
        let bytes = vec![0u8, 0, 1, 0, 2, 0, 3, 0]; // four s16 samples
        let err = decode_raw_pcm(
            Cursor::new(bytes),
            4_294_967_291, // coprime with 16_000; would blow up rubato's FFT sizing unchecked
            1,
            RawSampleFormat::S16Le,
        )
        .unwrap_err();
        assert!(matches!(err, AudioError::InvalidRate));
        assert_eq!(err.exit_code(), 2);
    }

    /// Hand-assembles a minimal PCM WAV with an arbitrary `sample_rate`
    /// field. `hound::WavWriter` itself panics building a header for a zero
    /// or absurd rate (`byte_rate` divide-by-zero / overflow in its own
    /// code), so a hostile header has to be built by hand to test that
    /// auris rejects one on read.
    fn hand_crafted_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let bits_per_sample: u16 = 16;
        let block_align = channels as u32 * (bits_per_sample as u32 / 8);
        let byte_rate = sample_rate.wrapping_mul(block_align);
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36u32 + data.len() as u32).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&(block_align as u16).to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&data);
        buf
    }

    #[test]
    fn wav_zero_rate_is_rejected() {
        let wav = hand_crafted_wav(0, 1, &[1, 2, 3, 4]);
        let err = decode(Cursor::new(wav)).unwrap_err();
        assert!(matches!(err, AudioError::UnsupportedRate(0)));
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn wav_absurd_rate_is_rejected() {
        // Far above MAX_SAMPLE_RATE but small enough that byte_rate stays
        // internally consistent, so hound's own fmt-chunk sanity check
        // doesn't intercept this before it reaches auris's rate validation.
        let wav = hand_crafted_wav(1_000_000, 1, &[1, 2, 3, 4]);
        let err = decode(Cursor::new(wav)).unwrap_err();
        assert!(matches!(err, AudioError::UnsupportedRate(_)));
        assert_eq!(err.exit_code(), 1);
    }

    /// A `Read` impl that yields endless zero bytes — used to exercise the
    /// size cap without actually allocating hundreds of megabytes.
    struct EndlessZeros;
    impl Read for EndlessZeros {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            buf.fill(0);
            Ok(buf.len())
        }
    }

    #[test]
    fn oversized_input_is_rejected() {
        let err = decode_raw_pcm(EndlessZeros, 16_000, 1, RawSampleFormat::S16Le).unwrap_err();
        assert!(matches!(err, AudioError::TooLarge));
        assert_eq!(err.exit_code(), 1);
    }

    /// Builds one second of a sine wave at `amplitude`, at `TARGET_SAMPLE_RATE`.
    fn sine_at(amplitude: f32) -> Vec<f32> {
        (0..TARGET_SAMPLE_RATE)
            .map(|i| {
                let t = i as f32 / TARGET_SAMPLE_RATE as f32;
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn all_zero_samples_are_silent() {
        assert!(is_silent(&vec![0.0f32; TARGET_SAMPLE_RATE as usize]));
    }

    #[test]
    fn ordinary_speech_level_is_not_silent() {
        assert!(!is_silent(&sine_at(0.1)));
    }

    #[test]
    fn very_low_level_signal_is_silent() {
        assert!(is_silent(&sine_at(1e-4)));
    }

    /// Pins that the gate is not over-aggressive: −45 dBFS is well under
    /// real quiet speech (the quietest fixture measures 1.1377e-1) but a
    /// sine of that amplitude still has an RMS of ~3.5e-3, ~3.5x above the
    /// 1e-3 threshold, so it must not trip the gate.
    #[test]
    fn low_but_real_signal_is_not_silent() {
        assert!(!is_silent(&sine_at(5e-3)));
    }

    /// The whole reason for a windowed max rather than an overall RMS: a
    /// short burst of real amplitude surrounded by silence must not be
    /// averaged away by the quiet samples around it.
    #[test]
    fn a_short_burst_amid_silence_is_not_silent() {
        let mut samples = vec![0.0f32; TARGET_SAMPLE_RATE as usize];
        let burst = sine_at(0.1);
        let start = samples.len() / 2;
        samples[start..start + burst.len().min(1600)]
            .copy_from_slice(&burst[..burst.len().min(1600)]);
        assert!(!is_silent(&samples));
    }

    #[test]
    fn empty_slice_is_silent() {
        assert!(is_silent(&[]));
    }
}
