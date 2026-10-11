//! Every supported width encodes and decodes back bit for bit.
//!
//! The fixture tests in `src/lib.rs` compare this encoder against recorded
//! output of itself, which catches a regression but cannot catch something
//! that was always wrong. Until now nothing was reachable except 16-bit, so
//! "always wrong" had nowhere to hide: the first thing this test found was
//! that all fourteen three-byte reads in `matrix.rs` were big-endian while
//! their 16- and 32-bit neighbours were little-endian. A 24-bit file written
//! that way is not corrupt in any way a container check would notice. It
//! decodes, it has the right duration, it plays - as noise.
//!
//! So the check is the samples, not the bytes of the file. Encode, decode,
//! compare every one. The decoder is Symphonia's, which did not write the
//! encoder: nothing here can agree with itself by sharing a bug.

use alac_encoder::{AlacEncoder, FormatDescription, SUPPORTED_BIT_DEPTHS};
use symphonia_codec_alac::AlacDecoder;
use symphonia_core::audio::{Audio, GenericAudioBufferRef};
use symphonia_core::codecs::audio::well_known::CODEC_ID_ALAC;
use symphonia_core::codecs::audio::{AudioCodecParameters, AudioDecoder, AudioDecoderOptions};
use symphonia_core::packet::PacketRef;

/// Frames per ALAC packet. Apple's default.
const FRAMES_PER_PACKET: u32 = 4096;

/// Test signal, interleaved, already in range for `bits`.
///
/// Three things at once, because each exercises a different arm. A ramp and a
/// tone compress, which is the normal path through the dynamic predictor.
/// Full-scale extremes at known offsets do not, which is how the escape path
/// ("frame not compressed", a different branch with its own per-width code)
/// gets run at all. And the channels are deliberately unalike, so a stereo
/// decorrelation that mixed them up could not come back looking right.
fn signal(frames: usize, channels: usize, bits: u32) -> Vec<i32> {
    let peak = i64::from(i32::MAX) >> (32 - bits);
    let mut out = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        for channel in 0..channels {
            let sample = match frame % 97 {
                // Full scale both ways, and the values either side of zero.
                // The negative extreme is one step further out than the
                // positive one at every width, and is where an off-by-one in
                // the sign extension shows up.
                0 => -peak - 1,
                1 => peak,
                2 => -1,
                3 => 0,
                _ => {
                    // A tone per channel at a different frequency. Integer
                    // maths, so the expectation is exact on every target.
                    let phase = (frame * (7 + 3 * channel)) % 360;
                    let swing = (phase as i64 - 180) * peak / 200;
                    swing + (frame as i64 % 11) - 5
                }
            };
            out.push(i32::try_from(sample.clamp(-peak - 1, peak)).expect("in range for width"));
        }
    }
    out
}

/// Interleaved samples as the little-endian bytes packed at `bits` that
/// `FormatDescription::pcm` documents.
fn pack(samples: &[i32], bits: u32) -> Vec<u8> {
    let width = ((bits + 7) / 8) as usize;
    let mut out = Vec::with_capacity(samples.len() * width);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes()[..width]);
    }
    out
}

/// Encodes, decodes, and returns the samples that came back, interleaved.
///
/// The magic cookie is read after the last packet on purpose: two of its
/// fields are running maxima the encoder only finishes filling in once it has
/// seen everything. Nothing in this decode depends on those two, but a muxer
/// writing its header first would be writing zeros into them.
fn round_trip(samples: &[i32], channels: usize, bits: u32) -> Vec<i32> {
    let out_format = FormatDescription::alac(44_100.0, FRAMES_PER_PACKET, channels as u32, bits);
    let in_format = FormatDescription::pcm(44_100.0, channels as u32, bits);
    let mut encoder = AlacEncoder::new(&out_format);

    let width = ((bits + 7) / 8) as usize;
    let pcm = pack(samples, bits);
    let packet_bytes = FRAMES_PER_PACKET as usize * channels * width;
    // Four bytes of slack past the declared maximum. `BitBuffer::write` puts
    // down four bytes at the current byte position even when fewer bits are
    // left to write, so the last word of a packet can reach past the size the
    // packet actually ends up being.
    let mut scratch = vec![0u8; out_format.max_packet_size() + 4];

    let mut packets = Vec::new();
    for chunk in pcm.chunks(packet_bytes) {
        let size = encoder.encode(&in_format, chunk, &mut scratch);
        packets.push(scratch[..size].to_vec());
    }

    let mut params = AudioCodecParameters::new();
    params.codec = CODEC_ID_ALAC;
    params.with_extra_data(encoder.magic_cookie().into_boxed_slice());
    let mut decoder =
        AlacDecoder::try_new(&params, &AudioDecoderOptions::default()).expect("cookie is readable");

    let mut got = Vec::with_capacity(samples.len());
    for (index, packet) in packets.iter().enumerate() {
        let reference = PacketRef::new(
            0,
            (index as u32).into(),
            u64::from(FRAMES_PER_PACKET).into(),
            packet,
        );
        let decoded = decoder.decode_ref(&reference).expect("packet decodes");
        let buffer = match decoded {
            GenericAudioBufferRef::S32(buffer) => buffer,
            _ => panic!("ALAC decodes to signed 32-bit, got something else"),
        };
        // Symphonia hands back every width left-aligned in an `i32`. Shifting
        // down is the whole conversion: it is exact, so a mismatch after it is
        // a real difference and not a rounding artefact.
        let shift = 32 - bits;
        for frame in 0..buffer.frames() {
            for channel in 0..channels {
                got.push(buffer.plane(channel).expect("channel is present")[frame] >> shift);
            }
        }
    }
    got
}

/// Driven off `SUPPORTED_BIT_DEPTHS` rather than a list written out here, so
/// that offering a new width without proving it is not possible.
#[test]
fn every_supported_width_survives_a_round_trip() {
    for bits in SUPPORTED_BIT_DEPTHS {
        for channels in [1usize, 2] {
            // Deliberately not a multiple of the packet size: the last packet
            // is short, which is a different branch in the encoder and a
            // different field in the frame header.
            let frames = FRAMES_PER_PACKET as usize * 2 + 1234;
            let sent = signal(frames, channels, bits);
            let got = round_trip(&sent, channels, bits);

            assert_eq!(
                got.len(),
                sent.len(),
                "{}-bit, {} channel(s): {} samples went in and {} came back",
                bits,
                channels,
                sent.len(),
                got.len()
            );
            if let Some(at) = (0..sent.len()).find(|&i| sent[i] != got[i]) {
                panic!(
                    "{}-bit, {} channel(s): frame {} channel {} went in as {} and came back as {}",
                    bits,
                    channels,
                    at / channels,
                    at % channels,
                    sent[at],
                    got[at]
                );
            }
        }
    }
}

/// The width reaches the cookie, which is the only place a decoder can read it.
///
/// Separate from the round trip because it is a different failure: an encoder
/// that compressed correctly at 24 bits but declared 16 in its
/// `ALACSpecificConfig` would produce a file every decoder misreads, and the
/// round trip above would still pass if it read the same wrong number back.
#[test]
fn the_cookie_declares_the_width_that_was_asked_for() {
    for bits in SUPPORTED_BIT_DEPTHS {
        let format = FormatDescription::alac(44_100.0, FRAMES_PER_PACKET, 2, bits);
        let cookie = AlacEncoder::new(&format).magic_cookie();
        // ALACSpecificConfig: frameLength u32, compatibleVersion u8, then
        // bitDepth u8. A stereo cookie has no channel layout after it.
        assert_eq!(cookie.len(), 24, "{}-bit cookie is the wrong length", bits);
        assert_eq!(
            u32::from_be_bytes([cookie[0], cookie[1], cookie[2], cookie[3]]),
            FRAMES_PER_PACKET,
            "{}-bit cookie declares the wrong frame length",
            bits
        );
        assert_eq!(u32::from(cookie[5]), bits, "cookie declares the wrong width");
    }
}

/// An input width that disagrees with the output width is a panic, not a file.
#[test]
#[should_panic(expected = "input is 16-bit and the encoder writes 24-bit")]
fn a_width_mismatch_is_refused_rather_than_encoded() {
    let out_format = FormatDescription::alac(44_100.0, FRAMES_PER_PACKET, 2, 24);
    let in_format = FormatDescription::pcm(44_100.0, 2, 16);
    let mut encoder = AlacEncoder::new(&out_format);
    let mut scratch = vec![0u8; out_format.max_packet_size() + 4];
    encoder.encode(&in_format, &[0u8; 64], &mut scratch);
}
