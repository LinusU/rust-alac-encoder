# ALAC Encoder for Rust

Rust port of [Apple's open source ALAC library](https://macosforge.github.io/alac/).

## Installation

This crate works with Cargo and is on [crates.io](https://crates.io/crates/alac-encoder).

## Usage

```rust
use alac_encoder::{AlacEncoder, FormatDescription};

// Specify the input format as signed 16-bit raw PCM, 44100 Hz & 2 channels
let input_format = FormatDescription::pcm::<i16>(44100.0, 2);

// Specify the output format as 44100 Hz ALAC with a frame size of 4096 & 2 channels
let output_format = FormatDescription::alac(44100.0, 4096, 2);

// Initialize the encoder
let mut encoder = AlacEncoder::new(&output_format);

// Allocate a buffer for the encoder to write chunks to.
let mut output = vec![0u8; output_format.max_packet_size()];

// Get a hold of the source data, e.g. from a file
let pcm = fs::read("foobar.pcm").unwrap();

// Iterate over chunks from the input
for chunk in pcm.chunks(frame_size as usize * channels as usize * 2) {
  // Feed the current chunk to the encoder
  let size = encoder.encode(&input_format, &chunk, &mut output);

  // Here you can do whatever you want with the result:
  Vec::from(&output[0..size]);
}
```

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
* MIT license ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
