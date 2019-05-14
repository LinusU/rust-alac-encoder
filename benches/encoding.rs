use std::fs;

use alac_encoder::{AlacEncoder, FormatDescription, MAX_ESCAPE_HEADER_BYTES};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn test_case (input: &str, frame_size: u32, channels: u32) {
    let input_format = FormatDescription::pcm::<i16>(44100.0, channels);
    let output_format = FormatDescription::alac(44100.0, frame_size, channels);

    let mut encoder = AlacEncoder::new(&output_format);

    let pcm = fs::read(format!("fixtures/{}", input)).unwrap();

    let mut output = vec![0u8; (frame_size as usize * channels as usize * 2) + MAX_ESCAPE_HEADER_BYTES];

    for chunk in pcm.chunks(frame_size as usize * channels as usize * 2) {
        let size = encoder.encode(&input_format, &chunk, &mut output);
        black_box(Vec::from(&output[0..size]));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("sample_352_2", |b| {
        b.iter(|| test_case("sample.pcm", 352, 2));
    });

    c.bench_function("sample_4096_2", |b| {
        b.iter(|| test_case("sample.pcm", 4096, 2));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
