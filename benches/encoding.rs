extern crate std;

use std::fs;

use alac_encoder::{AlacEncoder, FormatDescription};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn test_case (input: &[u8], output: &mut [u8], sample_rate: f64, frame_size: u32, channels: u32) {
    let input_format = FormatDescription::pcm::<i16>(sample_rate, channels);
    let output_format = FormatDescription::alac(sample_rate, frame_size, channels);

    let mut encoder = AlacEncoder::new(&output_format);

    for chunk in input.chunks(frame_size as usize * channels as usize * 2) {
        let size = encoder.encode(&input_format, chunk, output);

        black_box(&output[0..size]);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Like a Rolling Stone", |b| {
        const SAMPLE_RATE: f64 = 44100.0;
        const FRAME_SIZE: u32 = 352;
        const CHANNELS: u32 = 2;
        const BUFFER_SIZE: usize = FormatDescription::alac(SAMPLE_RATE, FRAME_SIZE, CHANNELS).max_packet_size();

        let input = fs::read("fixtures/like-a-rolling-stone.pcm").unwrap();
        let mut output = vec![0u8; BUFFER_SIZE];

        b.iter(|| test_case(&input, &mut output, SAMPLE_RATE, FRAME_SIZE, CHANNELS));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
