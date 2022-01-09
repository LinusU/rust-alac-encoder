#![no_std]

#[macro_use]
extern crate alloc;

mod ag;
mod bit_buffer;
mod dp;
mod matrix;

use alloc::vec::Vec;

use ag::AgParams;
use bit_buffer::BitBuffer;
use log::debug;
use matrix::Source;

pub const DEFAULT_FRAME_SIZE: usize = 4096;
pub const DEFAULT_FRAMES_PER_PACKET: u32 = 4096;

#[deprecated(note = "Use FormatDescription::max_packet_size instead")]
pub const MAX_ESCAPE_HEADER_BYTES: usize = max_packet_size(0, 8, 0);

const MAX_CHANNELS: usize = 8;
const MAX_SAMPLE_SIZE: usize = 32;
const MAX_SEARCHES: usize = 16;
const MAX_COEFS: usize = 16;

const DEFAULT_MIX_BITS: u32 = 2;
const MAX_RES: u32 = 4;
const DEFAULT_NUM_UV: u32 = 8;

const MIN_UV: usize = 4;
const MAX_UV: usize = 8;

const MAX_RUN_DEFAULT: u16 = 255;

const ALAC_COMPATIBLE_VERSION: u8 = 0;

#[derive(Clone, Copy, Debug)]
enum ElementType {
    /// Single channel element
    Sce = 0,
    /// Channel pair element
    Cpe = 1,
    /// End of frame marker
    End = 7,
}

const CHANNEL_MAPS: [[Option<ElementType>; MAX_CHANNELS]; MAX_CHANNELS] = [
    [Some(ElementType::Sce), None, None, None, None, None, None, None],
    [Some(ElementType::Cpe), None, None, None, None, None, None, None],
    [Some(ElementType::Sce), Some(ElementType::Cpe), None, None, None, None, None, None],
    [Some(ElementType::Sce), Some(ElementType::Cpe), None, Some(ElementType::Sce), None, None, None, None],
    [Some(ElementType::Sce), Some(ElementType::Cpe), None, Some(ElementType::Cpe), None, None, None, None],
    [Some(ElementType::Sce), Some(ElementType::Cpe), None, Some(ElementType::Cpe), None, Some(ElementType::Sce), None, None],
    [Some(ElementType::Sce), Some(ElementType::Cpe), None, Some(ElementType::Cpe), None, Some(ElementType::Sce), Some(ElementType::Sce), None],
    [Some(ElementType::Sce), Some(ElementType::Cpe), None, Some(ElementType::Cpe), None, Some(ElementType::Cpe), None, Some(ElementType::Sce)],
];

#[must_use]
const fn max_packet_size(bit_depth: usize, channels_per_frame: usize, frames_per_packet: usize) -> usize {
    let sce = 3 + 4 + 12 + 4 + (bit_depth * frames_per_packet);
    let cpe = 3 + 4 + 12 + 4 + (bit_depth * 2 * frames_per_packet);

    let channel_bits = match channels_per_frame {
        1 => sce,
        2 => cpe,
        3 => sce + cpe,
        4 => sce + cpe + sce,
        5 => sce + cpe + cpe,
        6 => sce + cpe + cpe + sce,
        7 => sce + cpe + cpe + sce + sce,
        8 => sce + cpe + cpe + cpe + sce,
        _ => unreachable!(),
    };

    (channel_bits + 3 + 7) / 8
}

pub trait PcmFormat {
    fn bits() -> u32;
    fn bytes() -> u32;
    fn flags() -> u32;
}

impl PcmFormat for i16 {
    fn bits() -> u32 { 16 }
    fn bytes() -> u32 { 2 }
    fn flags() -> u32 { 4 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FormatType {
    /// Apple Lossless
    AppleLossless,
    /// Linear PCM
    LinearPcm,
}

pub struct FormatDescription {
    sample_rate: f64,
    format_id: FormatType,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
}

impl FormatDescription {
    #[must_use]
    pub fn pcm<T: PcmFormat>(sample_rate: f64, channels: u32) -> FormatDescription {
        assert!(channels > 0 && channels <= MAX_CHANNELS as u32);

        FormatDescription {
            sample_rate,
            format_id: FormatType::LinearPcm,
            bytes_per_packet: channels * T::bytes(),
            frames_per_packet: 1,
            channels_per_frame: channels,
            bits_per_channel: T::bits(),
        }
    }

    #[must_use]
    pub const fn alac(sample_rate: f64, frames_per_packet: u32, channels: u32) -> FormatDescription {
        assert!(channels > 0 && channels <= MAX_CHANNELS as u32);

        FormatDescription {
            sample_rate,
            format_id: FormatType::AppleLossless,
            bytes_per_packet: 0,
            frames_per_packet,
            channels_per_frame: channels,
            bits_per_channel: 16,
        }
    }

    #[must_use]
    pub const fn max_packet_size(&self) -> usize {
        max_packet_size(self.bits_per_channel as usize, self.channels_per_frame as usize, self.frames_per_packet as usize)
    }
}

pub struct AlacEncoder {
    // ALAC encoder parameters
    bit_depth: usize,

    // encoding state
    last_mix_res: [i16; 8],

    // encoding buffers
    mix_buffer_u: Vec<i32>,
    mix_buffer_v: Vec<i32>,
    predictor_u: Vec<i32>,
    predictor_v: Vec<i32>,
    shift_buffer_uv: Vec<u16>,
    work_buffer: Vec<u8>,

    // per-channel coefficients buffers
    coefs_u: [[[i16; MAX_COEFS]; MAX_SEARCHES]; MAX_CHANNELS],
    coefs_v: [[[i16; MAX_COEFS]; MAX_SEARCHES]; MAX_CHANNELS],

    // encoding statistics
    total_bytes_generated: usize,
    avg_bit_rate: u32,
    max_frame_bytes: u32,
    frame_size: usize,
    num_channels: u32,
    output_sample_rate: u32,
}

impl AlacEncoder {
    pub fn new(output_format: &FormatDescription) -> AlacEncoder {
        assert_eq!(output_format.format_id, FormatType::AppleLossless);

        let frame_size = output_format.frames_per_packet as usize;
        let num_channels = output_format.channels_per_frame;
        let max_output_bytes = frame_size * (num_channels as usize) * ((10 + MAX_SAMPLE_SIZE) / 8) + 1;

        let mut coefs_u = [[[0i16; MAX_COEFS]; MAX_SEARCHES]; MAX_CHANNELS];
        let mut coefs_v = [[[0i16; MAX_COEFS]; MAX_SEARCHES]; MAX_CHANNELS];

        // allocate mix buffers
        let mix_buffer_u = vec![0i32; frame_size];
        let mix_buffer_v = vec![0i32; frame_size];

        // allocate dynamic predictor buffers
        let predictor_u = vec![0i32; frame_size];
        let predictor_v = vec![0i32; frame_size];

        // allocate combined shift buffer
        let shift_buffer_uv = vec![0u16; frame_size * 2];

        // allocate work buffer for search loop
        let work_buffer = vec![0u8; max_output_bytes];

        // initialize coefs arrays once b/c retaining state across blocks actually improves the encode ratio
        for channel in 0..(num_channels as usize) {
            for search in 0..MAX_SEARCHES {
                dp::init_coefs(&mut coefs_u[channel][search], dp::DENSHIFT_DEFAULT);
                dp::init_coefs(&mut coefs_v[channel][search], dp::DENSHIFT_DEFAULT);
            }
        }

        AlacEncoder {
            // ALAC encoder parameters
            bit_depth: output_format.bits_per_channel as usize,

            // encoding state
            last_mix_res: [0; 8],

            // encoding buffers
            mix_buffer_u,
            mix_buffer_v,
            predictor_u,
            predictor_v,
            shift_buffer_uv,
            work_buffer,

            // per-channel coefficients buffers
            coefs_u,
            coefs_v,

            // encoding statistics
            total_bytes_generated: 0,
            avg_bit_rate: 0,
            max_frame_bytes: 0,
            frame_size,
            num_channels,
            output_sample_rate: output_format.sample_rate as u32,
        }
    }

    pub fn bit_depth(&self) -> usize {
        self.bit_depth
    }

    pub fn channels(&self) -> usize {
        self.num_channels as usize
    }

    pub fn frames(&self) -> usize {
        self.frame_size
    }

    pub fn sample_rate(&self) -> usize {
        self.output_sample_rate as usize
    }

    pub fn magic_cookie(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(if self.num_channels > 2 { 48 } else { 24 });

        /* ALACSpecificConfig */
        result.extend(u32::to_be_bytes(self.frame_size as u32));
        result.push(ALAC_COMPATIBLE_VERSION);
        result.push(self.bit_depth as u8);
        result.push(ag::PB0 as u8);
        result.push(ag::MB0 as u8);
        result.push(ag::KB0 as u8);
        result.push(self.num_channels as u8);
        result.extend(u16::to_be_bytes(MAX_RUN_DEFAULT));
        result.extend(u32::to_be_bytes(self.max_frame_bytes));
        result.extend(u32::to_be_bytes(self.avg_bit_rate));
        result.extend(u32::to_be_bytes(self.output_sample_rate));

        /* ALACAudioChannelLayout */
        if self.num_channels > 2 {
            let channel_layout_tag = match self.num_channels {
                1 => [1, 0, 100, 0],
                2 => [2, 0, 101, 0],
                3 => [3, 0, 113, 0],
                4 => [4, 0, 116, 0],
                5 => [5, 0, 120, 0],
                6 => [6, 0, 124, 0],
                7 => [7, 0, 142, 0],
                8 => [8, 0, 127, 0],
                _ => panic!("Unsuported number of channels"),
            };

            result.extend(u32::to_be_bytes(24));
            result.extend(b"chan");
            result.extend(u32::to_be_bytes(0));
            result.extend(&channel_layout_tag);
            result.extend(u32::to_be_bytes(0));
            result.extend(u32::to_be_bytes(0));
        }

        result
    }

    pub fn encode(&mut self, input_format: &FormatDescription, input_data: &[u8], output_data: &mut [u8]) -> usize {
        assert_eq!(input_format.format_id, FormatType::LinearPcm);

        let num_frames = input_data.len() / (input_format.bytes_per_packet as usize);
        assert!(num_frames <= self.frame_size);

        let minimum_buffer_size = max_packet_size(self.bit_depth, self.num_channels as usize, self.frame_size);
        assert!(output_data.len() >= minimum_buffer_size);

        // create a bit buffer structure pointing to our output buffer
        let mut bitstream = BitBuffer::new(output_data);

        let input_increment = (self.bit_depth + 7) / 8;
        let mut input_position = 0usize;

        let mut channel_index = 0;
        let mut mono_element_tag = 0;
        let mut stereo_element_tag = 0;

        while channel_index < input_format.channels_per_frame {
            let tag = CHANNEL_MAPS[input_format.channels_per_frame as usize - 1][channel_index as usize].unwrap();

            bitstream.write_lte25(tag as u32, 3);

            match tag {
                ElementType::Sce => {
                    bitstream.write_lte25(mono_element_tag, 4);
                    let input_size = input_increment;
                    self.encode_mono(&mut bitstream, &input_data[input_position..], input_format.channels_per_frame as usize, channel_index as usize, num_frames);
                    input_position += input_size;
                    channel_index += 1;
                    mono_element_tag += 1;
                },
                ElementType::Cpe => {
                    bitstream.write_lte25(stereo_element_tag, 4);
                    let input_size = input_increment * 2;
                    self.encode_stereo(&mut bitstream, &input_data[input_position..], input_format.channels_per_frame as usize, channel_index as usize, num_frames);
                    input_position += input_size;
                    channel_index += 2;
                    stereo_element_tag += 1;
                },
                _ => panic!("Unexpected ElementTag {:?}", tag),
            }
        }

        // add 3-bit frame end tag: ID_END
        bitstream.write_lte25(ElementType::End as u32, 3);

        // byte-align the output data
        bitstream.byte_align();

        let output_size = bitstream.position() / 8;
        debug_assert!(output_size <= bitstream.len());
        debug_assert!(output_size <= minimum_buffer_size);

        self.total_bytes_generated += output_size;
        self.max_frame_bytes = core::cmp::max(self.max_frame_bytes, output_size as u32);

        output_size
    }

    fn encode_mono(&mut self, bitstream: &mut BitBuffer, input: &[u8], stride: usize, channel_index: usize, num_samples: usize) {
        let start_position = bitstream.position();

        // reload coefs array from previous frame
        let coefs_u = &mut self.coefs_u[channel_index];

        // pick bit depth for actual encoding
        // - we lop off the lower byte(s) for 24-/32-bit encodings
        let bytes_shifted: u8 = match self.bit_depth { 32 => 2, 24 => 1, _ => 0 };

        let shift = (bytes_shifted as usize) * 8;
        let mask: u32 = (1u32 << shift) - 1;
        let chan_bits = self.bit_depth - shift;

        // flag whether or not this is a partial frame
        let partial_frame: u8 = if num_samples == self.frame_size { 0 } else { 1 };

        match self.bit_depth {
            16 => {
                // convert 16-bit data to 32-bit for predictor
                let input16 = unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i16, num_samples * stride) };
                for index in 0..num_samples {
                    self.mix_buffer_u[index] = input16[index * stride] as i32;
                }
            },
            20 => {
                // convert 20-bit data to 32-bit for predictor
                matrix::copy20_to_predictor(input, stride, &mut self.mix_buffer_u, num_samples);
            },
            24 =>  {
                // convert 24-bit data to 32-bit for the predictor and extract the shifted off byte(s)
                matrix::copy24_to_predictor(input, stride, &mut self.mix_buffer_u, num_samples);
                for index in 0..num_samples {
                    self.shift_buffer_uv[index] = ((self.mix_buffer_u[index] as u32) & mask) as u16;
                    self.mix_buffer_u[index] >>= shift;
                }
            },
            32 => {
                // just copy the 32-bit input data for the predictor and extract the shifted off byte(s)
                let input32 = unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i32, num_samples * stride) };

                for index in 0..num_samples {
                    let val = input32[index * stride];

                    self.shift_buffer_uv[index] = ((val as u32) & mask) as u16;
                    self.mix_buffer_u[index] = val >> shift;
                }
            },
            _ => panic!("Invalid mBitDepth"),
        }

        // brute-force encode optimization loop (implied "encode depth" of 0 if comparing to cmd line tool)
        // - run over variations of the encoding params to find the best choice
        let min_u = 4;
        let max_u = 8;
        let pb_factor = 4;

        let mut min_bits = 1 << 31;
        let mut best_u = min_u;

        for num_u in (min_u..max_u).step_by(4) {
            let dilate = 32usize;
            for _ in 0..7 {
                dp::pc_block(&self.mix_buffer_u, &mut self.predictor_u, num_samples / dilate, &mut coefs_u[num_u - 1], num_u, chan_bits as usize, dp::DENSHIFT_DEFAULT);
            }

            let dilate = 8usize;
            dp::pc_block(&self.mix_buffer_u, &mut self.predictor_u, num_samples / dilate, &mut coefs_u[num_u - 1], num_u, chan_bits as usize, dp::DENSHIFT_DEFAULT);

            let mut work_bits = BitBuffer::new(&mut self.work_buffer);
            let ag_params = AgParams::new(ag::MB0, (pb_factor * (ag::PB0)) / 4, ag::KB0, (num_samples / dilate) as u32, (num_samples / dilate) as u32);
            ag::dyn_comp(&ag_params, &self.predictor_u, &mut work_bits, num_samples / dilate, chan_bits as usize);

            let num_bits = (dilate * work_bits.position()) + (16 * num_u);
            if num_bits < min_bits {
                best_u = num_u;
                min_bits = num_bits;
            }
        }

        // test for escape hatch if best calculated compressed size turns out to be more than the input size
        // - first, add bits for the header bytes mixRes/maxRes/shiftU/filterU
        min_bits += (4 /* mixRes/maxRes/etc. */ * 8) + (if partial_frame == (true as u8) { 32 } else { 0 });
        if bytes_shifted != 0 {
            min_bits += num_samples * ((bytes_shifted as usize) * 8);
        }

        let escape_bits = (num_samples * self.bit_depth) + (if partial_frame == (true as u8) { 32 } else { 0 }) + (2 * 8); /* 2 common header bytes */

        let mut do_escape = min_bits >= escape_bits;

        if !do_escape {
            // write bitstream header
            bitstream.write_lte25(0, 12);
            bitstream.write_lte25(((partial_frame as u32) << 3) | ((bytes_shifted as u32) << 1), 4);
            if partial_frame > 0 {
                bitstream.write(num_samples as u32, 32);
            }
            bitstream.write_lte25(0, 16); // mixBits = mixRes = 0

            // write the params and predictor coefs
            let mode_u = 0;
            bitstream.write_lte25((mode_u << 4) | dp::DENSHIFT_DEFAULT, 8);
            bitstream.write_lte25(((pb_factor as u32) << 5) | (best_u as u32), 8);
            for index in 0..best_u {
                bitstream.write_lte25(coefs_u[(best_u as usize) - 1][index] as u32, 16);
            }

            // if shift active, write the interleaved shift buffers
            if bytes_shifted != 0 {
                for index in 0..num_samples {
                    bitstream.write(self.shift_buffer_uv[index] as u32, shift);
                }
            }

            // run the dynamic predictor with the best result
            dp::pc_block(&self.mix_buffer_u, &mut self.predictor_u, num_samples, &mut coefs_u[best_u - 1], best_u, chan_bits as usize, dp::DENSHIFT_DEFAULT);

            // do lossless compression
            let ag_params = AgParams::new_standard(num_samples as u32, num_samples as u32);
            ag::dyn_comp(&ag_params, &self.predictor_u, bitstream, num_samples, chan_bits as usize);

            // if we happened to create a compressed packet that was actually bigger than an escape packet would be,
            // chuck it and do an escape packet
            let min_bits = bitstream.position() - start_position;
            if min_bits >= escape_bits {
                bitstream.set_position(start_position);
                do_escape = true;
                debug!("compressed frame too big: {} vs. {}", min_bits, escape_bits);
            }
        }

        if do_escape {
            // write bitstream header and coefs
            bitstream.write_lte25(0, 12);
            bitstream.write_lte25(((partial_frame as u32) << 3) | 1, 4); // LSB = 1 means "frame not compressed"
            if partial_frame > 0 {
                bitstream.write(num_samples as u32, 32);
            }

            // just copy the input data to the output buffer
            match self.bit_depth {
                16 => {
                    let input16 = unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i16, num_samples * stride) };
                    for index in (0..(num_samples * stride)).step_by(stride) {
                        bitstream.write_lte25(input16[index] as u32, 16);
                    }
                },
                20 => {
                    // convert 20-bit data to 32-bit for simplicity
                    matrix::copy20_to_predictor(input, stride, &mut self.mix_buffer_u, num_samples);
                    for index in 0..num_samples {
                        bitstream.write_lte25(self.mix_buffer_u[index] as u32, 20);
                    }
                },
                24 => {
                    // convert 24-bit data to 32-bit for simplicity
                    matrix::copy24_to_predictor(input, stride, &mut self.mix_buffer_u, num_samples);
                    for index in 0..num_samples {
                        bitstream.write_lte25(self.mix_buffer_u[index] as u32, 24);
                    }
                },
                32 => {
                    let input32 = unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i32, num_samples * stride) };
                    for index in (0..(num_samples * stride)).step_by(stride) {
                        bitstream.write(input32[index] as u32, 32);
                    }
                },
                _ => panic!("Invalid mBitDepth"),
            }
        }
    }

    fn encode_stereo_escape(&mut self, bitstream: &mut BitBuffer, input: &[u8], stride: usize, num_samples: usize) {
        // flag whether or not this is a partial frame
        let partial_frame: u8 = if num_samples == self.frame_size { 0 } else { 1 };

        // write bitstream header
        bitstream.write_lte25(0, 12);
        bitstream.write_lte25(((partial_frame as u32) << 3) | 1, 4); // LSB = 1 means "frame not compressed"
        if partial_frame > 0 {
            bitstream.write(num_samples as u32, 32);
        }

        // just copy the input data to the output buffer
        match self.bit_depth {
            16 => {
                let input16 = unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i16, num_samples * stride) };

                for index in (0..(num_samples * stride)).step_by(stride) {
                    bitstream.write_lte25(input16[index] as u32, 16);
                    bitstream.write_lte25(input16[index + 1] as u32, 16);
                }
            },
            20 => {
                // mix20() with mixres param = 0 means de-interleave so use it to simplify things
                matrix::mix20(Source { data: input, stride, num_samples }, &mut self.mix_buffer_u, &mut self.mix_buffer_v, 0, 0);
                for index in 0..num_samples {
                    bitstream.write_lte25(self.mix_buffer_u[index] as u32, 20);
                    bitstream.write_lte25(self.mix_buffer_v[index] as u32, 20);
                }
            },
            24 => {
                // mix24() with mixres param = 0 means de-interleave so use it to simplify things
                matrix::mix24(Source { data: input, stride, num_samples }, &mut self.mix_buffer_u, &mut self.mix_buffer_v, 0, 0, &mut self.shift_buffer_uv, 0);
                for index in 0..num_samples {
                    bitstream.write_lte25(self.mix_buffer_u[index] as u32, 24);
                    bitstream.write_lte25(self.mix_buffer_v[index] as u32, 24);
                }
            },
            32 => {
                let input32 = unsafe { core::slice::from_raw_parts(input.as_ptr() as *const i32, num_samples * stride) };

                for index in (0..(num_samples * stride)).step_by(stride) {
                    bitstream.write(input32[index] as u32, 32);
                    bitstream.write(input32[index + 1] as u32, 32);
                }
            },
            _ => panic!("Invalid mBitDepth"),
        }
    }

    fn encode_stereo(&mut self, bitstream: &mut BitBuffer, input: &[u8], stride: usize, channel_index: usize, num_samples: usize) {
        let start_position = bitstream.position();

        // reload coefs pointers for this channel pair
        // - note that, while you might think they should be re-initialized per block, retaining state across blocks
        //   actually results in better overall compression
        // - strangely, re-using the same coefs for the different passes of the "mixRes" search loop instead of using
        //   different coefs for the different passes of "mixRes" results in even better compression
        let coefs_u = &mut self.coefs_u[channel_index];
        let coefs_v = &mut self.coefs_v[channel_index];

        // matrix encoding adds an extra bit but 32-bit inputs cannot be matrixed b/c 33 is too many
        // so enable 16-bit "shift off" and encode in 17-bit mode
        // - in addition, 24-bit mode really improves with one byte shifted off
        let bytes_shifted: u8 = match self.bit_depth { 32 => 2, 24 => 1, _ => 0 };

        let chan_bits: u32 = (self.bit_depth as u32) - (bytes_shifted as u32 * 8) + 1;

        // flag whether or not this is a partial frame
        let partial_frame: u8 = if num_samples == self.frame_size { 0 } else { 1 };

        // brute-force encode optimization loop
        // - run over variations of the encoding params to find the best choice
        let mix_bits: i32 = DEFAULT_MIX_BITS as i32;
        let max_res: i32 = MAX_RES as i32;
        let num_u: u32 = DEFAULT_NUM_UV;
        let num_v: u32 = DEFAULT_NUM_UV;
        let mode: u32 = 0;
        let pb_factor: u32 = 4;

        let mut min_bits = 1 << 31;
        let mut best_res: i32 = self.last_mix_res[channel_index] as i32;

        for mix_res in 0..=max_res {
            let dilate = 8usize;
            let source = Source { data: input, stride, num_samples: num_samples / dilate };

            // mix the stereo inputs
            match self.bit_depth {
                16 => {
                    matrix::mix16(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res);
                },
                20 => {
                    matrix::mix20(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res);
                },
                24 => {
                    // includes extraction of shifted-off bytes
                    matrix::mix24(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res, &mut self.shift_buffer_uv, bytes_shifted);
                },
                32 => {
                    // includes extraction of shifted-off bytes
                    matrix::mix32(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res, &mut self.shift_buffer_uv, bytes_shifted);
                },
                _ => panic!("Invalid mBitDepth"),
            }

            // run the dynamic predictors
            dp::pc_block(&self.mix_buffer_u, &mut self.predictor_u, num_samples / dilate, &mut coefs_u[(num_u as usize) - 1], num_u as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);
            dp::pc_block(&self.mix_buffer_v, &mut self.predictor_v, num_samples / dilate, &mut coefs_v[(num_v as usize) - 1], num_v as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);

            // run the lossless compressor on each channel
            let mut work_bits = BitBuffer::new(&mut self.work_buffer);
            let ag_params = AgParams::new(ag::MB0, (pb_factor * (ag::PB0)) / 4, ag::KB0, (num_samples / dilate) as u32, (num_samples / dilate) as u32);
            ag::dyn_comp(&ag_params, &self.predictor_u, &mut work_bits, num_samples / dilate, chan_bits as usize);
            ag::dyn_comp(&ag_params, &self.predictor_v, &mut work_bits, num_samples / dilate, chan_bits as usize);

            // look for best match
            if work_bits.position() < min_bits {
                min_bits = work_bits.position();
                best_res = mix_res;
            }
        }

        self.last_mix_res[channel_index] = best_res as i16;

        // mix the stereo inputs with the current best mixRes
        let mix_res: i32 = self.last_mix_res[channel_index] as i32;
        let source = Source { data: input, stride, num_samples };
        match self.bit_depth {
            16 => {
                matrix::mix16(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res);
            },
            20 => {
                matrix::mix20(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res);
            },
            24 => {
                // also extracts the shifted off bytes into the shift buffers
                matrix::mix24(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res, &mut self.shift_buffer_uv, bytes_shifted);
            },
            32 => {
                // also extracts the shifted off bytes into the shift buffers
                matrix::mix32(source, &mut self.mix_buffer_u, &mut self.mix_buffer_v, mix_bits, mix_res, &mut self.shift_buffer_uv, bytes_shifted);
            },
            _ => panic!("Invalid mBitDepth"),
        }

        // now it's time for the predictor coefficient search loop
        let mut num_u: usize = MIN_UV;
        let mut num_v: usize = MIN_UV;
        let mut min_bits1: usize = 1 << 31;
        let mut min_bits2: usize = 1 << 31;

        for num_uv in (MIN_UV..=MAX_UV).step_by(4) {
            let dilate = 32usize;

            for _ in 0..8 {
                dp::pc_block(&self.mix_buffer_u, &mut self.predictor_u, num_samples / dilate, &mut coefs_u[(num_uv as usize) - 1], num_uv as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);
                dp::pc_block(&self.mix_buffer_v, &mut self.predictor_v, num_samples / dilate, &mut coefs_v[(num_uv as usize) - 1], num_uv as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);
            }

            let dilate = 8usize;
            let ag_params = AgParams::new(ag::MB0, (pb_factor * ag::PB0) / 4, ag::KB0, (num_samples / dilate) as u32, (num_samples / dilate) as u32);

            let mut work_bits = BitBuffer::new(&mut self.work_buffer);
            ag::dyn_comp(&ag_params, &self.predictor_u, &mut work_bits, num_samples / dilate, chan_bits as usize);
            let bits1 = work_bits.position();

            if (bits1 * dilate + 16 * num_uv) < min_bits1 {
                min_bits1 = bits1 * dilate + 16 * num_uv;
                num_u = num_uv;
            }

            let mut work_bits = BitBuffer::new(&mut self.work_buffer);
            ag::dyn_comp(&ag_params, &self.predictor_v, &mut work_bits, num_samples / dilate, chan_bits as usize);
            let bits2 = work_bits.position();

            if (bits2 * dilate + 16 * num_uv) < min_bits2 {
                min_bits2 = bits2 * dilate + 16 * num_uv;
                num_v = num_uv;
            }
        }

        // test for escape hatch if best calculated compressed size turns out to be more than the input size
        let mut min_bits = min_bits1 + min_bits2 + (8 /* mixRes/maxRes/etc. */ * 8) + (if partial_frame == (true as u8) { 32 } else { 0 });
        if bytes_shifted != 0 {
            min_bits += (num_samples) * ((bytes_shifted as usize) * 8) * 2;
        }

        let escape_bits = (num_samples * self.bit_depth * 2) + (if partial_frame == (true as u8) { 32 } else { 0 }) + (2 * 8); /* 2 common header bytes */

        let mut do_escape = min_bits >= escape_bits;

        if !do_escape {
            // write bitstream header and coefs
            bitstream.write_lte25(0, 12);
            bitstream.write_lte25(((partial_frame as u32) << 3) | ((bytes_shifted as u32) << 1), 4);
            if partial_frame > 0 {
                bitstream.write(num_samples as u32, 32);
            }
            bitstream.write_lte25(mix_bits as u32, 8);
            bitstream.write_lte25(mix_res as u32, 8);

            debug_assert!((mode < 16) && (dp::DENSHIFT_DEFAULT < 16));
            debug_assert!((pb_factor < 8) && (num_u < 32));
            debug_assert!((pb_factor < 8) && (num_v < 32));

            bitstream.write_lte25((mode << 4) | dp::DENSHIFT_DEFAULT, 8);
            bitstream.write_lte25((pb_factor << 5) | (num_u as u32), 8);
            for index in 0..num_u {
                bitstream.write_lte25(coefs_u[(num_u as usize) - 1][index as usize] as u32, 16);
            }

            bitstream.write_lte25((mode << 4) | dp::DENSHIFT_DEFAULT, 8);
            bitstream.write_lte25((pb_factor << 5) | (num_v as u32), 8);
            for index in 0..num_v {
                bitstream.write_lte25(coefs_v[(num_v as usize) - 1][index as usize] as u32, 16);
            }

            // if shift active, write the interleaved shift buffers
            if bytes_shifted != 0 {
                let bit_shift = (bytes_shifted as usize) * 8;
                debug_assert!(bit_shift <= 16);

                for index in (0..(num_samples * 2)).step_by(2) {
                    let shifted_val: u32 = ((self.shift_buffer_uv[index] as u32) << bit_shift) | (self.shift_buffer_uv[index + 1] as u32);
                    bitstream.write(shifted_val, bit_shift * 2);
                }
            }

            // run the dynamic predictor and lossless compression for the "left" channel
            // - note: to avoid allocating more buffers, we're mixing and matching between the available buffers instead
            //   of only using "U" buffers for the U-channel and "V" buffers for the V-channel
            if mode == 0 {
                dp::pc_block(&self.mix_buffer_u, &mut self.predictor_u, num_samples, &mut coefs_u[(num_u as usize) - 1], num_u as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);
            } else {
                dp::pc_block(&self.mix_buffer_u, &mut self.predictor_v, num_samples, &mut coefs_u[(num_u as usize) - 1], num_u as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);
                dp::pc_block(&self.predictor_v, &mut self.predictor_u, num_samples, &mut [], 31, chan_bits as usize, 0);
            }

            let ag_params = AgParams::new(ag::MB0, (pb_factor * ag::PB0) / 4, ag::KB0, num_samples as u32, num_samples as u32);
            ag::dyn_comp(&ag_params, &self.predictor_u, bitstream, num_samples, chan_bits as usize);

            // run the dynamic predictor and lossless compression for the "right" channel
            if mode == 0 {
                dp::pc_block(&self.mix_buffer_v, &mut self.predictor_v, num_samples, &mut coefs_v[(num_v as usize) - 1], num_v as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);
            } else {
                dp::pc_block(&self.mix_buffer_v, &mut self.predictor_u, num_samples, &mut coefs_v[(num_v as usize) - 1], num_v as usize, chan_bits as usize, dp::DENSHIFT_DEFAULT);
                dp::pc_block(&self.predictor_u, &mut self.predictor_v, num_samples, &mut [], 31, chan_bits as usize, 0);
            }

            let ag_params = AgParams::new(ag::MB0, (pb_factor * ag::PB0) / 4, ag::KB0, num_samples as u32, num_samples as u32);
            ag::dyn_comp(&ag_params, &self.predictor_v, bitstream, num_samples, chan_bits as usize);

            // if we happened to create a compressed packet that was actually bigger than an escape packet would be,
            // chuck it and do an escape packet
            let min_bits = bitstream.position() - start_position;
            if min_bits >= escape_bits {
                bitstream.set_position(start_position);
                do_escape = true;
                debug!("compressed frame too big: {} vs. {}", min_bits, escape_bits);
            }
        }

        if do_escape {
            self.encode_stereo_escape(bitstream, input, stride, num_samples);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{AlacEncoder, FormatDescription};

    use std::{fs, vec::Vec};

    use bincode::{deserialize};
    use serde::{Serialize, Deserialize};

    #[derive(Serialize, Deserialize, Eq, PartialEq, Debug)]
    struct EncodingResult {
        magic_cookie: Vec<u8>,
        alac_chunks: Vec<Vec<u8>>,
    }

    fn test_case (input: &str, expected: &str, frame_size: u32, channels: u32) {
        let input_format = FormatDescription::pcm::<i16>(44100.0, channels);
        let output_format = FormatDescription::alac(44100.0, frame_size, channels);

        let mut encoder = AlacEncoder::new(&output_format);

        let pcm = fs::read(format!("fixtures/{}", input)).unwrap();

        let mut output = vec![0u8; output_format.max_packet_size()];

        let mut result = EncodingResult {
            magic_cookie: encoder.magic_cookie(),
            alac_chunks: Vec::new(),
        };

        for chunk in pcm.chunks(frame_size as usize * channels as usize * 2) {
            let size = encoder.encode(&input_format, chunk, &mut output);
            result.alac_chunks.push(Vec::from(&output[0..size]));
        }

        // NOTE: Uncomment to write out actual result
        // fs::write(format!("actual_{}", expected), bincode::serialize(&result).unwrap()).unwrap();

        let expected: EncodingResult = deserialize(&fs::read(format!("fixtures/{}", expected)).unwrap()).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn it_encodes_sample_352_2() {
        test_case("sample.pcm", "sample_352_2.bin", 352, 2);
    }

    #[test]
    fn it_encodes_sample_4096_2() {
        test_case("sample.pcm", "sample_4096_2.bin", 4096, 2);
    }

    #[test]
    fn it_encodes_random_352_2() {
        test_case("random.pcm", "random_352_2.bin", 352, 2);
    }

    #[test]
    fn it_encodes_random_352_3() {
        test_case("random.pcm", "random_352_3.bin", 352, 3);
    }

    #[test]
    fn it_encodes_random_352_4() {
        test_case("random.pcm", "random_352_4.bin", 352, 4);
    }

    #[test]
    fn it_encodes_random_352_5() {
        test_case("random.pcm", "random_352_5.bin", 352, 5);
    }

    #[test]
    fn it_encodes_random_352_6() {
        test_case("random.pcm", "random_352_6.bin", 352, 6);
    }

    #[test]
    fn it_encodes_random_352_7() {
        test_case("random.pcm", "random_352_7.bin", 352, 7);
    }

    #[test]
    fn it_encodes_random_352_8() {
        test_case("random.pcm", "random_352_8.bin", 352, 8);
    }

    #[test]
    fn it_encodes_random_4096_2() {
        test_case("random.pcm", "random_4096_2.bin", 4096, 2);
    }

    #[test]
    fn it_encodes_random_4096_3() {
        test_case("random.pcm", "random_4096_3.bin", 4096, 3);
    }

    #[test]
    fn it_encodes_random_4096_4() {
        test_case("random.pcm", "random_4096_4.bin", 4096, 4);
    }

    #[test]
    fn it_encodes_random_4096_5() {
        test_case("random.pcm", "random_4096_5.bin", 4096, 5);
    }

    #[test]
    fn it_encodes_random_4096_6() {
        test_case("random.pcm", "random_4096_6.bin", 4096, 6);
    }

    #[test]
    fn it_encodes_random_4096_7() {
        test_case("random.pcm", "random_4096_7.bin", 4096, 7);
    }

    #[test]
    fn it_encodes_random_4096_8() {
        test_case("random.pcm", "random_4096_8.bin", 4096, 8);
    }
}
