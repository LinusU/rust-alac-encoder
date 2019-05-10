mod bindings;
mod bit_buffer;

use byteorder::{BE, WriteBytesExt};
use bit_buffer::BitBuffer;

pub const DEFAULT_FRAMES_PER_PACKET: u32 = bindings::kALACDefaultFramesPerPacket;

// FIXME: Adding some bytes here because the encoder does produce packages that large when encoding random data
// 4 & 5 channels seems to overflow by one byte
// 6 channels seems to overflow by four bytes
// 7 & 8 channels seems to overflow by seven bytes
pub const MAX_ESCAPE_HEADER_BYTES: usize = bindings::kALACMaxEscapeHeaderBytes as usize + 7;

const MAX_CHANNELS: usize = 8;
const MAX_SAMPLE_SIZE: usize = 32;
const MAX_SEARCHES: usize = 16;
const MAX_COEFS: usize = 16;

const DEFAULT_MIX_BITS: u32 = 2;
const MAX_RES: u32 = 4;
const DEFAULT_NUM_UV: u32 = 8;

const MIN_UV: u32 = 4;
const MAX_UV: u32 = 8;

const PB0: u8 = 40;
const MB0: u8 = 10;
const KB0: u8 = 14;
const MAX_RUN_DEFAULT: u16 = 255;

#[derive(Clone, Copy, Debug)]
enum ElementType {
    /// Single Channel Element
    SCE = 0,
    /// Channel Pair Element
    CPE = 1,
    /// Coupling Channel Element
    CCE = 2,
    /// LFE Channel Element
    LFE = 3,
    // not yet supported
    DSE = 4,
    PCE = 5,
    FIL = 6,
    END = 7,
    /// invalid
    NIL = 255,
}

const CHANNEL_MAPS: [[ElementType; MAX_CHANNELS]; MAX_CHANNELS] = [
    [ElementType::SCE, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL],
    [ElementType::CPE, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL],
    [ElementType::SCE, ElementType::CPE, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL],
    [ElementType::SCE, ElementType::CPE, ElementType::NIL, ElementType::SCE, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL],
    [ElementType::SCE, ElementType::CPE, ElementType::NIL, ElementType::CPE, ElementType::NIL, ElementType::NIL, ElementType::NIL, ElementType::NIL],
    [ElementType::SCE, ElementType::CPE, ElementType::NIL, ElementType::CPE, ElementType::NIL, ElementType::SCE, ElementType::NIL, ElementType::NIL],
    [ElementType::SCE, ElementType::CPE, ElementType::NIL, ElementType::CPE, ElementType::NIL, ElementType::SCE, ElementType::SCE, ElementType::NIL],
    [ElementType::SCE, ElementType::CPE, ElementType::NIL, ElementType::CPE, ElementType::NIL, ElementType::CPE, ElementType::NIL, ElementType::SCE],
];

#[derive(Debug)]
pub enum Error {
    Unimplemented,
    FileNotFound,
    Param,
    MemFull,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Unimplemented => write!(f, "kALAC_UnimplementedError"),
            Error::FileNotFound => write!(f, "kALAC_FileNotFoundError"),
            Error::Param => write!(f, "kALAC_ParamError"),
            Error::MemFull => write!(f, "kALAC_MemFullError"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    fn from_status(status: i32) -> Error {
        match status {
            bindings::kALAC_UnimplementedError => Error::Unimplemented,
            bindings::kALAC_FileNotFoundError => Error::FileNotFound,
            bindings::kALAC_ParamError => Error::Param,
            bindings::kALAC_MemFullError => Error::MemFull,
            _ => panic!("Unknown error status code"),
        }
    }
}

pub trait PcmFormat {
    fn bits() -> u32;
    fn bytes() -> u32;
    fn flags() -> u32;
}

impl PcmFormat for i16 {
    fn bits() -> u32 { 16 }
    fn bytes() -> u32 { 2 }
    fn flags() -> u32 { bindings::kALACFormatFlagsNativeEndian | bindings::kALACFormatFlagIsSignedInteger }
}

pub struct FormatDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

impl FormatDescription {
    pub fn pcm<T: PcmFormat>(sample_rate: f64, channels: u32) -> FormatDescription {
        FormatDescription {
            sample_rate,
            format_id: bindings::kALACFormatLinearPCM,
            format_flags: T::flags(),
            bytes_per_packet: channels * T::bytes(),
            frames_per_packet: 1,
            bytes_per_frame: channels * T::bytes(),
            channels_per_frame: channels,
            bits_per_channel: T::bits(),
            reserved: 0,
        }
    }

    pub fn alac(sample_rate: f64, frames_per_packet: u32, channels: u32) -> FormatDescription {
        FormatDescription {
            sample_rate,
            format_id: bindings::kALACFormatAppleLossless,
            format_flags: 1,
            bytes_per_packet: 0,
            frames_per_packet,
            bytes_per_frame: 0,
            channels_per_frame: channels,
            bits_per_channel: 0,
            reserved: 0,
        }
    }

    fn to_c_struct(&self) -> bindings::AudioFormatDescription {
        bindings::AudioFormatDescription {
            mSampleRate: self.sample_rate,
            mFormatID: self.format_id,
            mFormatFlags: self.format_flags,
            mBytesPerPacket: self.bytes_per_packet,
            mFramesPerPacket: self.frames_per_packet,
            mBytesPerFrame: self.bytes_per_frame,
            mChannelsPerFrame: self.channels_per_frame,
            mBitsPerChannel: self.bits_per_channel,
            mReserved: self.reserved,
        }
    }
}

pub struct AlacEncoder {
    c_handle: bindings::ALACEncoder,
}

impl AlacEncoder {
    pub fn new() -> AlacEncoder {
        AlacEncoder { c_handle: unsafe { bindings::ALACEncoder::new() } }
    }

    pub fn set_fast_mode(&mut self, fast_mode: bool) {
        self.c_handle.mFastMode = fast_mode;
    }

    pub fn set_frame_size(&mut self, frame_size: u32) {
        self.c_handle.mFrameSize = frame_size;
    }

    pub fn initialize_encoder(&mut self, output_format: &FormatDescription) -> Result<(), Error> {
        self.c_handle.mOutputSampleRate = output_format.sample_rate as u32;
        self.c_handle.mNumChannels = output_format.channels_per_frame;

        match output_format.format_flags {
            1 => { self.c_handle.mBitDepth = 16; },
            2 => { self.c_handle.mBitDepth = 20; },
            3 => { self.c_handle.mBitDepth = 24; },
            4 => { self.c_handle.mBitDepth = 32; },
            _ => {},
        }

        unsafe fn calloc(nobj: usize, size: usize) -> Result<*mut std::ffi::c_void, Error> {
            let result = libc::calloc(nobj, size);
            if result.is_null() { Err(Error::MemFull) } else { Ok(result) }
        }

        self.c_handle.mLastMixRes = unsafe { std::mem::zeroed() };

        self.c_handle.mMaxOutputBytes = self.c_handle.mFrameSize * self.c_handle.mNumChannels * ((10 + MAX_SAMPLE_SIZE as u32) / 8) + 1;

        // allocate mix buffers
        self.c_handle.mMixBufferU = unsafe { calloc(self.c_handle.mFrameSize as usize * std::mem::size_of::<i32>(), 1)? as *mut i32 };
        self.c_handle.mMixBufferV = unsafe { calloc(self.c_handle.mFrameSize as usize * std::mem::size_of::<i32>(), 1)? as *mut i32 };

        // allocate dynamic predictor buffers
        self.c_handle.mPredictorU = unsafe { calloc(self.c_handle.mFrameSize as usize * std::mem::size_of::<i32>(), 1)? as *mut i32 };
        self.c_handle.mPredictorV = unsafe { calloc(self.c_handle.mFrameSize as usize * std::mem::size_of::<i32>(), 1)? as *mut i32 };

        // allocate combined shift buffer
        self.c_handle.mShiftBufferUV = unsafe { calloc(self.c_handle.mFrameSize as usize * 2 * std::mem::size_of::<u16>(), 1)? as *mut u16 };

        // allocate work buffer for search loop
        self.c_handle.mWorkBuffer = unsafe { calloc(self.c_handle.mMaxOutputBytes as usize, 1)? as *mut u8 };

        // initialize coefs arrays once b/c retaining state across blocks actually improves the encode ratio
        for channel in 0..(self.c_handle.mNumChannels as usize) {
            for search in 0..MAX_SEARCHES {
                unsafe {
                    bindings::init_coefs(&mut self.c_handle.mCoefsU[channel][search][0], bindings::DENSHIFT_DEFAULT, MAX_COEFS as i32);
                    bindings::init_coefs(&mut self.c_handle.mCoefsV[channel][search][0], bindings::DENSHIFT_DEFAULT, MAX_COEFS as i32);
                }
            }
        }

        Ok(())
    }

    pub fn get_magic_cookie_size(num_channels: u32) -> usize {
        if num_channels > 2 {
            24 /* ALACSpecificConfig */ + 24 /* ALACAudioChannelLayout */
        } else {
            24 /* ALACSpecificConfig */
        }
    }

    pub fn get_magic_cookie(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(AlacEncoder::get_magic_cookie_size(self.c_handle.mNumChannels));

        /* ALACSpecificConfig */
        result.write_u32::<BE>(self.c_handle.mFrameSize).unwrap();
        result.write_u8(bindings::kALACCompatibleVersion as u8).unwrap();
        result.write_u8(self.c_handle.mBitDepth as u8).unwrap();
        result.write_u8(PB0).unwrap();
        result.write_u8(MB0).unwrap();
        result.write_u8(KB0).unwrap();
        result.write_u8(self.c_handle.mNumChannels as u8).unwrap();
        result.write_u16::<BE>(MAX_RUN_DEFAULT).unwrap();
        result.write_u32::<BE>(self.c_handle.mMaxFrameBytes).unwrap();
        result.write_u32::<BE>(self.c_handle.mAvgBitRate).unwrap();
        result.write_u32::<BE>(self.c_handle.mOutputSampleRate).unwrap();

        /* ALACAudioChannelLayout */
        if self.c_handle.mNumChannels > 2 {
            let channel_layout_tag = match self.c_handle.mNumChannels {
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

            result.write_u32::<BE>(24).unwrap();
            result.extend(&['c' as u8, 'h' as u8, 'a' as u8, 'n' as u8]);
            result.write_u32::<BE>(0).unwrap();
            result.extend(&channel_layout_tag);
            result.write_u32::<BE>(0).unwrap();
            result.write_u32::<BE>(0).unwrap();
        }

        result
    }

    pub fn encode(&mut self, input_format: &FormatDescription, output_format: &FormatDescription, input_data: &[u8], output_data: &mut [u8]) -> Result<usize, Error> {
        let num_frames = input_data.len() as u32 / input_format.bytes_per_packet;

        let minimum_buffer_size = (self.c_handle.mFrameSize * self.c_handle.mNumChannels * (((10 + output_format.bits_per_channel as u32) / 8) + 1)) as usize + MAX_ESCAPE_HEADER_BYTES;
        assert!(output_data.len() >= minimum_buffer_size);

        // create a bit buffer structure pointing to our output buffer
        // FIXME: mMaxOutputBytes is calculated from maximum sample size, and thus is usually larger than output_data length. Validate that this assumption holds...
        let mut bitstream = BitBuffer::new(&mut output_data[0..minimum_buffer_size]);

        match input_format.channels_per_frame {
            1 => {
                // add 3-bit frame start tag ID_SCE = mono channel & 4-bit element instance tag = 0
                bitstream.write(ElementType::SCE as u32, 3);
                bitstream.write(0, 4);

                // encode mono input buffer
                self.encode_mono(&mut bitstream, input_data, 1, 0, num_frames as usize)?;
            },
            2 => {
                // add 3-bit frame start tag ID_CPE = channel pair & 4-bit element instance tag = 0
                bitstream.write(ElementType::CPE as u32, 3);
                bitstream.write(0, 4);

                // encode stereo input buffer
                match self.c_handle.mFastMode {
                    false => self.encode_stereo(&mut bitstream, input_data, 2, 0, num_frames as usize)?,
                    true => self.encode_stereo_fast(&mut bitstream, input_data, 2, 0, num_frames)?,
                }
            },
            3...8 => {
                let input_increment = ((self.c_handle.mBitDepth + 7) / 8) as usize;
                let mut input_position = 0usize;

                let mut channel_index = 0;
                let mut mono_element_tag = 0;
                let mut stereo_element_tag = 0;
                let mut lfe_element_tag = 0;

                while channel_index < input_format.channels_per_frame {
                    let tag = CHANNEL_MAPS[input_format.channels_per_frame as usize - 1][channel_index as usize];

                    bitstream.write(tag as u32, 3);

                    match tag {
                        ElementType::SCE => {
                            // mono
                            bitstream.write(mono_element_tag, 4);
                            let input_size = input_increment * 1;
                            self.encode_mono(&mut bitstream, &input_data[input_position..(input_position + input_size)], input_format.channels_per_frame as usize, channel_index as usize, num_frames as usize)?;
                            input_position += input_size;
                            channel_index += 1;
                            mono_element_tag += 1;
                        },
                        ElementType::CPE => {
                            // stereo
                            bitstream.write(stereo_element_tag, 4);
                            let input_size = input_increment * 2;
                            self.encode_stereo(&mut bitstream, &input_data[input_position..(input_position + input_size)], input_format.channels_per_frame as usize, channel_index as usize, num_frames as usize)?;
                            input_position += input_size;
                            channel_index += 2;
                            stereo_element_tag += 1;
                        },
                        ElementType::LFE => {
                            // LFE channel (subwoofer)
                            bitstream.write(lfe_element_tag, 4);
                            let input_size = input_increment * 1;
                            self.encode_mono(&mut bitstream, &input_data[input_position..(input_position + input_size)], input_format.channels_per_frame as usize, channel_index as usize, num_frames as usize)?;
                            input_position += input_size;
                            channel_index += 1;
                            lfe_element_tag += 1;
                        },
                        _ => panic!("Unexpected ElementTag {:?}", tag),
                    }
                }
            },
            _ => {
                panic!("Unsuported number of channels");
            },
        }

        // add 3-bit frame end tag: ID_END
        bitstream.write(bindings::ELEMENT_TYPE_ID_END, 3);

        // byte-align the output data
        bitstream.byte_align(true);

        let output_size = bitstream.get_position() / 8;
        assert!(output_size <= bitstream.len());
        assert!(output_size <= self.c_handle.mMaxOutputBytes as usize);

        self.c_handle.mTotalBytesGenerated += output_size as u32;
        self.c_handle.mMaxFrameBytes = std::cmp::max(self.c_handle.mMaxFrameBytes, output_size as u32);

        Ok(output_size)
    }

    fn encode_mono(&mut self, bitstream: &mut BitBuffer, input: &[u8], stride: usize, channel_index: usize, num_samples: usize) -> Result<(), Error> {
        let start_bits = bitstream.save_state();
        let start_position = bitstream.get_position();

        match self.c_handle.mBitDepth {
            16 => {},
            20 => {},
            24 => {},
            32 => {},
            _ => return Err(Error::Param),
        }

        // reload coefs array from previous frame
        let coefs_u = &mut self.c_handle.mCoefsU[channel_index];

        // pick bit depth for actual encoding
        // - we lop off the lower byte(s) for 24-/32-bit encodings
        let bytes_shifted: u8 = match self.c_handle.mBitDepth { 32 => 2, 24 => 1, _ => 0 };

        let shift: u32 = (bytes_shifted as u32) * 8;
        let mask: u32 = (1u32 << shift) - 1;
        let chan_bits: u32 = (self.c_handle.mBitDepth as u32) - shift;

        // flag whether or not this is a partial frame
        let partial_frame: u8 = if num_samples == (self.c_handle.mFrameSize as usize) { 0 } else { 1 };

        match self.c_handle.mBitDepth {
            16 => {
                // convert 16-bit data to 32-bit for predictor
                let input16 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i16, num_samples * stride) };
                let output32 = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mMixBufferU, num_samples) };
                for index in 0..num_samples {
                    output32[index] = input16[index * stride] as i32;
                }
            },
            20 => {
                // convert 20-bit data to 32-bit for predictor
                unsafe { bindings::copy20ToPredictor(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, num_samples as i32); }
            },
            24 =>  {
                // convert 24-bit data to 32-bit for the predictor and extract the shifted off byte(s)
                unsafe { bindings::copy24ToPredictor(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, num_samples as i32); }
                let shift_buffer_uv = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mShiftBufferUV, num_samples) };
                let mix_buffer_u = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mMixBufferU, num_samples) };
                for index in 0..num_samples {
                    shift_buffer_uv[index] = ((mix_buffer_u[index] as u32) & mask) as u16;
                    mix_buffer_u[index] >>= shift;
                }
            },
            32 => {
                // just copy the 32-bit input data for the predictor and extract the shifted off byte(s)
                let input32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i32, num_samples * stride) };
                let shift_buffer_uv = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mShiftBufferUV, num_samples) };
                let mix_buffer_u = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mMixBufferU, num_samples) };

                for index in 0..num_samples {
                    let val = input32[index * stride];

                    shift_buffer_uv[index] = ((val as u32) & mask) as u16;
                    mix_buffer_u[index] = val >> shift;
                }
            },
            _ => panic!("Invalid mBitDepth"),
        }

        // brute-force encode optimization loop (implied "encode depth" of 0 if comparing to cmd line tool)
        // - run over variations of the encoding params to find the best choice
        let min_u = 4;
        let max_u = 8;
        let pb_factor = 4;

        let mut min_bits: u32 = 1 << 31;
        let mut best_u = min_u;

        let mut ag_params: bindings::AGParamRec = unsafe { std::mem::uninitialized() };
        let mut bits1: u32 = 0;

        for num_u in (min_u..max_u).step_by(4) {
            let mut work_bits = BitBuffer::new(unsafe { std::slice::from_raw_parts_mut(self.c_handle.mWorkBuffer, self.c_handle.mMaxOutputBytes as usize) });

            let dilate = 32;
            for _ in 0..7 {
                unsafe { bindings::pc_block(self.c_handle.mMixBufferU, self.c_handle.mPredictorU, (num_samples / dilate) as i32, coefs_u[num_u - 1].as_mut_ptr(), num_u as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
            }

            let dilate = 8;
            unsafe { bindings::pc_block(self.c_handle.mMixBufferU, self.c_handle.mPredictorU, (num_samples / dilate) as i32, coefs_u[num_u - 1].as_mut_ptr(), num_u as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }

            unsafe { bindings::set_ag_params(&mut ag_params, MB0 as u32, (pb_factor * (PB0 as u32)) / 4, KB0 as u32, (num_samples / dilate) as u32, (num_samples / dilate) as u32, bindings::MAX_RUN_DEFAULT); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorU, &mut work_bits.c_handle, (num_samples / dilate) as i32, chan_bits as i32, &mut bits1) };
            if status != 0 { return Err(Error::from_status(status)); }

            let num_bits = ((dilate as u32) * bits1) + (16 * num_u as u32);
            if num_bits < min_bits {
                best_u = num_u;
                min_bits = num_bits;
            }
        }

        // test for escape hatch if best calculated compressed size turns out to be more than the input size
        // - first, add bits for the header bytes mixRes/maxRes/shiftU/filterU
        min_bits += (4 /* mixRes/maxRes/etc. */ * 8) + (if partial_frame == (true as u8) { 32 } else { 0 });
        if bytes_shifted != 0 {
            min_bits += (num_samples as u32) * ((bytes_shifted as u32) * 8);
        }

        let escape_bits: u32 = ((num_samples as u32) * (self.c_handle.mBitDepth as u32)) + (if partial_frame == (true as u8) { 32 } else { 0 }) + (2 * 8); /* 2 common header bytes */

        let mut do_escape = min_bits >= escape_bits;

        if do_escape == false {
            // write bitstream header
            bitstream.write(0, 12);
            bitstream.write(((partial_frame as u32) << 3) | ((bytes_shifted as u32) << 1), 4);
            if partial_frame > 0 {
                bitstream.write(num_samples as u32, 32);
            }
            bitstream.write(0, 16); // mixBits = mixRes = 0

            // write the params and predictor coefs
            bitstream.write((0 << 4) | bindings::DENSHIFT_DEFAULT, 8); // modeU = 0
            bitstream.write(((pb_factor as u32) << 5) | (best_u as u32), 8);
            for index in 0..best_u {
                bitstream.write(coefs_u[(best_u as usize) - 1][index] as u32, 16);
            }

            // if shift active, write the interleaved shift buffers
            if bytes_shifted != 0 {
                let shift_buffer_uv = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mShiftBufferUV, num_samples) };

                for index in 0..num_samples {
                    bitstream.write(shift_buffer_uv[index] as u32, shift);
                }
            }

            // run the dynamic predictor with the best result
            unsafe { bindings::pc_block(self.c_handle.mMixBufferU, self.c_handle.mPredictorU, num_samples as i32, coefs_u[best_u - 1].as_mut_ptr(), best_u as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }

            // do lossless compression
            unsafe { bindings::set_standard_ag_params(&mut ag_params, num_samples as u32, num_samples as u32); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorU, &mut bitstream.c_handle, num_samples as i32, chan_bits as i32, &mut bits1) };
            if status != 0 { return Err(Error::from_status(status)); }

            // if we happened to create a compressed packet that was actually bigger than an escape packet would be,
            // chuck it and do an escape packet
            let min_bits = (bitstream.get_position() - start_position) as u32;
            if min_bits >= escape_bits {
                bitstream.load_state(start_bits); // reset bitstream state
                do_escape = true;
                println!("compressed frame too big: {} vs. {}", min_bits, escape_bits);
            }
        }

        if do_escape == true {
            // write bitstream header and coefs
            bitstream.write(0, 12);
            bitstream.write(((partial_frame as u32) << 3) | 1, 4); // LSB = 1 means "frame not compressed"
            if partial_frame > 0 {
                bitstream.write(num_samples as u32, 32);
            }

            // just copy the input data to the output buffer
            match self.c_handle.mBitDepth {
                16 => {
                    let input16 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i16 as *mut i16, num_samples * stride) };
                    for index in (0..(num_samples * stride)).step_by(stride) {
                        bitstream.write(input16[index] as u32, 16);
                    }
                },
                20 => {
                    // convert 20-bit data to 32-bit for simplicity
                    let mix_buffer_u = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mMixBufferU, num_samples) };
                    unsafe { bindings::copy20ToPredictor(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, num_samples as i32); }
                    for index in 0..num_samples {
                        bitstream.write(mix_buffer_u[index] as u32, 20);
                    }
                },
                24 => {
                    // convert 24-bit data to 32-bit for simplicity
                    let mix_buffer_u = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mMixBufferU, num_samples) };
                    unsafe { bindings::copy24ToPredictor(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, num_samples as i32); }
                    for index in 0..num_samples {
                        bitstream.write(mix_buffer_u[index] as u32, 24);
                    }
                },
                32 => {
                    let input32 = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i32, num_samples * stride) };
                    for index in (0..(num_samples * stride)).step_by(stride) {
                        bitstream.write(input32[index] as u32, 32);
                    }
                },
                _ => panic!("Invalid mBitDepth"),
            }
        }

        Ok(())
    }

    fn encode_stereo_escape(&mut self, bitstream: &mut BitBuffer, input: &[u8], stride: usize, num_samples: usize) -> Result<(), Error> {
        let status = unsafe { bindings::ALACEncoder_EncodeStereoEscape(&mut self.c_handle, &mut bitstream.c_handle, &input[0] as *const u8 as *mut u8 as *mut std::ffi::c_void, stride as u32, num_samples as u32) };
        if status == 0 { Ok(()) } else { Err(Error::from_status(status)) }
    }

    fn encode_stereo(&mut self, bitstream: &mut BitBuffer, input: &[u8], stride: usize, channel_index: usize, num_samples: usize) -> Result<(), Error> {
        let start_bits = bitstream.save_state();
        let start_position = bitstream.get_position();

        match self.c_handle.mBitDepth {
            16 => {},
            20 => {},
            24 => {},
            32 => {},
            _ => return Err(Error::Param),
        }

        // reload coefs pointers for this channel pair
        // - note that, while you might think they should be re-initialized per block, retaining state across blocks
        //   actually results in better overall compression
        // - strangely, re-using the same coefs for the different passes of the "mixRes" search loop instead of using
        //   different coefs for the different passes of "mixRes" results in even better compression
        let coefs_u = &mut self.c_handle.mCoefsU[channel_index];
        let coefs_v = &mut self.c_handle.mCoefsV[channel_index];

        // matrix encoding adds an extra bit but 32-bit inputs cannot be matrixed b/c 33 is too many
        // so enable 16-bit "shift off" and encode in 17-bit mode
        // - in addition, 24-bit mode really improves with one byte shifted off
        let bytes_shifted: u8 = match self.c_handle.mBitDepth { 32 => 2, 24 => 1, _ => 0 };

        let chan_bits: u32 = (self.c_handle.mBitDepth as u32) - (bytes_shifted as u32 * 8) + 1;

        // flag whether or not this is a partial frame
        let partial_frame: u8 = if num_samples == (self.c_handle.mFrameSize as usize) { 0 } else { 1 };

        // brute-force encode optimization loop
        // - run over variations of the encoding params to find the best choice
        let mix_bits: i32 = DEFAULT_MIX_BITS as i32;
        let max_res: i32 = MAX_RES as i32;
        let num_u: u32 = DEFAULT_NUM_UV;
        let num_v: u32 = DEFAULT_NUM_UV;
        let mode: u32 = 0;
        let pb_factor: u32 = 4;
        let mut dilate: u32 = 8;

        let mut min_bits: u32 = 1 << 31;
        let mut best_res: i32 = self.c_handle.mLastMixRes[channel_index] as i32;

        let mut ag_params: bindings::AGParamRec = unsafe { std::mem::uninitialized() };
        let mut bits1: u32 = 0;
        let mut bits2: u32 = 0;

        for mix_res in 0..=max_res {
            // mix the stereo inputs
            match self.c_handle.mBitDepth {
                16 => {
                    unsafe { bindings::mix16(input.as_ptr() as *const i16 as *mut i16, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, (num_samples as i32) / (dilate as i32), mix_bits, mix_res); }
                },
                20 => {
                    unsafe { bindings::mix20(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, (num_samples as i32) / (dilate as i32), mix_bits, mix_res); }
                },
                24 => {
                    // includes extraction of shifted-off bytes
                    unsafe { bindings::mix24(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, (num_samples as i32) / (dilate as i32), mix_bits, mix_res, self.c_handle.mShiftBufferUV, bytes_shifted as i32); }
                },
                32 => {
                    // includes extraction of shifted-off bytes
                    unsafe { bindings::mix32(input.as_ptr() as *const i32 as *mut i32, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, (num_samples as i32) / (dilate as i32), mix_bits, mix_res, self.c_handle.mShiftBufferUV, bytes_shifted as i32); }
                },
                _ => panic!("Invalid mBitDepth"),
            }

            let mut work_bits = BitBuffer::new(unsafe { std::slice::from_raw_parts_mut(self.c_handle.mWorkBuffer, self.c_handle.mMaxOutputBytes as usize) });

            // run the dynamic predictors
            unsafe { bindings::pc_block(self.c_handle.mMixBufferU, self.c_handle.mPredictorU, (num_samples as i32) / (dilate as i32), coefs_u[(num_u as usize) - 1].as_mut_ptr(), num_u as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
            unsafe { bindings::pc_block(self.c_handle.mMixBufferV, self.c_handle.mPredictorV, (num_samples as i32) / (dilate as i32), coefs_v[(num_v as usize) - 1].as_mut_ptr(), num_v as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }

            // run the lossless compressor on each channel
            unsafe { bindings::set_ag_params(&mut ag_params, MB0 as u32, (pb_factor * (PB0 as u32)) / 4, KB0 as u32, (num_samples as u32) / (dilate as u32), (num_samples as u32) / (dilate as u32), bindings::MAX_RUN_DEFAULT); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorU, &mut work_bits.c_handle, (num_samples as i32) / (dilate as i32), chan_bits as i32, &mut bits1) };
            if status != 0 { return Err(Error::from_status(status)); }

            unsafe { bindings::set_ag_params(&mut ag_params, MB0 as u32, (pb_factor * (PB0 as u32)) / 4, KB0 as u32, (num_samples as u32) / (dilate as u32), (num_samples as u32) / (dilate as u32), bindings::MAX_RUN_DEFAULT); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorV, &mut work_bits.c_handle, (num_samples as i32) / (dilate as i32), chan_bits as i32, &mut bits2) };
            if status != 0 { return Err(Error::from_status(status)); }

            // look for best match
            if (bits1 + bits2) < min_bits {
                min_bits = bits1 + bits2;
                best_res = mix_res;
            }
        }

        self.c_handle.mLastMixRes[channel_index] = best_res as i16;

        // mix the stereo inputs with the current best mixRes
        let mix_res: i32 = self.c_handle.mLastMixRes[channel_index] as i32;
        match self.c_handle.mBitDepth {
            16 => {
                unsafe { bindings::mix16(input.as_ptr() as *const i16 as *mut i16, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, num_samples as i32, mix_bits, mix_res); }
            },
            20 => {
                unsafe { bindings::mix20(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, num_samples as i32, mix_bits, mix_res); }
            },
            24 => {
                // also extracts the shifted off bytes into the shift buffers
                unsafe { bindings::mix24(input.as_ptr() as *mut u8, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, num_samples as i32, mix_bits, mix_res, self.c_handle.mShiftBufferUV, bytes_shifted as i32); }
            },
            32 => {
                // also extracts the shifted off bytes into the shift buffers
                unsafe { bindings::mix32(input.as_ptr() as *const i32 as *mut i32, stride as u32, self.c_handle.mMixBufferU, self.c_handle.mMixBufferV, num_samples as i32, mix_bits, mix_res, self.c_handle.mShiftBufferUV, bytes_shifted as i32); }
            },
            _ => panic!("Invalid mBitDepth"),
        }

        // now it's time for the predictor coefficient search loop
        let mut num_u: u32 = MIN_UV;
        let mut num_v: u32 = MIN_UV;
        let mut min_bits1: u32 = 1 << 31;
        let mut min_bits2: u32 = 1 << 31;

        for num_uv in (MIN_UV..=MAX_UV).step_by(4) {
            let mut work_bits = BitBuffer::new(unsafe { std::slice::from_raw_parts_mut(self.c_handle.mWorkBuffer, self.c_handle.mMaxOutputBytes as usize) });

            // let dilate: u32 = 32;
            dilate = 32;

            for _ in 0..8 {
                unsafe { bindings::pc_block(self.c_handle.mMixBufferU, self.c_handle.mPredictorU, (num_samples as i32) / (dilate as i32), coefs_u[(num_uv as usize) - 1].as_mut_ptr(), num_uv as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
                unsafe { bindings::pc_block(self.c_handle.mMixBufferV, self.c_handle.mPredictorV, (num_samples as i32) / (dilate as i32), coefs_v[(num_uv as usize) - 1].as_mut_ptr(), num_uv as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
            }

            dilate = 8;

            unsafe { bindings::set_ag_params(&mut ag_params, MB0 as u32, (pb_factor * PB0 as u32) / 4, KB0 as u32, (num_samples as u32) / dilate, (num_samples as u32) / dilate, bindings::MAX_RUN_DEFAULT); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorU, &mut work_bits.c_handle, (num_samples as i32) / (dilate as i32), chan_bits as i32, &mut bits1) };
            if status != 0 { return Err(Error::from_status(status)); }

            if (bits1 * dilate + 16 * num_uv) < min_bits1 {
                min_bits1 = bits1 * dilate + 16 * num_uv;
                num_u = num_uv;
            }

            unsafe { bindings::set_ag_params(&mut ag_params, MB0 as u32, (pb_factor * PB0 as u32) / 4, KB0 as u32, (num_samples as u32) / dilate, (num_samples as u32) / dilate, bindings::MAX_RUN_DEFAULT); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorV, &mut work_bits.c_handle, (num_samples as i32) / (dilate as i32), chan_bits as i32, &mut bits2) };
            if status != 0 { return Err(Error::from_status(status)); }

            if (bits2 * dilate + 16 * num_uv) < min_bits2 {
                min_bits2 = bits2 * dilate + 16 * num_uv;
                num_v = num_uv;
            }
        }

        // test for escape hatch if best calculated compressed size turns out to be more than the input size
        let mut min_bits = min_bits1 + min_bits2 + (8 /* mixRes/maxRes/etc. */ * 8) + (if partial_frame == (true as u8) { 32 } else { 0 });
        if bytes_shifted != 0 {
            min_bits += (num_samples as u32) * ((bytes_shifted as u32) * 8) * 2;
        }

        let escape_bits: u32 = ((num_samples as u32) * (self.c_handle.mBitDepth as u32) * 2) + (if partial_frame == (true as u8) { 32 } else { 0 }) + (2 * 8); /* 2 common header bytes */

        let mut do_escape = min_bits >= escape_bits;

        if do_escape == false {
            // write bitstream header and coefs
            bitstream.write(0, 12);
            bitstream.write(((partial_frame as u32) << 3) | ((bytes_shifted as u32) << 1), 4);
            if partial_frame > 0 {
                bitstream.write(num_samples as u32, 32);
            }
            bitstream.write(mix_bits as u32, 8);
            bitstream.write(mix_res as u32, 8);

            assert!((mode < 16) && (bindings::DENSHIFT_DEFAULT < 16));
            assert!((pb_factor < 8) && (num_u < 32));
            assert!((pb_factor < 8) && (num_v < 32));

            bitstream.write((mode << 4) | bindings::DENSHIFT_DEFAULT, 8);
            bitstream.write((pb_factor << 5) | num_u, 8);
            for index in 0..num_u {
                bitstream.write(coefs_u[(num_u as usize) - 1][index as usize] as u32, 16);
            }

            bitstream.write((mode << 4) | bindings::DENSHIFT_DEFAULT, 8);
            bitstream.write((pb_factor << 5) | num_v, 8);
            for index in 0..num_v {
                bitstream.write(coefs_v[(num_v as usize) - 1][index as usize] as u32, 16);
            }

            // if shift active, write the interleaved shift buffers
            if bytes_shifted != 0 {
                let bit_shift: u32 = (bytes_shifted as u32) * 8;
                assert!(bit_shift <= 16);

                let shift_buffer_uv = unsafe { std::slice::from_raw_parts_mut(self.c_handle.mShiftBufferUV, num_samples * 2) };

                for index in (0..(num_samples * 2)).step_by(2) {
                    let shifted_val: u32 = ((shift_buffer_uv[index + 0] as u32) << bit_shift) | (shift_buffer_uv[index + 1] as u32);
                    bitstream.write(shifted_val, bit_shift * 2);
                }
            }

            // run the dynamic predictor and lossless compression for the "left" channel
            // - note: to avoid allocating more buffers, we're mixing and matching between the available buffers instead
            //   of only using "U" buffers for the U-channel and "V" buffers for the V-channel
            if mode == 0 {
                unsafe { bindings::pc_block(self.c_handle.mMixBufferU, self.c_handle.mPredictorU, num_samples as i32, coefs_u[(num_u as usize) - 1].as_mut_ptr(), num_u as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
            } else {
                unsafe { bindings::pc_block(self.c_handle.mMixBufferU, self.c_handle.mPredictorV, num_samples as i32, coefs_u[(num_u as usize) - 1].as_mut_ptr(), num_u as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
                unsafe { bindings::pc_block(self.c_handle.mPredictorV, self.c_handle.mPredictorU, num_samples as i32, std::ptr::null_mut(), 31, chan_bits, 0); }
            }

            unsafe { bindings::set_ag_params(&mut ag_params, MB0 as u32, (pb_factor * PB0 as u32) / 4, KB0 as u32, num_samples as u32, num_samples as u32, bindings::MAX_RUN_DEFAULT); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorU, &mut bitstream.c_handle, num_samples as i32, chan_bits as i32, &mut bits1) };
            if status != 0 { return Err(Error::from_status(status)); }

            // run the dynamic predictor and lossless compression for the "right" channel
            if mode == 0 {
                unsafe { bindings::pc_block(self.c_handle.mMixBufferV, self.c_handle.mPredictorV, num_samples as i32, coefs_v[(num_v as usize) - 1].as_mut_ptr(), num_v as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
            } else {
                unsafe { bindings::pc_block(self.c_handle.mMixBufferV, self.c_handle.mPredictorU, num_samples as i32, coefs_v[(num_v as usize) - 1].as_mut_ptr(), num_v as i32, chan_bits, bindings::DENSHIFT_DEFAULT); }
                unsafe { bindings::pc_block(self.c_handle.mPredictorU, self.c_handle.mPredictorV, num_samples as i32, std::ptr::null_mut(), 31, chan_bits, 0); }
            }

            unsafe { bindings::set_ag_params(&mut ag_params, MB0 as u32, (pb_factor * PB0 as u32) / 4, KB0 as u32, num_samples as u32, num_samples as u32, bindings::MAX_RUN_DEFAULT); }
            let status = unsafe { bindings::dyn_comp(&mut ag_params, self.c_handle.mPredictorV, &mut bitstream.c_handle, num_samples as i32, chan_bits as i32, &mut bits2) };
            if status != 0 { return Err(Error::from_status(status)); }

            // if we happened to create a compressed packet that was actually bigger than an escape packet would be,
            // chuck it and do an escape packet
            let min_bits = (bitstream.get_position() - start_position) as u32;
            if min_bits >= escape_bits {
                bitstream.load_state(start_bits); // reset bitstream state
                do_escape = true;
                println!("compressed frame too big: {} vs. {}", min_bits, escape_bits);
            }
        }

        if do_escape == true {
            self.encode_stereo_escape(bitstream, input, stride, num_samples)?;
        }

        Ok(())
    }

    fn encode_stereo_fast(&mut self, bitstream: &mut BitBuffer, input: &[u8], stride: u32, channel_index: u32, num_samples: u32) -> Result<(), Error> {
        let status = unsafe { bindings::ALACEncoder_EncodeStereoFast(&mut self.c_handle, &mut bitstream.c_handle, &input[0] as *const u8 as *mut u8 as *mut std::ffi::c_void, stride, channel_index, num_samples) };
        if status == 0 { Ok(()) } else { Err(Error::from_status(status)) }
    }
}

impl Drop for AlacEncoder {
    fn drop(&mut self) {
        unsafe { bindings::ALACEncoder_ALACEncoder_destructor(&mut self.c_handle); }
    }
}

#[cfg(test)]
mod tests {
    use super::{AlacEncoder, FormatDescription, MAX_ESCAPE_HEADER_BYTES};

    use std::fs;

    use bincode::{deserialize};
    use serde::{Serialize, Deserialize};

    #[derive(Serialize, Deserialize, Eq, PartialEq, Debug)]
    struct EncodingResult {
        magic_cookie: Vec<u8>,
        alac_chunks: Vec<Vec<u8>>,
    }

    fn test_case (input: &str, expected: &str, frame_size: u32, channels: u32) {
        let mut encoder = AlacEncoder::new();

        let input_format = FormatDescription::pcm::<i16>(44100.0, channels);
        let output_format = FormatDescription::alac(44100.0, frame_size, channels);

        encoder.set_frame_size(frame_size);
        encoder.initialize_encoder(&output_format).unwrap();

        let pcm = fs::read(format!("fixtures/{}", input)).unwrap();

        let mut output = vec![0u8; (frame_size as usize * channels as usize * 2) + MAX_ESCAPE_HEADER_BYTES];

        let mut result = EncodingResult {
            magic_cookie: encoder.get_magic_cookie(),
            alac_chunks: Vec::new(),
        };

        for chunk in pcm.chunks(frame_size as usize * channels as usize * 2) {
            let size = encoder.encode(&input_format, &output_format, &chunk, &mut output).unwrap();
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
