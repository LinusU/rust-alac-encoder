mod bindings;

use byteorder::{BE, WriteBytesExt};

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

const PB0: u8 = 40;
const KB0: u8 = 10;
const MB0: u8 = 14;
const MAX_RUN_DEFAULT: u16 = 255;

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
    unsafe fn void_handle(&mut self) -> *mut std::ffi::c_void {
        &mut self.c_handle as *mut bindings::ALACEncoder as *mut std::ffi::c_void
    }

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
        result.write_u8(KB0).unwrap();
        result.write_u8(MB0).unwrap();
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
        let mut in_out_size = input_data.len() as i32;
        let status = unsafe { bindings::ALACEncoder_Encode(self.void_handle(), input_format.to_c_struct(), output_format.to_c_struct(), &input_data[0] as *const u8 as *mut u8, output_data.as_mut_ptr(), &mut in_out_size) };

        if status != 0 { Err(Error::from_status(status)) } else { Ok(in_out_size as usize) }
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
