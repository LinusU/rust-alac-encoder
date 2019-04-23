mod bindings;

pub const DEFAULT_FRAMES_PER_PACKET: u32 = bindings::kALACDefaultFramesPerPacket;

// FIXME: Adding some bytes here because the encoder does produce packages that large when encoding random data
// 4 & 5 channels seems to overflow by one byte
// 6 channels seems to overflow by four bytes
// 7 & 8 channels seems to overflow by seven bytes
pub const MAX_ESCAPE_HEADER_BYTES: usize = bindings::kALACMaxEscapeHeaderBytes as usize + 7;

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

    pub fn initialize_encoder(&mut self, output_format: &FormatDescription) -> i32 {
        unsafe { bindings::ALACEncoder_InitializeEncoder(self.void_handle(), output_format.to_c_struct()) }
    }

    pub fn get_magic_cookie_size(num_channels: u32) -> usize {
        if num_channels > 2 {
            24 /* ALACSpecificConfig */ + (bindings::kChannelAtomSize as usize) + 12 /* ALACAudioChannelLayout */
        } else {
            24 /* ALACSpecificConfig */
        }
    }

    pub fn get_magic_cookie(&self) -> Vec<u8> {
        let size = AlacEncoder::get_magic_cookie_size(self.c_handle.mNumChannels);
        let mut result = vec![0; size];

        unsafe {
            let mut in_out_size = size as u32;
            bindings::ALACEncoder_GetMagicCookie(&self.c_handle as *const bindings::ALACEncoder as *mut bindings::ALACEncoder, result.as_mut_ptr() as *mut std::ffi::c_void, &mut in_out_size);
            assert_eq!(size, in_out_size as usize);
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
        assert_eq!(encoder.initialize_encoder(&output_format), 0);

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
