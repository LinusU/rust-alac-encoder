mod bindings;

pub const DEFAULT_FRAMES_PER_PACKET: u32 = bindings::kALACDefaultFramesPerPacket;
pub const MAX_ESCAPE_HEADER_BYTES: usize = bindings::kALACMaxEscapeHeaderBytes as usize;

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

    pub fn get_magic_cookie_size(&self, num_channels: u32) -> usize {
        unsafe { bindings::ALACEncoder_GetMagicCookieSize(&self.c_handle as *const bindings::ALACEncoder as *mut bindings::ALACEncoder, num_channels) as usize }
    }

    pub fn get_magic_cookie(&self) -> Vec<u8> {
        let size = self.get_magic_cookie_size(self.c_handle.mNumChannels);
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
    use super::{AlacEncoder, FormatDescription, DEFAULT_FRAMES_PER_PACKET, MAX_ESCAPE_HEADER_BYTES};

    use std::fs;
    use std::io::Read;

    use byteorder::{LE, ReadBytesExt};

    #[test]
    fn it_works() {
        let mut encoder = AlacEncoder::new();
        let output_format = FormatDescription::alac(44100.0, DEFAULT_FRAMES_PER_PACKET, 2);

        encoder.initialize_encoder(&output_format);

        let cookie_size = encoder.get_magic_cookie_size(output_format.channels_per_frame);
        assert_eq!(cookie_size, 24);

        let cookie = encoder.get_magic_cookie();
        assert_eq!(cookie, vec![0, 0, 16, 0, 0, 16, 40, 10, 14, 2, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 172, 68]);
    }

    #[test]
    fn it_encodes_fixture() -> Result<(), Box<std::error::Error>> {
        let mut encoder = AlacEncoder::new();

        let input_format = FormatDescription::pcm::<i16>(44100.0, 2);
        let output_format = FormatDescription::alac(44100.0, 352, 2);

        let mut alac = fs::File::open("fixtures/sample.alac")?;
        let kuki = fs::read("fixtures/sample.kuki")?;
        let mut pakt = fs::File::open("fixtures/sample.pakt")?;
        let mut pcm = fs::File::open("fixtures/sample.pcm")?;

        let packet_count = (pakt.metadata().unwrap().len() as usize) / std::mem::size_of::<u32>();

        encoder.set_frame_size(352);
        encoder.initialize_encoder(&output_format);

        let cookie_size = encoder.get_magic_cookie_size(output_format.channels_per_frame);
        assert_eq!(cookie_size, kuki.len());

        let cookie = encoder.get_magic_cookie();
        assert_eq!(cookie, kuki);

        let mut input = vec![0u8; 1408];
        let mut output = vec![0u8; 1408 + MAX_ESCAPE_HEADER_BYTES];

        for _ in 1..packet_count {
            let packet_size = pakt.read_u32::<LE>()? as usize;

            // Read a chunk of PCM input
            pcm.read_exact(&mut input)?;

            // Convert that chunk to ALAC
            let size = encoder.encode(&input_format, &output_format, &input, &mut output)?;

            // Read the expected bytes
            let mut expected = vec![0u8; packet_size];
            alac.read_exact(&mut expected)?;

            // Compare encoded chunk to expected bytes
            assert_eq!(output[0..size], expected[0..packet_size]);
        }

        Ok(())
    }
}
