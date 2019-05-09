use crate::bindings;

pub struct BitBuffer<'a> {
    pub c_handle: bindings::BitBuffer,

    buffer: &'a mut [u8],
}

impl<'a> BitBuffer<'a> {
    pub fn new(buffer: &'a mut [u8]) -> BitBuffer<'a> {
        let mut c_handle: bindings::BitBuffer = unsafe { std::mem::uninitialized() };

        unsafe { bindings::BitBufferInit(&mut c_handle, buffer.as_mut_ptr(), buffer.len() as u32) };

        BitBuffer { c_handle, buffer }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn write(&mut self, bit_values: u32, num_bits: u32) {
        unsafe { bindings::BitBufferWrite(&mut self.c_handle, bit_values, num_bits) };
    }

    pub fn byte_align(&mut self, add_zeros: bool) {
        unsafe { bindings::BitBufferByteAlign(&mut self.c_handle, add_zeros as i32) };
    }

    pub fn get_position(&self) -> usize {
        unsafe { bindings::BitBufferGetPosition(&self.c_handle as *const bindings::BitBuffer as *mut bindings::BitBuffer) as usize }
    }

    pub fn save_state(&self) -> bindings::BitBuffer {
        self.c_handle
    }

    pub fn load_state(&mut self, state: bindings::BitBuffer) {
        self.c_handle = state;
    }
}
