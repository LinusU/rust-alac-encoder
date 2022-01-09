pub struct BitBuffer<'a> {
    pub buffer: &'a mut [u8],
    pub position: usize,
}

impl<'a> BitBuffer<'a> {
    pub fn new(buffer: &'a mut [u8]) -> BitBuffer<'a> {
        BitBuffer { buffer, position: 0 }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn write_lte25(&mut self, bit_values: u32, num_bits: u32) {
        assert!(num_bits > 0 && num_bits <= 25);

        let target = unsafe { self.buffer.as_mut_ptr().add(self.position >> 3) as *mut u32 };
        let shift = 32 - ((self.position as u32) & 7) - num_bits;

        let curr = u32::from_be(unsafe { core::ptr::read_unaligned(target) });

        let mask = ((!0u32) >> (32 - num_bits)) << shift;
        let main = ((bit_values << shift) & mask) | (curr & !mask);

        unsafe { core::ptr::write_unaligned(target, main.to_be()); }

        self.position += num_bits as usize;
    }

    pub fn write(&mut self, bit_values: u32, num_bits: u32) {
        assert!(num_bits > 0 && num_bits <= 32);

        let target = unsafe { self.buffer.as_mut_ptr().add(self.position >> 3) as *mut u32 };
        let shift = (32 - ((self.position as i32) & 7) - (num_bits as i32)) as i32;

        let curr = u32::from_be(unsafe { core::ptr::read_unaligned(target) });

        if shift < 0 {
            let mask = (!0u32) >> -shift;
            let main = (bit_values >> -shift) | (curr & !mask);
            let tail = ((bit_values << (8 + shift)) & 0xff) as u8;

            unsafe { core::ptr::write_unaligned(target, main.to_be()); }
            unsafe { core::ptr::write_unaligned(target.offset(1) as *mut u8, tail); }
        } else {
            let mask = ((!0u32) >> (32 - num_bits)) << shift;
            let main = ((bit_values << shift) & mask) | (curr & !mask);

            unsafe { core::ptr::write_unaligned(target, main.to_be()); }
        }

        self.position += num_bits as usize;
    }

    /// Align bit buffer to next byte boundary, writing zeros if requested
    pub fn byte_align(&mut self) {
        let bit = (self.position & 7) as u32;

        if bit == 0 { return; }

        self.write_lte25(0, 8 - bit);
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }
}
