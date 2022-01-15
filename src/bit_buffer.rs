use bitvec::{slice::BitSlice, order::Msb0, view::BitView, field::BitField};

pub struct BitBuffer<'a> {
    pub slice: &'a mut BitSlice<u8, Msb0>,
    pub position: usize,
}

impl<'a> BitBuffer<'a> {
    pub fn new(buffer: &'a mut [u8]) -> BitBuffer<'a> {
        BitBuffer { slice: buffer.view_bits_mut(), position: 0 }
    }

    pub fn len(&self) -> usize {
        self.slice.len() / 8
    }

    pub fn write(&mut self, bit_values: u32, num_bits: usize) {
        self.slice[self.position..(self.position + num_bits)].store_be(bit_values);
        self.position += num_bits;
    }

    /// Align bit buffer to next byte boundary, writing zeros if requested
    pub fn byte_align(&mut self) {
        let bit = self.position & 7;

        if bit == 0 { return; }

        self.write(0, 8 - bit);
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }
}
