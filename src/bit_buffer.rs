use std::cmp;

#[derive(Clone, Copy)]
pub struct Position {
    pub byte: usize,
    pub bit: u32,
}

pub struct BitBuffer<'a> {
    pub buffer: &'a mut [u8],
    pub position: Position,
}

impl<'a> BitBuffer<'a> {
    pub fn new(buffer: &'a mut [u8]) -> BitBuffer<'a> {
        BitBuffer { buffer, position: Position { byte: 0, bit: 0 } }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn write(&mut self, bit_values: u32, num_bits: u32) {
        if num_bits == 0 { return; }

        let mut inv_bit_index = 8 - self.position.bit;
        let mut bits_left = num_bits;

        while bits_left > 0 {
            let cur_num = cmp::min(inv_bit_index, bits_left);
            let tmp = (bit_values >> (bits_left - cur_num)) as u8;

            let shift = (inv_bit_index - cur_num) as u8;
            let mask = (0xffu8 >> (8 - cur_num)) << shift;

            // modify current byte
            self.buffer[self.position.byte] = (self.buffer[self.position.byte] & !mask) | ((tmp << shift) & mask);
            bits_left -= cur_num;

            // increment to next byte if need be
            inv_bit_index -= cur_num;
            if inv_bit_index == 0 {
                inv_bit_index = 8;
                self.position.byte += 1;
            }
        }

        self.position.bit = 8 - inv_bit_index;
    }

    /// Align bit buffer to next byte boundary, writing zeros if requested
    pub fn byte_align(&mut self, add_zeros: bool) {
        if self.position.bit == 0 { return; }

        match add_zeros {
            true => self.write(0, 8 - self.position.bit),
            false => self.advance(8 - self.position.bit),
        }
    }

    pub fn get_position(&self) -> usize {
        (self.position.byte * 8) + (self.position.bit as usize)
    }

    pub fn advance(&mut self, num_bits: u32) {
        if num_bits > 0 {
            self.position.bit += num_bits;
            self.position.byte += (self.position.bit as usize) >> 3;
            self.position.bit &= 7;
        }
    }

    pub fn save_position(&self) -> Position {
        self.position
    }

    pub fn load_position(&mut self, position: Position) {
        self.position = position;
    }
}
