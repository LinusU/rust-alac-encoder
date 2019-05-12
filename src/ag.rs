//! Adaptive Golomb

use std::cmp;
use std::ptr;

use crate::bit_buffer::BitBuffer;

const QBSHIFT: u32 = 9;
const QB: u32 = (1 << QBSHIFT);

const MMULSHIFT: u32 = 2;
const MDENSHIFT: u32 = (QBSHIFT - MMULSHIFT - 1);
const MOFF: u32 = ((1<<(MDENSHIFT-2)));

const BITOFF: u32 = 24;

// Max. prefix of 1's.
const MAX_PREFIX_16: u32 = 9;
const MAX_PREFIX_32: u32 = 9;

// Max. bits in 16-bit data type
const MAX_DATATYPE_BITS_16: u32 = 16;

const N_MAX_MEAN_CLAMP:u32 = 0xffff;
const N_MEAN_CLAMP_VAL:u32 = 0xffff;

pub const PB0: u32 = 40;
pub const MB0: u32 = 10;
pub const KB0: u32 = 14;

pub struct AgParams {
    mb: u32,
    pb: u32,
    kb: u32,
    wb: u32,
    /// Full width
    fw: u32,
    /// Sector width
    sw: u32,
}

impl AgParams {
    pub fn new(m: u32, p: u32, k: u32, f: u32, s: u32) -> AgParams {
        AgParams {
            mb: m,
            pb: p,
            kb: k,
            wb: (1 << k) - 1,
            fw: f,
            sw: s,
        }
    }

    pub fn new_standard(fullwidth: u32, sectorwidth: u32) -> AgParams {
        AgParams::new(MB0, PB0, KB0, fullwidth, sectorwidth)
    }
}

#[inline(always)]
fn dyn_code(m: u32, k: u32, n: u32, out_num_bits: &mut u32) -> u32 {
    let division = n / m;

    let mut num_bits: u32;
    let mut value: u32;

    if division < MAX_PREFIX_16 {
        let modulo = n % m;
        let de = (modulo == 0) as u32;
        num_bits = division + k + 1 - de;
        value = (((1 << division) - 1) << (num_bits - division)) + modulo + 1 - de;

        // if coding this way is bigger than doing escape, then do escape
        if num_bits > (MAX_PREFIX_16 + MAX_DATATYPE_BITS_16) {
            num_bits = MAX_PREFIX_16 + MAX_DATATYPE_BITS_16;
            value = (((1 << MAX_PREFIX_16) - 1) << MAX_DATATYPE_BITS_16) + n;
        }
    } else {
        num_bits = MAX_PREFIX_16 + MAX_DATATYPE_BITS_16;
        value = (((1 << MAX_PREFIX_16) - 1) << MAX_DATATYPE_BITS_16) + n;
    }

    *out_num_bits = num_bits;

    return value;
}

#[inline(always)]
fn dyn_code_32bit(maxbits: usize, m: u32, k: u32, n: u32, out_num_bits: &mut u32, out_value: &mut u32, overflow: &mut u32, overflowbits: &mut u32) -> bool {
    let division = (n / m) as u32;

    let mut num_bits: u32;
    let mut value: u32;

    let mut did_overflow = false;

    if division < MAX_PREFIX_32 {
        let modulo: u32 = n - (m * division);
        let de = (modulo == 0) as u32;

        num_bits = division + k + 1 - de;
        value = (((1<<division)-1)<<(num_bits-division)) + modulo + 1 - de;

        if num_bits > 25 {
            num_bits = MAX_PREFIX_32;
            value = (1 << MAX_PREFIX_32) - 1;
            *overflow = n;
            *overflowbits = maxbits as u32;
            did_overflow = true;
        }
    } else {
        num_bits = MAX_PREFIX_32;
        value = (1 << MAX_PREFIX_32) - 1;
        *overflow = n;
        *overflowbits = maxbits as u32;
        did_overflow = true;
    }

    *out_num_bits = num_bits;
    *out_value = value;

    return did_overflow;
}

#[inline(always)]
fn dyn_jam_no_deref(out: *mut u8, bit_pos: u32, num_bits: u32, value: u32) {
    assert!(num_bits > 0 && num_bits <= 32);

    let target = unsafe { out.offset((bit_pos as isize) >> 3) as *mut u32 };
    let shift = 32 - (bit_pos & 7) - num_bits;

    let curr = unsafe { ptr::read_unaligned(target).to_be() };

    let mask = ((!0u32) >> (32 - num_bits)) << shift;
    let main = ((value << shift) & mask) | (curr & !mask);

    unsafe { ptr::write_unaligned(target, u32::from_be(main)); }
}

#[inline(always)]
fn dyn_jam_no_deref_large(out: *mut u8, bit_pos: u32, num_bits: u32, value: u32) {
    assert!(num_bits > 0 && num_bits <= 32);

    let target = unsafe { out.offset((bit_pos as isize) >> 3) as *mut u32 };
    let shift = (32 - (bit_pos & 7) - num_bits) as i32;

    let curr = unsafe { ptr::read_unaligned(target).to_be() };

    if shift < 0 {
        let mask = (!0u32) >> -shift;
        let main = (value >> -shift) | (curr & !mask);
        let tail = ((value << ((8 + shift))) & 0xff) as u8;

        unsafe { ptr::write_unaligned(target, u32::from_be(main)); }
        unsafe { ptr::write_unaligned(target.offset(1) as *mut u8, tail); }
    } else {
        let mask = ((!0u32) >> (32 - num_bits)) << shift;
        let main = ((value << shift) & mask) | (curr & !mask);

        unsafe { ptr::write_unaligned(target, u32::from_be(main)); }
    }
}

pub fn dyn_comp(params: &AgParams, pc: &[i32], bitstream: &mut BitBuffer, num_samples: usize, bit_size: usize) -> u32 {
    assert!(bit_size > 0 && bit_size <= 32);

    let out: *mut u8 = unsafe { bitstream.buffer.as_mut_ptr().offset(bitstream.position.byte as isize) };
    let start_pos: u32 = bitstream.position.bit;
    let mut bit_pos: u32 = start_pos;

    let mut mb: u32 = params.mb;
    let pb: u32 = params.pb;
    let kb: u32 = params.kb;
    let wb: u32 = params.wb;
    let mut zmode: u32 = 0;

    let mut row_pos = 0usize;
    let row_size = params.sw as usize;
    let row_jump = params.fw as usize - row_size;
    let mut in_ptr = 0usize;

    let mut c: u32 = 0;
    while c < (num_samples as u32) {
        let k = cmp::min(31 - ((mb >> QBSHIFT) + 3).leading_zeros(), kb);
        let m = (1 << k) - 1;

        let del = pc[in_ptr];
        in_ptr += 1;
        row_pos += 1;

        let n: u32 = ((del.abs() << 1) - ((del >> 31) & 1)) as u32 - zmode;
        assert!(32 - n.leading_zeros() <= bit_size as u32);

        {
            let mut num_bits: u32 = 0;
            let mut value: u32 = 0;
            let mut overflow: u32 = 0;
            let mut overflowbits: u32 = 0;

            if dyn_code_32bit(bit_size, m, k, n, &mut num_bits, &mut value, &mut overflow, &mut overflowbits) {
                dyn_jam_no_deref(out, bit_pos, num_bits, value);
                bit_pos += num_bits;
                dyn_jam_no_deref_large(out, bit_pos, overflowbits, overflow);
                bit_pos += overflowbits;
            } else {
                dyn_jam_no_deref(out, bit_pos, num_bits, value);
                bit_pos += num_bits;
            }
        }

        c += 1;
        if row_pos >= row_size {
            row_pos = 0;
            in_ptr += row_jump;
        }

        mb = pb * (n + zmode) + mb - ((pb * mb) >> QBSHIFT);

        // update mean tracking if it's overflowed
        if n > N_MAX_MEAN_CLAMP {
            mb = N_MEAN_CLAMP_VAL;
        }

        zmode = 0;

        assert!(c <= (num_samples as u32));

        if ((mb << MMULSHIFT) < QB) && (c < (num_samples as u32)) {
            zmode = 1;
            let mut nz = 0u32;

            while (c < (num_samples as u32)) && (pc[in_ptr] == 0) {
                // Take care of wrap-around globals.
                in_ptr += 1;
                nz += 1;
                c += 1;
                row_pos += 1;

                if row_pos >= row_size {
                    row_pos = 0;
                    in_ptr += row_jump;
                }

                if nz >= 65535 {
                    zmode = 0;
                    break;
                }
            }

            let k = mb.leading_zeros() - BITOFF + ((mb + MOFF) >> MDENSHIFT);
            let mz = ((1 << k) - 1) & wb;

            let mut num_bits: u32 = 0;
            let value: u32 = dyn_code(mz, k, nz, &mut num_bits);
            dyn_jam_no_deref(out, bit_pos, num_bits, value);
            bit_pos += num_bits;

            mb = 0;
        }
    }

    let out_num_bits = bit_pos - start_pos;
    bitstream.advance(out_num_bits);

    out_num_bits
}
