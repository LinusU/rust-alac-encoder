//! ALAC mixing/matrixing routines to/from 32-bit predictor buffers.

// There is no plain middle-side option; instead there are various mixing
// modes including middle-side, each lossless, as embodied in the mix()
// and unmix() functions.  These functions exploit a generalized middle-side
// transformation:
//
// u := [(rL + (m-r)R)/m];
// v := L - R;
//
// where [ ] denotes integer floor.  The (lossless) inverse is
//
// L = u + v - [rV/m];
// R = L - v;

pub struct Source<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub num_samples: usize,
}

pub fn mix16(input: Source, u: &mut [i32], v: &mut [i32], mixbits: i32, mixres: i32) {
    if mixres != 0 {
        /* matrixed stereo */
        let modulo: i32 = 1 << mixbits;
        let m2: i32 = modulo - mixres;

        for j in 0..input.num_samples {
            let l = i16::from_le_bytes([input.data[j * input.stride * 2], input.data[j * input.stride * 2 + 1]]) as i32;
            let r = i16::from_le_bytes([input.data[j * input.stride * 2 + 2], input.data[j * input.stride * 2 + 3]]) as i32;

            u[j] = (mixres * l + m2 * r) >> mixbits;
            v[j] = l - r;
        }
    } else {
        /* Conventional separated stereo. */
        for j in 0..input.num_samples {
            u[j] = i16::from_le_bytes([input.data[j * input.stride * 2], input.data[j * input.stride * 2 + 1]]) as i32;
            v[j] = i16::from_le_bytes([input.data[j * input.stride * 2 + 2], input.data[j * input.stride * 2 + 3]]) as i32;
        }
    }
}

pub fn mix20(input: Source, u: &mut [i32], v: &mut [i32], mixbits: i32, mixres: i32) {
    if mixres != 0 {
        /* matrixed stereo */
        let modulo: i32 = 1 << mixbits;
        let m2: i32 = modulo - mixres;

        for j in 0..input.num_samples {
            let l = (i32::from_be_bytes([0, input.data[j * input.stride * 3], input.data[j * input.stride * 3 + 1], input.data[j * input.stride * 3 + 2]]) << 8) >> 12;
            let r = (i32::from_be_bytes([0, input.data[j * input.stride * 3 + 3], input.data[j * input.stride * 3 + 4], input.data[j * input.stride * 3 + 5]]) << 8) >> 12;

            u[j] = (mixres * l + m2 * r) >> mixbits;
            v[j] = l - r;
        }
    } else {
        /* Conventional separated stereo. */
        for j in 0..input.num_samples {
            u[j] = (i32::from_be_bytes([0, input.data[j * input.stride * 3], input.data[j * input.stride * 3 + 1], input.data[j * input.stride * 3 + 2]]) << 8) >> 12;
            v[j] = (i32::from_be_bytes([0, input.data[j * input.stride * 3 + 3], input.data[j * input.stride * 3 + 4], input.data[j * input.stride * 3 + 5]]) << 8) >> 12;
        }
    }
}

// 24-bit routines
// - 24-bit data sometimes compresses better by shifting off the bottom byte so these routines deal with
//	 the specified "unused lower bytes" in the combined "shift" buffer
pub fn mix24(input: Source, u: &mut [i32], v: &mut [i32], mixbits: i32, mixres: i32, shift_uv: &mut [u16], bytes_shifted: u8) {
    debug_assert!(bytes_shifted <= 2);

    let shift = bytes_shifted * 8;
    let mask = (1u32 << shift) - 1;

    if mixres != 0 {
        /* matrixed stereo */
        let modulo = 1 << mixbits;
        let m2 = modulo - mixres;

        if bytes_shifted != 0 {
            for j in 0..input.num_samples {
                let l = (i32::from_be_bytes([0, input.data[j * input.stride * 3], input.data[j * input.stride * 3 + 1], input.data[j * input.stride * 3 + 2]]) << 8) >> 8;
                let r = (i32::from_be_bytes([0, input.data[j * input.stride * 3 + 3], input.data[j * input.stride * 3 + 4], input.data[j * input.stride * 3 + 5]]) << 8) >> 8;

                shift_uv[j * 2] = ((l as u32) & mask) as u16;
                shift_uv[j * 2 + 1] = ((r as u32) & mask) as u16;

                let l = l >> shift;
                let r = r >> shift;

                u[j] = (mixres * l + m2 * r) >> mixbits;
                v[j] = l - r;
            }
        } else {
            for j in 0..input.num_samples {
                let l = (i32::from_be_bytes([0, input.data[j * input.stride * 3], input.data[j * input.stride * 3 + 1], input.data[j * input.stride * 3 + 2]]) << 8) >> 8;
                let r = (i32::from_be_bytes([0, input.data[j * input.stride * 3 + 3], input.data[j * input.stride * 3 + 4], input.data[j * input.stride * 3 + 5]]) << 8) >> 8;

                u[j] = (mixres * l + m2 * r) >> mixbits;
                v[j] = l - r;
            }
        }
    } else {
        /* Conventional separated stereo. */
        if bytes_shifted != 0 {
            for j in 0..input.num_samples {
                let l = (i32::from_be_bytes([0, input.data[j * input.stride * 3], input.data[j * input.stride * 3 + 1], input.data[j * input.stride * 3 + 2]]) << 8) >> 8;
                let r = (i32::from_be_bytes([0, input.data[j * input.stride * 3 + 3], input.data[j * input.stride * 3 + 4], input.data[j * input.stride * 3 + 5]]) << 8) >> 8;

                shift_uv[j * 2] = ((l as u32) & mask) as u16;
                shift_uv[j * 2 + 1] = ((r as u32) & mask) as u16;

                u[j] = l >> shift;
                v[j] = r >> shift;
            }
        } else {
            for j in 0..input.num_samples {
                u[j] = (i32::from_be_bytes([0, input.data[j * input.stride * 3], input.data[j * input.stride * 3 + 1], input.data[j * input.stride * 3 + 2]]) << 8) >> 8;
                v[j] = (i32::from_be_bytes([0, input.data[j * input.stride * 3 + 3], input.data[j * input.stride * 3 + 4], input.data[j * input.stride * 3 + 5]]) << 8) >> 8;
            }
        }
    }
}

// 32-bit routines
// - note that these really expect the internal data width to be < 32-bit but the arrays are 32-bit
// - otherwise, the calculations might overflow into the 33rd bit and be lost
// - therefore, these routines deal with the specified "unused lower" bytes in the combined "shift" buffer
pub fn mix32(input: Source, u: &mut [i32], v: &mut [i32], mixbits: i32, mixres: i32, shift_uv: &mut [u16], bytes_shifted: u8) {
    debug_assert!(bytes_shifted <= 2);

    let shift = bytes_shifted * 8;
    let mask = (1u32 << shift) - 1;

    if mixres != 0 {
        debug_assert!(bytes_shifted != 0);

        /* matrixed stereo with shift */
        let modulo = 1 << mixbits;
        let m2 = modulo - mixres;

        for j in 0..input.num_samples {
            let l = i32::from_le_bytes([input.data[j * input.stride * 4], input.data[j * input.stride * 4 + 1], input.data[j * input.stride * 4 + 2], input.data[j * input.stride * 4 + 3]]);
            let r = i32::from_le_bytes([input.data[j * input.stride * 4 + 4], input.data[j * input.stride * 4 + 5], input.data[j * input.stride * 4 + 6], input.data[j * input.stride * 4 + 7]]);

            shift_uv[j * 2] = ((l as u32) & mask) as u16;
            shift_uv[j * 2 + 1] = ((r as u32) & mask) as u16;

            let l = l >> shift;
            let r = r >> shift;

            u[j] = (mixres * l + m2 * r) >> mixbits;
            v[j] = l - r;
        }
    } else if bytes_shifted != 0 {
        /* de-interleaving with shift */
        for j in 0..input.num_samples {
            let l = i32::from_le_bytes([input.data[j * input.stride * 4], input.data[j * input.stride * 4 + 1], input.data[j * input.stride * 4 + 2], input.data[j * input.stride * 4 + 3]]);
            let r = i32::from_le_bytes([input.data[j * input.stride * 4 + 4], input.data[j * input.stride * 4 + 5], input.data[j * input.stride * 4 + 6], input.data[j * input.stride * 4 + 7]]);

            shift_uv[j * 2] = ((l as u32) & mask) as u16;
            shift_uv[j * 2 + 1] = ((r as u32) & mask) as u16;

            u[j] = l >> shift;
            v[j] = r >> shift;
        }
    } else {
        /* de-interleaving w/o shift */
        for j in 0..input.num_samples {
            u[j] = i32::from_le_bytes([input.data[j * input.stride * 4], input.data[j * input.stride * 4 + 1], input.data[j * input.stride * 4 + 2], input.data[j * input.stride * 4 + 3]]);
            v[j] = i32::from_le_bytes([input.data[j * input.stride * 4 + 4], input.data[j * input.stride * 4 + 5], input.data[j * input.stride * 4 + 6], input.data[j * input.stride * 4 + 7]]);
        }
    }
}

// 20/24-bit <-> 32-bit helper routines (not really matrixing but convenient to put here)

pub fn copy20_to_predictor(input: &[u8], stride: usize, output: &mut [i32], num_samples: usize) {
    for j in 0..num_samples {
        // 20-bit values are left-aligned in the 24-bit input buffer but right-aligned in the 32-bit output buffer
        let val = i32::from_be_bytes([0, input[j * stride * 3], input[j * stride * 3 + 1], input[j * stride * 3 + 2]]);
        output[j] = (val << 8) >> 12;
    }
}

pub fn copy24_to_predictor(input: &[u8], stride: usize, output: &mut [i32], num_samples: usize) {
    for j in 0..num_samples {
        let val = i32::from_be_bytes([0, input[j * stride * 3], input[j * stride * 3 + 1], input[j * stride * 3 + 2]]);
        output[j] = (val << 8) >> 8;
    }
}
