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
