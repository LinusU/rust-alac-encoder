//! Adaptive Golomb

use crate::bindings;

const QBSHIFT: u32 = 9;
const QB: u32 = (1 << QBSHIFT);

const MAX_RUN_DEFAULT: u32 = 255;

pub const PB0: u32 = 40;
pub const MB0: u32 = 10;
pub const KB0: u32 = 14;

pub struct AgParams {
    c_handle: bindings::AGParamRec,
}

impl AgParams {
    pub fn new(m: u32, p: u32, k: u32, f: u32, s: u32, maxrun: u32) -> AgParams {
        AgParams {
            c_handle: bindings::AGParamRec {
                mb: m,
                mb0: m,
                pb: p,
                kb: k,
                wb: (1 << k) - 1,
                qb: QB - p,
                fw: f,
                sw: s,
                maxrun: maxrun,
            }
        }
    }

    pub fn new_standard(fullwidth: u32, sectorwidth: u32) -> AgParams {
        AgParams::new(MB0, PB0, KB0, fullwidth, sectorwidth, MAX_RUN_DEFAULT)
    }

    pub fn c_handle(&self) -> *mut bindings::AGParamRec {
        &self.c_handle as *const bindings::AGParamRec as *mut bindings::AGParamRec
    }
}
