use std::num::Wrapping;

const AINIT: i32 = 38;
const BINIT: i32 = -29;
const CINIT: i32 = -2;

#[allow(dead_code)]
#[repr(i32)]
enum Sign {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}

trait Signed {
    fn sign(&self) -> Sign;
}

impl Signed for i32 {
    fn sign(&self) -> Sign {
        unsafe { std::mem::transmute(self.signum()) }
    }
}

pub fn init_coefs(coefs: &mut [i16], denshift: u32) {
    let den = 1i32 << denshift;

    coefs[0] = ((AINIT * den) >> 4) as i16;
    coefs[1] = ((BINIT * den) >> 4) as i16;
    coefs[2] = ((CINIT * den) >> 4) as i16;

    for index in 3..coefs.len() {
        coefs[index] = 0;
    }
}

pub fn pc_block(input: &[i32], pc1: &mut [i32], num: usize, coefs: &mut [i16], numactive: usize, chanbits: usize, denshift: u32) {
    let chanshift: u32 = 32 - (chanbits as u32);
    let denhalf: i32 = 1 << (denshift - 1);

    // just copy if numactive == 0
    if numactive == 0 {
        pc1.copy_from_slice(input);
        return;
    }

    pc1[0] = input[0];
    if numactive == 31 {
        // short-circuit if numactive == 31
        for j in 1..num {
            let del = input[j] - input[j-1];
            pc1[j] = (del << chanshift) >> chanshift;
        }
        return;
    }

    for j in 1..=numactive {
        let del = input[j] - input[j-1];
        pc1[j] = (del << chanshift) >> chanshift;
    }

    let lim = numactive + 1;

    match numactive {
        4 => {
            // optimization for numactive == 4
            let mut a0 = coefs[0];
            let mut a1 = coefs[1];
            let mut a2 = coefs[2];
            let mut a3 = coefs[3];

            for j in lim..num {
                let top = input[j - lim];

                let b0 = top - input[j - 1];
                let b1 = top - input[j - 2];
                let b2 = top - input[j - 3];
                let b3 = top - input[j - 4];

                let sum1: i32 = (Wrapping(denhalf) - Wrapping(a0 as i32) * Wrapping(b0) - Wrapping(a1 as i32) * Wrapping(b1) - Wrapping(a2 as i32) * Wrapping(b2) - Wrapping(a3 as i32) * Wrapping(b3)).0 >> denshift;
                let mut del = ((input[j] - top - sum1) << chanshift) >> chanshift;

                pc1[j] = del;

                match del.sign() {
                    Sign::Positive => {
                        let sgn = b3.signum();
                        a3 -= sgn as i16;
                        del -= (4 - 3) * ((sgn * b3) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b2.signum();
                        a2 -= sgn as i16;
                        del -= (4 - 2) * ((sgn * b2) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b1.signum();
                        a1 -= sgn as i16;
                        del -= (4 - 1) * ((sgn * b1) >> denshift);
                        if del <= 0 { continue; }

                        a0 -= b0.signum() as i16;
                    },
                    Sign::Negative => {
                        // note: to avoid unnecessary negations, we flip the value of "sgn"
                        let sgn = -(b3.signum());
                        a3 -= sgn as i16;
                        del -= (4 - 3) * ((sgn * b3) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b2.signum());
                        a2 -= sgn as i16;
                        del -= (4 - 2) * ((sgn * b2) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b1.signum());
                        a1 -= sgn as i16;
                        del -= (4 - 1) * ((sgn * b1) >> denshift);
                        if del >= 0 { continue; }

                        a0 += b0.signum() as i16;
                    },
                    Sign::Zero => {},
                }
            }

            coefs[0] = a0;
            coefs[1] = a1;
            coefs[2] = a2;
            coefs[3] = a3;
        },
        8 => {
            // optimization for numactive == 8
            let mut a0 = coefs[0];
            let mut a1 = coefs[1];
            let mut a2 = coefs[2];
            let mut a3 = coefs[3];
            let mut a4 = coefs[4];
            let mut a5 = coefs[5];
            let mut a6 = coefs[6];
            let mut a7 = coefs[7];

            for j in lim..num {
                let top = input[j - lim];

                let b0 = top - input[j - 1];
                let b1 = top - input[j - 2];
                let b2 = top - input[j - 3];
                let b3 = top - input[j - 4];
                let b4 = top - input[j - 5];
                let b5 = top - input[j - 6];
                let b6 = top - input[j - 7];
                let b7 = top - input[j - 8];

                let sum1: i32 = (Wrapping(denhalf) - Wrapping(a0 as i32) * Wrapping(b0) - Wrapping(a1 as i32) * Wrapping(b1) - Wrapping(a2 as i32) * Wrapping(b2) - Wrapping(a3 as i32) * Wrapping(b3) - Wrapping(a4 as i32) * Wrapping(b4) - Wrapping(a5 as i32) * Wrapping(b5) - Wrapping(a6 as i32) * Wrapping(b6) - Wrapping(a7 as i32) * Wrapping(b7)).0 >> denshift;
                let mut del = ((input[j] - top - sum1) << chanshift) >> chanshift;

                pc1[j] = del;

                match del.sign() {
                    Sign::Positive => {
                        let sgn = b7.signum();
                        a7 -= sgn as i16;
                        del -= 1 * ((sgn * b7) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b6.signum();
                        a6 -= sgn as i16;
                        del -= 2 * ((sgn * b6) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b5.signum();
                        a5 -= sgn as i16;
                        del -= 3 * ((sgn * b5) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b4.signum();
                        a4 -= sgn as i16;
                        del -= 4 * ((sgn * b4) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b3.signum();
                        a3 -= sgn as i16;
                        del -= 5 * ((sgn * b3) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b2.signum();
                        a2 -= sgn as i16;
                        del -= 6 * ((sgn * b2) >> denshift);
                        if del <= 0 { continue; }

                        let sgn = b1.signum();
                        a1 -= sgn as i16;
                        del -= 7 * ((sgn * b1) >> denshift);
                        if del <= 0 { continue; }

                        a0 -= b0.signum() as i16;
                    },
                    Sign::Negative => {
                        // note: to avoid unnecessary negations, we flip the value of "sgn"
                        let sgn = -(b7.signum());
                        a7 -= sgn as i16;
                        del -= 1 * ((sgn * b7) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b6.signum());
                        a6 -= sgn as i16;
                        del -= 2 * ((sgn * b6) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b5.signum());
                        a5 -= sgn as i16;
                        del -= 3 * ((sgn * b5) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b4.signum());
                        a4 -= sgn as i16;
                        del -= 4 * ((sgn * b4) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b3.signum());
                        a3 -= sgn as i16;
                        del -= 5 * ((sgn * b3) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b2.signum());
                        a2 -= sgn as i16;
                        del -= 6 * ((sgn * b2) >> denshift);
                        if del >= 0 { continue; }

                        let sgn = -(b1.signum());
                        a1 -= sgn as i16;
                        del -= 7 * ((sgn * b1) >> denshift);
                        if del >= 0 { continue; }

                        a0 += b0.signum() as i16;
                    },
                    Sign::Zero => {},
                }
            }

            coefs[0] = a0;
            coefs[1] = a1;
            coefs[2] = a2;
            coefs[3] = a3;
            coefs[4] = a4;
            coefs[5] = a5;
            coefs[6] = a6;
            coefs[7] = a7;
        },
        _ => {
            // general case
            for j in lim..num {
                let top = input[j - lim];

                let mut sum1: i32 = 0;
                for k in 0..numactive {
                    sum1 -= (coefs[k] as i32) * (top - input[j - 1 - k]);
                }

                let mut del = ((input[j] - top - ((sum1 + denhalf) >> denshift)) << chanshift) >> chanshift;

                pc1[j] = del;

                match del.sign() {
                    Sign::Positive => {
                        for k in (0..numactive).rev() {
                            let dd = top - input[j - 1 - k];
                            let sgn = dd.signum();
                            coefs[k] -= sgn as i16;
                            del -= (numactive - k) as i32 * ((sgn * dd) >> denshift);
                            if del <= 0 { break; }
                        }
                    },
                    Sign::Negative => {
                        for k in (0..numactive).rev() {
                            let dd = top - input[j - 1 - k];
                            let sgn = dd.signum();
                            coefs[k] += sgn as i16;
                            del -= (numactive - k) as i32 * ((-sgn * dd) >> denshift);
                            if del >= 0 { break; }
                        }
                    },
                    Sign::Zero => {}
                }
            }
        },
    }
}
