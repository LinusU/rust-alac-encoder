const AINIT: i32 = 38;
const BINIT: i32 = -29;
const CINIT: i32 = -2;

pub fn init_coefs(coefs: &mut [i16], denshift: u32) {
    let den = 1i32 << denshift;

    coefs[0] = ((AINIT * den) >> 4) as i16;
    coefs[1] = ((BINIT * den) >> 4) as i16;
    coefs[2] = ((CINIT * den) >> 4) as i16;

    for index in 3..coefs.len() {
        coefs[index] = 0;
    }
}
