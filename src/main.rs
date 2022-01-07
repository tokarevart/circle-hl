use std::{
    f64::consts::*, 
    ops::{Deref, DerefMut}
};

const SQRT_2_PI: f64 = 2.506628274631;

#[derive(Debug, Clone, Copy)]
struct PsiR {
    m: f64,
    s: f64,
}

impl PsiR {
    fn at(&self, r: f64) -> f64 {
        let (m, s) = (self.m, self.s);
        let r_minus_m = r - m;
        (-r_minus_m * r_minus_m / (2.0 * s * s)).exp() / (s * SQRT_2_PI)
    }

    fn at_with_scale(&self, a: f64, r: f64) -> f64 {
        let (m, s) = (self.m, self.s);
        let ia = 1.0 / a;
        let r_minus_m = r - m;
        ia * (-ia * r_minus_m * r_minus_m / (2.0 * s * s)).exp() / (s * SQRT_2_PI)
    }
}

#[derive(Debug, Clone, Copy)]
struct Psi(PsiR);

impl Psi {
    fn at(&self, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        self.0.at(r)
    }

    fn at_with_scale(&self, a: f64, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        self.0.at_with_scale(a, r)
    }
}

impl Deref for Psi {
    type Target = PsiR;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Psi {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn significant_points(psi: Psi) -> Vec<(i64, i64)> {
    let min_sqr_r = psi.m - 3.0 * psi.s;
    let max_sqr_r = psi.m + 3.0 * psi.s;
    let end = max_sqr_r.ceil() as i64 + 1;
    let beg = 1 - end;
    let max_size = (end - beg) * (end - beg);
    let mut res = Vec::with_capacity(max_size as usize);
    for x in beg..end {
        for y in beg..end {
            let sqr_r = (x * x + y * y) as f64;
            if sqr_r > min_sqr_r && sqr_r < max_sqr_r {
                res.push((x, y));
            }
        }
    }
    res
}

fn significant_points_with_scale(psi: Psi, a: f64) -> Vec<(i64, i64)> {
    let min_sqr_r = a * (psi.m - 3.0 * psi.s);
    let max_sqr_r = a * (psi.m + 3.0 * psi.s);
    let end = max_sqr_r.ceil() as i64 + 1;
    let beg = 1 - end;
    let max_size = (end - beg) * (end - beg);
    let mut res = Vec::with_capacity(max_size as usize);
    for x in beg..end {
        for y in beg..end {
            let sqr_r = (x * x + y * y) as f64;
            if sqr_r > min_sqr_r && sqr_r < max_sqr_r {
                res.push((x, y));
            }
        }
    }
    res
}

fn wavelet_transform_at_inner(
    f: &Vec<Vec<f64>>, psi: Psi, sigpts: &Vec<(i64, i64)>, bx: usize, by: usize
) -> f64 {
    sigpts.iter().map(
        |&(wx, wy)|
            f[bx + wx as usize][by + wy as usize] * psi.at(wx as f64, wy as f64)
    ).sum()
}

fn wavelet_transform_at_outer(
    f: &Vec<Vec<f64>>, psi: Psi, sigpts: &Vec<(i64, i64)>, bx: usize, by: usize
) -> f64 {
    let xlen = f.len();
    let ylen = f[0].len();

    let mut inside_count = 0;
    sigpts.iter().map(
        |&(wx, wy)| {
            let bwx = bx as i64 + wx;
            let bwy = by as i64 + wy;
            if bwx >= 0 && bwx < xlen as i64 && bwy >= 0 && bwy < ylen as i64 {
                inside_count += 1;
                f[bwx as usize][bwy as usize] * psi.at(wx as f64, wy as f64)
            } else {
                0.0
            }
        }
    ).sum::<f64>() / inside_count as f64 * sigpts.len() as f64
}

fn wavelet_transform(f: &Vec<Vec<f64>>, psi: Psi) -> Vec<Vec<f64>> {
    let sigpts = significant_points(psi);
    let xlen = f.len();
    let ylen = f[0].len();
    let mut res = vec![vec![-1.0; ylen]; xlen];

    let inner_xbeg = (psi.m + 3.0 * psi.s).ceil() as usize;
    let inner_xend = xlen - inner_xbeg;
    let inner_ybeg = inner_xbeg;
    let inner_yend = ylen - inner_ybeg;
    for bx in inner_xbeg..inner_xend {
        for by in inner_ybeg..inner_yend {
            res[bx][by] = wavelet_transform_at_inner(f, psi, &sigpts, bx, by);
        }
    }

    for by in 0..ylen {
        for bx in 0..inner_xbeg {
            res[bx][by] = wavelet_transform_at_outer(f, psi, &sigpts, bx, by);
        }
        for bx in inner_xend..xlen {
            res[bx][by] = wavelet_transform_at_outer(f, psi, &sigpts, bx, by);
        }
    }
    for bx in inner_xbeg..inner_xend {
        for by in 0..inner_ybeg {
            res[bx][by] = wavelet_transform_at_outer(f, psi, &sigpts, bx, by);
        }
        for by in inner_yend..ylen {
            res[bx][by] = wavelet_transform_at_outer(f, psi, &sigpts, bx, by);
        }
    }

    res
}

fn main() {
    println!("Hello, world!");
}
