use std::{
    ops::{Deref, DerefMut}, 
    fs,
    fs::File, 
    io::Write
};

const SQRT_2_PI: f64 = 2.506628274631;

#[derive(Debug, Clone, Copy)]
struct Gaussian {
    m: f64,
    s: f64,
}

impl Gaussian {
    fn at(&self, r: f64) -> f64 {
        let (m, s) = (self.m, self.s);
        let r_minus_m = r - m;
        (-r_minus_m * r_minus_m / (2.0 * s * s)).exp() / (s * SQRT_2_PI)
    }
}

trait RadialWavelet: Clone + Copy {
    fn radius(&self) -> f64;
    fn thickness(&self) -> f64;
    fn min_significant_radius(&self) -> f64;
    fn max_significant_radius(&self) -> f64;

    fn at(&self, x: f64, y: f64) -> f64;
    fn ati(&self, x: i64, y: i64) -> f64;
    fn prod_at_sigpt(&self, f: f64, x: i64, y: i64) -> f64;

    fn significant_points(&self) -> Vec<(i64, i64)> {
        let sqr_min_sr = self.min_significant_radius().powi(2);
        let sqr_max_sr = self.max_significant_radius().powi(2);
        let end = sqr_max_sr.sqrt().ceil() as i64 + 1;
        let beg = 1 - end;
        let max_size = (end - beg) * (end - beg);
        let mut res = Vec::with_capacity(max_size as usize);
        for x in beg..end {
            for y in beg..end {
                let sqr_r = (x * x + y * y) as f64;
                if sqr_r >= sqr_min_sr && sqr_r <= sqr_max_sr {
                    res.push((x, y));
                }
            }
        }
        res
    }
}

#[derive(Debug, Clone, Copy)]
struct RadialGaussian {
    gaussian: Gaussian,
    min_sr: f64,
    max_sr: f64,
}

impl RadialGaussian {
    fn new(m: f64, s: f64) -> Self {
        let min_sr = (m - 3.0 * s).max(0.0);
        let max_sr = m + 3.0 * s;
        Self {
            gaussian: Gaussian{ m, s },
            min_sr, max_sr
        }
    }
}

impl Deref for RadialGaussian {
    type Target = Gaussian;

    fn deref(&self) -> &Self::Target {
        &self.gaussian
    }
}

impl DerefMut for RadialGaussian {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.gaussian
    }
}

impl RadialWavelet for RadialGaussian {
    fn radius(&self) -> f64 {
        self.m
    }

    fn thickness(&self) -> f64 {
        3.0 * self.s
    }

    fn at(&self, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        self.gaussian.at(r)
    }

    fn ati(&self, x: i64, y: i64) -> f64 {
        let sqr_r = (x * x + y * y) as f64;
        self.gaussian.at(sqr_r.sqrt())
    }

    fn prod_at_sigpt(&self, f: f64, x: i64, y: i64) -> f64 {
        f * self.ati(x, y)
    }

    fn min_significant_radius(&self) -> f64 {
        self.min_sr
    }

    fn max_significant_radius(&self) -> f64 {
        self.max_sr
    }
}

#[derive(Debug, Clone, Copy)]
struct RadialHeaviside {
    radius: f64,
    thickness: f64,
    min_sr: f64,
    max_sr: f64,
}

impl RadialHeaviside {
    fn new(radius: f64, thickness: f64) -> Self {
        let min_sr = (radius - thickness).max(0.0);
        let max_sr = radius + thickness;
        Self {
            radius, thickness,
            min_sr, max_sr
        }
    }
}

impl RadialWavelet for RadialHeaviside {
    fn radius(&self) -> f64 {
        self.radius
    }

    fn thickness(&self) -> f64 {
        self.thickness
    }

    fn at(&self, x: f64, y: f64) -> f64 {
        let r = (x * x + y * y).sqrt();
        if r < self.min_sr || r > self.max_sr {
            0.0
        } else {
            1.0
        }
    }

    fn ati(&self, x: i64, y: i64) -> f64 {
        let r = ((x * x + y * y) as f64).sqrt();
        if r < self.min_sr || r > self.max_sr {
            0.0
        } else {
            1.0
        }
    }

    fn prod_at_sigpt(&self, f: f64, x: i64, y: i64) -> f64 {
        f
    }

    fn min_significant_radius(&self) -> f64 {
        self.min_sr
    }

    fn max_significant_radius(&self) -> f64 {
        self.max_sr
    }
}

// fn significant_points_with_scale(psi: Psi, a: f64) -> Vec<(i64, i64)> {
//     let min_r = a * (psi.m - 3.0 * psi.s).max(0.0);
//     let max_r = a * (psi.m + 3.0 * psi.s);
//     let sqr_min_r = min_r * min_r;
//     let sqr_max_r = max_r * max_r;
//     let end = max_r.ceil() as i64 + 1;
//     let beg = 1 - end;
//     let max_size = (end - beg) * (end - beg);
//     let mut res = Vec::with_capacity(max_size as usize);
//     for x in beg..end {
//         for y in beg..end {
//             let sqr_r = (x * x + y * y) as f64;
//             if sqr_r >= sqr_min_r && sqr_r <= sqr_max_r {
//                 res.push((x, y));
//             }
//         }
//     }
//     res
// }

fn wavelet_transform_at_inner(
    f: &Vec<Vec<f64>>, wavelet: impl RadialWavelet, sigpts: &Vec<(i64, i64)>, bx: usize, by: usize
) -> f64 {
    sigpts.iter().map(
        |&(wx, wy)| {
            let bwx = bx as i64 + wx;
            let bwy = by as i64 + wy;
            wavelet.prod_at_sigpt(f[bwx as usize][bwy as usize], wx, wy)
        }
    ).sum()
}

fn wavelet_transform_at_outer(
    f: &Vec<Vec<f64>>, wavelet: impl RadialWavelet, sigpts: &Vec<(i64, i64)>, bx: usize, by: usize
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
                wavelet.prod_at_sigpt(f[bwx as usize][bwy as usize], wx, wy)
            } else {
                0.0
            }
        }
    ).sum::<f64>() / inside_count as f64 * sigpts.len() as f64
}

fn wavelet_transform(f: &Vec<Vec<f64>>, wavelet: impl RadialWavelet) -> Vec<Vec<f64>> {
    let sigpts = wavelet.significant_points();
    let xlen = f.len();
    let ylen = f[0].len();
    let mut res = vec![vec![-1.0; ylen]; xlen];

    let inner_xbeg = wavelet.max_significant_radius().ceil() as usize;
    let inner_xend = xlen - inner_xbeg;
    let inner_ybeg = inner_xbeg;
    let inner_yend = ylen - inner_ybeg;
    for bx in inner_xbeg..inner_xend {
        for by in inner_ybeg..inner_yend {
            res[bx][by] = wavelet_transform_at_inner(f, wavelet, &sigpts, bx, by);
        }
    }

    for by in 0..ylen {
        for bx in 0..inner_xbeg {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by);
        }
        for bx in inner_xend..xlen {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by);
        }
    }
    for bx in inner_xbeg..inner_xend {
        for by in 0..inner_ybeg {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by);
        }
        for by in inner_yend..ylen {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by);
        }
    }

    res
}

fn wavelet_transform_with_pow(f: &Vec<Vec<f64>>, wavelet: impl RadialWavelet, pow: f64) -> Vec<Vec<f64>> {
    let sigpts = wavelet.significant_points();
    let xlen = f.len();
    let ylen = f[0].len();
    let mut res = vec![vec![-1.0; ylen]; xlen];

    let inner_xbeg = wavelet.max_significant_radius().ceil() as usize;
    let inner_xend = xlen - inner_xbeg;
    let inner_ybeg = inner_xbeg;
    let inner_yend = ylen - inner_ybeg;
    for bx in inner_xbeg..inner_xend {
        for by in inner_ybeg..inner_yend {
            res[bx][by] = wavelet_transform_at_inner(f, wavelet, &sigpts, bx, by).powf(pow);
        }
    }

    for by in 0..ylen {
        for bx in 0..inner_xbeg {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by).powf(pow);
        }
        for bx in inner_xend..xlen {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by).powf(pow);
        }
    }
    for bx in inner_xbeg..inner_xend {
        for by in 0..inner_ybeg {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by).powf(pow);
        }
        for by in inner_yend..ylen {
            res[bx][by] = wavelet_transform_at_outer(f, wavelet, &sigpts, bx, by).powf(pow);
        }
    }

    res
}

fn input_field(path: &str) -> Vec<Vec<f64>> {
    fs::read_to_string(path).unwrap()
        .lines()
        .map(|line| 
            line.split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        )
        .collect()
}

fn output_field(f: &Vec<Vec<f64>>, path: &str) {
    let mut file = File::create(path).unwrap();
    for x in 0..f.len() {
        writeln!(file, "{}", 
            f[x].iter()
                .map(|&v| v.to_string())
                .reduce(|acc, v| format!("{} {}", acc, v))
                .unwrap()
        ).unwrap();
    }
}

fn output_fields(fs: &Vec<Vec<Vec<f64>>>, path: &str) {
    let mut file = File::create(path).unwrap();
    let data = fs.iter()
        .map(|f|
            f.iter()
                .map(|row| 
                    row.iter()
                    .map(|&v| v.to_string())
                    .reduce(|acc, v| format!("{} {}", acc, v))
                    .unwrap()
                )
                .reduce(|acc, v| format!("{}\n{}", acc, v))
                .unwrap()
        )
        .reduce(|acc, v| format!("{}\n\n{}", acc, v))
        .unwrap();
    write!(&mut file, "{}", data).unwrap();
}

fn main() {
    let f = input_field("f.tsv");

    let mut wfs = Vec::<Vec<Vec<f64>>>::new();
    let mut wwfs = Vec::<Vec<Vec<f64>>>::new();
    let s = 0.3;
    for m in (0..100).step_by(1).map(|m| m as f64) {
        // let w = RadialGaussian::new(m, s);
        let w = RadialHeaviside::new(m, s * 3.0);
        println!(
            "radius:{}, thickness={}, sigpts num: {}", 
            w.radius(), w.thickness(), w.significant_points().len()
        );
        let wf = wavelet_transform_with_pow(&f, w, 4.0);
        let wwf = wavelet_transform(&wf, w);
        wfs.push(wf);
        wwfs.push(wwf);
    }
    output_fields(&wfs, "wfs.ssv");
    output_fields(&wwfs, "wwfs.ssv");

    // let psi = Psi::new(5.0, 0.2);
    // println!("sigpts num: {}", significant_points(psi).len());
    // let wf = wavelet_transform(&f, psi);
    // let wwf = wavelet_transform(&wf, psi);
    // output_field(&wf, "wf.ssv");
    // output_field(&wwf, "wwf.ssv");
}
