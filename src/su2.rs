/// This is ported from https://github.com/cmdawson/sk
// MIT License
// Copyright (c) 2005 Chris Dawson

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
use nalgebra::{Complex, Matrix2};

use crate::{
    gates::{x_dence_matrix, y_dence_matrix, z_dence_matrix},
    Qbit,
};

pub fn group_factor(u: &Matrix2<Qbit>) -> (Matrix2<Qbit>, Matrix2<Qbit>) {
    let cu = mat_to_cart3(u);
    let n = norm3(cu);

    let xu = cart3_to_mat((n, 0.0, 0.0));
    let s = similarity_matrix(u, &xu);
    let a_s = s.adjoint();

    let (a, b) = x_group_factor(&xu);
    (s * a * a_s, s * b * a_s)
}

fn x_group_factor(a: &Matrix2<Qbit>) -> (Matrix2<Qbit>, Matrix2<Qbit>) {
    let ac = mat_to_cart4(a);

    let st = (0.5 - 0.5 * ac.0).powf(0.25);
    let ct = (1.0 - st * st).sqrt();
    let theta = 2.0 * st.asin();
    let alpha = st.atan();

    let bc = (
        theta * st * alpha.cos(),
        theta * st * alpha.sin(),
        theta * ct,
    );
    let cc = (bc.0, bc.1, -bc.2);

    let b = cart3_to_mat(cc);
    let w = cart3_to_mat(bc);

    let a_b = b.adjoint();

    (b, similarity_matrix(&w, &a_b))
}

fn similarity_matrix(a: &Matrix2<Qbit>, b: &Matrix2<Qbit>) -> Matrix2<Qbit> {
    let ac = mat_to_cart3(a);
    let bc = mat_to_cart3(b);

    let na = norm3(ac);
    let nb = norm3(bc);

    let ab = ac.0 * bc.0 + ac.1 * bc.1 + ac.2 * bc.2;
    let s = (
        bc.1 * ac.2 - ac.1 * bc.2,
        ac.0 * bc.2 - bc.0 * ac.2,
        bc.0 * ac.1 - ac.0 * bc.1,
    );

    let ns = norm3(s);
    if ns.abs() < 1e-12 {
        Matrix2::identity()
    } else {
        let v = (ab / (na * nb)).acos() / ns;
        let s = (s.0 * v, s.1 * v, s.2 * v);
        cart3_to_mat(s)
    }
}

fn norm3(v: (f64, f64, f64)) -> f64 {
    dot3(v, v).sqrt()
}

fn dot3(v1: (f64, f64, f64), v2: (f64, f64, f64)) -> f64 {
    v1.0 * v2.0 + v1.1 * v2.1 + v1.2 * v2.2
}

fn mat_to_cart3(u: &Matrix2<Qbit>) -> (f64, f64, f64) {
    let sx1 = -1.0 * u[(0, 1)].im;
    let sx2 = u[(1, 0)].re;
    let sx3 = (u[(1, 1)].re - u[(0, 0)].re) / 2.0;

    let costh = (u[(0, 0)].re + u[(1, 1)].re) / 2.0;
    let sinth = (sx1 * sx1 + sx2 * sx2 + sx3 * sx3).sqrt();

    if sinth < 1e-10 {
        (2.0 * costh.acos(), 0.0, 0.0)
    } else {
        let th = sinth.atan2(costh);
        (
            2.0 * th * sx1 / sinth,
            2.0 * th * sx2 / sinth,
            2.0 * th * sx3 / sinth,
        )
    }
}

fn mat_to_cart4(u: &Matrix2<Qbit>) -> (f64, f64, f64, f64) {
    fn round(v: f64) -> f64 {
        if v.abs() < 1e-15 {
            0.0
        } else {
            v
        }
    }

    (
        round(u[(0, 0)].re),
        round(-1.0 * u[(0, 1)].im),
        round(u[(1, 0)].re),
        round(u[(1, 1)].im),
    )
}

fn cart3_to_mat(cart3: (f64, f64, f64)) -> Matrix2<Qbit> {
    let (a, b, c) = cart3;

    let th = (a * a + b * b + c * c).sqrt();

    if th < 1e-10 {
        Matrix2::identity()
    } else {
        let imag_sin = Complex::new(0.0, 1.0) * (th / 2.0).sin();
        let id = Matrix2::identity() * Complex::new((th / 2.0).cos(), 0.0);
        let x = x_dence_matrix() * (imag_sin * a / th);
        let y = y_dence_matrix() * (imag_sin * b / th);
        let z = z_dence_matrix() * (imag_sin * c / th);
        id + x + y + z
    }
}
