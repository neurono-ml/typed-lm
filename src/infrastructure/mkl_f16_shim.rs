//! Link-time shim for the missing Intel MKL `hgemm_` symbol.
//!
//! `intel-mkl-src` 0.8.1 bundles Intel MKL 2020.1, which does not export the
//! half-precision `hgemm_` routine that `candle-core`'s `mkl` feature declares
//! and references for `F16` matmuls. On the CPU this server always computes in
//! `F32`, so `hgemm_` is never executed; nevertheless the symbol reference makes
//! the linker fail. This module provides a `hgemm_` implemented on top of the
//! `sgemm_` routine MKL does export, so the `mkl` feature both links and still
//! returns the right result if a half-precision call ever reaches it.
//!
//! Compiled only when the `mkl` feature is enabled.

use core::ffi::c_char;
use half::f16;

extern "C" {
    fn sgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
}

/// Half-precision GEMM matching the Fortran BLAS `hgemm_` ABI.
///
/// Converts the operands to `F32`, delegates to MKL `sgemm_` and narrows the
/// result back to `F16`, honouring `transa`, `transb` and every leading
/// dimension. Column-major (Fortran) layout, like the routine it replaces.
///
/// # Safety
/// All pointers must be valid for the extents implied by `m`, `n`, `k`, `lda`,
/// `ldb` and `ldc`, matching the BLAS contract.
#[no_mangle]
pub unsafe extern "C" fn hgemm_(
    transa: *const c_char,
    transb: *const c_char,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const f16,
    a: *const f16,
    lda: *const i32,
    b: *const f16,
    ldb: *const i32,
    beta: *const f16,
    c: *mut f16,
    ldc: *const i32,
) {
    let rows = *m as usize;
    let columns = *n as usize;
    let inner = *k as usize;
    if rows == 0 || columns == 0 {
        return;
    }
    let leading_a = *lda as usize;
    let leading_b = *ldb as usize;
    let leading_c = *ldc as usize;
    let transpose_a = (*transa as u8) == b'T' || (*transa as u8) == b't';
    let transpose_b = (*transb as u8) == b'T' || (*transb as u8) == b't';

    // op(A) is rows x inner, column-major, packed with leading dimension rows.
    let mut dense_a = vec![0f32; rows * inner];
    for row in 0..rows {
        for column in 0..inner {
            let offset = if transpose_a {
                column + row * leading_a
            } else {
                row + column * leading_a
            };
            dense_a[row + column * rows] = (*a.add(offset)).to_f32();
        }
    }

    // op(B) is inner x columns, column-major, packed with leading dimension inner.
    let mut dense_b = vec![0f32; inner * columns];
    for row in 0..inner {
        for column in 0..columns {
            let offset = if transpose_b {
                column + row * leading_b
            } else {
                row + column * leading_b
            };
            dense_b[row + column * inner] = (*b.add(offset)).to_f32();
        }
    }

    // Seed with the original C so a non-zero beta scales it as BLAS requires.
    let mut dense_c = vec![0f32; rows * columns];
    for column in 0..columns {
        for row in 0..rows {
            dense_c[row + column * rows] = (*c.add(row + column * leading_c)).to_f32();
        }
    }

    let not_transposed = b'N' as c_char;
    let alpha = (*alpha).to_f32();
    let beta = (*beta).to_f32();
    sgemm_(
        &not_transposed,
        &not_transposed,
        &(rows as i32),
        &(columns as i32),
        &(inner as i32),
        &alpha,
        dense_a.as_ptr(),
        &(rows as i32),
        dense_b.as_ptr(),
        &(inner as i32),
        &beta,
        dense_c.as_mut_ptr(),
        &(rows as i32),
    );

    for column in 0..columns {
        for row in 0..rows {
            *c.add(row + column * leading_c) = f16::from_f32(dense_c[row + column * rows]);
        }
    }
}
