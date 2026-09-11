//! Print the generated CUDA C to stdout.
//!
//! Two uses, both real. It lets `nvcc -ptx` check the emitted source on a
//! machine with a toolkit but no free GPU — the generator is the half of this
//! crate most likely to be wrong, and it can be validated without touching the
//! device. And Nsight Compute needs the source on disk to correlate a profile
//! back to lines, which NVRTC's in-memory compilation otherwise denies it.
//!
//! ```text
//! cargo run -p ad-cuda --example dump_kernel > lbm.cu
//! nvcc -arch=compute_89 -ptx lbm.cu -o lbm.ptx
//! cargo run -p ad-cuda --example dump_kernel -- d3q27 bgk 128
//! ```
//!
//! Deliberately not feature-gated: the generator has no CUDA dependency, so this
//! runs anywhere.

use ad_cuda::kernel::{generate, KernelSpec};
use ad_gpu::types::VelocitySet;
use ad_solver::CollisionModel;

fn main() {
    let mut spec =
        KernelSpec { set: VelocitySet::D3Q19, collision: CollisionModel::Trt, block_x: 64 };
    for arg in std::env::args().skip(1) {
        match arg.to_ascii_lowercase().as_str() {
            "d3q19" => spec.set = VelocitySet::D3Q19,
            "d3q27" => spec.set = VelocitySet::D3Q27,
            "trt" => spec.collision = CollisionModel::Trt,
            "bgk" => spec.collision = CollisionModel::Bgk,
            "rbgk" => spec.collision = CollisionModel::RegularizedBgk,
            n => match n.parse::<u32>() {
                Ok(b) if b > 0 => spec.block_x = b,
                _ => {
                    eprintln!(
                        "usage: dump_kernel [d3q19|d3q27] [trt|bgk|rbgk] [<block size>]\n\
                         unrecognised argument: {arg}"
                    );
                    std::process::exit(2);
                }
            },
        }
    }
    print!("{}", generate(spec));
}
