// Placeholder. Later tasks land the real CLI here (argument parsing, the
// transcribe/serve subcommands, exit codes).

pub fn main() -> i32 {
    let mut args = std::env::args();
    let _bin = args.next();
    if args.next().as_deref() == Some("--version") {
        println!("auris {}", env!("CARGO_PKG_VERSION"));
    }
    0
}
