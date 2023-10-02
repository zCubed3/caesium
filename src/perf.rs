pub mod dropwatch;

pub fn print_progress(p: f32, width: usize) {
    if p < 1.0 {
        print!("[");

        for x in 0..width {
            let f = x as f32 / width as f32;

            if p > f {
                print!("=");
            } else {
                print!(" ");
            }
        }

        print!("] {}%\r", (p * 100.0).round() / 100.0);
    } else {
        print!("Done!");
        for _ in 0..width + 3 {
            print!(" ");
        }
        println!();
    }
}
