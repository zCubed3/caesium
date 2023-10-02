use std::time;

/// Measures time from the creation of this watch to when it exists scope
pub struct Dropwatch {
    start: time::Instant,
    id: String,
}

impl Dropwatch {
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self {
            start: time::Instant::now(),
            id: id.into(),
        }
    }
}

impl Drop for Dropwatch {
    fn drop(&mut self) {
        let elapsed = time::Instant::now() - self.start;
        println!(
            "{}: {}s ({}ms)",
            self.id,
            elapsed.as_secs_f32(),
            elapsed.as_millis()
        );
    }
}
