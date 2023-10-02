use rgml::prelude::*;

use rgml::vector::Vector;

use image::*;
use std::fs::*;
use std::path::Path;

use serde::*;

use bincode::*;

pub enum BufferShape {
    Shape1D,
    Shape2D,
    Shape3D,
}

pub trait DataBounds: Serialize {}

pub trait BufferData: Sized + Clone + Copy + Default + DataBounds {}

macro_rules! impl_data {
    ($ty:tt) => {
        impl DataBounds for $ty {}
        impl BufferData for $ty {}
    };
}

impl_data!(f32);
impl_data!(f64);
impl_data!(Vector2F32);
impl_data!(Vector3F32);
impl_data!(Vector4F32);
impl_data!(Vector2F64);
impl_data!(Vector3F64);
impl_data!(Vector4F64);

pub trait Buffer<T: BufferData>: Clone {
    fn get_buffer_shape(&self) -> BufferShape;
    fn get_buffer_width(&self) -> usize;
    fn get_buffer_height(&self) -> usize;
    fn get_buffer_depth(&self) -> usize;

    fn buffer_new(width: usize, height: usize, depth: usize) -> Self;

    fn buffer_write(&mut self, x: usize, y: usize, z: usize, value: T);
    fn buffer_read(&self, x: usize, y: usize, z: usize) -> T;

    fn buffer_save(&self, path: &Path);
}

/// A linear "line" shaped buffer
#[derive(Clone, Serialize)]
#[repr(C)]
pub struct LinearBuffer<T: BufferData> {
    _backing: Vec<T>,
    _size: usize,
}

impl<T: BufferData> LinearBuffer<T> {
    pub fn new(size: usize) -> Self {
        return Self {
            _backing: vec![T::default(); size],
            _size: size,
        };
    }
}

impl<T: BufferData> Buffer<T> for LinearBuffer<T> {
    fn get_buffer_shape(&self) -> BufferShape {
        return BufferShape::Shape1D;
    }

    fn get_buffer_width(&self) -> usize {
        return self._size;
    }

    fn get_buffer_height(&self) -> usize {
        return 1;
    }

    fn get_buffer_depth(&self) -> usize {
        return 1;
    }

    fn buffer_new(width: usize, _height: usize, _depth: usize) -> Self {
        return LinearBuffer::new(width);
    }

    fn buffer_write(&mut self, x: usize, _y: usize, _z: usize, value: T) {
        assert!(x < self._size);
        self._backing[x] = value;
    }

    fn buffer_read(&self, x: usize, _y: usize, _z: usize) -> T {
        assert!(x < self._size);
        return self._backing[x];
    }

    fn buffer_save(&self, _path: &Path) {
        todo!()
    }
}

/// A square "texture" shaped buffer
#[derive(Clone, Serialize)]
#[repr(C)]
pub struct SquareBuffer<T: BufferData> {
    _backing: Vec<T>,
    _width: usize,
    _height: usize,
}

impl<T: BufferData> SquareBuffer<T> {
    pub fn new(width: usize, height: usize) -> Self {
        return Self {
            _backing: vec![T::default(); width * height],
            _width: width,
            _height: height,
        };
    }

    fn coord_to_index(&self, x: usize, y: usize) -> usize {
        return x + (y * self._width);
    }
}

impl<T: BufferData> Buffer<T> for SquareBuffer<T> {
    fn get_buffer_shape(&self) -> BufferShape {
        return BufferShape::Shape2D;
    }

    fn get_buffer_width(&self) -> usize {
        return self._width;
    }

    fn get_buffer_height(&self) -> usize {
        return self._height;
    }

    fn get_buffer_depth(&self) -> usize {
        return 1;
    }

    fn buffer_new(width: usize, height: usize, _depth: usize) -> Self {
        return SquareBuffer::new(width, height);
    }

    fn buffer_write(&mut self, x: usize, y: usize, _z: usize, value: T) {
        let index = self.coord_to_index(x, y);
        assert!(index < self._width * self._height);
        self._backing[index] = value;
    }

    fn buffer_read(&self, x: usize, y: usize, _z: usize) -> T {
        let index = self.coord_to_index(x, y);
        assert!(index < self._width * self._height);
        return self._backing[index];
    }

    fn buffer_save(&self, path: &Path) {
        let mut file = File::open(path).expect("Failed to open file!");
        serialize_into(&mut file, self).expect("Failed to write buffer!");
    }
}

impl SquareBuffer<Vector<Real, 4>> {
    pub fn save_as_image<P: AsRef<Path>>(&self, path: P) {
        let mut image = RgbaImage::new(self._width as u32, self._height as u32);

        for x in 0..self._width {
            for y in 0..self._height {
                let raw_color = self.buffer_read(x, y, 0usize);
                image.put_pixel(
                    x as u32,
                    y as u32,
                    Rgba([
                        (raw_color[0].rl_saturate() * 255.0).rl_round() as u8,
                        (raw_color[1].rl_saturate() * 255.0).rl_round() as u8,
                        (raw_color[2].rl_saturate() * 255.0).rl_round() as u8,
                        (raw_color[3].rl_saturate() * 255.0).rl_round() as u8,
                    ]),
                );
            }
        }

        image.save(path).expect("Failed to save image");
    }
}
