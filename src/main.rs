pub mod buffer;
pub mod perf;

use rgml::prelude::*;
use std::fs::{create_dir, File};
use std::io::{BufRead, BufReader, Write};
use std::ops::{Add, Mul};
use std::time::Duration;
use image::codecs::gif::Repeat;
use image::Delay;
use pixels::{Pixels, SurfaceTexture};

use rand::*;

use winit::window::{Window, WindowBuilder};
use winit::event_loop::EventLoop;
use winit_input_helper::WinitInputHelper;

// TODO: Make tracy optional
use tracing;
use tracing::{event, span, Level};
use tracing_subscriber::layer::SubscriberExt;
use tracing_tracy::client::frame_mark;
use winit::dpi::LogicalSize;
use winit::event::Event;

use crate::buffer::{Buffer, SquareBuffer};
use crate::perf::dropwatch::Dropwatch;

const DEFAULT_WIDTH: usize = 640;
const DEFAULT_HEIGHT: usize = 480;

const NEAR: Real = 0.001;
const FAR: Real = 100.0;

const DEPTH_UNINIT: Real = 1.0;

#[inline]
pub fn edge(a: Vector2, b: Vector2, c: Vector2) -> Real {
    let c0 = c.x() - a.x();
    let c1 = b.y() - a.y();
    let c2 = c.y() - a.y();
    let c3 = b.x() - a.x();

    c0 * c1 - c2 * c3
}

#[derive(Default, Copy, Clone, PartialEq)]
pub struct Vertex {
    pub position: Vector3,
    pub normal: Vector3,
    pub tex_coord: Vector2,
}

#[derive(Default)]
pub struct Mesh {
    pub triangles: Vec<usize>,
    pub vertices: Vec<Vertex>,
}

pub fn load_obj(path: &str) -> Mesh {
    let mut mdl = Mesh::default();

    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut positions: Vec<Vector3> = Vec::new();
    let mut normals: Vec<Vector3> = Vec::new();
    let mut tex_coords: Vec<Vector2> = Vec::new();

    let mut faces: Vec<[usize; 3]> = Vec::new();

    for line in reader.lines() {
        let content = line.unwrap();

        if content.starts_with("v ") {
            let mut components: Vec<Real> = Vec::new();
            for word in content.split(" ") {
                let float_parse = word.parse::<Real>();

                if float_parse.is_ok() {
                    components.push(float_parse.unwrap());
                }
            }

            if components.len() == 3 {
                let pos = Vector3::new(components[0], components[1], components[2]);
                positions.push(pos);
            }
        }

        if content.starts_with("vn") {
            let mut components: Vec<Real> = Vec::new();
            for word in content.split(" ") {
                let float_parse = word.parse::<Real>();

                if float_parse.is_ok() {
                    components.push(float_parse.unwrap());
                }
            }

            if components.len() == 3 {
                let normal = Vector3::new(components[0], components[1], components[2]);
                normals.push(normal);
            }
        }

        if content.starts_with("vt") {
            let mut components: Vec<Real> = Vec::new();
            for word in content.split(" ") {
                let float_parse = word.parse::<Real>();

                if float_parse.is_ok() {
                    components.push(float_parse.unwrap());
                }
            }

            if components.len() == 2 {
                let tex_coord = Vector2::new(components[0], components[1]);
                tex_coords.push(tex_coord);
            }
        }

        if content.starts_with("f ") {
            let raw_face = content.replace("f ", "");

            for raw_indices in raw_face.split(" ") {
                let mut face: [usize; 3] = [0, 0, 0];
                let mut last_index = 0;
                for indice in raw_indices.split("/") {
                    let indice_parse = indice.parse::<usize>();

                    if indice_parse.is_ok() {
                        face[last_index] = indice_parse.unwrap() - 1;
                        last_index += 1;
                    }
                }

                faces.push(face);
            }
        }
    }

    for face in faces {
        let position = positions[face[0] as usize];
        let tex_coord = tex_coords[face[1] as usize];
        let normal = normals[face[2] as usize];

        let vertex = Vertex {
            position,
            normal,
            tex_coord,
        };

        let mut similar = false;
        for test_vertex_index in 0..mdl.vertices.len() {
            let test_vertex = mdl.vertices[test_vertex_index];
            if vertex == test_vertex {
                similar = true;
                mdl.triangles.push(test_vertex_index);
                break;
            }
        }

        if !similar {
            mdl.triangles.push(mdl.vertices.len());
            mdl.vertices.push(vertex);
        }
    }

    return mdl;
}

#[inline(always)]
fn bary_interpolate<E>((u, v, w): (Real, Real, Real), (v0, v1, v2): (E, E, E)) -> E
where
    E: Mul<Real, Output = E> + Add<E, Output = E>,
{
    (v0 * u) + (v1 * v) + (v2 * w)
}

#[derive(Default, Debug, Clone)]
pub struct RTAABB {
    pub min: Vector3,
    pub max: Vector3,
}

static mut AABB_ID: usize = 0usize;

impl RTAABB {
    pub fn new(min: Vector3, max: Vector3) -> Self {
        Self {
            min,
            max,
        }
    }

    pub fn contains_point(&self, point: Vector3) -> bool {
        point >= self.min && point <= self.max
    }

    pub fn union(&self, min: Vector3, max: Vector3) -> Self {
        Self {
            min: self.min.min(min),
            max: self.max.max(max),
        }
    }
}

const BEGIN_FRAME: usize = 0;
const END_FRAME: usize = 120;

const FRAMES_PER_SECOND: usize = 300;

const TIME_PER_FRAME: Real = 1.0 / (FRAMES_PER_SECOND as Real);

pub struct Framebuffer<'a> {
    pub color: &'a mut SquareBuffer<Vector4>,
    pub depth: &'a mut SquareBuffer<Real>
}

pub struct DrawContext<'a> {
    mesh: &'a Mesh,
    fb: Framebuffer<'a>,

    frame: usize,
    width: usize,
    height: usize
}

pub fn draw(ctx: DrawContext) {
    let tex_x = 1.0 / ctx.width as Real;
    let tex_y = 1.0 / ctx.height as Real;

    //
    // Calculate matrices
    //
    let time = ctx.frame as Real * TIME_PER_FRAME;

    let aspect = ctx.width as Real / ctx.height as Real;

    let mat_p = Matrix4x4::perspective(45.0, aspect, NEAR, FAR);
    let mat_v = {
        let sin = (time * Real::PI).rl_sin();
        let cos = (time * Real::PI).rl_cos();

        let horizontal = sin * 5.0;
        let vertical = cos * 5.0;

        let angle1 = horizontal.rl_atan2(-vertical).rl_to_degrees();

        let translation = Matrix4x4::translation(Vector3::new(horizontal, 0.0, vertical));
        let rotation = Matrix4x4::rotation(Vector3::new(0.0, angle1, 180.0));

        rotation * translation
    };

    let mat_vp = mat_p * mat_v;

    //
    // Render
    //
    let light = Vector3::new(1.0, 1.0, 1.0).normalize();

    for t in (0..ctx.mesh.triangles.len()).step_by(3) {
        //let span = span!(Level::TRACE, "Rasterize Triangle");
        //let _guard = span.enter();

        let v0 = ctx.mesh.vertices[ctx.mesh.triangles[t + 2]];
        let v1 = ctx.mesh.vertices[ctx.mesh.triangles[t + 1]];
        let v2 = ctx.mesh.vertices[ctx.mesh.triangles[t]];

        // Project the points
        let p0 = mat_vp * Vector4::from_w(v0.position, 1.0);
        let p1 = mat_vp * Vector4::from_w(v1.position, 1.0);
        let p2 = mat_vp * Vector4::from_w(v2.position, 1.0);

        // Invalid vertex?
        if p0.w() <= 0.0 || p1.w() <= 0.0 || p2.w() <= 0.0 {
            continue;
        }

        // Perspective correction
        let p0 = p0 / p0.w();
        let p1 = p1 / p1.w();
        let p2 = p2 / p2.w();

        // Lighting
        let l0 = v0.normal.dot(light).rl_saturate();
        let l1 = v1.normal.dot(light).rl_saturate();
        let l2 = v2.normal.dot(light).rl_saturate();

        let area = edge(p0.xy(), p1.xy(), p2.xy());

        // Calculate the bounding box of this triangle
        let aabb = {
            let mut bb_base = RTAABB::new(p0.xyz(), p0.xyz());
            bb_base = bb_base.union(p1.xyz(), p1.xyz());
            bb_base = bb_base.union(p2.xyz(), p2.xyz());

            bb_base
        };

        let min = (aabb.min + 1.0) * 0.5;
        let max = (aabb.max + 1.0) * 0.5;

        let min_pix_x = (ctx.width as Real * min.x()).rl_floor() as usize;
        let max_pix_x = (ctx.width as Real * max.x()).rl_ceil() as usize;

        let min_pix_y = (ctx.height as Real * min.y()).rl_floor() as usize;
        let max_pix_y = (ctx.height as Real * max.y()).rl_ceil() as usize;

        // Disables clip optimization
        /*
        let min_pix_x = 0;
        let min_pix_y = 0;
        let max_pix_y = WIDTH;
        let max_pix_y = HEIGHT;
         */

        let mut rng = thread_rng();
        let debug_col = Vector3::new(rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0));

        for x in min_pix_x..=max_pix_x {
            if x >= ctx.width {
                break;
            }

            let u = (x as Real) * tex_x;
            let vc_x = (u - 0.5) * 2.0;

            for y in min_pix_y..=max_pix_y {
                if y >= ctx.height {
                    break;
                }

                //let span = span!(Level::TRACE, "Draw Pixel");
                //let _guard = span.enter();

                let v = (y as Real) * tex_y;
                let vc_y = (v - 0.5) * 2.0;

                let point = Vector2::new(vc_x, vc_y);

                let w0 = edge(p1.xy(), p2.xy(), point);
                let w1 = edge(p2.xy(), p0.xy(), point);
                let w2 = edge(p0.xy(), p1.xy(), point);

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let u = w0 / area;
                    let v = w1 / area;
                    let w = w2 / area;

                    //
                    // Depth
                    //

                    // Check if the point we're intersecting is closer
                    // TODO: Make this faster?
                    let clip_pos = bary_interpolate((u, v, w), (p0, p1, p2));

                    // Normalize d across the clip plane
                    let depth = ctx.fb.depth.buffer_read(x, y, 0);

                    let clip_depth = clip_pos.z() / FAR;

                    if clip_depth > depth || clip_depth < 0.0 {
                        continue;
                    }

                    ctx.fb.depth.buffer_write(x, y, 0, clip_depth);

                    //
                    // Color
                    //
                    let norm = bary_interpolate(
                        (u, v, w),
                        (v0.normal, v1.normal, v2.normal)
                    ).normalize();

                    let fac = norm.dot(light).rl_saturate();
                    let fac = bary_interpolate(
                        (u, v, w),
                        (l0, l1, l2)
                    );

                    let global = Vector3::from_scalar(0.01);
                    let shaded = Vector3::from_scalar(fac).max(global);

                    let albedo = Vector3::new(1.0, 1.0, 1.0);

                    let lit = albedo * shaded;

                    let col = Vector4::from_w(lit, 1.0);

                    ctx.fb.color.buffer_write(x, y, 0, col)
                }

                //ctx.fb.color.buffer_write(x, y, 0, Vector4::from_w(debug_col, 1.0));
            }
        }
    }
}

pub fn create_pixels(window: &Window) -> Pixels {
    let window_size = window.inner_size();
    let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, window);

    Pixels::new(DEFAULT_WIDTH as u32, DEFAULT_HEIGHT as u32, surface_texture).unwrap()
}

pub fn main() {
    //
    // Tracing setup
    //
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::new()),
    )
    .expect("set up the subscriber");

    //
    // Output
    //
    create_dir("result");

    //
    // Window
    //
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();

    let window = {
        let size = LogicalSize::new(DEFAULT_WIDTH as f64, DEFAULT_HEIGHT as f64);
        let scaled_size = LogicalSize::new(DEFAULT_WIDTH as f64, DEFAULT_HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Caesium")
            .with_inner_size(scaled_size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    //
    // Actual program
    //

    let mut pixels = create_pixels(&window);

    // Our buffers
    let mut buffer = SquareBuffer::<Vector4>::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);
    let mut depth_buffer = SquareBuffer::<Real>::new(DEFAULT_WIDTH, DEFAULT_HEIGHT);

    // Our model
    let model = load_obj("test.obj");

    let mut frame = 0usize;

    event_loop.run(move |event, _, control_flow| {
        if let Event::RedrawRequested(_) = event {
            let size = window.inner_size();

            // Recreate our buffers if the size is wrong
            let old_width = buffer.get_buffer_width();
            let old_height = buffer.get_buffer_height();

            if size.width != old_width as u32 || size.height != old_height as u32 {
                buffer = SquareBuffer::<Vector4>::new(size.width as usize, size.height as usize);
                depth_buffer = SquareBuffer::<Real>::new(size.width as usize, size.height as usize);

                pixels.resize_buffer(size.width, size.height).unwrap();
                pixels.resize_surface(size.width, size.height).unwrap();
            }

            // Clear the buffers
            for x in 0..size.width as usize {
                for y in 0..size.height as usize {
                    buffer.buffer_write(x, y, 0, Vector4::new(0.0, 0.0, 0.0, 1.0));
                    depth_buffer.buffer_write(x, y, 0, DEPTH_UNINIT);
                }
            }

            let ctx = DrawContext {
                mesh: &model,
                frame: frame,
                fb: Framebuffer {
                    color: &mut buffer,
                    depth: &mut depth_buffer
                },

                width: size.width as usize,
                height: size.height as usize
            };


            let span = span!(Level::TRACE, "Draw");
            let guard = span.enter();

            draw(ctx);

            drop(guard);
            drop(span);

            let pix = pixels.frame_mut();

            let mut x = 0;
            let mut y = 0;

            let span = span!(Level::TRACE, "Blit to Screen");
            let guard = span.enter();

            for pixel in pix.chunks_exact_mut(4) {
                let col = buffer.buffer_read(x, y, 0);

                pixel[0] = (col.x() * 255.0).rl_floor() as u8;
                pixel[1] = (col.y() * 255.0).rl_floor() as u8;
                pixel[2] = (col.z() * 255.0).rl_floor() as u8;
                pixel[3] = (col.w() * 255.0).rl_floor() as u8;

                x += 1;

                if x >= size.width as usize {
                    x = 0;
                    y += 1;
                }
            }

            drop(guard);
            drop(span);

            pixels.render().unwrap();

            frame += 1;

            frame_mark();
        }

        if input.update(&event) {


            window.request_redraw();
        }
    });

    /*
    for frame in BEGIN_FRAME..END_FRAME {
        let span = span!(Level::TRACE, "Draw Screen");
        let _guard = span.enter();

        let _dw = Dropwatch::new("Draw Screen");

        //
        // Clear the buffers first
        //
        {
            let span = span!(Level::TRACE, "Clear Buffers");
            let _guard = span.enter();

            for x in 0..WIDTH {
                for y in 0..HEIGHT {
                    buffer.buffer_write(x, y, 0, Vector4::new(0.0, 0.0, 0.0, 1.0));
                    depth_buffer.buffer_write(x, y, 0, DEPTH_UNINIT);
                }
            }
        }

        //
        // Calculate matrices
        //
        let time = frame as Real * TIME_PER_FRAME;

        let aspect = WIDTH as Real / HEIGHT as Real;

        let mat_p = Matrix4x4::perspective(90.0, aspect, NEAR, FAR);
        let mat_v = {
            let sin = (time * Real::PI).rl_sin();
            let cos = (time * Real::PI).rl_cos();

            let angle1 = sin * 10.0;
            let angle2 = cos * 10.0;

            let translation = Matrix4x4::translation(Vector3::new(0.0, 0.0, -1.0));
            let rotation = Matrix4x4::rotation(Vector3::new(angle1, angle2, 180.0));

            rotation * translation
        };

        let mat_vp = mat_p * mat_v;

        //
        // Render
        //
        let light = Vector3::new(1.0, 1.0, 1.0).normalize();

        for t in (0..model.triangles.len()).step_by(3) {
            //let span = span!(Level::TRACE, "Rasterize Triangle");
            //let _guard = span.enter();

            let v0 = model.vertices[model.triangles[t + 2]];
            let v1 = model.vertices[model.triangles[t + 1]];
            let v2 = model.vertices[model.triangles[t]];

            // Project the points
            let p0 = mat_vp * Vector4::from_w(v0.position, 1.0);
            let p1 = mat_vp * Vector4::from_w(v1.position, 1.0);
            let p2 = mat_vp * Vector4::from_w(v2.position, 1.0);

            // Invalid vertex?
            if p0.w() <= 0.0 || p1.w() <= 0.0 || p2.w() <= 0.0 {
                continue;
            }

            // Perspective correction
            let p0 = p0 / p0.w();
            let p1 = p1 / p1.w();
            let p2 = p2 / p2.w();

            // Lighting
            let l0 = v0.normal.dot(light).rl_saturate();
            let l1 = v1.normal.dot(light).rl_saturate();
            let l2 = v2.normal.dot(light).rl_saturate();

            let area = edge(p0.xy(), p1.xy(), p2.xy());

            // Calculate the bounding box of this triangle
            let aabb = {
                let mut bb_base = RTAABB::new(p0.xyz(), p0.xyz());
                bb_base = bb_base.union(p1.xyz(), p1.xyz());
                bb_base = bb_base.union(p2.xyz(), p2.xyz());

                bb_base
            };

            let min = (aabb.min + 1.0) * 0.5;
            let max = (aabb.max + 1.0) * 0.5;

            let min_pix_x = (WIDTH as Real * min.x()).rl_floor() as usize;
            let max_pix_x = (WIDTH as Real * max.x()).rl_ceil() as usize;

            let min_pix_y = (HEIGHT as Real * min.y()).rl_floor() as usize;
            let max_pix_y = (HEIGHT as Real * max.y()).rl_ceil() as usize;

            // Disables clip optimization
            /*
            let min_pix_x = 0;
            let min_pix_y = 0;
            let max_pix_y = WIDTH;
            let max_pix_y = HEIGHT;
             */

            let mut rng = thread_rng();
            let debug_col = Vector3::new(rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0));

            for x in min_pix_x..=max_pix_x {
                if x >= WIDTH {
                    break;
                }

                let u = (x as Real) * tex_x;
                let vc_x = (u - 0.5) * 2.0;

                for y in min_pix_y..=max_pix_y {
                    if y >= HEIGHT {
                        break;
                    }

                    //let span = span!(Level::TRACE, "Draw Pixel");
                    //let _guard = span.enter();

                    let v = (y as Real) * tex_y;
                    let vc_y = (v - 0.5) * 2.0;

                    let point = Vector2::new(vc_x, vc_y);

                    let w0 = edge(p1.xy(), p2.xy(), point);
                    let w1 = edge(p2.xy(), p0.xy(), point);
                    let w2 = edge(p0.xy(), p1.xy(), point);

                    if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                        let u = w0 / area;
                        let v = w1 / area;
                        let w = w2 / area;

                        //
                        // Depth
                        //

                        // Check if the point we're intersecting is closer
                        // TODO: Make this faster?
                        let clip_pos = bary_interpolate((u, v, w), (p0, p1, p2));

                        // Normalize d across the clip plane
                        let depth = depth_buffer.buffer_read(x, y, 0);

                        let clip_depth = clip_pos.z() / FAR;

                        if clip_depth > depth || clip_depth < 0.0 {
                            continue;
                        }

                        depth_buffer.buffer_write(x, y, 0, clip_depth);

                        //
                        // Color
                        //
                        let norm = bary_interpolate(
                            (u, v, w),
                            (v0.normal, v1.normal, v2.normal)
                        ).normalize();

                        let fac = norm.dot(light).rl_saturate();
                        let fac = bary_interpolate(
                            (u, v, w),
                            (l0, l1, l2)
                        );

                        let global = Vector3::from_scalar(0.01);
                        let shaded = Vector3::from_scalar(fac).max(global);

                        let albedo = Vector3::from_scalar(1.0);

                        let lit = albedo * shaded;

                        let col = Vector4::from_w(lit, 1.0);

                        buffer.buffer_write(x, y, 0, col)
                    }

                    //buffer.buffer_write(x, y, 0, Vector4::from_w(debug_col, 1.0));
                }
            }
        }

        /*
        //
        // Save the output
        //
        {
            let span = span!(Level::TRACE, "Save Image");
            let _guard = span.enter();

            let name = format!("./result/output.{}.png", frame);

            buffer.save_as_image(name);
        }
         */
    }

    {
        let _gif_time = Dropwatch::new("Create GIF");

        // Jank, let's do this better :P
        let image = File::create("./result/animation.gif").unwrap();
        let mut gif = image::codecs::gif::GifEncoder::new(image);

        gif.set_repeat(Repeat::Infinite).expect("Failed to set repeat!");

        for frame in BEGIN_FRAME..END_FRAME {
            let img = image::io::Reader::open(format!("./result/output.{}.png", frame)).expect("Failed to load frame!").decode();
            let dynamic_image = img.unwrap();
            let rgba = dynamic_image.as_rgba8().expect("Failed to convert to rgba!");

            let duration = Duration::from_secs_f64(TIME_PER_FRAME as f64);
            let delay = Delay::from_saturating_duration(duration);

            let frame = image::Frame::from_parts(rgba.clone(), 0, 0, delay);

            gif.encode_frame(frame).expect("Failed to encode frame!");
        }
    }
     */
}
