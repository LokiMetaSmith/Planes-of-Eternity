use image::GenericImageView;

// --- Procedural Generation Helpers ---

fn hash(x: i32, y: i32) -> f32 {
    let mut n = x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263);
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n as f32) / (std::i32::MAX as f32)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn noise_2d(x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - x.floor();
    let yf = y - y.floor();

    let bl = hash(xi, yi);
    let br = hash(xi + 1, yi);
    let tl = hash(xi, yi + 1);
    let tr = hash(xi + 1, yi + 1);

    // Smoothstep
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);

    let b = lerp(bl, br, u);
    let t = lerp(tl, tr, u);
    lerp(b, t, v)
}

fn fbm(x: f32, y: f32, octaves: i32) -> f32 {
    let mut v = 0.0;
    let mut a = 0.5;
    let mut scale = 1.0;
    for _ in 0..octaves {
        v += a * noise_2d(x * scale, y * scale);
        a *= 0.5;
        scale *= 2.0;
    }
    // Normalize roughly to -1..1 then 0..1
    // Simplification: just return abs value or clamp
    (v + 1.0) * 0.5
}

// -------------------------------------

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self, image::ImageError> {
        let img = match image::load_from_memory(bytes) {
            Ok(img) => img,
            Err(e) => {
                 // Try forcing PNG if format detection failed
                 image::load_from_memory_with_format(bytes, image::ImageFormat::Png).map_err(|_| e)?
            }
        };
        Self::from_image(device, queue, &img, label)
    }

    pub fn create_fallback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &[255, 0, 255, 255], // Magenta
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: &str,
    ) -> Result<Self, image::ImageError> {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }

    pub fn create_procedural_atlas(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let size = 256;
        let mut pixels = vec![0u8; size * size * 4];

        for y in 0..size {
            for x in 0..size {
                let idx = (x + y * size) * 4;
                let u = x as f32 / size as f32;
                let v = y as f32 / size as f32;

                // 4 Quadrants
                // TL (0,0): Stone
                // TR (1,0): Lava
                // BL (0,1): Fire
                // BR (1,1): Wood

                if u < 0.5 && v < 0.5 {
                    // TL: Stone
                    let lx = (x % 128) as f32 * 0.1;
                    let ly = (y % 128) as f32 * 0.1;

                    let n = fbm(lx, ly, 3);
                    let val = (100.0 + n * 50.0) as u8;

                    // Simple "Cobble" grid
                    let gx = (lx * 2.0).fract();
                    let gy = (ly * 2.0).fract();
                    let edge = if gx < 0.1 || gy < 0.1 { 0.5 } else { 1.0 };

                    let f = (val as f32 * edge) as u8;

                    pixels[idx] = f;
                    pixels[idx+1] = f;
                    pixels[idx+2] = f;
                    pixels[idx+3] = 255;

                } else if u >= 0.5 && v < 0.5 {
                    // TR: Lava
                    let lx = ((x - 128) % 128) as f32 * 0.05;
                    let ly = (y % 128) as f32 * 0.05;

                    let n = fbm(lx, ly, 4);
                    // Veins
                    let heat = (n * 2.0).clamp(0.0, 1.0);

                    // Dark crust (50,0,0) to Bright Orange (255, 200, 0)
                    let r = lerp(50.0, 255.0, heat) as u8;
                    let g = lerp(0.0, 200.0, heat * heat) as u8;
                    let b = 0;

                    pixels[idx] = r;
                    pixels[idx+1] = g;
                    pixels[idx+2] = b;
                    pixels[idx+3] = 255;

                } else if u < 0.5 && v >= 0.5 {
                     // BL: Fire
                    let lx = (x % 128) as f32 * 0.05;
                    let ly = ((y - 128) % 128) as f32 * 0.1;

                    let n = fbm(lx, ly - (lx * 0.5), 3); // Skew

                    let r = (200.0 + 55.0 * n) as u8;
                    let g = (100.0 + 100.0 * n) as u8;
                    pixels[idx] = r;
                    pixels[idx+1] = g;
                    pixels[idx+2] = 0;
                    pixels[idx+3] = 255;

                } else {
                     // BR: Wood
                    let lx = ((x - 128) % 128) as f32 * 0.1;
                    let ly = ((y - 128) % 128) as f32 * 0.02; // Stretched

                    let grain = noise_2d(lx * 2.0, ly * 0.5);
                    let wave = ((lx * 2.0 + grain).sin() + 1.0) * 0.5;

                    let base = 80.0;
                    let var = 40.0 * wave;

                    pixels[idx] = (base + var) as u8;
                    pixels[idx+1] = ((base + var) * 0.6) as u8;
                    pixels[idx+2] = ((base + var) * 0.3) as u8;
                    pixels[idx+3] = 255;
                }
            }
        }

        let texture_size = wgpu::Extent3d {
            width: size as u32,
            height: size as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel Atlas"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * size as u32),
                rows_per_image: Some(size as u32),
            },
            texture_size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}
