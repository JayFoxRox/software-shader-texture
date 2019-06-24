Implementing the OpenGL texture lookup in software.
Intended to prototype software rendering using Vulkan or OpenGL.

This might be useful for video game emulators which need more fine grained control over the renderer, but still want to benefit from the hardware rasterizer.

Example applications could be GPU implementations of upscaling (HQ4x or xBRZ), unswizzling, texture decompression, per-component LUTs and other effects which require raw-pixel data.

As this is written in GLSL (or possibly Spir-V later), it should be possible to run this code as part of a compute-shader; possibly with custom rasterizer.

Very incomplete and hacky.
