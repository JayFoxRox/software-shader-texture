// This code often cites https://www.khronos.org/registry/OpenGL/specs/gl/glspec20.pdf
// It is written based on https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/textureQueryLod.xhtml

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

unsigned int min_filter;
unsigned int mag_filter;

void APIENTRY DebugCallbackGL(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam) {
  fprintf(stderr, "DebugCallbackGL: %s\n", message);
}

int main() {

  if (!glfwInit()) {
      fprintf(stderr, "Error: GLFW initialization failed\n");
  }

  //FIXME: Only necessary for textureQueryLod debug purposes
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(640, 480, "My Title", NULL, NULL);
  if (!window) {
    fprintf(stderr, "Error: Window or context creation failed\n");
  }

  glfwMakeContextCurrent(window);

  GLenum err = glewInit();
  if (GLEW_OK != err) {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }
  fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));


  // Enable OpenGL debugging
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(DebugCallbackGL, 0);

  printf("GL_VENDOR: %s\n", glGetString(GL_VENDOR));
  printf("GL_RENDERER: %s\n", glGetString(GL_RENDERER));
  printf("GL_VERSION: %s\n", glGetString(GL_VERSION));
  printf("\n");

  const unsigned int width = 1024;
  const unsigned int height = width;
  uint32_t* pixels = malloc(width * height * 4);
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      uint32_t color = 0;
      color = 0x00000000 | (x << 16) | (y << 8);

      //FIXME: Checkerboard is interesting to see interpolation
      color = ((x + (y & 8)) & 8) ? 0xFFFFFFFF : 0xFF0000FF;

      pixels[y * width + x] = color;
    }
  }

  // Initialize the texture with raw 8-bit data in it.
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
  glGenerateMipmap(GL_TEXTURE_2D);

#if 1
  // Compute shader that just converts every 8-bit value into a float and stores the result in a buffer.

  const char* vs_src[] = {
R"GLSL(
#version 400

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 uv;

out vec2 g_uv;

void main() {
  gl_Position = position;
  g_uv = uv;
}
)GLSL"
  };

  const char* fs_src[] = {
R"GLSL(
#version 400

uniform sampler2D pixels;

uniform bool hw;

// Must match GL.h
#define NEAREST				0x2600
#define LINEAR				0x2601
#define NEAREST_MIPMAP_NEAREST		0x2700
#define NEAREST_MIPMAP_LINEAR		0x2702
#define LINEAR_MIPMAP_NEAREST		0x2701
#define LINEAR_MIPMAP_LINEAR			0x2703

uniform uint TEXTURE_MIN_FILTER;
uniform uint TEXTURE_MAG_FILTER;

in vec2 g_uv;

float scale_factor_polygon(vec2 P) {
  return max(length(dFdx(P)), length(dFdy(P)));
}

float lambda_polygon(vec2 P) {
  float p = scale_factor_polygon(P);
  return log2(p);
}

float lambda_prime_polygon(vec2 P) {
  return lambda_polygon(P); //FIXME: + bias
}

float scale_factor_line(vec2 P, vec2 p1, vec2 p2) {
  vec2 d = p2 - p1;
  vec2 dx = dFdx(P)*d.x;
  vec2 dy = dFdy(P)*d.y;
  return length(dx + dy) / length(d);
}

float lambda_line(vec2 P, vec2 p1, vec2 p2) {
  float p = scale_factor_line(P, p1, p2);
  return log2(p);
}

float lambda_prime_line(vec2 P, vec2 p1, vec2 p2) {
  return lambda_line(P, p1, p2); //FIXME: + bias
}

float get_c() {
  // If the magnification-filter is given by LINEAR and the minification-filter is given by NEAREST_MIPMAP_NEAREST or NEAREST_MIPMAP_LINEAR, then c=0.5.
  // This is done to ensure that a minified texture does not appear "sharper" than a magnified texture.
  bool mag_hack = (TEXTURE_MAG_FILTER == LINEAR);
  bool min_hack = (TEXTURE_MIN_FILTER == NEAREST_MIPMAP_NEAREST) ||
                  (TEXTURE_MIN_FILTER == NEAREST_MIPMAP_LINEAR);
  if (mag_hack && min_hack) {
    return 0.5;
  }

  // Otherwise c=0
  return 0.0;
}

float ComputeAccessedLod(float computedLod) {

#define TEXTURE_MIN_LOD 0
#define TEXTURE_MAX_LOD 100
#define maxAccessibleLevel 100

  // Clamp the computed LOD according to the texture LOD clamps.
  computedLod = clamp(computedLod, float(TEXTURE_MIN_LOD), float(TEXTURE_MAX_LOD));

  // Clamp the computed LOD to the range of accessible levels.
  computedLod = clamp(computedLod, 0.0, float(maxAccessibleLevel));

  // Return a value according to the min filter.
  if ((TEXTURE_MIN_FILTER == NEAREST) ||
      (TEXTURE_MIN_FILTER == LINEAR)) {
    return 0.0;
  }

  if ((TEXTURE_MIN_FILTER == NEAREST_MIPMAP_NEAREST) ||
      (TEXTURE_MIN_FILTER == LINEAR_MIPMAP_NEAREST)) {
    return ceil(computedLod + 0.5) - 1.0;
  }

  // Default case
  return computedLod;
}

// This isn't available below GLSL 4.0, so we manually compute it
vec2 sw_textureQueryLod_polygon(sampler2D sampler, vec2 P) {
  //FIXME: calculate tex-coord properly
  float lambda_prime = lambda_prime_polygon(P * textureSize(sampler, 0));
  return vec2(ComputeAccessedLod(lambda_prime), lambda_prime);
}

//FIXME: where p1 and p2 are window coordinates.. can probably isolate them
vec2 sw_textureQueryLod_line(sampler2D sampler, vec2 P, vec2 p1, vec2 p2) {
  float lambda_prime = lambda_prime_line(P * textureSize(sampler, 0), p1, p2);
  return vec2(ComputeAccessedLod(lambda_prime), lambda_prime);
}

vec4 texelFetchOffset_Wrap(sampler2D sampler, ivec2 P, int lod, ivec2 offset) {
  ivec2 s = textureSize(sampler, lod);
  return texelFetch(sampler, (P + offset) % s, lod);
}

vec4 sw_texture_nearest_lod(sampler2D sampler, vec2 P, int lod) {
  ivec2 s = textureSize(sampler, lod);
  vec2 t = s + P * (s - 1) + 0.5;
  ivec2 i = ivec2(t);

  return texelFetchOffset_Wrap(sampler, i, lod, ivec2(0, 0));
}

vec4 sw_texture_linear_lod(sampler2D sampler, vec2 P, int lod) {
  ivec2 s = textureSize(sampler, lod);
  vec2 t = s + P * s - 0.5;
  ivec2 i = ivec2(t);
  vec2 f = fract(t);

  vec4 v00 = texelFetchOffset_Wrap(sampler, i, lod, ivec2(0, 0));
  vec4 v10 = texelFetchOffset_Wrap(sampler, i, lod, ivec2(1, 0));
  vec4 v11 = texelFetchOffset_Wrap(sampler, i, lod, ivec2(1, 1));
  vec4 v01 = texelFetchOffset_Wrap(sampler, i, lod, ivec2(0, 1));

  vec4 v0 = mix(v00, v01, f.y);
  vec4 v1 = mix(v10, v11, f.y);
  return mix(v0, v1, f.x);
}

// Returns the value of the texture element that is nearest (in Manhattan distance) to the center of the pixel being textured.
vec4 sw_texture_nearest(sampler2D sampler, vec2 P) {
  return sw_texture_nearest_lod(sampler, P, 0);
}

// Returns the weighted average of the four texture elements that are closest to the center of the pixel being textured. These can include border texture elements, depending on the values of GL_TEXTURE_WRAP_S and GL_TEXTURE_WRAP_T, and on the exact mapping.
vec4 sw_texture_linear(sampler2D sampler, vec2 P) {
  return sw_texture_linear_lod(sampler, P, 0);
}

// Chooses the mipmap that most closely matches the size of the pixel being textured and uses the GL_NEAREST criterion (the texture element nearest to the center of the pixel) to produce a texture value.
vec4 sw_texture_nearest_mipmap_nearest(sampler2D sampler, vec2 P) {
  float lod_t = sw_textureQueryLod_polygon(sampler, P).x;
  int lod_i = int(lod_t);

  return sw_texture_nearest_lod(sampler, P, lod_i);
}

// Chooses the mipmap that most closely matches the size of the pixel being textured and uses the GL_LINEAR criterion (a weighted average of the four texture elements that are closest to the center of the pixel) to produce a texture value.
vec4 sw_texture_linear_mipmap_nearest(sampler2D sampler, vec2 P) {
  float lod_t = sw_textureQueryLod_polygon(sampler, P).x;
  int lod_i = int(lod_t);

  return sw_texture_linear_lod(sampler, P, lod_i);
}

// Chooses the two mipmaps that most closely match the size of the pixel being textured and uses the GL_NEAREST criterion (the texture element nearest to the center of the pixel) to produce a texture value from each mipmap. The final texture value is a weighted average of those two values.
vec4 sw_texture_nearest_mipmap_linear(sampler2D sampler, vec2 P) {
  float lod_t = sw_textureQueryLod_polygon(sampler, P).x;
  int lod_i = int(lod_t);
  float lod_f = fract(lod_t);

  vec4 v0 = sw_texture_nearest_lod(sampler, P, lod_i);
  vec4 v1 = sw_texture_nearest_lod(sampler, P, lod_i + 1); 

  return mix(v0, v1, lod_f);
}

// Chooses the two mipmaps that most closely match the size of the pixel being textured and uses the GL_LINEAR criterion (a weighted average of the four texture elements that are closest to the center of the pixel) to produce a texture value from each mipmap. The final texture value is a weighted average of those two values.
vec4 sw_texture_linear_mipmap_linear(sampler2D sampler, vec2 P) {
  float lod_t = sw_textureQueryLod_polygon(sampler, P).x;
  int lod_i = int(lod_t);
  float lod_f = fract(lod_t);

  vec4 v0 = sw_texture_linear_lod(sampler, P, lod_i);
  vec4 v1 = sw_texture_linear_lod(sampler, P, lod_i + 1); 

  return mix(v0, v1, lod_f);
}

vec4 sw_texture(sampler2D sampler, vec2 P) {

  // If lambda(x, y) is less than or equal  to  the  constantc(described below in section 3.8.9) the texture is said to be magnified; if it is greater, the texture is minified.
  float lambda = lambda_polygon(P * textureSize(sampler, 0));
  uint chosen_filter = (lambda <= get_c()) ? TEXTURE_MAG_FILTER : TEXTURE_MIN_FILTER;

  if (chosen_filter == NEAREST) {
    return sw_texture_nearest(sampler, P);
  } else if (chosen_filter == LINEAR) {
    return sw_texture_linear(sampler, P);
  } else if (chosen_filter == LINEAR_MIPMAP_LINEAR) {
    return sw_texture_linear_mipmap_linear(sampler, P);
  } else {
    //FIXME: Add other filters
    return vec4(1.0, 0.0, 0.0, 1.0);
  }
}

void main() {
  vec2 uv = gl_FragCoord.xy;

  vec4 color = sw_texture(pixels, g_uv);
  //color = vec4(sw_textureQueryLod_polygon(pixels, g_uv).y / 5.0);

  if (hw) { // || gl_FragCoord.x > 320) {
    color = texture(pixels, g_uv);
    //color = vec4(textureQueryLod(pixels, g_uv).y / 5.0);
  }

  gl_FragColor = color;
}
)GLSL"
  };

  
  GLuint program = glCreateProgram();
  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(vs, 1, (const GLchar**)&vs_src, NULL);
  glCompileShader(vs);
  glShaderSource(fs, 1, (const GLchar**)&fs_src, NULL);
  glCompileShader(fs);
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  GLint logLength;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
  if (logLength > 0) {
    GLchar* infolog = (GLchar*)malloc(logLength);
    glGetProgramInfoLog(program, logLength, NULL, infolog);
    fprintf(stderr, "%s", infolog);
    free(infolog);
    return 1;
  }
#endif

  float w = 1.0f;
  float zn = 1.0f;
  float zf = zn;

  typedef struct {
    float x, y, z, w;
    float u, v;
  } Vertex;
  Vertex vertices[4] = {
    { -0.9f, -1.0f, zn, w, 0.0f, 0.0f },
    {  0.9f, -1.0f, zn, w, 1.0f, 0.0f },
    { -0.3f,  1.0f, zf, w, 0.0f, 1.0f },
    {  0.3f,  1.0f, zf, w, 1.0f, 1.0f }
  };

  GLint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  GLint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)16);

  glDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);

  while(1) {
    glfwPollEvents();
    bool hw = (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS);
    bool linear = (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);

    if (linear) {

      //FIXME: GL_LINEAR_MIPMAP_LINEAR

      min_filter = GL_LINEAR_MIPMAP_LINEAR; //GL_LINEAR;
      mag_filter = GL_LINEAR;
    } else {
      min_filter = GL_NEAREST;
      mag_filter = GL_NEAREST;      
    }

    glClearColor(1.0f, hw ? 0.5f : 0.0f, 1.0f, 1.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    // Set up texture unit
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);

    // Set up the program
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "pixels"), 0);
    glUniform1ui(glGetUniformLocation(program, "TEXTURE_MIN_FILTER"), min_filter);
    glUniform1ui(glGetUniformLocation(program, "TEXTURE_MAG_FILTER"), mag_filter);
    glUniform1i(glGetUniformLocation(program, "hw"), hw);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFinish();
    glfwSwapBuffers(window);
  }
}