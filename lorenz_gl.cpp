// lorenz_gl.cpp — 로렌츠 끌개: HIP(ROCm)로 계산, OpenGL로 렌더링, 둘이 같은 GPU 버퍼를 공유 (interop)
//
// 의존성 (Ubuntu):  sudo apt install libglfw3-dev libgl-dev
//
// GPU(ROCm) 빌드 (Strix Halo iGPU = gfx1151):
//   hipcc -O3 -std=c++17 -fopenmp -DUSE_HIP --offload-arch=gfx1151 -x hip lorenz_gl.cpp -o lorenz_gl -lglfw -lGL
// CPU 전용 빌드 (OpenMP로 계산, OpenGL로 렌더링):
//   g++ -O3 -march=native -std=c++17 -fopenmp lorenz_gl.cpp -o lorenz_gl -lglfw -lGL
//
// 실행:  ./lorenz_gl [입자수] [장면]     기본: GPU면 4M, CPU면 512K / 장면 1
//        ./lorenz_gl 16000000 2
//
// 조작:  좌클릭 드래그 = 회전   휠 = 줌   A = 자동 회전   Space = 일시정지
//        1 = 나비효과 (지름 0.01짜리 공에서 출발)   2 = 공간 전체에 무작위로 뿌리기   R = 리셋
//        [ ] = 입자 수 절반/두 배   , . = rho -/+   - = = 시간 속도   C = 색 모드   T = 잔상
//        B N = 밝기 -/+   P = 점 크기   G = GPU <-> CPU   V = vsync   F = 전체화면   Q/Esc = 종료
//        (상태는 창 제목에 표시)
//
// 데이터 흐름 (GPU, interop):
//   OpenGL VBO (입자 xyz + 색 파라미터) ── hipGraphicsMapResources ──> HIP 커널이 그 자리에서 RK4 적분
//   ── unmap ──> OpenGL이 같은 VBO를 GL_POINTS로 그림. 호스트로 복사 0번.
// 렌더링:
//   ① 누적 텍스처(RGBA32F)를 decay배로 어둡게 (잔상)
//   ② 입자를 점으로, 가산 블렌딩(GL_ONE, GL_ONE)으로 누적 -> 같은 픽셀에 몰려도 ROP가 처리, atomic 경합 없음
//   ③ 밉맵 생성 -> 흐린 밉 레벨을 섞어 글로우
//   ④ 통계 패스(l, l², 켜진 픽셀)를 밉맵으로 평균 내서 자동 노출
//   ⑤ 톤매핑해서 화면에

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#if __has_include(<hip/hip_gl_interop.h>)
#include <hip/hip_gl_interop.h>
#else
#include <hip/amd_detail/amd_hip_gl_interop.h>
#endif
#define HD __host__ __device__
#else
#define HD
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ───────────────────────── 적분 (CPU/GPU 공용) ─────────────────────────
struct V4 {
    float x, y, z, w;  // w = 색 파라미터 (초기 순서 0~1)
};

struct LP {
    float s, r, b, dt;
    int K;
};

HD inline void deriv(float x, float y, float z, const LP& L, float& dx, float& dy, float& dz) {
    dx = L.s * (y - x);
    dy = x * (L.r - z) - y;
    dz = x * y - L.b * z;
}

HD inline void advance(V4& p, const LP& L) {
    float x = p.x, y = p.y, z = p.z, h = L.dt;
    for (int k = 0; k < L.K; ++k) {
        float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
        deriv(x, y, z, L, k1x, k1y, k1z);
        deriv(x + 0.5f * h * k1x, y + 0.5f * h * k1y, z + 0.5f * h * k1z, L, k2x, k2y, k2z);
        deriv(x + 0.5f * h * k2x, y + 0.5f * h * k2y, z + 0.5f * h * k2z, L, k3x, k3y, k3z);
        deriv(x + h * k3x, y + h * k3y, z + h * k3z, L, k4x, k4y, k4z);
        x += h / 6.f * (k1x + 2.f * k2x + 2.f * k3x + k4x);
        y += h / 6.f * (k1y + 2.f * k2y + 2.f * k3y + k4y);
        z += h / 6.f * (k1z + 2.f * k2z + 2.f * k3z + k4z);
    }
    p.x = x;
    p.y = y;
    p.z = z;
}

#ifdef USE_HIP
#define HIPCHECK(e)                                                                    \
    do {                                                                               \
        hipError_t e_ = (e);                                                           \
        if (e_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP error: %s (%s:%d)\n", hipGetErrorString(e_),     \
                         __FILE__, __LINE__);                                          \
            std::exit(1);                                                              \
        }                                                                              \
    } while (0)

__global__ void k_lorenz(V4* P, int n, LP L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        V4 p = P[i];
        advance(p, L);
        P[i] = p;
    }
}
#endif

// ─────────────────────────── 셰이더 ───────────────────────────
static const char* VS_POINTS = R"(#version 330 core
layout(location = 0) in vec4 aPos;
uniform mat4 uMVP;
uniform int uCMode;
uniform float uS, uR, uB, uPointSize, uDist;
out vec3 vCol;
vec3 hsv(float h) { return clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0); }
void main() {
    vec3 p = aPos.xyz;
    vec4 c = uMVP * vec4(p.x, p.z - 25.0, p.y, 1.0);   // 로렌츠 z축이 화면 위쪽
    gl_Position = c;
    gl_PointSize = uPointSize;
    float hue;
    if (uCMode == 0) {
        hue = 0.85 * aPos.w;
    } else {
        vec3 d = vec3(uS * (p.y - p.x), p.x * (uR - p.z) - p.y, p.x * p.y - uB * p.z);
        hue = 0.66 - 0.66 * clamp(length(d) / 160.0, 0.0, 1.0);
    }
    float fog = clamp(1.0 + (uDist - c.w) / 60.0, 0.45, 1.6);  // 가까울수록 밝게
    vCol = (0.25 + 0.75 * hsv(hue)) * fog;
}
)";

static const char* FS_POINTS = R"(#version 330 core
in vec3 vCol;
out vec4 o;
void main() { o = vec4(vCol, 1.0); }
)";

static const char* VS_FULL = R"(#version 330 core
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);   // 화면을 덮는 삼각형 하나
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

static const char* FS_DECAY = R"(#version 330 core
out vec4 o;
void main() { o = vec4(0.0); }   // 색은 안 쓰고 블렌드 상수로 dst *= decay
)";

static const char* FS_STATS = R"(#version 330 core
in vec2 uv;
uniform sampler2D uAcc;
out vec4 o;
void main() {
    vec3 a = textureLod(uAcc, uv, 0.0).rgb;
    float l = max(a.r, max(a.g, a.b));
    o = vec4(l, l * l, l > 1e-4 ? 1.0 : 0.0, 0.0);   // 밉맵 꼭대기 = 화면 전체 평균
}
)";

static const char* FS_TONE = R"(#version 330 core
in vec2 uv;
uniform sampler2D uAcc;
uniform float uExpo;
out vec4 o;
vec3 tm(vec3 a) {
    float l = max(a.r, max(a.g, a.b));
    float s = l > 1e-6 ? (1.0 - exp(-l * uExpo)) / l : 0.0;    // 밝기 기준 -> 색상 유지
    vec3 v = a * s;
    float hot = clamp((l * uExpo - 2.5) / 6.0, 0.0, 1.0) * 0.55;  // 아주 밀집한 곳만 하얗게
    return v + (1.0 - v) * hot;
}
void main() {
    vec3 a = textureLod(uAcc, uv, 0.0).rgb;
    vec3 glow = textureLod(uAcc, uv, 2.0).rgb * 0.55 + textureLod(uAcc, uv, 4.0).rgb * 0.45;
    vec3 c = tm(a) + 0.30 * tm(glow * 0.6) + vec3(0.010, 0.012, 0.028);
    o = vec4(clamp(c, 0.0, 1.0), 1.0);
}
)";

static GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "셰이더 컴파일 실패:\n%s\n", log);
        std::exit(1);
    }
    return s;
}

static GLuint program(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    GLuint a = compile(GL_VERTEX_SHADER, vs), b = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, a);
    glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        std::fprintf(stderr, "셰이더 링크 실패:\n%s\n", log);
        std::exit(1);
    }
    glDeleteShader(a);
    glDeleteShader(b);
    return p;
}

// ─────────────────────────── 행렬 (열 우선) ───────────────────────────
struct M4 {
    float m[16];
};
static M4 mul(const M4& a, const M4& b) {
    M4 r{};
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}
static M4 ident() {
    M4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1;
    return r;
}
static M4 persp(float fovy, float aspect, float n, float f) {
    M4 r{};
    float t = 1.f / std::tan(fovy * 0.5f);
    r.m[0] = t / aspect;
    r.m[5] = t;
    r.m[10] = (f + n) / (n - f);
    r.m[11] = -1;
    r.m[14] = 2 * f * n / (n - f);
    return r;
}
static M4 rotX(float a) {
    M4 r = ident();
    r.m[5] = std::cos(a); r.m[6] = std::sin(a);
    r.m[9] = -std::sin(a); r.m[10] = std::cos(a);
    return r;
}
static M4 rotY(float a) {
    M4 r = ident();
    r.m[0] = std::cos(a); r.m[2] = -std::sin(a);
    r.m[8] = std::sin(a); r.m[10] = std::cos(a);
    return r;
}
static M4 trans(float x, float y, float z) {
    M4 r = ident();
    r.m[12] = x; r.m[13] = y; r.m[14] = z;
    return r;
}

// ─────────────────────────── 앱 ───────────────────────────
struct App {
    GLFWwindow* win = nullptr;
    int fbw = 0, fbh = 0;
    GLuint vao = 0, vbo = 0, emptyVao = 0;
    GLuint accTex = 0, accFbo = 0, statTex = 0, statFbo = 0;
    GLuint progPts = 0, progDecay = 0, progStats = 0, progTone = 0;

    int n = 0, mode = 1, cmode = 0;
    std::vector<V4> host;
    LP L{10.f, 28.f, 8.f / 3.f, 0.004f, 3};
    float yaw = 0.6f, pitch = 0.25f, zoom = 1.f, psize = 1.f, gain = 1.f, expo = -1.f;
    bool paused = false, autorot = true, trails = true, vsync = true, fullscreen = false;
    bool dragging = false, cam_moved = false;
    double lastx = 0, lasty = 0, simt = 0, sim_ms = 0;
    int wx = 100, wy = 100, ww = 1600, wh = 900;

    bool gpu_ok = false, use_gpu = false, interop = false;
    std::string gpu_name;
#ifdef USE_HIP
    hipGraphicsResource_t res = nullptr;
    V4* dBuf = nullptr;  // interop이 안 될 때 쓰는 일반 GPU 버퍼
#endif

    // ── 입자 초기화 ──
    void fill(int m) {
        mode = m;
        host.resize(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            std::mt19937 rng((m == 1 ? 1234u : 99u) + (unsigned)i);
            std::uniform_real_distribution<float> U(0.f, 1.f);
            V4 p;
            if (m == 1) {  // 지름 0.01 공, x 오프셋 순서 = 색
                float u = 2.f * (i + 0.5f) / n - 1.f, r = 0.005f, q = std::sqrt(std::max(0.f, 1.f - u * u));
                p = {-8.f + r * u, 7.f + r * q * (2 * U(rng) - 1), 27.f + r * q * (2 * U(rng) - 1), (float)i / n};
            } else {
                p = {-25.f + 50.f * U(rng), -30.f + 60.f * U(rng), 55.f * U(rng), (float)i / n};
            }
            host[i] = p;
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(V4) * n, host.data());
#ifdef USE_HIP
        if (dBuf) HIPCHECK(hipMemcpy(dBuf, host.data(), sizeof(V4) * n, hipMemcpyHostToDevice));
#endif
        clear_acc();
        simt = 0;
    }

    // ── VBO (입자 상태) 생성 + HIP 등록 ──
    void make_particles(int N) {
        gpu_release();
        if (vbo) glDeleteBuffers(1, &vbo);
        n = N;
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(V4) * n, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(V4), nullptr);
        gpu_register();
        fill(mode);
    }

    void gpu_register() {
#ifdef USE_HIP
        if (!gpu_ok) return;
        glFinish();
        hipError_t e = hipGraphicsGLRegisterBuffer(&res, vbo, hipGraphicsRegisterFlagsNone);
        interop = (e == hipSuccess);
        if (!interop) {
            res = nullptr;
            (void)hipGetLastError();
            HIPCHECK(hipMalloc((void**)&dBuf, sizeof(V4) * n));
            static bool warned = false;
            if (!warned) {
                std::fprintf(stderr, "HIP-GL interop 등록 실패 (%s) -> GPU 계산 후 복사 방식으로 동작\n",
                             hipGetErrorString(e));
                warned = true;
            }
        }
#endif
    }

    void gpu_release() {
#ifdef USE_HIP
        if (res) {
            HIPCHECK(hipGraphicsUnregisterResource(res));
            res = nullptr;
        }
        if (dBuf) {
            HIPCHECK(hipFree(dBuf));
            dBuf = nullptr;
        }
#endif
    }

    // ── 한 프레임 적분 ──
    void simulate() {
        LP Lf = L;
        if (paused) return;
        double t0 = glfwGetTime();
        if (use_gpu && gpu_ok) {
#ifdef USE_HIP
            const int BS = 256, G = (n + BS - 1) / BS;
            if (interop) {
                V4* p = nullptr;
                size_t sz = 0;
                HIPCHECK(hipGraphicsMapResources(1, &res, 0));
                HIPCHECK(hipGraphicsResourceGetMappedPointer((void**)&p, &sz, res));
                hipLaunchKernelGGL(k_lorenz, dim3(G), dim3(BS), 0, 0, p, n, Lf);
                HIPCHECK(hipGetLastError());
                HIPCHECK(hipDeviceSynchronize());
                HIPCHECK(hipGraphicsUnmapResources(1, &res, 0));
            } else {
                hipLaunchKernelGGL(k_lorenz, dim3(G), dim3(BS), 0, 0, dBuf, n, Lf);
                HIPCHECK(hipGetLastError());
                HIPCHECK(hipMemcpy(host.data(), dBuf, sizeof(V4) * n, hipMemcpyDeviceToHost));
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(V4) * n, host.data());
            }
#endif
        } else {
            V4* P = host.data();
            const int N = n;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; ++i) advance(P[i], Lf);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(V4) * n, host.data());
        }
        sim_ms = 0.9 * sim_ms + 0.1 * (glfwGetTime() - t0) * 1000;
        simt += L.dt * L.K;
    }

    void set_gpu(bool on) {
        if (on && !gpu_ok) return;
        if (on == use_gpu) return;
#ifdef USE_HIP
        if (!on && interop) {  // GPU -> CPU: VBO에 있는 최신 상태를 호스트로
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(V4) * n, host.data());
        }
        if (on && dBuf) HIPCHECK(hipMemcpy(dBuf, host.data(), sizeof(V4) * n, hipMemcpyHostToDevice));
#endif
        use_gpu = on;
    }

    // ── 누적 텍스처 ──
    void make_targets() {
        for (GLuint* t : {&accTex, &statTex})
            if (*t) glDeleteTextures(1, t);
        for (GLuint* f : {&accFbo, &statFbo})
            if (*f) glDeleteFramebuffers(1, f);
        auto mk = [&](GLuint& tex, GLuint& fbo) {
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, fbw, fbh, 0, GL_RGBA, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glGenerateMipmap(GL_TEXTURE_2D);
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                std::fprintf(stderr, "RGBA32F 프레임버퍼 생성 실패\n");
                std::exit(1);
            }
        };
        mk(accTex, accFbo);
        mk(statTex, statFbo);
        clear_acc();
    }

    void clear_acc() {
        if (!accFbo) return;
        glBindFramebuffer(GL_FRAMEBUFFER, accFbo);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    // ── 렌더링 ──
    void render() {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        if (w != fbw || h != fbh) {
            fbw = std::max(1, w);
            fbh = std::max(1, h);
            make_targets();
        }
        float dist = 72.f / zoom;
        M4 P = persp(0.8f, (float)fbw / fbh, 1.f, 500.f);
        M4 V = mul(trans(0, 0, -dist), mul(rotX(pitch), rotY(-yaw)));
        M4 MVP = mul(P, V);

        // ① 잔상 감쇠: dst = dst * decay (블렌드 상수 사용)
        glBindFramebuffer(GL_FRAMEBUFFER, accFbo);
        glViewport(0, 0, fbw, fbh);
        glBindVertexArray(emptyVao);
        float decay = !trails ? 0.f : (cam_moved && !autorot ? 0.35f : 0.82f);
        if (decay <= 0.f) {
            glClearColor(0, 0, 0, 0);
            glClear(GL_COLOR_BUFFER_BIT);
        } else {
            glEnable(GL_BLEND);
            glBlendColor(decay, decay, decay, decay);
            glBlendFunc(GL_ZERO, GL_CONSTANT_COLOR);
            glUseProgram(progDecay);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        }

        // ② 입자를 가산 블렌딩으로 누적
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUseProgram(progPts);
        glUniformMatrix4fv(glGetUniformLocation(progPts, "uMVP"), 1, GL_FALSE, MVP.m);
        glUniform1i(glGetUniformLocation(progPts, "uCMode"), cmode);
        glUniform1f(glGetUniformLocation(progPts, "uS"), L.s);
        glUniform1f(glGetUniformLocation(progPts, "uR"), L.r);
        glUniform1f(glGetUniformLocation(progPts, "uB"), L.b);
        glUniform1f(glGetUniformLocation(progPts, "uPointSize"), psize);
        glUniform1f(glGetUniformLocation(progPts, "uDist"), dist);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, n);
        glDisable(GL_BLEND);

        // ③ 글로우용 밉맵
        glBindTexture(GL_TEXTURE_2D, accTex);
        glGenerateMipmap(GL_TEXTURE_2D);

        // ④ 자동 노출 통계: (l, l², 켜짐) -> 밉맵 꼭대기 1픽셀 = 전체 평균
        glBindFramebuffer(GL_FRAMEBUFFER, statFbo);
        glBindVertexArray(emptyVao);
        glUseProgram(progStats);
        glUniform1i(glGetUniformLocation(progStats, "uAcc"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accTex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindTexture(GL_TEXTURE_2D, statTex);
        glGenerateMipmap(GL_TEXTURE_2D);
        int top = (int)std::floor(std::log2((double)std::max(fbw, fbh)));
        float st[4] = {0, 0, 0, 0};
        glGetTexImage(GL_TEXTURE_2D, top, GL_RGBA, GL_FLOAT, st);
        if (st[0] > 1e-9f && st[2] > 1e-9f) {
            // 밝기 가중 평균(Σl²/Σl)과 켜진 픽셀 평균(Σl/켜진 수)의 기하평균 기준
            float ref = std::sqrt((st[1] / st[0]) * (st[0] / st[2]));
            float target = 1.2f / ref;
            expo = expo <= 0 ? target : 0.85f * expo + 0.15f * target;
        }

        // ⑤ 톤매핑해서 화면에
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbw, fbh);
        glUseProgram(progTone);
        glUniform1i(glGetUniformLocation(progTone, "uAcc"), 0);
        glUniform1f(glGetUniformLocation(progTone, "uExpo"), std::max(expo, 1e-3f) * gain);
        glBindTexture(GL_TEXTURE_2D, accTex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    void init_gl() {
        progPts = program(VS_POINTS, FS_POINTS);
        progDecay = program(VS_FULL, FS_DECAY);
        progStats = program(VS_FULL, FS_STATS);
        progTone = program(VS_FULL, FS_TONE);
        glGenVertexArrays(1, &vao);
        glGenVertexArrays(1, &emptyVao);
        glfwGetFramebufferSize(win, &fbw, &fbh);
        make_targets();
    }

    void probe_gpu() {
#ifdef USE_HIP
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return;
        HIPCHECK(hipSetDevice(0));
        hipDeviceProp_t prop;
        HIPCHECK(hipGetDeviceProperties(&prop, 0));
        gpu_name = prop.name;
        gpu_ok = true;
#endif
    }

    void update_title(double fps) {
        char nstr[32], t[320];
        if (n >= (1 << 20)) std::snprintf(nstr, sizeof nstr, "%.1fM", n / 1048576.0);
        else std::snprintf(nstr, sizeof nstr, "%dK", n / 1024);
        std::string be = use_gpu ? std::string(interop ? "HIP-GL interop " : "HIP+copy ") + gpu_name
                                 : "CPU OpenMP " + std::to_string(omp_get_max_threads()) + "T";
        std::snprintf(t, sizeof t, "Lorenz | %s | %s particles x %d RK4 | sim %.2f ms | %.0f fps | rho %.0f | t=%.1f%s",
                      be.c_str(), nstr, L.K, sim_ms, fps, L.r, simt, paused ? " | PAUSED" : "");
        glfwSetWindowTitle(win, t);
    }

    void toggle_fullscreen() {
        GLFWmonitor* mon = glfwGetPrimaryMonitor();
        if (!fullscreen) {
            glfwGetWindowPos(win, &wx, &wy);
            glfwGetWindowSize(win, &ww, &wh);
            const GLFWvidmode* vm = glfwGetVideoMode(mon);
            glfwSetWindowMonitor(win, mon, 0, 0, vm->width, vm->height, vm->refreshRate);
        } else {
            glfwSetWindowMonitor(win, nullptr, wx, wy, ww, wh, 0);
        }
        fullscreen = !fullscreen;
    }
};

// ─────────────────────────── 입력 콜백 ───────────────────────────
static void on_key(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    App& A = *(App*)glfwGetWindowUserPointer(w);
    switch (key) {
        case GLFW_KEY_ESCAPE: case GLFW_KEY_Q: glfwSetWindowShouldClose(w, 1); break;
        case GLFW_KEY_SPACE: A.paused = !A.paused; break;
        case GLFW_KEY_A: A.autorot = !A.autorot; break;
        case GLFW_KEY_T: A.trails = !A.trails; break;
        case GLFW_KEY_C: A.cmode ^= 1; break;
        case GLFW_KEY_G: A.set_gpu(!A.use_gpu); break;
        case GLFW_KEY_V: A.vsync = !A.vsync; glfwSwapInterval(A.vsync ? 1 : 0); break;
        case GLFW_KEY_F: A.toggle_fullscreen(); break;
        case GLFW_KEY_P: A.psize = A.psize >= 3.f ? 1.f : A.psize + 1.f; break;
        case GLFW_KEY_B: A.gain /= 1.25f; break;
        case GLFW_KEY_N: A.gain *= 1.25f; break;
        case GLFW_KEY_1: A.fill(1); A.cmode = 0; break;
        case GLFW_KEY_2: A.fill(2); A.cmode = 1; break;
        case GLFW_KEY_R: A.fill(A.mode); break;
        case GLFW_KEY_COMMA: A.L.r = std::max(0.5f, A.L.r - 1.f); break;
        case GLFW_KEY_PERIOD: A.L.r = std::min(200.f, A.L.r + 1.f); break;
        case GLFW_KEY_MINUS: A.L.K = std::max(1, A.L.K - 1); break;
        case GLFW_KEY_EQUAL: A.L.K = std::min(40, A.L.K + 1); break;
        case GLFW_KEY_LEFT_BRACKET:
        case GLFW_KEY_RIGHT_BRACKET: {
            int nn = key == GLFW_KEY_RIGHT_BRACKET ? A.n * 2 : A.n / 2;
            nn = std::clamp(nn, 1024, A.gpu_ok ? (1 << 27) : (1 << 24));
            if (nn != A.n) {
                bool g = A.use_gpu;
                A.set_gpu(false);
                A.make_particles(nn);
                A.set_gpu(g);
            }
            break;
        }
    }
}

static void on_button(GLFWwindow* w, int button, int action, int) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        A.dragging = action == GLFW_PRESS;
        if (A.dragging) {
            A.autorot = false;
            glfwGetCursorPos(w, &A.lastx, &A.lasty);
        }
    }
}

static void on_cursor(GLFWwindow* w, double x, double y) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    if (A.dragging) {
        A.yaw += (float)(x - A.lastx) * 0.006f;
        A.pitch = std::clamp(A.pitch + (float)(y - A.lasty) * 0.006f, -1.5f, 1.5f);
        A.cam_moved = true;
    }
    A.lastx = x;
    A.lasty = y;
}

static void on_scroll(GLFWwindow* w, double, double dy) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    A.zoom = std::clamp(A.zoom * (float)std::pow(1.1, dy), 0.2f, 10.f);
    A.cam_moved = true;
}

// ─────────────────────────── 메인 ───────────────────────────
int main(int argc, char** argv) {
    int N = 0, scene = 1, shot_frames = 0;
    const char* shot = nullptr;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shot") && i + 2 < argc) {  // --shot 파일.ppm 프레임수 : 렌더링 확인용
            shot = argv[++i];
            shot_frames = std::atoi(argv[++i]);
        } else {
            pos.push_back(argv[i]);
        }
    }
    if (pos.size() > 0) N = std::atoi(pos[0].c_str());
    if (pos.size() > 1) scene = std::atoi(pos[1].c_str()) == 2 ? 2 : 1;

    if (!glfwInit()) {
        std::fprintf(stderr, "GLFW 초기화 실패 (디스플레이가 없나요?)\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    App A;
    A.win = glfwCreateWindow(A.ww, A.wh, "Lorenz", nullptr, nullptr);
    if (!A.win) {
        std::fprintf(stderr, "OpenGL 3.3 창 생성 실패\n");
        return 1;
    }
    glfwMakeContextCurrent(A.win);
    glfwSwapInterval(shot ? 0 : 1);
    glfwSetWindowUserPointer(A.win, &A);
    glfwSetKeyCallback(A.win, on_key);
    glfwSetMouseButtonCallback(A.win, on_button);
    glfwSetCursorPosCallback(A.win, on_cursor);
    glfwSetScrollCallback(A.win, on_scroll);
    std::printf("OpenGL: %s / %s\n", (const char*)glGetString(GL_RENDERER), (const char*)glGetString(GL_VERSION));

    A.probe_gpu();  // GL 컨텍스트를 만든 뒤에 HIP 초기화 (interop 순서)
    if (N <= 0) N = A.gpu_ok ? (1 << 22) : (1 << 19);
    A.mode = scene;
    A.cmode = scene == 2 ? 1 : 0;
    A.init_gl();
    A.make_particles(N);
    A.use_gpu = false;
    A.set_gpu(A.gpu_ok);
    if (A.gpu_ok) std::printf("HIP: %s, interop %s\n", A.gpu_name.c_str(), A.interop ? "OK" : "실패 -> 복사 방식");

    double last_title = 0, last_t = glfwGetTime(), fps = 60;
    long frame = 0;
    while (!glfwWindowShouldClose(A.win)) {
        glfwPollEvents();
        if (A.autorot) {
            A.yaw += 0.004f;
        }
        A.simulate();
        A.render();
        A.cam_moved = false;
        glfwSwapBuffers(A.win);

        double now = glfwGetTime();
        fps = 0.9 * fps + 0.1 / std::max(1e-6, now - last_t);
        last_t = now;
        if (now - last_title > 0.4) {
            A.update_title(fps);
            last_title = now;
        }
        ++frame;
        if (shot && frame >= shot_frames) {
            std::vector<unsigned char> px((size_t)A.fbw * A.fbh * 3);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glReadBuffer(GL_BACK);
            A.render();
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, A.fbw, A.fbh, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            FILE* f = std::fopen(shot, "wb");
            std::fprintf(f, "P6 %d %d 255\n", A.fbw, A.fbh);
            for (int y = A.fbh - 1; y >= 0; --y) std::fwrite(&px[(size_t)y * A.fbw * 3], 1, (size_t)A.fbw * 3, f);
            std::fclose(f);
            std::printf("saved %s (frame %ld, sim %.2f ms, t=%.1f)\n", shot, frame, A.sim_ms, A.simt);
            break;
        }
    }
    A.gpu_release();
    glfwTerminate();
    return 0;
}
