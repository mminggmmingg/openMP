// lorenz.cpp — 터미널 3D 로렌츠 끌개: 입자 수백만 개, CPU(OpenMP) + GPU(ROCm/HIP)
//
// CPU 전용 빌드:
//   g++ -O3 -march=native -std=c++17 -fopenmp lorenz.cpp -o lorenz
// GPU(ROCm) 포함 빌드 (Strix Halo iGPU = gfx1151):
//   hipcc -O3 -std=c++17 -fopenmp -DUSE_HIP --offload-arch=gfx1151 -x hip lorenz.cpp -o lorenz
//
// 실행:  ./lorenz [입자수]      기본: GPU 있으면 2M, 없으면 256K
//        ./lorenz N 2           2번 장면(무작위 뿌리기)으로 시작
//        ./lorenz --bench [N]   화면 없이 CPU 스레드별 / GPU 속도 비교
//
// 조작:  좌클릭 드래그 = 회전   휠 = 줌   a = 자동 회전   space = 일시정지
//        1 = 나비효과 (지름 0.01짜리 공 하나에서 출발)   2 = 공간 전체에 무작위 뿌리기
//        [ ] = 입자 수 절반/두 배   , . = rho -/+   - = = 시간 속도   c = 색 모드
//        t = 잔상 on/off   g = CPU <-> GPU   r = 리셋   q = 종료
//
// 구조:  dx/dt = σ(y - x),  dy/dt = x(ρ - z) - y,  dz/dt = xy - βz   (σ=10, ρ=28, β=8/3)
//   ① 입자마다 RK4로 K스텝 적분      (입자끼리 독립 -> 완전 병렬)
//   ② 카메라로 3D -> 2D 투영
//   ③ 픽셀 버퍼에 빛 누적             (CPU: 배열 리덕션, GPU: atomicAdd)
//   ④ 버퍼를 지수 감쇠시켜 잔상 효과, 자동 노출로 톤매핑

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define HD __host__ __device__
#else
#define HD
#endif

#include <omp.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ───────────────────────── 공용 계산 (CPU/GPU) ─────────────────────────
struct LP {      // 로렌츠 파라미터
    float s, r, b, dt;
    int K;       // 프레임당 RK4 스텝 수
};

struct Cam {
    float cyaw, syaw, cpit, spit;
    float scale, D, cx, cy;
    int PW, PH;
    int cmode;   // 0: 입자 번호(초기 위치) 색, 1: 속도 색
    int n;
};

HD inline void deriv(float x, float y, float z, const LP& L, float& dx, float& dy, float& dz) {
    dx = L.s * (y - x);
    dy = x * (L.r - z) - y;
    dz = x * y - L.b * z;
}

HD inline void rk4(float& x, float& y, float& z, const LP& L) {
    float h = L.dt;
    float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    deriv(x, y, z, L, k1x, k1y, k1z);
    deriv(x + 0.5f * h * k1x, y + 0.5f * h * k1y, z + 0.5f * h * k1z, L, k2x, k2y, k2z);
    deriv(x + 0.5f * h * k2x, y + 0.5f * h * k2y, z + 0.5f * h * k2z, L, k3x, k3y, k3z);
    deriv(x + h * k3x, y + h * k3y, z + h * k3z, L, k4x, k4y, k4z);
    x += h / 6.f * (k1x + 2.f * k2x + 2.f * k3x + k4x);
    y += h / 6.f * (k1y + 2.f * k2y + 2.f * k3y + k4y);
    z += h / 6.f * (k1z + 2.f * k2z + 2.f * k3z + k4z);
}

HD inline void hsv(float h, float& r, float& g, float& b) {
    h = h - floorf(h);
    float t = h * 6.f;
    int k = (int)t;
    float f = t - k, q = 1.f - f;
    switch (k) {
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = q; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = q; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        default: r = 1; g = 0; b = q; break;
    }
}

// 입자 하나: 적분 -> 투영 -> 색 결정. 픽셀 번호(없으면 -1)와 RGB 기여를 돌려줌
HD inline int advance_and_project(int i, float* X, float* Y, float* Z, const LP& L, const Cam& C,
                                  float& cr, float& cg, float& cb) {
    float x = X[i], y = Y[i], z = Z[i];
    for (int k = 0; k < L.K; ++k) rk4(x, y, z, L);
    X[i] = x;
    Y[i] = y;
    Z[i] = z;

    // 로렌츠의 z축을 화면 위쪽으로: (x, z-25, y) 후 yaw(세로축) -> pitch(가로축) 회전
    float X0 = x, Y0 = z - 25.f, Z0 = y;
    float X1 = C.cyaw * X0 + C.syaw * Z0;
    float Z1 = -C.syaw * X0 + C.cyaw * Z0;
    float Y2 = C.cpit * Y0 - C.spit * Z1;
    float Z2 = C.spit * Y0 + C.cpit * Z1;
    float den = C.D - Z2;
    if (den < 1.f) return -1;
    float persp = C.D / den;
    int px = (int)(C.cx + C.scale * X1 * persp);
    int py = (int)(C.cy - C.scale * Y2 * persp);
    if (px < 0 || py < 0 || px >= C.PW || py >= C.PH) return -1;

    float hue;
    if (C.cmode == 0) {
        hue = 0.85f * (float)i / (float)C.n;
    } else {
        float dx, dy, dz;
        deriv(x, y, z, L, dx, dy, dz);
        float sp = sqrtf(dx * dx + dy * dy + dz * dz);
        float t = sp / 160.f;
        hue = 0.66f - 0.66f * (t > 1.f ? 1.f : t);
    }
    hsv(hue, cr, cg, cb);
    float fog = 0.55f + 0.45f * persp;  // 가까울수록 밝게
    cr = (0.25f + 0.75f * cr) * fog;
    cg = (0.25f + 0.75f * cg) * fog;
    cb = (0.25f + 0.75f * cb) * fog;
    return py * C.PW + px;
}

// ─────────────────────────── GPU 커널 ───────────────────────────
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

// 누적 버퍼는 64비트 고정소수점 정수. RDNA3에는 글로벌 float atomic add 명령이 없어서
// atomicAdd(float*)가 CAS 재시도 루프가 되고, 수백만 스레드가 한 픽셀에 몰리면(나비효과 초반) 거의 멈춤.
// 정수 atomic add는 하드웨어 명령이라 재시도가 없음.
static const float ACC_SCALE = 4096.f;
typedef unsigned long long u64;

__global__ void k_decay(u64* acc, int m, u64 d16) {  // acc *= decay (d16 = decay * 65536)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) acc[i] = (acc[i] * d16) >> 16;
}

__global__ void k_step(float* X, float* Y, float* Z, int n, LP L, Cam C, u64* acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float r = 0.f, g = 0.f, b = 0.f;
    int pix = -1;
    if (i < n) pix = advance_and_project(i, X, Y, Z, L, C, r, g, b);  // early return 금지: 아래 wave 연산에 전원 참여

    // wavefront 집계: wave 안 모든 스레드가 같은 픽셀이면 shuffle로 합쳐서 대표 1개만 atomic
    int lane = threadIdx.x & (warpSize - 1);
    int p0 = __shfl(pix, 0);
    if (__all(pix == p0)) {
        if (p0 < 0) return;
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            r += __shfl_xor(r, off);
            g += __shfl_xor(g, off);
            b += __shfl_xor(b, off);
        }
        if (lane == 0) {
            atomicAdd(&acc[3 * p0 + 0], (u64)(r * ACC_SCALE));
            atomicAdd(&acc[3 * p0 + 1], (u64)(g * ACC_SCALE));
            atomicAdd(&acc[3 * p0 + 2], (u64)(b * ACC_SCALE));
        }
    } else if (pix >= 0) {
        atomicAdd(&acc[3 * pix + 0], (u64)(r * ACC_SCALE));
        atomicAdd(&acc[3 * pix + 1], (u64)(g * ACC_SCALE));
        atomicAdd(&acc[3 * pix + 2], (u64)(b * ACC_SCALE));
    }
}
#endif

// ─────────────────────────── 시뮬레이션 ───────────────────────────
struct Sim {
    int n = 0, PW = 0, PH = 0, mode = 1;
    std::vector<float> x, y, z, acc;
    bool use_gpu = false, gpu_ok = false;
    std::string gpu_name;
#ifdef USE_HIP
    float *dX = nullptr, *dY = nullptr, *dZ = nullptr;
    u64* dAcc = nullptr;
    std::vector<u64> accI;  // GPU 누적 버퍼의 호스트 사본
#endif

    void init(int N, int pw, int ph) {
        n = N;
        PW = pw;
        PH = ph;
        x.assign(n, 0.f);
        y.assign(n, 0.f);
        z.assign(n, 0.f);
        acc.assign((size_t)PW * PH * 3, 0.f);
        gpu_probe();
        gpu_alloc();
        reset(mode);
    }

    void reset(int m) {
        mode = m;
        if (mode == 1) {
            // 나비효과: 지름 0.01 공. x 오프셋 순으로 번호를 매겨서, 색 = 초기 위치가 되게
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                std::mt19937 rng(1234u + (unsigned)i);
                std::uniform_real_distribution<float> U(-1.f, 1.f);
                float u = 2.f * (i + 0.5f) / n - 1.f;
                float r = 0.005f;
                x[i] = -8.f + r * u;
                y[i] = 7.f + r * std::sqrt(std::max(0.f, 1.f - u * u)) * U(rng);
                z[i] = 27.f + r * std::sqrt(std::max(0.f, 1.f - u * u)) * U(rng);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                std::mt19937 rng(99u + (unsigned)i);
                std::uniform_real_distribution<float> U(0.f, 1.f);
                x[i] = -25.f + 50.f * U(rng);
                y[i] = -30.f + 60.f * U(rng);
                z[i] = 0.f + 55.f * U(rng);
            }
        }
        std::fill(acc.begin(), acc.end(), 0.f);
        upload();
    }

    // ── CPU: 적분 + 투영 + OpenMP 배열 리덕션으로 픽셀 누적 ──
    void cpu_frame(const LP& L, const Cam& C, float decay) {
        float* a = acc.data();
        const int M3 = PW * PH * 3, N = n;
        float *X = x.data(), *Y = y.data(), *Z = z.data();
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < M3; ++k) a[k] *= decay;

        // reduction(+: a[:M3]) -> 스레드마다 픽셀 버퍼 사본을 따로 갖고 누적, 끝에 한꺼번에 합침
        // (atomic 없이 경쟁을 피하는 OpenMP 4.5 배열 섹션 리덕션)
        #pragma omp parallel for schedule(static) reduction(+ : a[:M3])
        for (int i = 0; i < N; ++i) {
            float r, g, b;
            int pix = advance_and_project(i, X, Y, Z, L, C, r, g, b);
            if (pix >= 0) {
                a[3 * pix + 0] += r;
                a[3 * pix + 1] += g;
                a[3 * pix + 2] += b;
            }
        }
    }

    void frame(const LP& L, const Cam& C, float decay) {
        if (use_gpu && gpu_ok)
            gpu_frame(L, C, decay);
        else
            cpu_frame(L, C, decay);
    }

    // ── GPU ──
    void gpu_probe() {
#ifdef USE_HIP
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return;
        hipDeviceProp_t prop;
        HIPCHECK(hipGetDeviceProperties(&prop, 0));
        gpu_name = prop.name;
        gpu_ok = true;
#endif
    }

    void gpu_alloc() {
#ifdef USE_HIP
        if (!gpu_ok) return;
        gpu_free();
        HIPCHECK(hipMalloc((void**)&dX, sizeof(float) * n));
        HIPCHECK(hipMalloc((void**)&dY, sizeof(float) * n));
        HIPCHECK(hipMalloc((void**)&dZ, sizeof(float) * n));
        HIPCHECK(hipMalloc((void**)&dAcc, sizeof(u64) * acc.size()));
        accI.assign(acc.size(), 0);
#endif
    }

    void gpu_free() {
#ifdef USE_HIP
        if (dX || dAcc) (void)hipDeviceSynchronize();  // 실행 중인 커널이 끝난 뒤 해제
        for (float* p : {dX, dY, dZ})
            if (p) (void)hipFree(p);
        if (dAcc) (void)hipFree(dAcc);
        dX = dY = dZ = nullptr;
        dAcc = nullptr;
#endif
    }

    void upload() {  // 호스트 -> GPU (리셋, CPU에서 GPU로 전환할 때)
#ifdef USE_HIP
        if (!gpu_ok) return;
        size_t b = sizeof(float) * n;
        HIPCHECK(hipMemcpy(dX, x.data(), b, hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(dY, y.data(), b, hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(dZ, z.data(), b, hipMemcpyHostToDevice));
        const int M3 = (int)acc.size();
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < M3; ++k) accI[k] = (u64)(acc[k] * ACC_SCALE);
        HIPCHECK(hipMemcpy(dAcc, accI.data(), sizeof(u64) * M3, hipMemcpyHostToDevice));
#endif
    }

    void download() {  // GPU -> 호스트 (GPU에서 CPU로 전환할 때)
#ifdef USE_HIP
        if (!gpu_ok) return;
        size_t b = sizeof(float) * n;
        HIPCHECK(hipMemcpy(x.data(), dX, b, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(y.data(), dY, b, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(z.data(), dZ, b, hipMemcpyDeviceToHost));
#endif
    }

    void gpu_frame(const LP& L, const Cam& C, float decay) {
#ifdef USE_HIP
        int M3 = (int)acc.size();
        u64 d16 = (u64)(decay * 65536.f);
        hipLaunchKernelGGL(k_decay, dim3((M3 + 255) / 256), dim3(256), 0, 0, dAcc, M3, d16);
        hipLaunchKernelGGL(k_step, dim3((n + 255) / 256), dim3(256), 0, 0, dX, dY, dZ, n, L, C, dAcc);
        HIPCHECK(hipGetLastError());
        // 입자 수백만 개가 아니라 픽셀 버퍼(수십 KB)만 가져오면 됨
        HIPCHECK(hipMemcpy(accI.data(), dAcc, sizeof(u64) * M3, hipMemcpyDeviceToHost));
        const float inv = 1.f / ACC_SCALE;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < M3; ++k) acc[k] = (float)accI[k] * inv;
#else
        (void)L; (void)C; (void)decay;
#endif
    }

    void set_gpu(bool on) {
        if (on && !gpu_ok) return;
        if (on && !use_gpu) upload();
        if (!on && use_gpu) download();
        use_gpu = on;
    }

    ~Sim() { gpu_free(); }
};

// ─────────────────────────── 렌더링 ───────────────────────────
static void render(const std::vector<float>& acc, int PW, int TR, float expo,
                   std::vector<std::string>& lines) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < TR; ++row) {
        std::string& s = lines[row];
        s.clear();
        char tmp[48];
        std::snprintf(tmp, sizeof tmp, "\x1b[%d;1H", row + 1);
        s += tmp;
        unsigned pf = ~0u, pb = ~0u;
        for (int c = 0; c < PW; ++c) {
            unsigned col[2];
            for (int h = 0; h < 2; ++h) {
                const float* a = &acc[3 * ((size_t)(2 * row + h) * PW + c)];
                // 밝기(최대 채널) 기준 톤매핑 -> 색상은 유지, 아주 밀집한 곳만 하얗게 달아오름
                float l = std::max(a[0], std::max(a[1], a[2]));
                float sc = l > 1e-6f ? (1.f - std::exp(-l * expo)) / l : 0.f;
                float hot = std::clamp((l * expo - 2.5f) / 6.f, 0.f, 1.f) * 0.55f;
                unsigned rgb = 0;
                for (int k = 0; k < 3; ++k) {
                    float v = a[k] * sc;
                    v += (1.f - v) * hot;
                    int base = k == 2 ? 10 : 4;  // 살짝 남색 배경
                    unsigned q = (unsigned)std::min(255.f, base + (255 - base) * v) & ~3u;
                    rgb = (rgb << 8) | q;
                }
                col[h] = rgb;
            }
            if (col[0] != pf) {
                std::snprintf(tmp, sizeof tmp, "\x1b[38;2;%u;%u;%um", col[0] >> 16, (col[0] >> 8) & 255, col[0] & 255);
                s += tmp;
                pf = col[0];
            }
            if (col[1] != pb) {
                std::snprintf(tmp, sizeof tmp, "\x1b[48;2;%u;%u;%um", col[1] >> 16, (col[1] >> 8) & 255, col[1] & 255);
                s += tmp;
                pb = col[1];
            }
            s += "▀";
        }
    }
}

// 자동 노출: 희미한 픽셀이 많아도 밝은 구조가 안 날아가고, 희박할 때도 너무 어둡지 않게
static float auto_exposure(const std::vector<float>& acc) {
    double s1 = 0, s2 = 0;
    long cnt = 0;
    const int M = (int)(acc.size() / 3);
    #pragma omp parallel for reduction(+ : s1, s2, cnt) schedule(static)
    for (int p = 0; p < M; ++p) {
        float l = std::max(acc[3 * p], std::max(acc[3 * p + 1], acc[3 * p + 2]));
        s1 += l;
        s2 += (double)l * l;
        cnt += l > 1e-4f;
    }
    if (s1 < 1e-6 || cnt == 0) return 1.f;
    // 밝기 가중 평균과 단순 평균의 기하평균: 희박할 때(나비효과 초반)와 빽빽할 때 모두 무난하게
    double ref = std::sqrt((s2 / s1) * (s1 / cnt));
    return (float)(1.2 / ref);
}

// ─────────────────────────── 터미널 ───────────────────────────
static termios g_orig;
static bool g_raw = false;
static volatile sig_atomic_t g_quit = 0;

static void restore_term() {
    if (!g_raw) return;
    std::printf("\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[0m\x1b[?25h\x1b[?1049l");
    std::fflush(stdout);
    tcsetattr(STDIN_FILENO, TCSANOW, &g_orig);
    g_raw = false;
}
static void on_signal(int) { g_quit = 1; }

static void setup_term() {
    if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &g_orig) != 0) return;
    termios raw = g_orig;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    g_raw = true;
    std::atexit(restore_term);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::printf("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[?1000h\x1b[?1002h\x1b[?1006h");
    std::fflush(stdout);
}

// ─────────────────────────── 카메라 ───────────────────────────
struct View {
    float yaw = 0.6f, pitch = 0.25f, zoom = 1.f;
    Cam cam(int PW, int PH, int cmode, int n) const {
        Cam C;
        C.cyaw = std::cos(yaw);
        C.syaw = std::sin(yaw);
        C.cpit = std::cos(pitch);
        C.spit = std::sin(pitch);
        C.D = 110.f;
        C.scale = zoom * PH / 62.f;
        C.cx = PW * 0.5f;
        C.cy = PH * 0.5f;
        C.PW = PW;
        C.PH = PH;
        C.cmode = cmode;
        C.n = n;
        return C;
    }
};

// ─────────────────────────── 벤치마크 ───────────────────────────
static void bench_scene(Sim& S, int scene, const LP& L, const View& V, int PW, int PH) {
    const int WARM = 3, FR = 20;
    const char* name = scene == 1 ? "장면 1 (나비효과: 전 입자가 한 점에 몰림)" : "장면 2 (무작위: 고르게 퍼짐)";
    std::printf("[%s]\n", name);
    int maxT = omp_get_max_threads();
    std::vector<int> ts{1};
    for (int t = 2; t <= maxT; t *= 2) ts.push_back(t);
    if (ts.back() != maxT) ts.push_back(maxT);
    double base = 0;
    Cam C = V.cam(PW, PH, scene == 2, S.n);
    for (int t : ts) {
        omp_set_num_threads(t);
        S.set_gpu(false);
        S.reset(scene);
        for (int f = 0; f < WARM; ++f) S.frame(L, C, 0.8f);
        double t0 = omp_get_wtime();
        for (int f = 0; f < FR; ++f) S.frame(L, C, 0.8f);
        double ms = (omp_get_wtime() - t0) / FR * 1000;
        if (t == 1) base = ms;
        std::printf("  CPU %2d threads: %8.2f ms/frame  (%7.1f M 입자-스텝/s)   speedup x%.2f\n",
                    t, ms, (double)S.n * L.K / ms / 1e3, base / ms);
    }
    omp_set_num_threads(maxT);
    if (S.gpu_ok) {
        S.reset(scene);
        S.set_gpu(true);
        for (int f = 0; f < WARM; ++f) S.frame(L, C, 0.8f);
        double t0 = omp_get_wtime();
        for (int f = 0; f < FR; ++f) S.frame(L, C, 0.8f);
        double ms = (omp_get_wtime() - t0) / FR * 1000;
        std::printf("  GPU %s: %8.2f ms/frame  (%7.1f M 입자-스텝/s)   speedup x%.2f (vs 1 thread)\n",
                    S.gpu_name.c_str(), ms, (double)S.n * L.K / ms / 1e3, base / ms);
        S.set_gpu(false);
    } else {
        std::printf("  GPU: 사용 불가 (USE_HIP 없이 빌드됐거나 장치 없음)\n");
    }
}

static int run_bench(int N) {
    const int PW = 200, PH = 100;
    Sim S;
    S.mode = 2;
    S.init(N, PW, PH);
    LP L{10.f, 28.f, 8.f / 3.f, 0.004f, 4};
    View V;
    std::printf("로렌츠 벤치마크: 입자 %d개, 프레임당 RK4 %d스텝, 화면 %dx%d px\n", N, L.K, PW, PH);
    bench_scene(S, 1, L, V, PW, PH);
    bench_scene(S, 2, L, V, PW, PH);
    return 0;
}

// ─────────────────────────── 메인 ───────────────────────────
int main(int argc, char** argv) {
    bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int N = 0;
    if (bench && argc > 2) N = std::atoi(argv[2]);
    if (!bench && argc > 1) N = std::atoi(argv[1]);
    int start_mode = (!bench && argc > 2 && std::atoi(argv[2]) == 2) ? 2 : 1;
    if (bench) return run_bench(N > 0 ? N : (1 << 20));

    int TW = 120, rows = 40;
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        TW = ws.ws_col;
        rows = ws.ws_row;
    }
    int TR = std::max(4, rows - 1);
    int PW = TW, PH = TR * 2;

    Sim S;
    S.gpu_probe();
    if (N <= 0) N = S.gpu_ok ? (1 << 21) : (1 << 18);
    N = std::clamp(N, 1024, 1 << 26);
    S.mode = start_mode;
    S.init(N, PW, PH);
    S.set_gpu(S.gpu_ok);
    setup_term();

    LP L{10.f, 28.f, 8.f / 3.f, 0.004f, 3};
    View V;
    int cmode = start_mode == 2 ? 1 : 0;  // 나비효과는 번호 색, 무작위는 속도 색
    bool paused = false, autorot = true, trails = true;
    float expo = 1.f;
    std::vector<std::string> lines(TR);
    std::string out, inbuf;
    bool drag = false;
    int lastx = 0, lasty = 0;
    double sim_ms = 0, fps = 0, simt = 0;
    long frame = 0;

    while (!g_quit) {
        double tf = omp_get_wtime();
        bool cam_moved = false;

        // ── 입력 ──
        char buf[4096];
        ssize_t nr;
        while (g_raw && (nr = read(STDIN_FILENO, buf, sizeof buf)) > 0) inbuf.append(buf, (size_t)nr);
        size_t pos = 0;
        while (pos < inbuf.size()) {
            if (inbuf[pos] == '\x1b') {
                if (pos + 2 >= inbuf.size()) break;
                if (inbuf[pos + 1] == '[' && inbuf[pos + 2] == '<') {
                    size_t end = inbuf.find_first_of("Mm", pos + 3);
                    if (end == std::string::npos) break;
                    int b, mx, my;
                    if (std::sscanf(inbuf.c_str() + pos + 3, "%d;%d;%d", &b, &mx, &my) == 3) {
                        bool press = inbuf[end] == 'M';
                        if (b == 64 || b == 65) {
                            V.zoom *= b == 64 ? 1.1f : 1.f / 1.1f;
                            V.zoom = std::clamp(V.zoom, 0.2f, 8.f);
                            cam_moved = true;
                        } else if ((b & 3) == 0) {
                            if (!press) {
                                drag = false;
                            } else if (!(b & 32)) {
                                drag = true;
                                autorot = false;
                            } else if (drag) {
                                V.yaw += (mx - lastx) * 0.03f;
                                V.pitch = std::clamp(V.pitch + (my - lasty) * 0.06f, -1.5f, 1.5f);
                                cam_moved = true;
                            }
                            lastx = mx;
                            lasty = my;
                        }
                    }
                    pos = end + 1;
                    continue;
                }
                ++pos;
                continue;
            }
            switch (inbuf[pos]) {
                case 'q': case 'Q': g_quit = 1; break;
                case ' ': paused = !paused; break;
                case 'a': case 'A': autorot = !autorot; break;
                case 't': case 'T': trails = !trails; break;
                case 'c': case 'C': cmode ^= 1; break;
                case 'g': case 'G': S.set_gpu(!S.use_gpu); break;
                case '1': S.reset(1); simt = 0; cmode = 0; break;
                case '2': S.reset(2); simt = 0; cmode = 1; break;
                case 'r': case 'R': S.reset(S.mode); simt = 0; break;
                case ',': case '<': L.r = std::max(0.5f, L.r - 1.f); break;
                case '.': case '>': L.r = std::min(200.f, L.r + 1.f); break;
                case '-': case '_': L.K = std::max(1, L.K - 1); break;
                case '=': case '+': L.K = std::min(40, L.K + 1); break;
                case '[': case ']': {
                    int nn = inbuf[pos] == ']' ? S.n * 2 : S.n / 2;
                    nn = std::clamp(nn, 1024, S.gpu_ok ? (1 << 26) : (1 << 24));
                    if (nn != S.n) {
                        bool g = S.use_gpu;
                        S.use_gpu = false;
                        S.init(nn, PW, PH);
                        S.set_gpu(g);
                        simt = 0;
                    }
                    break;
                }
            }
            ++pos;
        }
        inbuf.erase(0, pos);

        if (autorot) {
            V.yaw += 0.006f;
            cam_moved = true;
        }

        // ── 시뮬레이션 + 누적 ──
        LP Lf = L;
        if (paused) Lf.K = 0;  // 멈춰도 회전은 되게: 적분 없이 투영만
        Cam C = V.cam(PW, PH, cmode, S.n);
        // 잔상: 카메라가 크게 움직이면 번짐이 생기므로 감쇠를 키움
        float decay = !trails ? 0.f : (cam_moved && !autorot ? 0.35f : 0.82f);
        double t0 = omp_get_wtime();
        S.frame(Lf, C, decay);
        double t1 = omp_get_wtime();
        sim_ms = 0.9 * sim_ms + 0.1 * (t1 - t0) * 1000;
        if (!paused) simt += L.dt * L.K;

        // ── 렌더 ──
        float target = auto_exposure(S.acc);
        expo = frame == 0 ? target : 0.85f * expo + 0.15f * target;
        render(S.acc, PW, TR, expo, lines);
        out.clear();
        for (auto& l : lines) out += l;
        char status[320];
        std::string be = S.use_gpu ? "GPU " + S.gpu_name : "CPU " + std::to_string(omp_get_max_threads()) + "T";
        char nstr[32];
        if (S.n >= (1 << 20)) std::snprintf(nstr, sizeof nstr, "%.1fM", S.n / 1048576.0);
        else std::snprintf(nstr, sizeof nstr, "%dK", S.n / 1024);
        int len = std::snprintf(status, sizeof status,
            " %s | %s particles x %d RK4 | sim %.1fms | %.0f fps | t=%.1f rho=%.0f | %s%s%s"
            " | drag:rotate wheel:zoom 1 2 [ ] , . - = c t a g r q",
            be.c_str(), nstr, L.K, sim_ms, fps, simt, L.r, cmode ? "speed" : "index",
            autorot ? " auto" : "", paused ? " PAUSED" : "");
        len = std::clamp(len, 0, TW);
        out += "\x1b[" + std::to_string(TR + 1) + ";1H\x1b[0m\x1b[7m";
        out.append(status, (size_t)len);
        out += "\x1b[0m\x1b[K";
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);

        ++frame;
        double el = omp_get_wtime() - tf;
        if (el < 1.0 / 60) usleep((useconds_t)((1.0 / 60 - el) * 1e6));
        fps = 0.9 * fps + 0.1 / (omp_get_wtime() - tf);

        if (!g_raw && frame >= 900) break;  // 파이프 출력일 때는 잠깐 돌고 종료
    }
    restore_term();
    return 0;
}
