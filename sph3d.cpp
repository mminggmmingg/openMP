// sph3d.cpp — 3D SPH 물 시뮬레이션: HIP(ROCm)/OpenMP로 계산, OpenGL로 물 표면 렌더링 (interop)
//
// 의존성 (Ubuntu):  sudo apt install libglfw3-dev libgl-dev
//
// GPU(ROCm) 빌드 (Strix Halo iGPU = gfx1151):
//   hipcc -O3 -std=c++17 -fopenmp -DUSE_HIP --offload-arch=gfx1151 -x hip sph3d.cpp -o sph3d -lglfw -lGL
// CPU 전용 빌드:
//   g++ -O3 -march=native -std=c++17 -fopenmp sph3d.cpp -o sph3d -lglfw -lGL
//
// 실행:  ./sph3d [입자수] [장면]       기본: GPU면 256K, CPU면 32K / 장면 1
//        ./sph3d --bench [입자수]      창 없이 CPU 스레드별 / GPU 속도 비교
//
// 조작:  좌클릭 드래그 = 카메라 회전     휠 = 줌     우클릭 드래그 = 손으로 물 휘젓기
//        방향키 = 중력 기울이기 (상자 기울이기)   0 = 중력 원위치   S = 출렁이기(파도 기계) on/off
//        1 = 댐 붕괴   2 = 수영장에 물덩어리 낙하   3 = 양쪽 물기둥 충돌   R = 리셋
//        [ ] = 입자 수 절반/두 배   M = 물 표면 / 입자 공 렌더링   A = 자동 회전
//        G = GPU <-> CPU   P = 일시정지   V = vsync   F = 전체화면   Q/Esc = 종료   (상태는 창 제목)
//
// 시뮬레이션 (서브스텝마다, 2D sph.cpp와 같은 구조를 3D로):
//   ① 셀 카운팅 (atomicAdd로 셀 안 순번)  ② prefix scan  ③ 셀 순서로 재배치
//   ④ 밀도/압력: 주변 3x3x3 셀. 같은 (y,z) 행의 x 3칸은 정렬 배열에서 연속 -> 9개 구간만 훑음
//   ⑤ 압력 + 점성 + 중력 + 손 -> 적분. 입자 위치는 OpenGL VBO 안에 있고 HIP이 그 자리에서 갱신
//
// 렌더링 (screen-space fluid):
//   ① 배경(하늘 + 바닥)을 텍스처로   ② 입자를 구 스프라이트로 그려 화면 깊이 버퍼
//   ③ 같은 입자를 가산 블렌딩해서 두께 버퍼 (+ 속도 가중 -> 거품)
//   ④ 깊이를 양방향(bilateral) 블러로 매끈하게 -> 입자 알갱이가 표면으로
//   ⑤ 깊이에서 법선 복원 -> 굴절(배경 왜곡 + 두께에 따른 흡수) + 프레넬 반사 + 하이라이트 + 거품

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
#include <string>
#include <vector>

// ───────────────────────── 데이터 ─────────────────────────
struct alignas(16) F4 {
    float x, y, z, w;  // 위치: w = 속력 (렌더링 색/거품용),  속도: w 미사용
};

struct Params {
    float Lx, Ly, Lz;            // 상자 크기 (입자 간격 = 1)
    float h, h2, cellInv;
    int gw, gh, gd;              // 이웃 탐색 격자 (셀 크기 = h)
    float m, rho0, c2, nu, dt, maxv, eps;
    float poly6, spikyG, viscL;
    float gx, gy, gz, rep;       // 중력, 손 반발 세기
    float hx, hy, hz, hvx, hvy, hvz, hR;
    int hand;
};

struct Bufs {
    F4 *pos, *vel;     // 상태 (force 단계의 출력). pos는 interop이면 OpenGL VBO
    F4 *poss, *vels;   // 셀 순서로 정렬된 사본
    float *rho, *p;
    int *cell, *rank, *cnt, *start;
};

// ───────────────────── 입자 단위 계산 (CPU/GPU 공용) ─────────────────────
HD inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

HD inline int cell_of(const F4& q, const Params& P) {
    int cx = clampi((int)(q.x * P.cellInv), 0, P.gw - 1);
    int cy = clampi((int)(q.y * P.cellInv), 0, P.gh - 1);
    int cz = clampi((int)(q.z * P.cellInv), 0, P.gd - 1);
    return (cz * P.gh + cy) * P.gw + cx;
}

HD inline void scatter_i(int i, const Bufs& B) {
    int d = B.start[B.cell[i]] + B.rank[i];
    B.poss[d] = B.pos[i];
    B.vels[d] = B.vel[i];
}

// 3x3x3 이웃 셀 = (y,z) 9개 행 x 연속된 x 3칸
#define FOR_NEIGHBORS(P, B, xi, yi, zi, ...)                                              \
    {                                                                                     \
        int cx_ = clampi((int)((xi) * (P).cellInv), 0, (P).gw - 1);                       \
        int cy_ = clampi((int)((yi) * (P).cellInv), 0, (P).gh - 1);                       \
        int cz_ = clampi((int)((zi) * (P).cellInv), 0, (P).gd - 1);                       \
        int x0_ = cx_ > 0 ? cx_ - 1 : 0, x1_ = cx_ < (P).gw - 1 ? cx_ + 1 : cx_;          \
        int y0_ = cy_ > 0 ? cy_ - 1 : 0, y1_ = cy_ < (P).gh - 1 ? cy_ + 1 : cy_;          \
        int z0_ = cz_ > 0 ? cz_ - 1 : 0, z1_ = cz_ < (P).gd - 1 ? cz_ + 1 : cz_;          \
        for (int zz_ = z0_; zz_ <= z1_; ++zz_)                                            \
            for (int yy_ = y0_; yy_ <= y1_; ++yy_) {                                      \
                int row_ = (zz_ * (P).gh + yy_) * (P).gw;                                 \
                int b_ = (B).start[row_ + x0_], e_ = (B).start[row_ + x1_ + 1];           \
                for (int j = b_; j < e_; ++j) {                                           \
                    F4 q_ = (B).poss[j];                                                  \
                    float dx = (xi) - q_.x, dy = (yi) - q_.y, dz = (zi) - q_.z;           \
                    float r2 = dx * dx + dy * dy + dz * dz;                               \
                    if (r2 < (P).h2) { __VA_ARGS__ }                                      \
                }                                                                         \
            }                                                                             \
    }

HD inline void density_i(int i, const Params& P, const Bufs& B) {
    F4 a = B.poss[i];
    float sum = 0.f;
    FOR_NEIGHBORS(P, B, a.x, a.y, a.z, {
        float t = P.h2 - r2;
        sum += t * t * t;
    })
    float rho = P.m * P.poly6 * sum;
    B.rho[i] = rho;
    float pr = P.c2 * (rho - P.rho0);
    B.p[i] = pr > 0.f ? pr : 0.f;  // 음압 제거 (표면 뭉침 방지)
}

HD inline void force_i(int i, const Params& P, const Bufs& B) {
    F4 a = B.poss[i], v = B.vels[i];
    float rhoi = B.rho[i];
    float pri = B.p[i] / (rhoi * rhoi);
    float ax = 0.f, ay = 0.f, az = 0.f, lx = 0.f, ly = 0.f, lz = 0.f;
    FOR_NEIGHBORS(P, B, a.x, a.y, a.z, {
        if (r2 > 1e-12f) {
            float r = sqrtf(r2), hr = P.h - r;
            float rhoj = B.rho[j];
            float fp = -P.m * (pri + B.p[j] / (rhoj * rhoj)) * P.spikyG * hr * hr / r;
            ax += fp * dx;
            ay += fp * dy;
            az += fp * dz;
            F4 vj = B.vels[j];
            float fv = P.m * P.viscL * hr / rhoj;
            lx += fv * (vj.x - v.x);
            ly += fv * (vj.y - v.y);
            lz += fv * (vj.z - v.z);
        }
    })
    ax += P.nu * lx + P.gx;
    ay += P.nu * ly + P.gy;
    az += P.nu * lz + P.gz;

    if (P.hand) {  // 손: 손 속도로 끌고 가면서 안쪽 입자는 밀어냄
        float dx = P.hx - a.x, dy = P.hy - a.y, dz = P.hz - a.z;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < P.hR * P.hR) {
            float d = sqrtf(d2) + 1e-6f, w = 1.f - d / P.hR;
            ax += w * (0.8f * (P.hvx - v.x) - P.rep * dx / d);
            ay += w * (0.8f * (P.hvy - v.y) - P.rep * dy / d);
            az += w * (0.8f * (P.hvz - v.z) - P.rep * dz / d);
        }
    }

    v.x += P.dt * ax;
    v.y += P.dt * ay;
    v.z += P.dt * az;
    float sp2 = v.x * v.x + v.y * v.y + v.z * v.z;
    if (sp2 > P.maxv * P.maxv) {
        float s = P.maxv / sqrtf(sp2);
        v.x *= s;
        v.y *= s;
        v.z *= s;
        sp2 = P.maxv * P.maxv;
    }
    a.x += P.dt * v.x;
    a.y += P.dt * v.y;
    a.z += P.dt * v.z;
    const float e = P.eps, k = -0.3f;
    if (a.x < e)        { a.x = e;        if (v.x < 0) v.x *= k; }
    if (a.x > P.Lx - e) { a.x = P.Lx - e; if (v.x > 0) v.x *= k; }
    if (a.y < e)        { a.y = e;        if (v.y < 0) v.y *= k; }
    if (a.y > P.Ly - e) { a.y = P.Ly - e; if (v.y > 0) v.y *= k; }
    if (a.z < e)        { a.z = e;        if (v.z < 0) v.z *= k; }
    if (a.z > P.Lz - e) { a.z = P.Lz - e; if (v.z > 0) v.z *= k; }
    a.w = sqrtf(sp2);
    B.pos[i] = a;
    B.vel[i] = v;
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

static const int SCAN_BS = 1024;

__global__ void k_zero(int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0;
}

__global__ void k_cell(Params P, Bufs B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = cell_of(B.pos[i], P);
    B.cell[i] = c;
    B.rank[i] = atomicAdd(&B.cnt[c], 1);
}

__global__ void k_scan_block(const int* in, int* out, int* bsum, int n) {
    __shared__ int s[SCAN_BS];
    int t = threadIdx.x, gid = blockIdx.x * SCAN_BS + t;
    s[t] = gid < n ? in[gid] : 0;
    __syncthreads();
    for (int off = 1; off < SCAN_BS; off <<= 1) {
        int v = t >= off ? s[t - off] : 0;
        __syncthreads();
        s[t] += v;
        __syncthreads();
    }
    if (gid < n) out[gid] = s[t];
    if (t == SCAN_BS - 1) bsum[blockIdx.x] = s[t];
}

__global__ void k_scan_single(const int* in, int* out, int n) {
    __shared__ int s[SCAN_BS];
    __shared__ int carry;
    int t = threadIdx.x;
    if (t == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += SCAN_BS) {
        int gid = base + t;
        s[t] = gid < n ? in[gid] : 0;
        __syncthreads();
        for (int off = 1; off < SCAN_BS; off <<= 1) {
            int v = t >= off ? s[t - off] : 0;
            __syncthreads();
            s[t] += v;
            __syncthreads();
        }
        if (gid < n) out[gid] = s[t] + carry;
        __syncthreads();
        if (t == SCAN_BS - 1) carry += s[t];
        __syncthreads();
    }
}

__global__ void k_scan_add(int* out, const int* boff, int n) {
    int gid = blockIdx.x * SCAN_BS + threadIdx.x;
    if (gid < n) out[gid] += boff[blockIdx.x];
}

__global__ void k_scatter(Bufs B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) scatter_i(i, B);
}
__global__ void k_density(Params P, Bufs B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) density_i(i, P, B);
}
__global__ void k_force(Params P, Bufs B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) force_i(i, P, B);
}
#endif

// ─────────────────────────── 시뮬레이션 ───────────────────────────
struct Sim {
    Params P{};
    int n = 0, cap = 0, M = 0, sub = 1;
    float k = 0, g0 = 0, vref = 0;
    std::vector<F4> pos, vel, poss, vels;
    std::vector<float> rho, p;
    std::vector<int> cell, rank, cnt, start, partial;
    bool gpu_ok = false;
    std::string gpu_name;
#ifdef USE_HIP
    Bufs db{};
    F4* dPosOwn = nullptr;  // interop이 아닐 때(또는 벤치마크) 쓰는 위치 버퍼
    int *bsum = nullptr, *boff = nullptr;
    int nb = 0;
    bool alloc = false;
#endif

    void probe() {
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

    // 장면 1(댐 붕괴)이 대략 N개가 되도록 상자 크기 결정. 상자 = k * (2, 1.1, 1), 입자 간격 1
    void init(int N) {
        k = std::max(10.f, std::cbrt(N / 0.5775f));
        P.Lx = 2.f * k;
        P.Ly = 1.1f * k;
        P.Lz = k;
        P.h = 2.f;
        P.h2 = P.h * P.h;
        P.cellInv = 1.f / P.h;
        P.gw = (int)std::ceil(P.Lx / P.h);
        P.gh = (int)std::ceil(P.Ly / P.h);
        P.gd = (int)std::ceil(P.Lz / P.h);
        const double PI = 3.14159265358979;
        P.poly6 = (float)(315.0 / (64.0 * PI * std::pow((double)P.h, 9.0)));
        P.spikyG = (float)(-45.0 / (PI * std::pow((double)P.h, 6.0)));
        P.viscL = (float)(45.0 / (PI * std::pow((double)P.h, 6.0)));

        // 간격 1 정육면체 격자에서 밀도가 정확히 1이 되도록 질량 결정
        double sum = 0;
        for (int a = -2; a <= 2; ++a)
            for (int b = -2; b <= 2; ++b)
                for (int c = -2; c <= 2; ++c) {
                    double r2 = a * a + b * b + c * c;
                    if (r2 < P.h2) sum += std::pow(P.h2 - r2, 3.0);
                }
        P.rho0 = 1.f;
        P.m = (float)(1.0 / (P.poly6 * sum));

        // 시간 단위 = 1프레임. 상자 높이를 ~70프레임에 떨어지도록 중력을 상자 크기에 맞춤
        const float T = 70.f;
        g0 = 2.f * P.Ly / (T * T);
        vref = std::sqrt(2.f * g0 * P.Ly);
        float c = 7.f * vref;  // 약압축: 음속 = 최대 유속의 ~7배 (밀도 오차 ~2%)
        P.c2 = c * c;
        P.maxv = 0.5f * c;
        P.nu = 0.06f * P.h * c / 8.f;
        sub = (int)std::ceil(c / (0.4f * P.h));  // CFL
        P.dt = 1.f / sub;
        P.eps = 0.5f;
        P.gx = P.gz = 0.f;
        P.gy = -g0;
        P.rep = 6.f * g0;
        P.hand = 0;
        P.hR = 0.12f * P.Ly;

        M = P.gw * P.gh * P.gd;
        cap = (int)(0.36f * P.Lx * P.Ly * P.Lz) + 4096;
        pos.assign(cap, F4{0, 0, 0, 0});
        vel.assign(cap, F4{0, 0, 0, 0});
        poss.assign(cap, F4{0, 0, 0, 0});
        vels.assign(cap, F4{0, 0, 0, 0});
        rho.assign(cap, 0.f);
        p.assign(cap, 0.f);
        cell.assign(cap, 0);
        rank.assign(cap, 0);
        cnt.assign(M, 0);
        start.assign(M + 1, 0);
        partial.assign(omp_get_max_threads() + 1, 0);
        gpu_alloc();
    }

    void fill(int sc) {
        n = 0;
        unsigned seed = 7u;
        auto jit = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return ((seed >> 8) / 16777216.f - 0.5f) * 0.1f;
        };
        auto add = [&](float x, float y, float z) {
            if (n >= cap) return;
            pos[n] = F4{x + jit(), y + jit(), z + jit(), 0.f};
            vel[n] = F4{0, 0, 0, 0};
            ++n;
        };
        auto block = [&](float x0, float y0, float z0, float x1, float y1, float z1) {
            for (float z = z0 + 0.5f; z < z1; z += 1.f)
                for (float y = y0 + 0.5f; y < y1; y += 1.f)
                    for (float x = x0 + 0.5f; x < x1; x += 1.f) add(x, y, z);
        };
        auto ball = [&](float cx, float cy, float cz, float R) {
            for (float z = cz - R; z < cz + R; z += 1.f)
                for (float y = cy - R; y < cy + R; y += 1.f)
                    for (float x = cx - R; x < cx + R; x += 1.f) {
                        float dx = x - cx, dy = y - cy, dz = z - cz;
                        if (dx * dx + dy * dy + dz * dz < R * R) add(x, y, z);
                    }
        };
        const float e = P.eps, X = P.Lx, Y = P.Ly, Z = P.Lz;
        if (sc == 2) {
            block(e, e, e, X - e, 0.2f * Y, Z - e);
            ball(0.5f * X, 0.62f * Y, 0.5f * Z, 0.24f * Y);
        } else if (sc == 3) {
            block(e, e, e, 0.2f * X, 0.7f * Y, 0.6f * Z);
            block(0.8f * X, e, 0.4f * Z, X - e, 0.7f * Y, Z - e);
        } else {
            block(e, e, e, 0.35f * X, 0.75f * Y, Z - e);
        }
    }

    // ── CPU ──
    void cpu_step() {
        Params Pl = P;
        Bufs B{pos.data(), vel.data(), poss.data(), vels.data(), rho.data(), p.data(),
               cell.data(), rank.data(), cnt.data(), start.data()};
        const int N = n, MM = M;
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int c = 0; c < MM; ++c) B.cnt[c] = 0;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                int c = cell_of(B.pos[i], Pl);
                B.cell[i] = c;
                B.rank[i] = __atomic_fetch_add(&B.cnt[c], 1, __ATOMIC_RELAXED);
            }

            int T = omp_get_num_threads(), t = omp_get_thread_num();
            int lo = (int)((long long)MM * t / T), hi = (int)((long long)MM * (t + 1) / T);
            int s = 0;
            for (int c = lo; c < hi; ++c) s += B.cnt[c];
            partial[t + 1] = s;
            #pragma omp barrier
            #pragma omp single
            {
                partial[0] = 0;
                for (int q = 0; q < T; ++q) partial[q + 1] += partial[q];
            }
            int run = partial[t];
            for (int c = lo; c < hi; ++c) {
                B.start[c] = run;
                run += B.cnt[c];
            }
            if (t == T - 1) B.start[MM] = run;
            #pragma omp barrier

            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) scatter_i(i, B);
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) density_i(i, Pl, B);
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) force_i(i, Pl, B);
        }
    }

    void cpu_frame() {
        for (int s = 0; s < sub; ++s) cpu_step();
    }

    // ── GPU ──
    void gpu_alloc() {
#ifdef USE_HIP
        if (!gpu_ok) return;
        gpu_free();
        nb = (M + SCAN_BS - 1) / SCAN_BS;
        auto f4 = [&](F4** q) { HIPCHECK(hipMalloc((void**)q, sizeof(F4) * cap)); };
        auto fl = [&](float** q) { HIPCHECK(hipMalloc((void**)q, sizeof(float) * cap)); };
        auto in = [&](int** q, size_t c) { HIPCHECK(hipMalloc((void**)q, sizeof(int) * c)); };
        f4(&dPosOwn); f4(&db.vel); f4(&db.poss); f4(&db.vels);
        fl(&db.rho); fl(&db.p);
        in(&db.cell, cap); in(&db.rank, cap); in(&db.cnt, M); in(&db.start, M + 1);
        in(&bsum, nb); in(&boff, nb + 1);
        HIPCHECK(hipMemset(db.start, 0, sizeof(int) * (M + 1)));  // start[0] = 0
        HIPCHECK(hipMemset(boff, 0, sizeof(int) * (nb + 1)));     // boff[0] = 0
        alloc = true;
#endif
    }

    void gpu_free() {
#ifdef USE_HIP
        if (!alloc) return;
        HIPCHECK(hipDeviceSynchronize());
        for (F4* q : {dPosOwn, db.vel, db.poss, db.vels}) (void)hipFree(q);
        for (float* q : {db.rho, db.p}) (void)hipFree(q);
        for (int* q : {db.cell, db.rank, db.cnt, db.start, bsum, boff}) (void)hipFree(q);
        alloc = false;
#endif
    }

#ifdef USE_HIP
    void gpu_frame(F4* dpos) {
        db.pos = dpos;
        const int BS = 256, G = (n + BS - 1) / BS;
        for (int s = 0; s < sub; ++s) {
            hipLaunchKernelGGL(k_zero, dim3((M + BS - 1) / BS), dim3(BS), 0, 0, db.cnt, M);
            hipLaunchKernelGGL(k_cell, dim3(G), dim3(BS), 0, 0, P, db, n);
            hipLaunchKernelGGL(k_scan_block, dim3(nb), dim3(SCAN_BS), 0, 0, (const int*)db.cnt, db.start + 1, bsum, M);
            hipLaunchKernelGGL(k_scan_single, dim3(1), dim3(SCAN_BS), 0, 0, (const int*)bsum, boff + 1, nb);
            hipLaunchKernelGGL(k_scan_add, dim3(nb), dim3(SCAN_BS), 0, 0, db.start + 1, (const int*)boff, M);
            hipLaunchKernelGGL(k_scatter, dim3(G), dim3(BS), 0, 0, db, n);
            hipLaunchKernelGGL(k_density, dim3(G), dim3(BS), 0, 0, P, db, n);
            hipLaunchKernelGGL(k_force, dim3(G), dim3(BS), 0, 0, P, db, n);
        }
        HIPCHECK(hipGetLastError());
        HIPCHECK(hipDeviceSynchronize());
    }
    void upload(F4* dpos) {
        HIPCHECK(hipMemcpy(dpos, pos.data(), sizeof(F4) * n, hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(db.vel, vel.data(), sizeof(F4) * n, hipMemcpyHostToDevice));
    }
    void download(F4* dpos, bool with_vel) {
        HIPCHECK(hipMemcpy(pos.data(), dpos, sizeof(F4) * n, hipMemcpyDeviceToHost));
        if (with_vel) HIPCHECK(hipMemcpy(vel.data(), db.vel, sizeof(F4) * n, hipMemcpyDeviceToHost));
    }
#endif

    ~Sim() { gpu_free(); }
};

// ─────────────────────────── 셰이더 ───────────────────────────
static const char* GLSL_HDR = "#version 330 core\n";

static const char* GLSL_SKY = R"(
const vec3 SUN = normalize(vec3(-0.45, 0.85, 0.30));
vec3 sky(vec3 d) {
    float t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 c = mix(vec3(0.05, 0.055, 0.075), vec3(0.32, 0.45, 0.62), pow(t, 1.4));
    float s = max(dot(d, SUN), 0.0);
    c += vec3(1.0, 0.92, 0.75) * pow(s, 500.0) * 6.0 + vec3(0.30, 0.25, 0.18) * pow(s, 6.0) * 0.35;
    return c;
}
)";

static const char* VS_FULL = R"(
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

// 배경: 하늘 + 바닥 체커 (픽셀마다 광선 추적)
static const char* FS_BG = R"(
in vec2 uv;
uniform vec3 uCam, uRight, uUp, uBack, uBox;
uniform vec2 uTan;
out vec4 o;
void main() {
    vec3 d = normalize(uRight * (uv.x * 2.0 - 1.0) * uTan.x + uUp * (uv.y * 2.0 - 1.0) * uTan.y - uBack);
    vec3 c = sky(d);
    if (d.y < -1e-4 && uCam.y > 0.0) {
        float t = -uCam.y / d.y;
        vec3 p = uCam + t * d;
        vec2 q = p.xz / (uBox.y / 10.0);
        float chk = mod(floor(q.x) + floor(q.y), 2.0);
        bool inside = p.x > 0.0 && p.x < uBox.x && p.z > 0.0 && p.z < uBox.z;
        vec3 fl = inside ? mix(vec3(0.20, 0.22, 0.25), vec3(0.30, 0.32, 0.36), chk)
                         : mix(vec3(0.07, 0.075, 0.085), vec3(0.09, 0.095, 0.11), chk);
        fl *= 0.55 + 0.45 * max(SUN.y, 0.0);
        float fade = exp(-t / (uBox.x * 4.0));
        c = mix(c, fl, fade);
    }
    o = vec4(c, 1.0);
}
)";

static const char* VS_SPRITE = R"(
layout(location = 0) in vec4 aPos;
uniform mat4 uView, uProj;
uniform float uRadius, uPxScale;
out vec3 vView;
out float vSpeed;
void main() {
    vec4 v = uView * vec4(aPos.xyz, 1.0);
    vView = v.xyz;
    vSpeed = aPos.w;
    gl_Position = uProj * v;
    gl_PointSize = max(1.0, 2.0 * uRadius * uPxScale / max(-v.z, 1e-3));
}
)";

// 구 스프라이트 -> 화면 선형 깊이
static const char* FS_DEPTH = R"(
in vec3 vView;
uniform mat4 uProj;
uniform float uRadius;
out float oDepth;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    c.y = -c.y;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    vec3 p = vView + vec3(c, sqrt(1.0 - r2)) * uRadius;
    vec4 clip = uProj * vec4(p, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    oDepth = -p.z;
}
)";

// 두께 (가산) + 속도 가중 두께
static const char* FS_THICK = R"(
in float vSpeed;
uniform float uRadius;
out vec4 o;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float t = 2.0 * sqrt(1.0 - r2) * uRadius;
    o = vec4(t, t * vSpeed, 0.0, 0.0);
}
)";

// 깊이 양방향 블러 (분리형): 공간 가우시안 x 깊이 차이 가우시안 -> 가장자리는 유지
static const char* FS_BLUR = R"(
uniform sampler2D uTex;
uniform ivec2 uDir;
uniform float uWorldR, uPxScale, uFall;
out float o;
void main() {
    ivec2 pc = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(uTex, 0);
    float d = texelFetch(uTex, pc, 0).r;
    if (d <= 0.0) { o = 0.0; return; }
    float rad = clamp(uWorldR * uPxScale / d, 1.0, 24.0);
    float sig = rad * 0.5;
    float sum = 0.0, ws = 0.0;
    for (int i = -24; i <= 24; ++i) {
        if (abs(float(i)) > rad) continue;
        float s = texelFetch(uTex, clamp(pc + uDir * i, ivec2(0), sz - 1), 0).r;
        if (s <= 0.0) continue;
        float dd = (s - d) * uFall;
        float w = exp(-float(i * i) / (2.0 * sig * sig) - dd * dd);
        sum += s * w;
        ws += w;
    }
    o = sum / ws;
}
)";

// 합성: 법선 복원 -> 굴절/흡수 + 프레넬 반사 + 하이라이트 + 거품
static const char* FS_COMP = R"(
in vec2 uv;
uniform sampler2D uDepth, uThick, uBg;
uniform vec2 uTan;
uniform mat3 uV2W;
uniform vec3 uLightV;
uniform float uA, uB, uThickScale, uLy, uVref;
out vec4 o;
vec3 vpos(vec2 q, float d) { return vec3((q * 2.0 - 1.0) * uTan * d, -d); }
void main() {
    ivec2 pc = ivec2(gl_FragCoord.xy);
    ivec2 sz = textureSize(uDepth, 0);
    vec2 px = 1.0 / vec2(sz);
    float d = texelFetch(uDepth, pc, 0).r;
    vec3 bg = texelFetch(uBg, pc, 0).rgb;
    if (d <= 0.0) { o = vec4(bg, 1.0); gl_FragDepth = 1.0; return; }

    vec3 P = vpos(uv, d);
    float dr = texelFetch(uDepth, min(pc + ivec2(1, 0), sz - 1), 0).r;
    float dl = texelFetch(uDepth, max(pc - ivec2(1, 0), ivec2(0)), 0).r;
    float du = texelFetch(uDepth, min(pc + ivec2(0, 1), sz - 1), 0).r;
    float dn = texelFetch(uDepth, max(pc - ivec2(0, 1), ivec2(0)), 0).r;
    vec3 ddx  = dr > 0.0 ? vpos(uv + vec2(px.x, 0.0), dr) - P : vec3(0.0);
    vec3 ddx2 = dl > 0.0 ? P - vpos(uv - vec2(px.x, 0.0), dl) : vec3(0.0);
    if (dr <= 0.0 || (dl > 0.0 && abs(ddx2.z) < abs(ddx.z))) ddx = ddx2;
    vec3 ddy  = du > 0.0 ? vpos(uv + vec2(0.0, px.y), du) - P : vec3(0.0);
    vec3 ddy2 = dn > 0.0 ? P - vpos(uv - vec2(0.0, px.y), dn) : vec3(0.0);
    if (du <= 0.0 || (dn > 0.0 && abs(ddy2.z) < abs(ddy.z))) ddy = ddy2;
    vec3 cr = cross(ddx, ddy);
    vec3 n = dot(cr, cr) > 1e-12 ? normalize(cr) : vec3(0.0, 0.0, 1.0);
    vec3 V = normalize(-P);
    if (dot(n, V) < 0.0) n = -n;

    vec4 th = texelFetch(uThick, pc, 0);
    float T = th.r * uThickScale / uLy;              // 두께 (상자 높이 대비)
    float spd = th.g / max(th.r, 1e-4) / uVref;      // 평균 속력 (최대 유속 대비)

    // 굴절: 법선만큼 배경을 비틀어 보고, 두께만큼 빨강부터 흡수
    vec2 ruv = clamp(uv + n.xy * 0.05 * min(T * 5.0, 1.0), vec2(0.0), vec2(1.0));
    vec3 refr = texture(uBg, ruv).rgb;
    vec3 water;
    vec3 tr = exp(-vec3(5.0, 1.3, 0.45) * T);
    float lit = 0.45 + 0.55 * max(dot(uV2W * n, SUN), 0.0);
    vec3 inscatter = vec3(0.05, 0.36, 0.58) * lit;   // 물 속에서 산란된 빛 (두꺼울수록 이 색)
    water = vec3(0.0);
    water = refr * tr + inscatter * (1.0 - tr);

    // 반사 (프레넬) + 태양 하이라이트
    vec3 R = uV2W * reflect(-V, n);
    float F = 0.02 + 0.98 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
    vec3 c = mix(water, sky(R), F);
    vec3 H = normalize(uLightV + V);
    float edge = smoothstep(0.08, 0.35, dot(n, V));   // 실루엣 가장자리의 번쩍임 억제
    c += vec3(1.0, 0.95, 0.85) * pow(max(dot(n, H), 0.0), 220.0) * 1.6 * edge;

    // 거품: 빠르게 움직이는 물
    float foam = smoothstep(0.55, 1.15, spd) * smoothstep(0.0, 0.02, T);
    vec3 fc = vec3(0.90, 0.95, 1.0) * (0.55 + 0.45 * max(dot(n, uLightV), 0.0));
    c = mix(c, fc, foam * 0.8);

    o = vec4(c, 1.0);
    gl_FragDepth = ((uA * -d + uB) / d) * 0.5 + 0.5;
}
)";

// 입자를 그냥 공으로 (M 키), 손 구체에도 사용
static const char* FS_SPHERE = R"(
in vec3 vView;
in float vSpeed;
uniform mat4 uProj;
uniform float uRadius, uVref, uAlpha;
uniform vec3 uLightV, uColor;
uniform int uSolid;
out vec4 o;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    c.y = -c.y;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    vec3 n = vec3(c, sqrt(1.0 - r2));
    vec3 p = vView + n * uRadius;
    vec4 clip = uProj * vec4(p, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    vec3 base;
    if (uSolid == 1) {
        base = uColor;
    } else {
        float s = vSpeed / uVref;
        base = mix(vec3(0.04, 0.20, 0.60), vec3(0.15, 0.65, 0.95), smoothstep(0.0, 0.6, s));
        base = mix(base, vec3(0.92, 0.96, 1.0), smoothstep(0.7, 1.2, s));
    }
    float diff = max(dot(n, uLightV), 0.0);
    vec3 H = normalize(uLightV + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(n, H), 0.0), 60.0);
    o = vec4(base * (0.25 + 0.75 * diff) + 0.35 * spec, uSolid == 1 ? uAlpha : 1.0);
}
)";

static const char* VS_LINE = R"(
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
)";
static const char* FS_LINE = R"(
uniform vec4 uColor;
out vec4 o;
void main() { o = uColor; }
)";

static GLuint compile(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "셰이더 컴파일 실패:\n%s\n", log);
        std::exit(1);
    }
    return s;
}

static GLuint program(const char* vs, const char* fs, bool sky = false) {
    std::string v = std::string(GLSL_HDR) + vs;
    std::string f = std::string(GLSL_HDR) + (sky ? GLSL_SKY : "") + fs;
    GLuint p = glCreateProgram();
    GLuint a = compile(GL_VERTEX_SHADER, v), b = compile(GL_FRAGMENT_SHADER, f);
    glAttachShader(p, a);
    glAttachShader(p, b);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
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
            for (int q = 0; q < 4; ++q) s += a.m[q * 4 + rr] * b.m[c * 4 + q];
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

struct V3 {
    float x, y, z;
};
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 norm(V3 a) { return a * (1.f / std::sqrt(dot(a, a))); }

// ─────────────────────────── 앱 ───────────────────────────
struct App {
    GLFWwindow* win = nullptr;
    Sim S;
    int fbw = 0, fbh = 0;
    GLuint vao = 0, vbo = 0, emptyVao = 0, boxVao = 0, boxVbo = 0, handVao = 0, handVbo = 0;
    GLuint bgTex = 0, depthTex = 0, tmpTex = 0, thickTex = 0, depthRbo = 0;
    GLuint bgFbo = 0, depthFbo = 0, tmpFbo = 0, thickFbo = 0;
    GLuint pBg = 0, pDepth = 0, pThick = 0, pBlur = 0, pComp = 0, pSph = 0, pLine = 0;

    int scene = 1, rmode = 0;  // rmode 0: 물 표면, 1: 입자 공
    float yaw = 0.55f, pitch = 0.40f, zoom = 1.f;
    float tiltX = 0.f, tiltZ = 0.f;
    bool paused = false, autorot = false, slosh = false, vsync = true, fullscreen = false;
    bool ldrag = false, rdrag = false, hand_valid = false;
    double lastx = 0, lasty = 0, sim_ms = 0;
    long simframe = 0;
    V3 hand{0, 0, 0}, handv{0, 0, 0};
    int wx = 100, wy = 100, ww = 1600, wh = 900;

    // 카메라 (render에서 갱신, 손 광선에 사용)
    V3 camPos{}, camRight{}, camUp{}, camBack{}, target{};
    float tanX = 0, tanY = 0;

    bool use_gpu = false, interop = false, dirty = true;
#ifdef USE_HIP
    hipGraphicsResource_t res = nullptr;
#endif

    // ── 입자 버퍼 ──
    void make_particles(int N) {
        gpu_release();
        S.init(N);
        if (!vbo) glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(F4) * S.cap, nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(F4), nullptr);
        gpu_register();
        make_box();
        fill(scene);
    }

    void fill(int sc) {
        scene = sc;
        S.fill(sc);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4) * S.n, S.pos.data());
        dirty = true;
        tiltX = tiltZ = 0.f;
        simframe = 0;
    }

    void gpu_register() {
#ifdef USE_HIP
        if (!S.gpu_ok) return;
        glFinish();
        hipError_t e = hipGraphicsGLRegisterBuffer(&res, vbo, hipGraphicsRegisterFlagsNone);
        interop = (e == hipSuccess);
        if (!interop) {
            res = nullptr;
            (void)hipGetLastError();
            static bool warned = false;
            if (!warned) {
                std::fprintf(stderr, "HIP-GL interop 등록 실패 (%s) -> GPU 계산 후 복사 방식\n", hipGetErrorString(e));
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
#endif
    }

    // ── 한 프레임 ──
    void simulate() {
        if (paused) return;
        if (slosh) tiltX = 0.38f * std::sin(simframe * 0.045f);
        V3 g = norm(V3{std::sin(tiltX), -std::cos(tiltX) * std::cos(tiltZ), std::sin(tiltZ)}) * S.g0;
        S.P.gx = g.x;
        S.P.gy = g.y;
        S.P.gz = g.z;
        S.P.hand = rdrag && hand_valid;
        S.P.hx = hand.x;
        S.P.hy = hand.y;
        S.P.hz = hand.z;
        S.P.hvx = handv.x;
        S.P.hvy = handv.y;
        S.P.hvz = handv.z;

        double t0 = glfwGetTime();
        if (use_gpu) {
#ifdef USE_HIP
            if (interop) {
                F4* ptr = nullptr;
                size_t sz = 0;
                HIPCHECK(hipGraphicsMapResources(1, &res, 0));
                HIPCHECK(hipGraphicsResourceGetMappedPointer((void**)&ptr, &sz, res));
                if (dirty) S.upload(ptr);
                S.gpu_frame(ptr);
                HIPCHECK(hipGraphicsUnmapResources(1, &res, 0));
            } else {
                if (dirty) S.upload(S.dPosOwn);
                S.gpu_frame(S.dPosOwn);
                S.download(S.dPosOwn, false);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4) * S.n, S.pos.data());
            }
            dirty = false;
#endif
        } else {
            S.cpu_frame();
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4) * S.n, S.pos.data());
        }
        sim_ms = 0.9 * sim_ms + 0.1 * (glfwGetTime() - t0) * 1000;
        ++simframe;
    }

    void set_gpu(bool on) {
        if (on && !S.gpu_ok) return;
        if (on == use_gpu) return;
#ifdef USE_HIP
        if (!on && !dirty) {  // GPU -> CPU: 최신 상태를 호스트로
            if (interop) {
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4) * S.n, S.pos.data());
                HIPCHECK(hipMemcpy(S.vel.data(), S.db.vel, sizeof(F4) * S.n, hipMemcpyDeviceToHost));
            } else {
                S.download(S.dPosOwn, true);
            }
        }
#endif
        if (on) dirty = true;
        use_gpu = on;
    }

    // ── 렌더 타깃 ──
    static void mktex(GLuint& tex, GLint ifmt, GLenum fmt, GLenum type, GLint filter, int w, int h) {
        if (tex) glDeleteTextures(1, &tex);
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    static void mkfbo(GLuint& fbo, GLuint tex, GLuint rbo = 0) {
        if (fbo) glDeleteFramebuffers(1, &fbo);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        if (rbo) glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::fprintf(stderr, "프레임버퍼 생성 실패\n");
            std::exit(1);
        }
    }
    void make_targets() {
        mktex(bgTex, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_LINEAR, fbw, fbh);
        mktex(depthTex, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST, fbw, fbh);
        mktex(tmpTex, GL_R32F, GL_RED, GL_FLOAT, GL_NEAREST, fbw, fbh);
        mktex(thickTex, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_NEAREST, fbw, fbh);
        if (depthRbo) glDeleteRenderbuffers(1, &depthRbo);
        glGenRenderbuffers(1, &depthRbo);
        glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, fbw, fbh);
        mkfbo(bgFbo, bgTex);
        mkfbo(depthFbo, depthTex, depthRbo);
        mkfbo(tmpFbo, tmpTex);
        mkfbo(thickFbo, thickTex);
    }

    void make_box() {
        float X = S.P.Lx, Y = S.P.Ly, Z = S.P.Lz;
        float v[] = {0, 0, 0, X, 0, 0,  X, 0, 0, X, 0, Z,  X, 0, Z, 0, 0, Z,  0, 0, Z, 0, 0, 0,
                     0, Y, 0, X, Y, 0,  X, Y, 0, X, Y, Z,  X, Y, Z, 0, Y, Z,  0, Y, Z, 0, Y, 0,
                     0, 0, 0, 0, Y, 0,  X, 0, 0, X, Y, 0,  X, 0, Z, X, Y, Z,  0, 0, Z, 0, Y, Z};
        glBindVertexArray(boxVao);
        glBindBuffer(GL_ARRAY_BUFFER, boxVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, nullptr);
    }

    void init_gl() {
        pBg = program(VS_FULL, FS_BG, true);
        pDepth = program(VS_SPRITE, FS_DEPTH);
        pThick = program(VS_SPRITE, FS_THICK);
        pBlur = program(VS_FULL, FS_BLUR);
        pComp = program(VS_FULL, FS_COMP, true);
        pSph = program(VS_SPRITE, FS_SPHERE);
        pLine = program(VS_LINE, FS_LINE);
        glGenVertexArrays(1, &vao);
        glGenVertexArrays(1, &emptyVao);
        glGenVertexArrays(1, &boxVao);
        glGenBuffers(1, &boxVbo);
        glGenVertexArrays(1, &handVao);
        glGenBuffers(1, &handVbo);
        glBindVertexArray(handVao);
        glBindBuffer(GL_ARRAY_BUFFER, handVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(F4), nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(F4), nullptr);
        glfwGetFramebufferSize(win, &fbw, &fbh);
        make_targets();
        glEnable(GL_PROGRAM_POINT_SIZE);
    }

    // ── 손: 커서 광선과 (상자 중심을 지나는, 화면과 평행한) 평면의 교점 ──
    void update_hand() {
        if (!rdrag) {
            hand_valid = false;
            return;
        }
        double mx, my;
        int w, h;
        glfwGetCursorPos(win, &mx, &my);
        glfwGetWindowSize(win, &w, &h);
        float x = (float)(mx / std::max(1, w)) * 2.f - 1.f, y = 1.f - (float)(my / std::max(1, h)) * 2.f;
        V3 dir = norm(camRight * (x * tanX) + camUp * (y * tanY) - camBack);
        float den = dot(dir, camBack);
        if (std::fabs(den) < 1e-5f) return;
        float t = dot(target - camPos, camBack) / den;
        V3 p = camPos + dir * t;
        if (hand_valid) {
            V3 v = p - hand;
            float sp = std::sqrt(dot(v, v)), mx_ = S.P.maxv * 0.6f;
            if (sp > mx_) v = v * (mx_ / sp);
            handv = handv * 0.5f + v * 0.5f;
        } else {
            handv = {0, 0, 0};
        }
        hand = p;
        hand_valid = true;
    }

    // ── 렌더링 ──
    void render() {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        if (w <= 0 || h <= 0) return;
        if (w != fbw || h != fbh) {
            fbw = w;
            fbh = h;
            make_targets();
        }
        const float fovy = 0.8f, aspect = (float)fbw / fbh;
        tanY = std::tan(fovy * 0.5f);
        tanX = tanY * aspect;
        float pxScale = fbh / (2.f * tanY);
        target = {S.P.Lx * 0.5f, S.P.Ly * 0.33f, S.P.Lz * 0.5f};
        float dist = 1.15f * S.P.Lx / zoom;
        M4 R = mul(rotX(pitch), rotY(-yaw));
        camRight = {R.m[0], R.m[4], R.m[8]};
        camUp = {R.m[1], R.m[5], R.m[9]};
        camBack = {R.m[2], R.m[6], R.m[10]};
        camPos = target + camBack * dist;
        M4 Vw = mul(trans(0, 0, -dist), mul(R, trans(-target.x, -target.y, -target.z)));
        M4 Pj = persp(fovy, aspect, 0.5f, 10.f * S.P.Lx);
        M4 MVP = mul(Pj, Vw);
        V3 sun = norm(V3{-0.45f, 0.85f, 0.30f});
        float lv[3] = {dot(camRight, sun), dot(camUp, sun), dot(camBack, sun)};
        float v2w[9] = {camRight.x, camRight.y, camRight.z, camUp.x, camUp.y, camUp.z, camBack.x, camBack.y, camBack.z};

        auto U = [](GLuint p, const char* n) { return glGetUniformLocation(p, n); };
        auto sprite_uniforms = [&](GLuint p, float radius) {
            glUseProgram(p);
            glUniformMatrix4fv(U(p, "uView"), 1, GL_FALSE, Vw.m);
            glUniformMatrix4fv(U(p, "uProj"), 1, GL_FALSE, Pj.m);
            glUniform1f(U(p, "uRadius"), radius);
            glUniform1f(U(p, "uPxScale"), pxScale);
        };
        auto bg_pass = [&](GLuint fbo) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glViewport(0, 0, fbw, fbh);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);
            glUseProgram(pBg);
            glUniform3f(U(pBg, "uCam"), camPos.x, camPos.y, camPos.z);
            glUniform3f(U(pBg, "uRight"), camRight.x, camRight.y, camRight.z);
            glUniform3f(U(pBg, "uUp"), camUp.x, camUp.y, camUp.z);
            glUniform3f(U(pBg, "uBack"), camBack.x, camBack.y, camBack.z);
            glUniform3f(U(pBg, "uBox"), S.P.Lx, S.P.Ly, S.P.Lz);
            glUniform2f(U(pBg, "uTan"), tanX, tanY);
            glBindVertexArray(emptyVao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
        };

        if (rmode == 0) {
            // ① 배경
            bg_pass(bgFbo);

            // ② 깊이
            glBindFramebuffer(GL_FRAMEBUFFER, depthFbo);
            glClearColor(0, 0, 0, 0);
            glClearDepth(1.0);
            glDepthMask(GL_TRUE);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            sprite_uniforms(pDepth, 0.85f);
            glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, S.n);
            glDisable(GL_DEPTH_TEST);

            // ③ 두께 (가산)
            glBindFramebuffer(GL_FRAMEBUFFER, thickFbo);
            glClear(GL_COLOR_BUFFER_BIT);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            sprite_uniforms(pThick, 1.0f);
            glDrawArrays(GL_POINTS, 0, S.n);
            glDisable(GL_BLEND);

            // ④ 깊이 블러 (가로 -> 세로) x 2
            glUseProgram(pBlur);
            glUniform1f(U(pBlur, "uWorldR"), 2.2f);
            glUniform1f(U(pBlur, "uPxScale"), pxScale);
            glUniform1f(U(pBlur, "uFall"), 0.55f);
            glUniform1i(U(pBlur, "uTex"), 0);
            glActiveTexture(GL_TEXTURE0);
            glBindVertexArray(emptyVao);
            for (int it = 0; it < 2; ++it) {
                glBindFramebuffer(GL_FRAMEBUFFER, tmpFbo);
                glBindTexture(GL_TEXTURE_2D, depthTex);
                glUniform2i(U(pBlur, "uDir"), 1, 0);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glBindFramebuffer(GL_FRAMEBUFFER, depthFbo);
                glBindTexture(GL_TEXTURE_2D, tmpTex);
                glUniform2i(U(pBlur, "uDir"), 0, 1);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            }

            // ⑤ 합성 -> 화면 (깊이도 써서 상자 선이 물 뒤로 가려지게)
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, fbw, fbh);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glDepthMask(GL_TRUE);
            glUseProgram(pComp);
            glUniform1i(U(pComp, "uDepth"), 0);
            glUniform1i(U(pComp, "uThick"), 1);
            glUniform1i(U(pComp, "uBg"), 2);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, depthTex);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, thickTex);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, bgTex);
            glActiveTexture(GL_TEXTURE0);
            glUniform2f(U(pComp, "uTan"), tanX, tanY);
            glUniformMatrix3fv(U(pComp, "uV2W"), 1, GL_FALSE, v2w);
            glUniform3fv(U(pComp, "uLightV"), 1, lv);
            glUniform1f(U(pComp, "uA"), Pj.m[10]);
            glUniform1f(U(pComp, "uB"), Pj.m[14]);
            glUniform1f(U(pComp, "uThickScale"), 1.f / (4.f / 3.f * 3.14159265f));  // 입자 1개 = 반지름 1 구
            glUniform1f(U(pComp, "uLy"), S.P.Ly);
            glUniform1f(U(pComp, "uVref"), S.vref);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glDepthFunc(GL_LESS);
        } else {
            bg_pass(0);
            glDepthMask(GL_TRUE);
            glClear(GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LESS);
            sprite_uniforms(pSph, 0.6f);
            glUniform1f(U(pSph, "uVref"), S.vref);
            glUniform3fv(U(pSph, "uLightV"), 1, lv);
            glUniform1i(U(pSph, "uSolid"), 0);
            glBindVertexArray(vao);
            glDrawArrays(GL_POINTS, 0, S.n);
        }

        // 손 구체 (반투명)
        if (rdrag && hand_valid) {
            F4 hp{hand.x, hand.y, hand.z, 0};
            glBindBuffer(GL_ARRAY_BUFFER, handVbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4), &hp);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_FALSE);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            sprite_uniforms(pSph, S.P.hR);
            glUniform3fv(U(pSph, "uLightV"), 1, lv);
            glUniform1i(U(pSph, "uSolid"), 1);
            glUniform3f(U(pSph, "uColor"), 1.0f, 0.55f, 0.25f);
            glUniform1f(U(pSph, "uAlpha"), 0.35f);
            glBindVertexArray(handVao);
            glDrawArrays(GL_POINTS, 0, 1);
            glDisable(GL_BLEND);
        }

        // 상자 모서리
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(pLine);
        glUniformMatrix4fv(U(pLine, "uMVP"), 1, GL_FALSE, MVP.m);
        glUniform4f(U(pLine, "uColor"), 0.75f, 0.85f, 1.0f, 0.35f);
        glBindVertexArray(boxVao);
        glDrawArrays(GL_LINES, 0, 24);
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        glDisable(GL_DEPTH_TEST);
    }

    void update_title(double fps) {
        char nstr[32], t[400];
        if (S.n >= 1000000) std::snprintf(nstr, sizeof nstr, "%.2fM", S.n / 1e6);
        else std::snprintf(nstr, sizeof nstr, "%dK", S.n / 1000);
        std::string be = use_gpu ? std::string(interop ? "HIP-GL interop " : "HIP+copy ") + S.gpu_name
                                 : "CPU OpenMP " + std::to_string(omp_get_max_threads()) + "T";
        std::snprintf(t, sizeof t, "SPH 3D | %s | %s particles x %d substeps | sim %.1f ms | %.0f fps | %s%s%s",
                      be.c_str(), nstr, S.sub, sim_ms, fps, rmode ? "spheres" : "fluid",
                      slosh ? " | slosh" : "", paused ? " | PAUSED" : "");
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

// ─────────────────────────── 입력 ───────────────────────────
static void on_key(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    App& A = *(App*)glfwGetWindowUserPointer(w);
    const float step = 0.12f, lim = 1.2f;
    switch (key) {
        case GLFW_KEY_ESCAPE: case GLFW_KEY_Q: glfwSetWindowShouldClose(w, 1); break;
        case GLFW_KEY_P: case GLFW_KEY_SPACE: A.paused = !A.paused; break;
        case GLFW_KEY_A: A.autorot = !A.autorot; break;
        case GLFW_KEY_S: A.slosh = !A.slosh; if (!A.slosh) A.tiltX = 0; break;
        case GLFW_KEY_M: A.rmode ^= 1; break;
        case GLFW_KEY_G: A.set_gpu(!A.use_gpu); break;
        case GLFW_KEY_V: A.vsync = !A.vsync; glfwSwapInterval(A.vsync ? 1 : 0); break;
        case GLFW_KEY_F: A.toggle_fullscreen(); break;
        case GLFW_KEY_1: case GLFW_KEY_2: case GLFW_KEY_3: A.fill(key - GLFW_KEY_0); break;
        case GLFW_KEY_R: A.fill(A.scene); break;
        case GLFW_KEY_0: A.tiltX = A.tiltZ = 0; A.slosh = false; break;
        case GLFW_KEY_LEFT: A.tiltX = std::max(-lim, A.tiltX - step); break;
        case GLFW_KEY_RIGHT: A.tiltX = std::min(lim, A.tiltX + step); break;
        case GLFW_KEY_UP: A.tiltZ = std::max(-lim, A.tiltZ - step); break;
        case GLFW_KEY_DOWN: A.tiltZ = std::min(lim, A.tiltZ + step); break;
        case GLFW_KEY_LEFT_BRACKET:
        case GLFW_KEY_RIGHT_BRACKET: {
            int target = (int)(0.5775f * A.S.k * A.S.k * A.S.k);
            int nn = key == GLFW_KEY_RIGHT_BRACKET ? target * 2 : target / 2;
            nn = std::clamp(nn, 4000, A.S.gpu_ok ? 8000000 : 2000000);
            bool g = A.use_gpu;
            A.set_gpu(false);
            A.make_particles(nn);
            A.set_gpu(g);
            break;
        }
    }
}

static void on_button(GLFWwindow* w, int button, int action, int) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        A.ldrag = action == GLFW_PRESS;
        if (A.ldrag) A.autorot = false;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        A.rdrag = action == GLFW_PRESS;
        A.hand_valid = false;
    }
    glfwGetCursorPos(w, &A.lastx, &A.lasty);
}

static void on_cursor(GLFWwindow* w, double x, double y) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    if (A.ldrag) {
        A.yaw += (float)(x - A.lastx) * 0.006f;
        A.pitch = std::clamp(A.pitch + (float)(y - A.lasty) * 0.006f, -0.2f, 1.45f);
    }
    A.lastx = x;
    A.lasty = y;
}

static void on_scroll(GLFWwindow* w, double, double dy) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    A.zoom = std::clamp(A.zoom * (float)std::pow(1.1, dy), 0.3f, 6.f);
}

// ─────────────────────────── 벤치마크 (창 없이) ───────────────────────────
static int run_bench(int N) {
    Sim S;
    S.probe();
    S.init(N);
    S.fill(1);
    std::printf("SPH 3D 벤치마크: 상자 %.0fx%.0fx%.0f, 장면 1, 입자 %d개, 서브스텝 %d/프레임, 셀 %d\n",
                S.P.Lx, S.P.Ly, S.P.Lz, S.n, S.sub, S.M);
    const int WARM = 1, FR = 3;
    int maxT = omp_get_max_threads();
    std::vector<int> ts{1};
    for (int t = 2; t <= maxT; t *= 2) ts.push_back(t);
    if (ts.back() != maxT) ts.push_back(maxT);
    double base = 0;
    for (int t : ts) {
        omp_set_num_threads(t);
        S.fill(1);
        for (int f = 0; f < WARM; ++f) S.cpu_frame();
        double t0 = omp_get_wtime();
        for (int f = 0; f < FR; ++f) S.cpu_frame();
        double ms = (omp_get_wtime() - t0) / FR * 1000;
        if (t == 1) base = ms;
        std::printf("  CPU %2d threads: %9.2f ms/frame  (%6.1f fps)   speedup x%.2f\n", t, ms, 1000 / ms, base / ms);
    }
    omp_set_num_threads(maxT);
#ifdef USE_HIP
    if (S.gpu_ok) {
        S.fill(1);
        S.upload(S.dPosOwn);
        for (int f = 0; f < WARM + 2; ++f) S.gpu_frame(S.dPosOwn);
        double t0 = omp_get_wtime();
        const int GF = 10;
        for (int f = 0; f < GF; ++f) S.gpu_frame(S.dPosOwn);
        double ms = (omp_get_wtime() - t0) / GF * 1000;
        std::printf("  GPU %s: %9.2f ms/frame  (%6.1f fps)   speedup x%.2f (vs 1 thread)\n",
                    S.gpu_name.c_str(), ms, 1000 / ms, base / ms);
        return 0;
    }
#endif
    std::printf("  GPU: 사용 불가 (USE_HIP 없이 빌드됐거나 장치 없음)\n");
    return 0;
}

// ─────────────────────────── 메인 ───────────────────────────
int main(int argc, char** argv) {
    int N = 0, scene = 1, shot_frames = 0, shot_mode = 0;
    const char* shot = nullptr;
    bool bench = false, shot_slosh = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (!std::strcmp(argv[i], "--shot") && i + 2 < argc) {  // 렌더링 확인용: --shot 파일.ppm 프레임수
            shot = argv[++i];
            shot_frames = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--spheres")) shot_mode = 1;
        else if (!std::strcmp(argv[i], "--slosh")) shot_slosh = true;
        else pos.push_back(argv[i]);
    }
    if (pos.size() > 0) N = std::atoi(pos[0].c_str());
    if (pos.size() > 1) scene = std::clamp(std::atoi(pos[1].c_str()), 1, 3);
    if (bench) return run_bench(N > 0 ? N : 131072);

    if (!glfwInit()) {
        std::fprintf(stderr, "GLFW 초기화 실패 (디스플레이가 없나요?)\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);
    App A;
    A.win = glfwCreateWindow(A.ww, A.wh, "SPH 3D", nullptr, nullptr);
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

    A.S.probe();  // GL 컨텍스트 다음에 HIP 초기화
    if (N <= 0) N = A.S.gpu_ok ? 262144 : 32768;
    A.scene = scene;
    A.rmode = shot_mode;
    A.slosh = shot_slosh;
    A.init_gl();
    A.make_particles(N);
    A.set_gpu(A.S.gpu_ok);
    if (A.S.gpu_ok) std::printf("HIP: %s, interop %s\n", A.S.gpu_name.c_str(), A.interop ? "OK" : "실패 -> 복사 방식");
    std::printf("상자 %.0fx%.0fx%.0f, 입자 %d개, 서브스텝 %d/프레임\n", A.S.P.Lx, A.S.P.Ly, A.S.P.Lz, A.S.n, A.S.sub);

    double last_title = 0, last_t = glfwGetTime(), fps = 60;
    long frame = 0;
    while (!glfwWindowShouldClose(A.win)) {
        glfwPollEvents();
        if (A.autorot) A.yaw += 0.003f;
        A.update_hand();
        A.simulate();
        A.render();
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
            A.render();
            std::vector<unsigned char> px((size_t)A.fbw * A.fbh * 3);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glReadBuffer(GL_BACK);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, A.fbw, A.fbh, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            FILE* f = std::fopen(shot, "wb");
            std::fprintf(f, "P6 %d %d 255\n", A.fbw, A.fbh);
            for (int y = A.fbh - 1; y >= 0; --y) std::fwrite(&px[(size_t)y * A.fbw * 3], 1, (size_t)A.fbw * 3, f);
            std::fclose(f);
            std::printf("saved %s (frame %ld, sim %.1f ms)\n", shot, frame, A.sim_ms);
            break;
        }
    }
    A.gpu_release();
    glfwTerminate();
    return 0;
}
