// sph.cpp — 터미널 인터랙티브 SPH 물 시뮬레이션: CPU(OpenMP) + GPU(ROCm/HIP)
//
// CPU 전용 빌드:
//   g++ -O3 -march=native -std=c++17 -fopenmp sph.cpp -o sph
// GPU(ROCm) 포함 빌드 (Strix Halo iGPU = gfx1151):
//   hipcc -O3 -std=c++17 -fopenmp -DUSE_HIP --offload-arch=gfx1151 -x hip sph.cpp -o sph
//
// 실행:  ./sph [res]          res = 1픽셀당 입자 배율 (기본: GPU 있으면 4, 없으면 2. 최대 10)
//                              입자 수 ∝ res², 서브스텝 수 ∝ res -> 계산량 ∝ res³
//        ./sph --bench [res]  화면 없이 CPU 스레드별 / GPU 속도 비교
//
// 조작:  좌클릭 드래그 = 물 밀기     우클릭 누르고 있기 = 커서로 빨아들이기
//        방향키 = 중력 방향 (상자 기울이기)     g = CPU <-> GPU 전환
//        [ ] = 입자 수(해상도) 줄이기/늘리기   1 = 댐 붕괴   2 = 물덩어리 낙하   3 = 수영장 + 양쪽 물기둥
//        휠/+ - = 손 크기   r = 장면 리셋   p = 일시정지   q = 종료
//
// 구조:  매 서브스텝마다
//   ① 셀 카운팅   각 입자의 격자 셀 번호 계산 + atomicAdd로 셀 안 순번(rank) 확보
//   ② prefix scan 셀별 개수 -> 셀 시작 위치 (CPU: 스레드별 2단계 scan, GPU: 블록 scan 3단계)
//   ③ scatter     입자를 셀 순서대로 재배치 (같은 셀 입자가 메모리상 연속 -> 캐시/코얼레싱)
//   ④ density     주변 3x3 셀만 보고 밀도/압력 계산
//   ⑤ force       압력 + 점성 + 중력 + 마우스 -> 적분 (정렬본을 읽고 원본에 써서 경쟁 없음)
// ①~⑤의 입자 단위 계산은 HD 함수 하나를 CPU 루프와 GPU 커널이 같이 씀.

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
#include <string>
#include <vector>

// ───────────────────────── 파라미터 / 버퍼 ─────────────────────────
struct Params {
    float W, H;                  // 월드 크기 (픽셀 단위, y는 아래 방향)
    float h, h2, cellInv;        // 커널 반경 h = 셀 크기
    int gw, gh;                  // 이웃 탐색 격자
    float m, rho0, c2, nu, dt, maxv, eps;
    float poly6, spikyG, viscL;  // 커널 정규화 상수
    float gx, gy;                // 중력
    float mx, my, mvx, mvy, mR;  // 마우스
    int mmode;                   // 0 없음, 1 밀기, 2 당기기
};

struct Bufs {
    float *x, *y, *vx, *vy;      // 상태 (force 단계의 출력)
    float *xs, *ys, *vxs, *vys;  // 셀 순서로 정렬된 사본 (density/force의 입력)
    float *rho, *p;
    int *cell, *rank, *cnt, *start;  // start 길이 = 셀 수 + 1
};

// ───────────────────── 입자 단위 계산 (CPU/GPU 공용) ─────────────────────
HD inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

HD inline int cell_of(float x, float y, const Params& P) {
    int cx = clampi((int)(x * P.cellInv), 0, P.gw - 1);
    int cy = clampi((int)(y * P.cellInv), 0, P.gh - 1);
    return cy * P.gw + cx;
}

HD inline void scatter_i(int i, const Bufs& B) {
    int d = B.start[B.cell[i]] + B.rank[i];
    B.xs[d] = B.x[i];
    B.ys[d] = B.y[i];
    B.vxs[d] = B.vx[i];
    B.vys[d] = B.vy[i];
}

// 같은 행의 3개 셀은 정렬된 배열에서 연속 구간이라 한 번의 루프로 훑을 수 있음
#define FOR_NEIGHBORS(P, B, xi, yi, ...)                                            \
    {                                                                               \
        int cx_ = clampi((int)((xi) * (P).cellInv), 0, (P).gw - 1);                 \
        int cy_ = clampi((int)((yi) * (P).cellInv), 0, (P).gh - 1);                 \
        int x0_ = cx_ > 0 ? cx_ - 1 : 0, x1_ = cx_ < (P).gw - 1 ? cx_ + 1 : cx_;    \
        int y0_ = cy_ > 0 ? cy_ - 1 : 0, y1_ = cy_ < (P).gh - 1 ? cy_ + 1 : cy_;    \
        for (int yy_ = y0_; yy_ <= y1_; ++yy_) {                                    \
            int b_ = (B).start[yy_ * (P).gw + x0_], e_ = (B).start[yy_ * (P).gw + x1_ + 1]; \
            for (int j = b_; j < e_; ++j) {                                         \
                float dx = (xi) - (B).xs[j], dy = (yi) - (B).ys[j];                 \
                float r2 = dx * dx + dy * dy;                                       \
                if (r2 < (P).h2) { __VA_ARGS__ }                                           \
            }                                                                       \
        }                                                                           \
    }

HD inline void density_i(int i, const Params& P, const Bufs& B) {
    float xi = B.xs[i], yi = B.ys[i];
    float sum = 0.f;
    FOR_NEIGHBORS(P, B, xi, yi, {
        float t = P.h2 - r2;
        sum += t * t * t;
    })
    float rho = P.m * P.poly6 * sum;
    B.rho[i] = rho;
    float pr = P.c2 * (rho - P.rho0);
    B.p[i] = pr > 0.f ? pr : 0.f;  // 음압 제거 -> 표면에서 입자 뭉침(tensile instability) 방지
}

HD inline void force_i(int i, const Params& P, const Bufs& B) {
    float xi = B.xs[i], yi = B.ys[i], vxi = B.vxs[i], vyi = B.vys[i];
    float rhoi = B.rho[i];
    float pri = B.p[i] / (rhoi * rhoi);
    float ax = 0.f, ay = 0.f, lx = 0.f, ly = 0.f;
    FOR_NEIGHBORS(P, B, xi, yi, {
        if (r2 > 1e-12f) {
            float r = sqrtf(r2), hr = P.h - r;
            float rhoj = B.rho[j];
            // 압력: 대칭형 -Σ m (p_i/ρ_i² + p_j/ρ_j²) ∇W_spiky  (운동량 보존)
            float fp = -P.m * (pri + B.p[j] / (rhoj * rhoj)) * P.spikyG * hr * hr / r;
            ax += fp * dx;
            ay += fp * dy;
            // 점성: ν Σ m/ρ_j (v_j - v_i) ∇²W_visc
            float fv = P.m * P.viscL * hr / rhoj;
            lx += fv * (B.vxs[j] - vxi);
            ly += fv * (B.vys[j] - vyi);
        }
    })
    ax += P.nu * lx + P.gx;
    ay += P.nu * ly + P.gy;

    if (P.mmode) {
        float dx = P.mx - xi, dy = P.my - yi;
        float d2 = dx * dx + dy * dy;
        if (d2 < P.mR * P.mR) {
            float d = sqrtf(d2) + 1e-6f, w = 1.f - d / P.mR;
            if (P.mmode == 1) {  // 밀기: 손 속도로 끌고 감
                ax += w * 0.8f * (P.mvx - vxi);
                ay += w * 0.8f * (P.mvy - vyi);
            } else {             // 당기기: 커서 쪽으로 + 감쇠
                ax += w * (0.35f * dx / d - 0.15f * vxi);
                ay += w * (0.35f * dy / d - 0.15f * vyi);
            }
        }
    }

    // 반-암시적 오일러 적분
    vxi += P.dt * ax;
    vyi += P.dt * ay;
    float sp2 = vxi * vxi + vyi * vyi;
    if (sp2 > P.maxv * P.maxv) {
        float s = P.maxv / sqrtf(sp2);
        vxi *= s;
        vyi *= s;
    }
    xi += P.dt * vxi;
    yi += P.dt * vyi;
    if (xi < P.eps)       { xi = P.eps;       if (vxi < 0) vxi *= -0.3f; }
    if (xi > P.W - P.eps) { xi = P.W - P.eps; if (vxi > 0) vxi *= -0.3f; }
    if (yi < P.eps)       { yi = P.eps;       if (vyi < 0) vyi *= -0.3f; }
    if (yi > P.H - P.eps) { yi = P.H - P.eps; if (vyi > 0) vyi *= -0.3f; }
    B.x[i] = xi;
    B.y[i] = yi;
    B.vx[i] = vxi;
    B.vy[i] = vyi;
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

__global__ void k_cell(Params P, Bufs B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = cell_of(B.x[i], B.y[i], P);
    B.cell[i] = c;
    B.rank[i] = atomicAdd(&B.cnt[c], 1);
}

// 블록 단위 inclusive scan (Hillis-Steele). 블록 합계는 bsum에.
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

// 블록 합계처럼 길이가 1024를 넘을 수 있는 배열을 블록 하나로 scan: 1024개씩 타일을 돌며 carry 누적
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
    float res = 2.f;
    int n = 0, cap = 0, M = 0, sub = 1;
    std::vector<float> x, y, vx, vy, xs, ys, vxs, vys, rho, p;
    std::vector<int> cell, rank, cnt, start, partial;
    Bufs hb{};
    bool use_gpu = false, gpu_ok = false, dev_dirty = true;
    std::string gpu_name;
#ifdef USE_HIP
    Bufs db{};
    int *bsum = nullptr, *boff = nullptr;
    int nb = 0;
    bool gpu_alloc = false;
#endif
    ~Sim() { gpu_free(); }

    void init(float W, float H, float r) {
        res = r;
        float s = 1.f / res;  // 입자 간격 (픽셀)
        P.W = W;
        P.H = H;
        P.h = 2.f * s;
        P.h2 = P.h * P.h;
        P.cellInv = 1.f / P.h;
        P.gw = (int)std::ceil(W / P.h);
        P.gh = (int)std::ceil(H / P.h);
        const float PI = 3.14159265f;
        P.poly6 = 4.f / (PI * std::pow(P.h, 8.f));
        P.spikyG = -30.f / (PI * std::pow(P.h, 5.f));
        P.viscL = 40.f / (PI * std::pow(P.h, 5.f));

        // 정사각 격자 배치(간격 s)에서 밀도가 정확히 1이 되도록 질량 결정
        double sum = 0;
        int K = (int)std::ceil(P.h / s);
        for (int a = -K; a <= K; ++a)
            for (int b = -K; b <= K; ++b) {
                double r2 = (a * s) * (a * s) + (b * s) * (b * s);
                if (r2 < P.h2) sum += std::pow(P.h2 - r2, 3.0);
            }
        P.rho0 = 1.f;
        P.m = (float)(1.0 / (P.poly6 * sum));

        // 약압축(WCSPH): 음속을 최대 유속의 ~8배로 -> 밀도 오차 ~1.5%
        float g = 0.04f;  // 픽셀/프레임²
        float vmax = std::sqrt(2.f * g * H);
        float c = 8.f * vmax;
        P.c2 = c * c;
        P.maxv = 0.5f * c;
        P.nu = 0.08f * P.h * c / 8.f;  // Monaghan 인공점성 α=0.08 상당
        sub = (int)std::ceil(c / (0.35f * P.h));  // CFL: dt <= 0.35 h / c
        P.dt = 1.f / sub;
        P.eps = 0.5f * s;
        P.gx = 0.f;
        P.gy = g;
        P.mmode = 0;
        P.mR = 6.f;

        M = P.gw * P.gh;
        cap = (int)(W * H * res * res * 0.70f) + 1024;
        for (auto* v : {&x, &y, &vx, &vy, &xs, &ys, &vxs, &vys, &rho, &p}) v->assign(cap, 0.f);
        cell.assign(cap, 0);
        rank.assign(cap, 0);
        cnt.assign(M, 0);
        start.assign(M + 1, 0);
        partial.assign(omp_get_max_threads() + 1, 0);
        hb = Bufs{x.data(), y.data(), vx.data(), vy.data(), xs.data(), ys.data(), vxs.data(),
                  vys.data(), rho.data(), p.data(), cell.data(), rank.data(), cnt.data(), start.data()};
        gpu_setup();
    }

    void scene(int k) {
        float s = 1.f / res;
        n = 0;
        unsigned seed = 12345;
        auto jit = [&]() { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.f - 0.5f) * 0.1f * s; };
        auto fill = [&](float x0, float y0, float x1, float y1) {
            for (float yy = y0 + 0.5f * s; yy < y1; yy += s)
                for (float xx = x0 + 0.5f * s; xx < x1; xx += s) {
                    if (n >= cap) return;
                    x[n] = xx + jit();
                    y[n] = yy + jit();
                    vx[n] = vy[n] = 0.f;
                    ++n;
                }
        };
        float W = P.W, H = P.H, e = P.eps;
        if (k == 2) {
            fill(e, H * 0.78f, W - e, H - e);              // 얕은 수조
            fill(W * 0.36f, H * 0.08f, W * 0.64f, H * 0.42f);  // 떨어지는 물덩어리
        } else if (k == 3) {
            fill(e, H * 0.50f, W - e, H - e);              // 상자 절반 수영장
            fill(e, H * 0.08f, W * 0.25f, H * 0.45f);      // 양쪽에서 쏟아지는 물기둥
            fill(W * 0.75f, H * 0.08f, W - e, H * 0.45f);
        } else {
            fill(e, H * 0.22f, W * 0.40f, H - e);          // 댐 붕괴
        }
        P.gx = 0.f;
        P.gy = 0.04f;
        dev_dirty = true;
    }

    // ── CPU: 서브스텝 하나를 parallel 영역 하나 안에서 ──
    void cpu_step() {
        Params Pl = P;
        Bufs B = hb;
        const int N = n, MM = M;
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int c = 0; c < MM; ++c) B.cnt[c] = 0;

            // ① 셀 카운팅: atomic fetch-add가 셀 안 순번을 줌
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                int c = cell_of(B.x[i], B.y[i], Pl);
                B.cell[i] = c;
                B.rank[i] = __atomic_fetch_add(&B.cnt[c], 1, __ATOMIC_RELAXED);
            }

            // ② 2단계 병렬 exclusive scan: 스레드별 구간 합 -> 구간 합의 scan -> 구간 내 scan
            int T = omp_get_num_threads(), t = omp_get_thread_num();
            int lo = (int)((long long)MM * t / T), hi = (int)((long long)MM * (t + 1) / T);
            int s = 0;
            for (int c = lo; c < hi; ++c) s += B.cnt[c];
            partial[t + 1] = s;
            #pragma omp barrier
            #pragma omp single
            {
                partial[0] = 0;
                for (int k = 0; k < T; ++k) partial[k + 1] += partial[k];
            }
            int run = partial[t];
            for (int c = lo; c < hi; ++c) {
                B.start[c] = run;
                run += B.cnt[c];
            }
            if (t == T - 1) B.start[MM] = run;
            #pragma omp barrier

            // ③ 셀 순서로 재배치
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) scatter_i(i, B);

            // ④ 밀도/압력  ⑤ 힘 + 적분
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) density_i(i, Pl, B);
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) force_i(i, Pl, B);
        }
    }

    void frame() {
        if (use_gpu && gpu_ok) {
            gpu_frame();
        } else {
            for (int s = 0; s < sub; ++s) cpu_step();
        }
    }

    // ── GPU ──
    void gpu_setup() {
#ifdef USE_HIP
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return;
        hipDeviceProp_t prop;
        HIPCHECK(hipGetDeviceProperties(&prop, 0));
        gpu_name = prop.name;
        gpu_free();  // 해상도 변경 시 이전 버퍼 해제
        nb = (M + SCAN_BS - 1) / SCAN_BS;
        auto fal = [&](float** pp) { HIPCHECK(hipMalloc((void**)pp, sizeof(float) * cap)); };
        auto ial = [&](int** pp, size_t k) { HIPCHECK(hipMalloc((void**)pp, sizeof(int) * k)); };
        fal(&db.x); fal(&db.y); fal(&db.vx); fal(&db.vy);
        fal(&db.xs); fal(&db.ys); fal(&db.vxs); fal(&db.vys);
        fal(&db.rho); fal(&db.p);
        ial(&db.cell, cap); ial(&db.rank, cap); ial(&db.cnt, M); ial(&db.start, M + 1);
        ial(&bsum, nb); ial(&boff, nb + 1);
        HIPCHECK(hipMemset(db.start, 0, sizeof(int) * (M + 1)));  // start[0] = 0 고정
        HIPCHECK(hipMemset(boff, 0, sizeof(int) * (nb + 1)));     // boff[0] = 0 고정
        gpu_ok = gpu_alloc = true;
#endif
    }

    void gpu_free() {
#ifdef USE_HIP
        if (!gpu_alloc) return;
        for (float* f : {db.x, db.y, db.vx, db.vy, db.xs, db.ys, db.vxs, db.vys, db.rho, db.p}) (void)hipFree(f);
        for (int* q : {db.cell, db.rank, db.cnt, db.start, bsum, boff}) (void)hipFree(q);
        gpu_alloc = false;
#endif
    }

    void gpu_frame() {
#ifdef USE_HIP
        const int BS = 256, G = (n + BS - 1) / BS;
        if (dev_dirty) {
            size_t bytes = sizeof(float) * n;
            HIPCHECK(hipMemcpy(db.x, x.data(), bytes, hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(db.y, y.data(), bytes, hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(db.vx, vx.data(), bytes, hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(db.vy, vy.data(), bytes, hipMemcpyHostToDevice));
            dev_dirty = false;
        }
        for (int s = 0; s < sub; ++s) {
            HIPCHECK(hipMemsetAsync(db.cnt, 0, sizeof(int) * M, 0));
            hipLaunchKernelGGL(k_cell, dim3(G), dim3(BS), 0, 0, P, db, n);
            // 3단계 scan: 블록별 scan -> 블록 합계 scan(블록 1개, 길이 제한 없음) -> 블록 오프셋 더하기
            hipLaunchKernelGGL(k_scan_block, dim3(nb), dim3(SCAN_BS), 0, 0, (const int*)db.cnt, db.start + 1, bsum, M);
            hipLaunchKernelGGL(k_scan_single, dim3(1), dim3(SCAN_BS), 0, 0, (const int*)bsum, boff + 1, nb);
            hipLaunchKernelGGL(k_scan_add, dim3(nb), dim3(SCAN_BS), 0, 0, db.start + 1, (const int*)boff, M);
            hipLaunchKernelGGL(k_scatter, dim3(G), dim3(BS), 0, 0, db, n);
            hipLaunchKernelGGL(k_density, dim3(G), dim3(BS), 0, 0, P, db, n);
            hipLaunchKernelGGL(k_force, dim3(G), dim3(BS), 0, 0, P, db, n);
        }
        HIPCHECK(hipGetLastError());
        size_t bytes = sizeof(float) * n;  // 렌더링용으로 상태를 호스트에 복사 (동기화 포함)
        HIPCHECK(hipMemcpy(x.data(), db.x, bytes, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(y.data(), db.y, bytes, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(vx.data(), db.vx, bytes, hipMemcpyDeviceToHost));
        HIPCHECK(hipMemcpy(vy.data(), db.vy, bytes, hipMemcpyDeviceToHost));
#endif
    }

    void set_gpu(bool on) {
        if (on && !gpu_ok) return;
        if (on && !use_gpu) dev_dirty = true;  // 호스트 상태가 최신 -> 업로드
        use_gpu = on;
    }
};

// ─────────────────────────── 렌더링 ───────────────────────────
struct Raster {
    int PW, PH;
    std::vector<float> cnt, spd;
};

static void rasterize(const Sim& S, Raster& R) {
    std::fill(R.cnt.begin(), R.cnt.end(), 0.f);
    std::fill(R.spd.begin(), R.spd.end(), 0.f);
    float* cnt = R.cnt.data();
    float* spd = R.spd.data();
    const int PW = R.PW, PH = R.PH;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < S.n; ++i) {
        int px = clampi((int)S.x[i], 0, PW - 1), py = clampi((int)S.y[i], 0, PH - 1);
        int id = py * PW + px;
        float sp = std::sqrt(S.vx[i] * S.vx[i] + S.vy[i] * S.vy[i]);
        #pragma omp atomic
        cnt[id] += 1.f;
        #pragma omp atomic
        spd[id] += sp;
    }
}

static inline unsigned shade(const Raster& R, int px, int py, float norm, float vref) {
    static const int kw[3] = {1, 2, 1};
    float f = 0.f, wsum = 0.f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = px + dx, yy = py + dy;
            if (xx < 0 || yy < 0 || xx >= R.PW || yy >= R.PH) continue;
            float w = (float)(kw[dx + 1] * kw[dy + 1]);
            f += w * R.cnt[yy * R.PW + xx];
            wsum += w;
        }
    f = f / wsum * norm;
    int id = py * R.PW + px;
    float sp = R.cnt[id] > 0 ? R.spd[id] / R.cnt[id] / vref : 0.f;
    float a = std::min(1.f, f * 1.25f);
    float foam = std::clamp((sp - 0.35f) / 0.65f, 0.f, 1.f) * a * 0.85f;
    float depth = std::min(1.f, f * 0.9f);  // 꽉 찬 곳일수록 진한 파랑
    float wr = 30 - 18 * depth, wg = 130 - 70 * depth, wb = 230 - 60 * depth;
    float r = 7 + (wr - 7) * a, g = 9 + (wg - 9) * a, b = 18 + (wb - 18) * a;
    r += (230 - r) * foam;
    g += (242 - g) * foam;
    b += (255 - b) * foam;
    unsigned ri = (unsigned)r & ~3u, gi = (unsigned)g & ~3u, bi = (unsigned)b & ~3u;
    return (ri << 16) | (gi << 8) | bi;
}

static void render(const Raster& R, std::vector<std::string>& lines, int TW, int TR, float res, float vref) {
    float norm = 1.f / (res * res);
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < TR; ++row) {
        std::string& s = lines[row];
        s.clear();
        char tmp[48];
        std::snprintf(tmp, sizeof tmp, "\x1b[%d;1H", row + 1);
        s += tmp;
        unsigned pf = ~0u, pb = ~0u;
        for (int c = 0; c < TW; ++c) {
            unsigned f = shade(R, c, 2 * row, norm, vref);
            unsigned b = shade(R, c, 2 * row + 1, norm, vref);
            if (f != pf) {
                std::snprintf(tmp, sizeof tmp, "\x1b[38;2;%u;%u;%um", f >> 16, (f >> 8) & 255, f & 255);
                s += tmp;
                pf = f;
            }
            if (b != pb) {
                std::snprintf(tmp, sizeof tmp, "\x1b[48;2;%u;%u;%um", b >> 16, (b >> 8) & 255, b & 255);
                s += tmp;
                pb = b;
            }
            s += "▀";
        }
    }
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

// ─────────────────────────── 벤치마크 ───────────────────────────
static int run_bench(float res) {
    Sim S;
    S.init(200.f, 100.f, res);
    const int WARM = 3, FR = 10;
    std::printf("SPH 벤치마크: 200x100 px, res %.1f, 장면 3(수영장), 서브스텝 %d/프레임, 셀 %d\n",
                res, S.sub, S.M);
    S.scene(3);
    std::printf("  입자 수 %d\n", S.n);

    int maxT = omp_get_max_threads();
    std::vector<int> ts{1};
    for (int t = 2; t <= maxT; t *= 2) ts.push_back(t);
    if (ts.back() != maxT) ts.push_back(maxT);
    double base = 0;
    for (int t : ts) {
        omp_set_num_threads(t);
        S.scene(3);
        S.set_gpu(false);
        for (int f = 0; f < WARM; ++f) S.frame();
        double t0 = omp_get_wtime();
        for (int f = 0; f < FR; ++f) S.frame();
        double ms = (omp_get_wtime() - t0) / FR * 1000;
        if (t == 1) base = ms;
        std::printf("  CPU %2d threads: %8.2f ms/frame   speedup x%.2f\n", t, ms, base / ms);
    }
    omp_set_num_threads(maxT);
    if (S.gpu_ok) {
        S.scene(3);
        S.set_gpu(true);
        for (int f = 0; f < WARM; ++f) S.frame();
        double t0 = omp_get_wtime();
        for (int f = 0; f < FR; ++f) S.frame();
        double ms = (omp_get_wtime() - t0) / FR * 1000;
        std::printf("  GPU %-11s: %8.2f ms/frame   speedup x%.2f (vs 1 thread)\n",
                    S.gpu_name.c_str(), ms, base / ms);
    } else {
        std::printf("  GPU: 사용 불가 (USE_HIP 없이 빌드됐거나 장치 없음)\n");
    }
    return 0;
}

// ─────────────────────────── 메인 ───────────────────────────
struct Mouse {
    float x = 0, y = 0, px = 0, py = 0, vx = 0, vy = 0;
    bool left = false, right = false;
};

int main(int argc, char** argv) {
    bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    float res = 0.f;  // 0 = 자동 (GPU 있으면 4, 없으면 2)
    if (bench && argc > 2) res = (float)std::atof(argv[2]);
    if (!bench && argc > 1) res = (float)std::atof(argv[1]);
    if (bench && res <= 0) res = 3.f;
    if (res > 0) res = std::clamp(res, 0.5f, 10.f);
    if (bench) return run_bench(res);

    int TW = 120, rows = 40;
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        TW = ws.ws_col;
        rows = ws.ws_row;
    }
    int TR = std::max(4, rows - 1);
    Sim S;
    S.init((float)TW, (float)(TR * 2), res > 0 ? res : 2.f);
    if (res <= 0 && S.gpu_ok) S.init((float)TW, (float)(TR * 2), 4.f);  // GPU면 입자 4배로 시작
    res = S.res;
    int cur_scene = 1;
    S.scene(cur_scene);
    S.set_gpu(S.gpu_ok);  // GPU 있으면 GPU로 시작

    // [ ] 키: 해상도(입자 수) 실시간 변경
    static const float RES_STEPS[] = {1.f, 1.5f, 2.f, 2.5f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 10.f};
    const int NSTEP = sizeof(RES_STEPS) / sizeof(RES_STEPS[0]);
    auto change_res = [&](int dir) {
        int k = 0;
        while (k < NSTEP - 1 && RES_STEPS[k] < res - 1e-3f) ++k;
        k = std::clamp(k + dir, 0, NSTEP - 1);
        if (RES_STEPS[k] == res) return;
        bool g = S.use_gpu;
        float mR = S.P.mR;
        res = RES_STEPS[k];
        S.init(S.P.W, S.P.H, res);
        S.P.mR = mR;
        S.scene(cur_scene);
        S.use_gpu = false;
        S.set_gpu(g);
    };
    setup_term();

    Raster R{TW, TR * 2, std::vector<float>(TW * TR * 2), std::vector<float>(TW * TR * 2)};
    float vref = std::sqrt(2.f * 0.04f * S.P.H);
    std::vector<std::string> lines(TR);
    std::string out, inbuf;
    Mouse m;
    bool paused = false;
    double sim_ms = 0, ren_ms = 0, fps = 0;
    long frame = 0;
    const char* gdir = "v";

    while (!g_quit) {
        double tf = omp_get_wtime();

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
                    int b, x, y;
                    if (std::sscanf(inbuf.c_str() + pos + 3, "%d;%d;%d", &b, &x, &y) == 3) {
                        bool press = inbuf[end] == 'M';
                        m.x = x - 0.5f;
                        m.y = 2.f * (y - 1) + 1.f;
                        if (b == 64 || b == 65) {
                            if (press) S.P.mR = std::clamp(S.P.mR + (b == 64 ? 1.f : -1.f), 2.f, 40.f);
                        } else {
                            int btn = b & 3;
                            bool motion = b & 32;
                            if (!press) {
                                if (btn == 0) m.left = false;
                                if (btn == 2) m.right = false;
                            } else if (!motion) {
                                if (btn == 0) m.left = true;
                                if (btn == 2) m.right = true;
                                m.px = m.x;
                                m.py = m.y;
                            }
                        }
                    }
                    pos = end + 1;
                    continue;
                }
                if (inbuf[pos + 1] == '[') {  // 방향키 = 중력 방향
                    float g = 0.04f;
                    switch (inbuf[pos + 2]) {
                        case 'A': S.P.gx = 0; S.P.gy = -g; gdir = "^"; break;
                        case 'B': S.P.gx = 0; S.P.gy = g;  gdir = "v"; break;
                        case 'C': S.P.gx = g; S.P.gy = 0;  gdir = ">"; break;
                        case 'D': S.P.gx = -g; S.P.gy = 0; gdir = "<"; break;
                    }
                    pos += 3;
                    continue;
                }
                ++pos;
                continue;
            }
            switch (inbuf[pos]) {
                case 'q': case 'Q': g_quit = 1; break;
                case 'g': case 'G': S.set_gpu(!S.use_gpu); break;
                case '1': case '2': case '3':
                    cur_scene = inbuf[pos] - '0';
                    S.scene(cur_scene);
                    gdir = "v";
                    break;
                case 'r': case 'R': S.scene(cur_scene); gdir = "v"; break;
                case ']': change_res(+1); break;
                case '[': change_res(-1); break;
                case 'p': case 'P': paused = !paused; break;
                case '+': case '=': S.P.mR = std::min(40.f, S.P.mR + 1.f); break;
                case '-': case '_': S.P.mR = std::max(2.f, S.P.mR - 1.f); break;
            }
            ++pos;
        }
        inbuf.erase(0, pos);

        // ── 마우스 -> 시뮬레이션 파라미터 ──
        m.vx = 0.5f * m.vx + 0.5f * (m.x - m.px);
        m.vy = 0.5f * m.vy + 0.5f * (m.y - m.py);
        m.px = m.x;
        m.py = m.y;
        S.P.mmode = m.left ? 1 : (m.right ? 2 : 0);
        S.P.mx = m.x;
        S.P.my = m.y;
        S.P.mvx = m.vx;
        S.P.mvy = m.vy;

        // ── 시뮬레이션 / 렌더 ──
        double t0 = omp_get_wtime();
        if (!paused) {
            S.frame();
            ++frame;
        }
        double t1 = omp_get_wtime();
        rasterize(S, R);
        render(R, lines, TW, TR, res, vref);
        double t2 = omp_get_wtime();
        if (!paused) sim_ms = 0.9 * sim_ms + 0.1 * (t1 - t0) * 1000;
        ren_ms = 0.9 * ren_ms + 0.1 * (t2 - t1) * 1000;

        out.clear();
        for (auto& l : lines) out += l;
        char status[320];
        std::string be = S.use_gpu ? "GPU " + S.gpu_name : "CPU " + std::to_string(omp_get_max_threads()) + "T";
        int len = std::snprintf(status, sizeof status,
            " %s | res %.1f: %d particles x %d substeps | sim %.1fms draw %.1fms | %.0f fps | g %s%s"
            " | [ ]:particles L:push R:pull arrows:gravity g:CPU/GPU 1 2 3 +- p q",
            be.c_str(), res, S.n, S.sub, sim_ms, ren_ms, fps, gdir, paused ? " PAUSED" : "");
        len = std::clamp(len, 0, TW);
        out += "\x1b[" + std::to_string(TR + 1) + ";1H\x1b[0m\x1b[7m";
        out.append(status, (size_t)len);
        out += "\x1b[0m\x1b[K";
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);

        double el = omp_get_wtime() - tf;
        if (el < 1.0 / 60) usleep((useconds_t)((1.0 / 60 - el) * 1e6));
        fps = 0.9 * fps + 0.1 / (omp_get_wtime() - tf);

        if (!g_raw && frame >= 240) break;  // 파이프 출력일 때는 잠깐 돌고 종료
    }
    restore_term();
    return 0;
}
