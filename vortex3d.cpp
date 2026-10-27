// vortex3d.cpp — 3D 소용돌이(와류) 유체역학: 와류 필라멘트법 + 연기 추적 입자, OpenMP/HIP로 계산, OpenGL로 렌더링
//
// 의존성 (Ubuntu):  sudo apt install libglfw3-dev libgl-dev
//
// GPU(ROCm) 빌드 (Strix Halo iGPU = gfx1151):
//   hipcc -O3 -march=native -std=c++17 -fopenmp -DUSE_HIP --offload-arch=gfx1151 -x hip vortex3d.cpp -o vortex3d -lglfw -lGL
//   (-march=native가 없으면 CPU 쪽이 SSE2로만 컴파일돼서 OpenMP 경로가 1.5배쯤 느림.
//    /usr/bin/hipcc가 시스템 clang을 잡아서 omp.h를 못 찾으면 앞에 HIP_CLANG_PATH=/opt/rocm/lib/llvm/bin)
// CPU 전용 빌드:
//   g++ -O3 -march=native -std=c++17 -fopenmp vortex3d.cpp -o vortex3d -lglfw -lGL
//
// 실행:  ./vortex3d [연기입자수] [장면]       기본: GPU면 512K, CPU면 256K / 장면 1
//        ./vortex3d --bench [연기입자수]      창 없이 CPU 스레드별 / GPU 속도 비교
//        ./vortex3d --diag [장면] [프레임]    창 없이 필라멘트 상태(고리 중심/반지름, 재연결 수) 출력
//        OpenMP 스레드 기본값 = 물리 코어 수 (SMT 끔). OMP_NUM_THREADS=32 ./vortex3d 처럼 바꿀 수 있음
//
// 조작:  좌클릭 드래그 = 카메라 회전   휠 = 줌   우클릭 = 커서 방향으로 연기 고리 발사   E = 화면 중앙으로 발사
//        S = 자동 연속 발사   1 = 립프로깅(두 고리가 번갈아 서로를 통과)   2 = 정면 충돌   3 = 비스듬한 충돌(합쳐짐)
//        4 = 세잎 매듭 와류   5 = 고리 하나(과녁)   R = 리셋   K = 재연결 on/off   L = 와류 심(필라멘트) 선   M = 연기 / 발광
//        C = 카메라 따라가기   A = 자동 회전   [ ] = 연기 입자 절반/두 배   - = = 시간 속도   , . = 점 크기
//        B N = 밝기 -/+   G = GPU <-> CPU   P/Space = 일시정지   V = vsync   F = 전체화면   Q/Esc = 종료
//        (상태는 창 제목에 표시)
//
// 물리 (비점성 비압축 유체에서 와도가 가는 관 = 필라멘트에 몰려 있다고 보는 모델):
//   속도장은 와도로부터 비오-사바르 법칙으로 결정됨   u(x) = Γ/4π ∮ dl × (x - y) / |x - y|³
//   필라멘트 = 닫힌 꺾은선. 정칙화(Rosenhead-Moore) 커널 dl × r / (|r|² + δ²)^{3/2}을 직선 조각 위에서 적분한 닫힌 식:
//     u = Γ/4π · (r1 × r2) · (r0·r1/√(|r1|²+δ²) - r0·r2/√(|r2|²+δ²)) / (|r1 × r2|² + δ²|r0|²)
//     (r1, r2 = 양 끝점 -> x, r0 = 조각, δ = 심 반지름. 고리 속도 ≈ Γ/4πR (ln(8R/δ) - 1/2))
//   필라멘트 노드는 자기들이 만든 속도로 같이 움직임 (헬름홀츠 정리: 와선은 유체에 얼어붙어 있음)
//   -> 와선이 늘어나면 그만큼 와도가 세짐(와류 신장)이 저절로 들어감. 순환 Γ는 모든 필라멘트 공통.
//   연기 입자는 같은 속도장에 실려 가기만 하는 수동 추적자 (화면에 보이는 건 이것)
//
// 한 프레임 (시뮬레이션 시간 h):
//   ① 필라멘트 RK4 서브스텝: 노드 N개 x 조각 N개 n-body (OpenMP). t, t+h/2, t+h에서 조각 배열 스냅샷
//   ② 연기 입자 RK4 한 스텝: k1 = 스냅샷(t), k2,k3 = 스냅샷(t+h/2), k4 = 스냅샷(t+h)의 비오-사바르 합
//      입자 M개 x 조각 N개 (CPU: OpenMP + SIMD, GPU: 조각을 LDS에 타일로 올려 블록 전체가 공유)
//      GPU면 입자 위치는 OpenGL VBO 안에 있고 HIP이 그 자리에서 갱신. 그동안 CPU는 ③을 동시에 진행
//   ③ 위상 정리: 재연결(반평행으로 가까워진 두 조각의 연결을 맞바꿈 -> 고리가 갈라지거나 합쳐짐)
//      -> 늘어난 조각 쪼개기 / 짧은 조각 합치기 (Catmull-Rom) -> 노드 크기 꺾임만 지우는 4차 평활화
//      -> 너무 작은 고리 삭제, 노드가 예산을 넘으면 작은 고리부터 삭제 (점성 대신: 작은 고리일수록 빨리 사라짐)
//   ①②에서 다른 고리가 충분히 멀면 조각을 다 더하지 않고 그 고리의 쌍극자 장 하나로 (난류 상태에서 2~4배)
// 렌더링:
//   ① 입자를 점으로 가산 누적 (rgb = 색 x 밀도, a = 밀도, 원근 보정 가중치)  ② 밉맵 + 자동 노출
//   ③ 합성: 연기 = 1 - exp(-밀도)만큼 배경을 가림 + 흐린 밀도의 기울기로 가짜 조명 / 발광 = 톤매핑 + 글로우

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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(__HIP_DEVICE_COMPILE__)
#define RSQRT(x) __builtin_amdgcn_rsqf(x)        // v_rsq_f32 한 명령 (오차 ~1ulp)
#define FDIV(a, b) ((a) * __builtin_amdgcn_rcpf(b))
#else
#define RSQRT(x) (1.f / std::sqrt(x))
#define FDIV(a, b) ((a) / (b))
#endif

// ───────────────────────── 상수 / 데이터 ─────────────────────────
constexpr float PI_F = 3.14159265358979f;
constexpr float GAMMA = 1.f;             // 순환 (모든 필라멘트 공통 -> 재연결이 잘 정의됨)
constexpr float DELTA = 0.15f;           // 와류 심 반지름 = 정칙화 길이 (기본 고리 반지름 1 기준)
constexpr float DELTA2 = DELTA * DELTA;
constexpr float L0 = 0.6f * DELTA;       // 필라멘트 조각 목표 길이
constexpr int MAXN = 8192;               // 필라멘트 노드 상한 (메모리)
constexpr int NBUDGET = 1200;            // 노드가 이보다 많으면 작은 고리부터 지움 (작은 고리일수록 점성으로 빨리 사라지는 셈)
constexpr int MIN_NODES = 6;             // 이보다 노드가 적은 고리는 삭제
constexpr float MIN_LEN = 8.f * DELTA;   // 이보다 짧은(반지름 ~1.3δ) 고리는 삭제
constexpr float RMAX = 30.f;             // 카메라 중심에서 이만큼 멀어진 고리는 삭제
constexpr float DREC = DELTA;            // 재연결 후보: 두 조각 중점 거리
constexpr float RECON_RATIO = 0.9f;      // 재연결 조건: 바꾼 뒤 두 조각 길이 합이 이 비율 미만으로 줄어듦
constexpr int RECON_WIN = 5;             // 한 번 재연결한 곳 앞뒤 노드는 이번 프레임에 또 안 함
constexpr int RECON_PIECE = 16;          // 고리가 갈라질 때 두 조각 다 이만큼 노드가 있어야 함 (뾰족한 끝에서 잘게 썰리는 것 방지)
constexpr float FLOOR_Y = -3.5f;         // 바닥 격자 높이 (그림용)

struct alignas(16) F4 {
    float x, y, z, w;  // 위치 + w = 색상(hue 0~1)
};

// 필라멘트 조각 배열 (SoA, CPU용): 시작점 a, 조각 벡터 r0 = b - a, l2 = |r0|²
// GPU에는 조각 하나를 float4 두 개 (ax ay az l2)(rx ry rz 0)로 묶어서 보냄 (LDS 읽기 2번)
struct Segs {
    const float *ax, *ay, *az, *rx, *ry, *rz, *l2;
};

// 고리 하나의 요약: 중심, 반지름(중심에서 가장 먼 노드), 벡터 넓이 m = ½∮ y × dl, 조각 구간 [s, s+n)
// 고리에서 충분히 멀면(거리 > FAR x 반지름) 조각을 다 더하는 대신 쌍극자 장 (3(m·r̂)r̂ - m)/r³ 하나로 (Barnes-Hut 비슷).
// 닫힌 고리는 Σdl = 0이라 단극자 항이 없고 쌍극자가 첫 항. 잘게 부서진 고리 수십 개가 흩어진 난류 상태에서 몇 배 빨라짐.
struct alignas(16) LInfo {
    float cx, cy, cz, rho;
    float mx, my, mz, pad;
    int s, n, pad2, pad3;
};
constexpr float FAR_TR = 3.f;   // 연기 입자
constexpr float FAR_FIL = 4.f;  // 필라멘트 노드 (고리 운동이 더 민감하니 넉넉히)

HD inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static inline float u01(uint32_t& s) {
    s = hash32(s + 0x9e3779b9U);
    return (s >> 8) * (1.f / 16777216.f);
}

// ───────────────────── 비오-사바르: 직선 조각 하나 (CPU/GPU 공용) ─────────────────────
// Γ/4π는 호출하는 쪽에서 곱함. 점이 조각의 연장선 위에 있으면 r1 × r2 = 0 -> 기여 0 (자기 조각 포함)
// (|r1|, |r2|에도 δ²를 넣어야 가까운 조각의 기여가 제대로 들어감. 빼면 고리 속도가 절반쯤으로 떨어짐)
// r2 = r1 - r0 이라서 r1 × r2 = r0 × r1,  r0·r2 = r0·r1 - |r0|²
// (|r2|²는 꼭 제곱합으로: q1 - 2a + l2로 쓰면 컴파일러가 sqrt 인자가 음수일 수 있다고 보고 errno 처리 때문에 벡터화를 포기함 -> 5배 느림)
HD inline void bs_add(float px, float py, float pz, float ax, float ay, float az, float rx, float ry, float rz,
                      float l2, float& ux, float& uy, float& uz) {
    float x = px - ax, y = py - ay, z = pz - az;  // r1
    float u = x - rx, v = y - ry, w = z - rz;     // r2
    float a = rx * x + ry * y + rz * z;           // r0·r1
    float cx = ry * z - rz * y;                   // r1 × r2
    float cy = rz * x - rx * z;
    float cz = rx * y - ry * x;
    float t = a * RSQRT(x * x + y * y + z * z + DELTA2) - (a - l2) * RSQRT(u * u + v * v + w * w + DELTA2);
    float k = FDIV(t, cx * cx + cy * cy + cz * cz + DELTA2 * l2 + 1e-12f);
    ux += k * cx;
    uy += k * cy;
    uz += k * cz;
}

// 고리 쌍극자 장 (d = 점 - 고리 중심)
HD inline void dip_add(float dx, float dy, float dz, const LInfo& l, float& ux, float& uy, float& uz) {
    float r2 = dx * dx + dy * dy + dz * dz;
    float ir2 = FDIV(1.f, r2), ir3 = ir2 * RSQRT(r2);
    float md = 3.f * (l.mx * dx + l.my * dy + l.mz * dz) * ir2;
    ux += (md * dx - l.mx) * ir3;
    uy += (md * dy - l.my) * ir3;
    uz += (md * dz - l.mz) * ir3;
}

// CPU: 조각 [o, o+ns) 전부의 합. 안쪽 루프는 SIMD로 (조각 SoA라 그대로 벡터화됨)
static inline void vel_cpu(float px, float py, float pz, const Segs& S, int o, int ns, float& ux, float& uy, float& uz) {
    const float *ax = S.ax + o, *ay = S.ay + o, *az = S.az + o;
    const float *rx = S.rx + o, *ry = S.ry + o, *rz = S.rz + o, *l2 = S.l2 + o;
    float x = 0.f, y = 0.f, z = 0.f;
    #pragma omp simd reduction(+ : x, y, z)
    for (int s = 0; s < ns; ++s) bs_add(px, py, pz, ax[s], ay[s], az[s], rx[s], ry[s], rz[s], l2[s], x, y, z);
    ux = x;
    uy = y;
    uz = z;
}

// 고리마다: 가까우면 조각 전부(SIMD), 멀면 쌍극자. own = 이 고리는 항상 정확히 (필라멘트 노드 자신의 고리)
static inline void vel_cull(float px, float py, float pz, const Segs& S, const LInfo* L, int nl, float far, int own,
                            float& ux, float& uy, float& uz) {
    float x = 0.f, y = 0.f, z = 0.f;
    for (int k = 0; k < nl; ++k) {
        const LInfo& l = L[k];
        float dx = px - l.cx, dy = py - l.cy, dz = pz - l.cz, lim = far * l.rho;
        if (k != own && dx * dx + dy * dy + dz * dz > lim * lim) {
            dip_add(dx, dy, dz, l, x, y, z);
        } else {
            float a, b, c;
            vel_cpu(px, py, pz, S, l.s, l.n, a, b, c);
            x += a;
            y += b;
            z += c;
        }
    }
    ux = x;
    uy = y;
    uz = z;
}

// 연기 입자 RK4 한 스텝: 고리 요약 L[loff[0]..] = 스냅샷(t), [1] = (t+h/2), [2] = (t+h)
static void tracers_cpu(F4* P, int n, const Segs& S, const LInfo* L, const int* loff, const int* lcnt, float h, float c,
                        float far) {
    // dynamic: 다른 프로그램이 코어 몇 개를 쓰고 있어도 느린 스레드가 덩어리를 덜 가져가서 전체가 안 기다림
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < n; ++i) {
        F4 p = P[i];
        float ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz;
        const float hh = 0.5f * h * c, hf = h * c;
        const LInfo *L0 = L + loff[0], *L1 = L + loff[1], *L2 = L + loff[2];
        vel_cull(p.x, p.y, p.z, S, L0, lcnt[0], far, -1, ax, ay, az);
        vel_cull(p.x + hh * ax, p.y + hh * ay, p.z + hh * az, S, L1, lcnt[1], far, -1, bx, by, bz);
        vel_cull(p.x + hh * bx, p.y + hh * by, p.z + hh * bz, S, L1, lcnt[1], far, -1, cx, cy, cz);
        vel_cull(p.x + hf * cx, p.y + hf * cy, p.z + hf * cz, S, L2, lcnt[2], far, -1, dx, dy, dz);
        const float k = hf / 6.f;
        p.x += k * (ax + 2.f * bx + 2.f * cx + dx);
        p.y += k * (ay + 2.f * by + 2.f * cy + dy);
        p.z += k * (az + 2.f * bz + 2.f * cz + dz);
        P[i] = p;
    }
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

constexpr int TILE = 256;

// 고리마다: 블록 전체(입자 256개의 바운딩 박스)가 고리에서 멀면 쌍극자, 아니면 그 고리의 조각을 LDS 타일로 올려
// 블록이 같이 훑음 (n-body 타일링). 멀다/가깝다를 블록 단위로 정하니 같은 웨이브 안에서 분기가 안 갈라짐.
__device__ inline void vel_loops(float px, float py, float pz, const float4* G, const LInfo* L, int nl, float bx,
                                 float by, float bz, float bh, float far, float4* sh, float& ux, float& uy, float& uz) {
    float x = 0.f, y = 0.f, z = 0.f;
    for (int k = 0; k < nl; ++k) {
        const LInfo l = L[k];
        float ex = l.cx - bx, ey = l.cy - by, ez = l.cz - bz, lim = far * l.rho + bh;
        if (ex * ex + ey * ey + ez * ez > lim * lim) {
            dip_add(px - l.cx, py - l.cy, pz - l.cz, l, x, y, z);
            continue;
        }
        for (int base = 0; base < l.n; base += TILE) {
            __syncthreads();  // 이전 타일을 다 쓴 다음에 덮어쓰기
            int t = threadIdx.x;
            if (base + t < l.n) {
                int g = 2 * (l.s + base + t);
                sh[2 * t] = G[g];
                sh[2 * t + 1] = G[g + 1];
            }
            __syncthreads();
            int m = min(TILE, l.n - base);
            #pragma unroll 8
            for (int q = 0; q < m; ++q) {
                float4 A = sh[2 * q], R = sh[2 * q + 1];
                bs_add(px, py, pz, A.x, A.y, A.z, R.x, R.y, R.z, A.w, x, y, z);
            }
        }
    }
    ux = x;
    uy = y;
    uz = z;
}

__global__ void __launch_bounds__(TILE) k_tracers(F4* P, int n, const float4* G, const LInfo* L, int l0, int m0,
                                                  int l1, int m1, int l2, int m2, float h, float c, float far) {
    __shared__ float4 sh[2 * TILE];
    const int t = threadIdx.x, i = blockIdx.x * TILE + t;
    F4 p = i < n ? P[i] : F4{0.f, 0.f, 0.f, 0.f};  // 범위 밖 스레드도 타일 적재에는 참여

    // 블록 바운딩 박스 (sh를 잠깐 빌려서 min/max 트리 리덕션)
    const float BIG = 1e30f;
    sh[t] = i < n ? make_float4(p.x, p.y, p.z, 0.f) : make_float4(BIG, BIG, BIG, 0.f);
    sh[TILE + t] = i < n ? make_float4(p.x, p.y, p.z, 0.f) : make_float4(-BIG, -BIG, -BIG, 0.f);
    __syncthreads();
    for (int s = TILE / 2; s > 0; s >>= 1) {
        if (t < s) {
            float4 a = sh[t], b = sh[t + s];
            sh[t] = make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), 0.f);
            a = sh[TILE + t];
            b = sh[TILE + t + s];
            sh[TILE + t] = make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), 0.f);
        }
        __syncthreads();
    }
    const float4 lo = sh[0], hi = sh[TILE];
    const float bx = 0.5f * (lo.x + hi.x), by = 0.5f * (lo.y + hi.y), bz = 0.5f * (lo.z + hi.z);
    const float hx = 0.5f * (hi.x - lo.x), hy = 0.5f * (hi.y - lo.y), hz = 0.5f * (hi.z - lo.z);
    const float bh = sqrtf(hx * hx + hy * hy + hz * hz) + 0.15f;  // + 한 스텝 동안 움직일 여유

    float ax, ay, az, qx, qy, qz, cx, cy, cz, dx, dy, dz;
    const float hh = 0.5f * h * c, hf = h * c;
    vel_loops(p.x, p.y, p.z, G, L + l0, m0, bx, by, bz, bh, far, sh, ax, ay, az);
    vel_loops(p.x + hh * ax, p.y + hh * ay, p.z + hh * az, G, L + l1, m1, bx, by, bz, bh, far, sh, qx, qy, qz);
    vel_loops(p.x + hh * qx, p.y + hh * qy, p.z + hh * qz, G, L + l1, m1, bx, by, bz, bh, far, sh, cx, cy, cz);
    vel_loops(p.x + hf * cx, p.y + hf * cy, p.z + hf * cz, G, L + l2, m2, bx, by, bz, bh, far, sh, dx, dy, dz);
    if (i < n) {
        const float k = hf / 6.f;
        p.x += k * (ax + 2.f * qx + 2.f * cx + dx);
        p.y += k * (ay + 2.f * qy + 2.f * cy + dy);
        p.z += k * (az + 2.f * qz + 2.f * cz + dz);
        P[i] = p;
    }
}
#endif

// ─────────────────────────── 벡터 ───────────────────────────
struct V3 {
    float x, y, z;
};
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float len(V3 a) { return std::sqrt(dot(a, a)); }
static V3 norm(V3 a) { return a * (1.f / std::max(len(a), 1e-20f)); }
static V3 v3(const F4& p) { return {p.x, p.y, p.z}; }
static F4 f4(V3 p, float w) { return {p.x, p.y, p.z, w}; }

// 균일 Catmull-Rom: p1 -> p2 사이 (u = 0..1)
static V3 catmull(V3 p0, V3 p1, V3 p2, V3 p3, float u) {
    float u2 = u * u, u3 = u2 * u;
    return (p1 * 2.f + (p2 - p0) * u + (p0 * 2.f - p1 * 5.f + p2 * 4.f - p3) * u2 + (p1 * 3.f - p0 - p2 * 3.f + p3) * u3) *
           0.5f;
}

// 조각 배열 (호스트). 스냅샷 여러 개를 이어붙여 담기도 함
struct SegBuf {
    std::vector<float> v[7];  // ax ay az rx ry rz l2
    int n = 0;
    void clear() {
        for (auto& a : v) a.clear();
        n = 0;
    }
    void add(const F4& a, const F4& b) {
        float rx = b.x - a.x, ry = b.y - a.y, rz = b.z - a.z;
        v[0].push_back(a.x);
        v[1].push_back(a.y);
        v[2].push_back(a.z);
        v[3].push_back(rx);
        v[4].push_back(ry);
        v[5].push_back(rz);
        v[6].push_back(rx * rx + ry * ry + rz * rz);
        ++n;
    }
    Segs soa() const { return {v[0].data(), v[1].data(), v[2].data(), v[3].data(), v[4].data(), v[5].data(), v[6].data()}; }
};

// ─────────────────────────── 와류 필라멘트 ───────────────────────────
// 노드는 고리마다 연속 구간에 저장 (ls[k]부터 ln[k]개). nx[i] = 다음 노드.
// 재연결은 nx만 맞바꾸고, canon()이 다시 고리를 따라가며 연속 구간으로 정리함.
struct Fil {
    std::vector<F4> x;        // 노드 위치, w = 색상
    std::vector<int> nx;      // 다음 노드 (고리의 마지막 -> 처음)
    std::vector<int> ls, ln;  // 고리 시작 / 노드 수
    std::vector<F4> xs, k1, k2, k3, k4;
    SegBuf work;
    long recon = 0;

    void clear() {
        x.clear();
        nx.clear();
        ls.clear();
        ln.clear();
        recon = 0;
    }
    int loops() const { return (int)ls.size(); }

    void relink() {
        nx.resize(x.size());
        for (size_t k = 0; k < ls.size(); ++k)
            for (int t = 0; t < ln[k]; ++t) nx[ls[k] + t] = t + 1 < ln[k] ? ls[k] + t + 1 : ls[k];
    }

    int add_loop(const std::vector<F4>& p) {
        ls.push_back((int)x.size());
        ln.push_back((int)p.size());
        x.insert(x.end(), p.begin(), p.end());
        relink();
        return (int)ls.size() - 1;
    }

    // 중심 c, 축 n, 반지름 R. 축 둘레로 오른손 방향 순환 -> 고리는 +n 쪽으로 나아감 (중심의 유속 = +n Γ/2R)
    // amp = 반지름/축 방향으로 섞는 작은 요동 (모드 2~9, 대칭을 깨서 불안정성이 자랄 씨앗)
    int add_ring(V3 c, V3 n, float R, float hue, float amp, uint32_t seed) {
        n = norm(n);
        V3 e1 = norm(cross(n, std::fabs(n.y) < 0.9f ? V3{0, 1, 0} : V3{1, 0, 0}));
        V3 e2 = cross(n, e1);  // e1 x e2 = n
        int m = std::max(2 * MIN_NODES, (int)std::ceil(2.f * PI_F * R / L0));
        float phr[10], pha[10];
        uint32_t s = seed;
        for (int k = 0; k < 10; ++k) {
            phr[k] = 2.f * PI_F * u01(s);
            pha[k] = 2.f * PI_F * u01(s);
        }
        std::vector<F4> p(m);
        for (int j = 0; j < m; ++j) {
            float f = 2.f * PI_F * j / m, dr = 0.f, da = 0.f;
            for (int k = 2; k < 10; ++k) {
                dr += amp / k * std::cos(k * f + phr[k]);
                da += amp / k * std::cos(k * f + pha[k]);
            }
            p[j] = f4(c + (e1 * std::cos(f) + e2 * std::sin(f)) * (R + dr) + n * da, hue);
        }
        return add_loop(p);
    }

    // (p, q) 토러스 매듭. 대칭축 = x. 색은 매듭을 따라 무지개 -> 재연결 뒤 어느 부분이 어디로 갔는지 보임
    int add_knot(V3 c, float R0, float r1, int pp, int qq) {
        auto at = [&](float f) {
            float r = R0 + r1 * std::cos(qq * f);
            return c + V3{r1 * std::sin(qq * f), r * std::cos(pp * f), r * std::sin(pp * f)};
        };
        float L = 0.f;
        V3 prev = at(0.f);
        for (int k = 1; k <= 4000; ++k) {
            V3 q = at(2.f * PI_F * k / 4000);
            L += len(q - prev);
            prev = q;
        }
        int m = (int)std::ceil(L / L0);
        std::vector<F4> p(m);
        for (int j = 0; j < m; ++j) p[j] = f4(at(2.f * PI_F * j / m), (float)j / m);
        return add_loop(p);
    }

    // 조각 길이로 가중한 와도 중심
    bool centroid(V3& c) const {
        V3 s{0, 0, 0};
        float w = 0.f;
        for (size_t i = 0; i < x.size(); ++i) {
            V3 a = v3(x[i]), b = v3(x[nx[i]]);
            float l = len(b - a);
            s = s + (a + b) * (0.5f * l);
            w += l;
        }
        if (w <= 0.f) return false;
        c = s * (1.f / w);
        return true;
    }

    // 위치 p로 고리 요약 (조각 번호 = 노드 번호 + base, 고리는 연속 구간이라 조각도 연속)
    void loopinfo(const std::vector<F4>& p, int base, std::vector<LInfo>& out) const {
        for (size_t k = 0; k < ls.size(); ++k) {
            const int s = ls[k], m = ln[k];
            V3 c{0, 0, 0};
            for (int t = 0; t < m; ++t) c = c + v3(p[s + t]);
            c = c * (1.f / m);
            float r2 = 0.f;
            V3 mm{0, 0, 0};
            for (int t = 0; t < m; ++t) {
                V3 a = v3(p[s + t]) - c, b = v3(p[s + (t + 1) % m]) - c;
                r2 = std::max(r2, dot(a, a));
                mm = mm + cross(a, b - a);
            }
            out.push_back(LInfo{c.x, c.y, c.z, std::sqrt(r2), 0.5f * mm.x, 0.5f * mm.y, 0.5f * mm.z, 0.f, base + s, m, 0, 0});
        }
    }

    // ── 비오-사바르 n-body: 위치 p(연결은 nx)에서 노드 속도 u. 다른 고리가 멀면 쌍극자로 ──
    std::vector<LInfo> wl;
    std::vector<int> own;
    void vel(const std::vector<F4>& p, std::vector<F4>& u) {
        const int N = (int)p.size();
        work.clear();
        for (int i = 0; i < N; ++i) work.add(p[i], p[nx[i]]);
        wl.clear();
        loopinfo(p, 0, wl);
        own.resize(N);
        for (size_t k = 0; k < ls.size(); ++k)
            for (int t = 0; t < ln[k]; ++t) own[ls[k] + t] = (int)k;
        const Segs S = work.soa();
        const LInfo* L = wl.data();
        const int nl = (int)wl.size();
        const float c = GAMMA / (4.f * PI_F);
        u.resize(N);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            float ux, uy, uz;
            vel_cull(p[i].x, p[i].y, p[i].z, S, L, nl, FAR_FIL, own[i], ux, uy, uz);
            u[i] = F4{c * ux, c * uy, c * uz, 0.f};
        }
    }

    void rk4(float dt) {
        const int N = (int)x.size();
        if (!N) return;
        auto stage = [&](const std::vector<F4>& k, float s) {
            xs.resize(N);
            for (int i = 0; i < N; ++i) xs[i] = F4{x[i].x + s * k[i].x, x[i].y + s * k[i].y, x[i].z + s * k[i].z, x[i].w};
        };
        vel(x, k1);
        stage(k1, 0.5f * dt);
        vel(xs, k2);
        stage(k2, 0.5f * dt);
        vel(xs, k3);
        stage(k3, dt);
        vel(xs, k4);
        const float s = dt / 6.f;
        for (int i = 0; i < N; ++i) {
            x[i].x += s * (k1[i].x + 2.f * k2[i].x + 2.f * k3[i].x + k4[i].x);
            x[i].y += s * (k1[i].y + 2.f * k2[i].y + 2.f * k3[i].y + k4[i].y);
            x[i].z += s * (k1[i].z + 2.f * k2[i].z + 2.f * k3[i].z + k4[i].z);
        }
    }

    void snapshot(SegBuf& b, std::vector<LInfo>& L) const {
        loopinfo(x, b.n, L);
        for (size_t i = 0; i < x.size(); ++i) b.add(x[i], x[nx[i]]);
    }

    // ── 재연결 ──
    // 조각 i(a -> a')와 j(b -> b')가 가깝고, 연결을 a -> b', b -> a'로 바꾸면 길이 합이 확 줄어들 때만 맞바꿈.
    // 길이가 줄어드는 건 두 조각이 반평행일 때뿐이고(실제 와류 재연결 조건), 되돌리면 다시 길어지므로 왔다갔다 안 함.
    // 같은 고리 안에서 바꾸면 고리가 둘로 갈라지고, 다른 고리끼리 바꾸면 하나로 합쳐짐.
    int reconnect() {
        const int N = (int)x.size();
        if (N < 2 * MIN_NODES) return 0;
        std::vector<int> lid(N), lpos(N);
        for (size_t k = 0; k < ls.size(); ++k)
            for (int t = 0; t < ln[k]; ++t) {
                lid[ls[k] + t] = (int)k;
                lpos[ls[k] + t] = t;
            }
        std::vector<V3> mid(N);
        std::vector<float> sl(N);
        for (int i = 0; i < N; ++i) {
            V3 a = v3(x[i]), b = v3(x[nx[i]]);
            mid[i] = (a + b) * 0.5f;
            sl[i] = len(b - a);
        }
        std::vector<float> br(N);
        std::vector<int> bj(N);
        const float r2 = DREC * DREC;
        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i) {
            const V3 mi = mid[i], xi = v3(x[i]), xin = v3(x[nx[i]]);
            float best = RECON_RATIO;
            int jb = -1;
            for (int j = i + 1; j < N; ++j) {
                V3 d = mid[j] - mi;
                if (dot(d, d) > r2) continue;
                if (lid[j] == lid[i]) {  // 같은 고리: 갈라진 두 고리가 너무 작아지면 제외
                    int dp = std::abs(lpos[j] - lpos[i]), m = ln[lid[i]];
                    if (std::min(dp, m - dp) < RECON_PIECE) continue;
                }
                float r = (len(v3(x[nx[j]]) - xi) + len(xin - v3(x[j]))) / (sl[i] + sl[j] + 1e-12f);
                if (r < best) {
                    best = r;
                    jb = j;
                }
            }
            br[i] = best;
            bj[i] = jb;
        }
        std::vector<std::pair<float, int>> cand;
        for (int i = 0; i < N; ++i)
            if (bj[i] >= 0) cand.push_back({br[i], i});
        if (cand.empty()) return 0;
        std::sort(cand.begin(), cand.end());
        std::vector<char> used(N, 0);
        auto mark = [&](int i) {
            int k = lid[i], m = ln[k];
            for (int d = -RECON_WIN; d <= RECON_WIN; ++d) used[ls[k] + ((lpos[i] + d) % m + m) % m] = 1;
        };
        int cnt = 0;
        for (auto& c : cand) {
            int i = c.second, j = bj[i];
            if (used[i] || used[j]) continue;
            std::swap(nx[i], nx[j]);
            mark(i);
            mark(j);
            ++cnt;
        }
        return cnt;
    }

    // nx를 따라 고리를 다시 모아 연속 구간으로. 너무 작거나 멀리 간 고리는 버리고, 노드 예산을 넘으면 작은 고리부터 버림
    void canon(V3 ref, float rmax) {
        const int N = (int)x.size();
        std::vector<char> vis(N, 0);
        std::vector<int> order;                       // 살아남은 고리들의 노드를 고리 순서대로
        std::vector<std::pair<float, int>> lp;        // (길이, order 안 시작 위치)
        std::vector<int> lcnt;
        order.reserve(N);
        int total = 0;
        for (int i = 0; i < N; ++i) {
            if (vis[i]) continue;
            const int st = (int)order.size();
            for (int j = i; !vis[j]; j = nx[j]) {
                vis[j] = 1;
                order.push_back(j);
            }
            const int m = (int)order.size() - st;
            float L = 0.f;
            V3 c{0, 0, 0};
            for (int t = 0; t < m; ++t) {
                V3 a = v3(x[order[st + t]]);
                L += len(v3(x[order[st + (t + 1) % m]]) - a);
                c = c + a;
            }
            c = c * (1.f / m);
            if (m < MIN_NODES || L < MIN_LEN || dot(c - ref, c - ref) > rmax * rmax) {
                order.resize(st);
                continue;
            }
            lp.push_back({L, st});
            lcnt.push_back(m);
            total += m;
        }
        std::vector<char> keep(lp.size(), 1);
        if (total > NBUDGET) {
            std::vector<int> by(lp.size());
            for (size_t k = 0; k < by.size(); ++k) by[k] = (int)k;
            std::sort(by.begin(), by.end(), [&](int a, int b) { return lp[a].first < lp[b].first; });
            for (size_t q = 0; q + 1 < by.size() && total > NBUDGET; ++q) {
                keep[by[q]] = 0;
                total -= lcnt[by[q]];
            }
        }
        std::vector<F4> y;
        y.reserve(total);
        std::vector<int> nls, nln;
        for (size_t k = 0; k < lp.size(); ++k) {
            if (!keep[k]) continue;
            nls.push_back((int)y.size());
            nln.push_back(lcnt[k]);
            for (int t = 0; t < lcnt[k]; ++t) y.push_back(x[order[lp[k].second + t]]);
        }
        x.swap(y);
        ls.swap(nls);
        ln.swap(nln);
        relink();
    }

    // 조각 길이를 [0.45, 1.5] x 목표 길이로 유지. 늘어난 곳은 Catmull-Rom으로 노드 삽입, 뭉친 곳은 노드 삭제
    void resample() {
        float total = 0.f;
        for (size_t i = 0; i < x.size(); ++i) total += len(v3(x[nx[i]]) - v3(x[i]));
        const float l0 = std::max(L0, total / (0.8f * MAXN));  // 노드 상한에 가까워지면 조각을 굵게
        const float lmin = 0.45f * l0, lmax = 1.5f * l0;
        std::vector<F4> y;
        y.reserve(x.size() + 64);
        std::vector<int> nls, nln, kept;
        for (size_t k = 0; k < ls.size(); ++k) {
            const int s = ls[k], m = ln[k];
            kept.clear();
            kept.push_back(s);
            for (int t = 1; t < m; ++t)
                if (len(v3(x[s + t]) - v3(x[kept.back()])) >= lmin) kept.push_back(s + t);
            while (kept.size() > 3 && len(v3(x[kept.back()]) - v3(x[kept[0]])) < lmin) kept.pop_back();
            const int M = (int)kept.size();
            if (M < 4) continue;
            const int start = (int)y.size();
            for (int t = 0; t < M; ++t) {
                const F4& a = x[kept[t]];
                V3 pa = v3(a), pb = v3(x[kept[(t + 1) % M]]);
                y.push_back(a);
                float L = len(pb - pa);
                if (L > lmax) {
                    V3 p0 = v3(x[kept[(t + M - 1) % M]]), p3 = v3(x[kept[(t + 2) % M]]);
                    int q = (int)std::ceil(L / l0);
                    for (int u = 1; u < q; ++u) y.push_back(f4(catmull(p0, pa, pb, p3, (float)u / q), a.w));
                }
            }
            const int cnt = (int)y.size() - start;
            if (cnt < MIN_NODES) {
                y.resize(start);
                continue;
            }
            nls.push_back(start);
            nln.push_back(cnt);
        }
        x.swap(y);
        ls.swap(nls);
        ln.swap(nln);
        relink();
    }

    // 4차 차분 평활화: 노드 간격 크기의 꺾임만 지우고 고리 모양(긴 파장)은 거의 안 건드림 (일종의 초점성)
    void smooth(float eps) {
        xs.resize(x.size());
        for (size_t k = 0; k < ls.size(); ++k) {
            const int s = ls[k], m = ln[k];
            for (int t = 0; t < m; ++t) {
                auto P = [&](int d) { return v3(x[s + (t + d + m) % m]); };
                V3 d4 = P(-2) - P(-1) * 4.f + P(0) * 6.f - P(1) * 4.f + P(2);
                xs[s + t] = f4(P(0) - d4 * eps, x[s + t].w);
            }
        }
        x.swap(xs);
    }

    double rec_ms = 0;
    int topology(V3 ref, bool rec, float eps) {
        double t0 = omp_get_wtime();
        int c = rec ? reconnect() : 0;
        rec_ms += (omp_get_wtime() - t0) * 1000;
        recon += c;
        canon(ref, RMAX);
        resample();
        smooth(eps);
        return c;
    }
};

// ─────────────────────────── 시뮬레이션 ───────────────────────────
struct Sim {
    Fil F;
    int n = 0;                       // 연기 입자 수
    std::vector<F4> tr;              // 연기 입자 (CPU 모드에선 원본, GPU 모드에선 사본)
    SegBuf snap;                     // 이번 프레임 스냅샷들을 이어붙인 것
    std::vector<LInfo> lin;          // 스냅샷들의 고리 요약 (조각 번호는 snap 전체 기준)
    std::vector<int> loff, lcnt;     // 스냅샷별 lin 시작 / 고리 수
    std::vector<float> pack;
    float h = 0.05f;                 // 프레임당 시뮬레이션 시간
    float far = FAR_TR;              // 연기 입자: 고리 반지름의 이 배수보다 멀면 쌍극자 (아주 크게 하면 근사 끔)
    int T = 1, msub = 2;             // 연기 RK4 스텝 수 / 반 스텝당 필라멘트 서브스텝 수
    int chunk = 0, fired = 0;
    bool recon_on = true;
    V3 ref{0, 0, 0};                 // 이 점에서 RMAX 넘게 멀어진 고리는 삭제
    double simt = 0, fil_ms = 0, rk_acc = 0, topo_acc = 0;
    bool gpu_ok = false;
    std::string gpu_name;
#ifdef USE_HIP
    F4* dPosOwn = nullptr;  // interop이 아닐 때(또는 벤치마크) 쓰는 입자 버퍼
    float* dSeg = nullptr;
    LInfo* dLin = nullptr;
    size_t segCap = 0, linCap = 0;
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

    void alloc(int N) {
        n = N;
        tr.assign(N, F4{0, 0, 0, 0});
        chunk = 0;
#ifdef USE_HIP
        if (gpu_ok) {
            if (dPosOwn) HIPCHECK(hipFree(dPosOwn));
            HIPCHECK(hipMalloc((void**)&dPosOwn, sizeof(F4) * std::max(1, N)));
        }
#endif
    }

    void scene(int sc) {
        F.clear();
        fired = 0;
        if (sc == 2) {         // 정면 충돌: 마주 보고 달려오는 두 고리 (순환 방향 반대)
            F.add_ring({-1.5f, 0, 0}, {1, 0, 0}, 1.f, 0.58f, 0.012f, 21);
            F.add_ring({1.5f, 0, 0}, {-1, 0, 0}, 1.f, 0.02f, 0.012f, 22);
        } else if (sc == 3) {  // 비스듬한 충돌: 나란히 가면서 안쪽으로 기울어진 두 고리 -> 안쪽 테두리가 반평행
            const float th = 0.32f;
            F.add_ring({0, 0, 1.25f}, {std::cos(th), 0, -std::sin(th)}, 1.f, 0.36f, 0.004f, 31);
            F.add_ring({0, 0, -1.25f}, {std::cos(th), 0, std::sin(th)}, 1.f, 0.86f, 0.004f, 32);
        } else if (sc == 4) {  // 세잎 매듭 (2,3 토러스 매듭)
            F.add_knot({0, 0, 0}, 1.25f, 0.5f, 2, 3);
        } else if (sc == 5) {  // 고리 하나 (우클릭으로 고리를 쏴서 맞혀 보기)
            F.add_ring({0, 0, 0}, {1, 0, 0}, 1.2f, 0.58f, 0.002f, 51);
        } else {               // 립프로깅: 같은 축 위에 같은 방향 두 고리
            F.add_ring({-0.3f, 0, 0}, {1, 0, 0}, 1.f, 0.58f, 0.002f, 11);
            F.add_ring({0.3f, 0, 0}, {1, 0, 0}, 1.f, 0.07f, 0.002f, 12);
        }
        V3 c;
        if (F.centroid(c)) ref = c;
        simt = 0;
        chunk = 0;
        seed(tr.data(), n, 0, F.loops(), 1000u + sc);
    }

    // 고리 [k0, k1)을 따라 연기 입자 cnt개를 뿌림: 길이에 비례해 조각을 고르고, 심 둘레로 가우시안.
    // 필라멘트를 따라 순서대로 뿌려서(층화) 번호가 이웃한 입자는 공간에서도 이웃 -> GPU 블록 바운딩 박스가 작아짐
    void seed(F4* dst, int cnt, int k0, int k1, uint32_t salt) {
        std::vector<int> si;
        std::vector<float> cum;
        float tot = 0.f;
        for (int k = k0; k < k1; ++k)
            for (int t = 0; t < F.ln[k]; ++t) {
                int i = F.ls[k] + t;
                tot += len(v3(F.x[F.nx[i]]) - v3(F.x[i]));
                cum.push_back(tot);
                si.push_back(i);
            }
        if (si.empty()) {  // 필라멘트가 없으면 화면 밖으로
            for (int q = 0; q < cnt; ++q) dst[q] = F4{0.f, 1e7f, 0.f, 0.f};
            return;
        }
        const int ns = (int)si.size();
        #pragma omp parallel for schedule(static)
        for (int q = 0; q < cnt; ++q) {
            uint32_t s = hash32(salt * 0x9E3779B1u ^ ((uint32_t)q * 0x85EBCA77u));
            float u = ((float)q + u01(s)) / (float)cnt * tot;
            int k = std::min((int)(std::lower_bound(cum.begin(), cum.end(), u) - cum.begin()), ns - 1);
            int i = si[k];
            V3 a = v3(F.x[i]), d = v3(F.x[F.nx[i]]) - a;
            V3 t = norm(d);
            V3 n1 = norm(cross(t, std::fabs(t.y) < 0.9f ? V3{0, 1, 0} : V3{1, 0, 0}));
            V3 n2 = cross(t, n1);
            float along = u01(s);
            float sig = u01(s) < 0.75f ? 0.45f * DELTA : 1.25f * DELTA;  // 심 + 바깥 대기층
            float rho = sig * std::sqrt(-2.f * std::log(std::max(u01(s), 1e-7f)));
            float ph = 2.f * PI_F * u01(s);
            V3 p = a + d * along + (n1 * std::cos(ph) + n2 * std::sin(ph)) * rho;
            dst[q] = f4(p, F.x[i].w);
        }
    }

    // c에서 dir 방향으로 고리 발사. 연기는 입자 풀의 1/8 덩어리를 돌아가며 재활용 (오래된 연기부터 사라짐)
    bool fire(V3 c, V3 dir, int& off, int& cnt) {
        uint32_t s = 777u + 31u * (uint32_t)fired;
        float R = 0.55f + 0.25f * u01(s);
        if ((int)F.x.size() + (int)(2.f * PI_F * R / L0) + 8 > MAXN / 2) return false;
        float hue = std::fmod(0.12f + 0.618034f * fired, 1.f);
        int k = F.add_ring(c, dir, R, hue, 0.004f, s);
        ++fired;
        const int K = 8;
        off = (int)((long long)n * chunk / K);
        cnt = (int)((long long)n * (chunk + 1) / K) - off;
        chunk = (chunk + 1) % K;
        seed(tr.data() + off, cnt, k, k + 1, s);
        return true;
    }

    // ── 한 프레임 ──
    void frame(bool gpu, F4* dpos) {
        (void)dpos;  // CPU 전용 빌드에서는 안 씀
        const float omega = GAMMA / (2.f * PI_F * DELTA2);  // 심 안쪽 강체 회전 각속도 (연기 입자 RK4가 버틸 만큼 쪼갬)
        T = std::max(1, (int)std::ceil(h * omega / 0.5f));
        msub = std::max(1, (int)std::ceil(h / (2 * T) / 0.0125f));
        const float c = GAMMA / (4.f * PI_F), hs = h / T, dtf = hs / (2 * msub);

        double t0 = omp_get_wtime();
        snap.clear();
        lin.clear();
        loff.clear();
        lcnt.clear();
        auto take = [&]() {
            loff.push_back((int)lin.size());
            F.snapshot(snap, lin);
            lcnt.push_back((int)lin.size() - loff.back());
        };
        take();
        for (int k = 0; k < 2 * T; ++k) {
            for (int m = 0; m < msub; ++m) F.rk4(dtf);
            take();
        }
        double t1 = omp_get_wtime();
        if (gpu) {
#ifdef USE_HIP
            gpu_launch(dpos, hs, c);
#endif
        } else {
            const Segs S = snap.soa();
            for (int s = 0; s < T; ++s) tracers_cpu(tr.data(), n, S, lin.data(), &loff[2 * s], &lcnt[2 * s], hs, c, far);
        }
        double t2 = omp_get_wtime();
        F.topology(ref, recon_on, std::min(0.05f, 0.04f * h / 0.05f));  // GPU가 연기를 계산하는 동안
        double t3 = omp_get_wtime();
#ifdef USE_HIP
        if (gpu) HIPCHECK(hipDeviceSynchronize());
#endif
        fil_ms = 0.9 * fil_ms + 0.1 * ((t1 - t0) + (t3 - t2)) * 1000;
        rk_acc += (t1 - t0) * 1000;
        topo_acc += (t3 - t2) * 1000;
        simt += h;
    }

#ifdef USE_HIP
    void gpu_launch(F4* dpos, float hs, float c) {
        const int tot = snap.n;
        if (tot == 0 || n == 0) return;
        const size_t need = (size_t)8 * tot;
        if (need > segCap) {
            if (dSeg) HIPCHECK(hipFree(dSeg));
            segCap = need * 2;
            HIPCHECK(hipMalloc((void**)&dSeg, sizeof(float) * segCap));
        }
        pack.resize(need);
        for (int q = 0; q < tot; ++q) {  // (ax ay az l2)(rx ry rz 0)
            float* d = pack.data() + (size_t)8 * q;
            d[0] = snap.v[0][q]; d[1] = snap.v[1][q]; d[2] = snap.v[2][q]; d[3] = snap.v[6][q];
            d[4] = snap.v[3][q]; d[5] = snap.v[4][q]; d[6] = snap.v[5][q]; d[7] = 0.f;
        }
        HIPCHECK(hipMemcpy(dSeg, pack.data(), sizeof(float) * need, hipMemcpyHostToDevice));
        if (lin.size() > linCap) {
            if (dLin) HIPCHECK(hipFree(dLin));
            linCap = lin.size() * 2;
            HIPCHECK(hipMalloc((void**)&dLin, sizeof(LInfo) * linCap));
        }
        HIPCHECK(hipMemcpy(dLin, lin.data(), sizeof(LInfo) * lin.size(), hipMemcpyHostToDevice));
        const float4* G4 = (const float4*)dSeg;
        const int G = (n + TILE - 1) / TILE;
        for (int s = 0; s < T; ++s)
            hipLaunchKernelGGL(k_tracers, dim3(G), dim3(TILE), 0, 0, dpos, n, G4, (const LInfo*)dLin, loff[2 * s],
                               lcnt[2 * s], loff[2 * s + 1], lcnt[2 * s + 1], loff[2 * s + 2], lcnt[2 * s + 2], hs, c, far);
        HIPCHECK(hipGetLastError());
    }
#endif

    ~Sim() {
#ifdef USE_HIP
        if (dPosOwn) (void)hipFree(dPosOwn);
        if (dSeg) (void)hipFree(dSeg);
        if (dLin) (void)hipFree(dLin);
#endif
    }
};

// ─────────────────────────── 셰이더 ───────────────────────────
static const char* GLSL_HDR = "#version 330 core\n";

static const char* GLSL_HSV = R"(
vec3 hsv(float h, float s) {
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), k, s);
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

// 연기 입자: 점 하나 = 작은 부피. 화면 한 픽셀이 덮는 부피는 거리² 에 비례하니 가중치 (거리 기준/z)²
static const char* VS_PTS = R"(
layout(location = 0) in vec4 aPos;
uniform mat4 uView, uProj;
uniform float uPointSize, uDist, uSat;
out vec4 vCol;
void main() {
    vec4 v = uView * vec4(aPos.xyz, 1.0);
    gl_Position = uProj * v;
    gl_PointSize = uPointSize;
    float r = uDist / max(-v.z, 1e-3);
    float w = min(r * r, 16.0);
    vCol = vec4(hsv(aPos.w, uSat) * w, w);
}
)";
static const char* FS_PTS = R"(
in vec4 vCol;
out vec4 o;
void main() { o = vCol; }
)";

static const char* FS_STATS = R"(
in vec2 uv;
uniform sampler2D uAcc;
out vec4 o;
void main() {
    float l = textureLod(uAcc, uv, 0.0).a;
    o = vec4(l, l * l, l > 1e-6 ? 1.0 : 0.0, 0.0);   // 밉맵 꼭대기 = 화면 전체 평균
}
)";

// 합성: 배경(하늘 + 바닥 격자, 픽셀마다 광선) 위에 연기 / 발광
static const char* FS_COMP = R"(
in vec2 uv;
uniform sampler2D uAcc;
uniform vec3 uCam, uRight, uUp, uBack;
uniform vec2 uTan;
uniform float uExpo, uFloorY;
uniform int uMode;
out vec4 o;
vec3 background(vec3 d) {
    float t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 c = mix(vec3(0.010, 0.012, 0.018), vec3(0.060, 0.075, 0.100), pow(t, 1.6));
    if (d.y < -1e-4 && uCam.y > uFloorY) {
        float s = (uFloorY - uCam.y) / d.y;
        vec2 q = (uCam + s * d).xz;
        vec2 w = max(fwidth(q), vec2(1e-4));
        vec2 g1 = abs(fract(q - 0.5) - 0.5) / w;
        vec2 g5 = abs(fract(q / 5.0 - 0.5) - 0.5) / (w / 5.0);
        float l1 = 1.0 - min(min(g1.x, g1.y), 1.0);
        float l5 = 1.0 - min(min(g5.x, g5.y), 1.0);
        float fade = exp(-s / 45.0) * smoothstep(0.0, 0.05, -d.y);
        vec3 fl = vec3(0.018, 0.021, 0.028) + vec3(0.045, 0.055, 0.070) * l1 + vec3(0.08, 0.10, 0.13) * l5;
        c = mix(c, fl, fade);
    }
    return c;
}
vec3 tm(vec3 a) {
    float l = max(a.r, max(a.g, a.b));
    float s = l > 1e-6 ? (1.0 - exp(-l)) / l : 0.0;    // 밝기 기준 -> 색상 유지
    vec3 v = a * s;
    float hot = clamp((l - 2.5) / 6.0, 0.0, 1.0) * 0.55;
    return v + (1.0 - v) * hot;
}
void main() {
    vec3 d = normalize(uRight * (uv.x * 2.0 - 1.0) * uTan.x + uUp * (uv.y * 2.0 - 1.0) * uTan.y - uBack);
    vec3 bg = background(d);
    vec4 a = mix(textureLod(uAcc, uv, 0.0), textureLod(uAcc, uv, 1.0), 0.4);
    if (uMode == 0) {
        float D = a.a * uExpo;                                  // 광학 두께
        vec3 tint = a.a > 1e-8 ? a.rgb / a.a : vec3(1.0);
        float alpha = 1.0 - exp(-D);
        // 흐린 밀도의 화면 기울기 = 연기 덩어리의 표면 법선 근사 -> 왼쪽 위에서 오는 빛
        vec2 e = 6.0 / vec2(textureSize(uAcc, 0));
        float L = 2.5;
        float gx = textureLod(uAcc, uv + vec2(e.x, 0.0), L).a - textureLod(uAcc, uv - vec2(e.x, 0.0), L).a;
        float gy = textureLod(uAcc, uv + vec2(0.0, e.y), L).a - textureLod(uAcc, uv - vec2(0.0, e.y), L).a;
        vec3 n = normalize(vec3(-gx * uExpo * 0.8, -gy * uExpo * 0.8, 1.0));
        float diff = max(dot(n, normalize(vec3(-0.45, 0.65, 0.6))), 0.0);
        float dense = 1.0 - exp(-D * 0.35);                     // 두꺼운 곳은 스스로 그늘
        vec3 col = tint * (0.30 + 0.85 * diff) * (1.0 - 0.45 * dense);
        o = vec4(mix(bg, col, alpha), 1.0);
    } else {
        vec3 glow = textureLod(uAcc, uv, 2.0).rgb * 0.55 + textureLod(uAcc, uv, 4.0).rgb * 0.45;
        vec3 c = bg + tm(a.rgb * uExpo) + 0.30 * tm(glow * uExpo * 0.6);
        o = vec4(clamp(c, 0.0, 1.0), 1.0);
    }
}
)";

static const char* VS_LINE = R"(
layout(location = 0) in vec4 aPos;
uniform mat4 uMVP;
out float vHue;
void main() { gl_Position = uMVP * vec4(aPos.xyz, 1.0); vHue = aPos.w; }
)";
static const char* FS_LINE = R"(
in float vHue;
uniform float uAlpha;
out vec4 o;
void main() { o = vec4(hsv(vHue, 0.45) * 1.1, uAlpha); }
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

static GLuint program(const char* vs, const char* fs) {
    std::string v = std::string(GLSL_HDR) + GLSL_HSV + vs;
    std::string f = std::string(GLSL_HDR) + GLSL_HSV + fs;
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

// ─────────────────────────── 앱 ───────────────────────────
struct App {
    GLFWwindow* win = nullptr;
    Sim S;
    int fbw = 0, fbh = 0;
    GLuint vao = 0, vbo = 0, emptyVao = 0, lineVao = 0, lineVbo = 0;
    GLuint accTex = 0, accFbo = 0, statTex = 0, statFbo = 0, outTex = 0, outFbo = 0;
    GLuint pPts = 0, pStats = 0, pComp = 0, pLine = 0;

    int scene = 1, rmode = 0;  // rmode 0: 연기, 1: 발광
    float yaw = 0.35f, pitch = 0.22f, zoom = 1.f, psize = 1.f, gain = 1.f, expo = -1.f;
    bool paused = false, autorot = false, follow = true, lines = false, autofire = false;
    bool vsync = true, fullscreen = false, offscreen = false, ldrag = false;
    double lastx = 0, lasty = 0, sim_ms = 0, last_fire = 0;
    int wx = 100, wy = 100, ww = 1600, wh = 900;
    std::vector<F4> lineBuf;

    // 카메라 (render에서 갱신, 고리 발사 광선에 사용)
    V3 camPos{0, 0, 7}, camRight{1, 0, 0}, camUp{0, 1, 0}, camBack{0, 0, 1}, target{0, 0, 0};
    float tanX = 0.6f, tanY = 0.4f, dist = 7.5f;

    bool use_gpu = false, interop = false;
#ifdef USE_HIP
    hipGraphicsResource_t res = nullptr;
#endif

    // ── 연기 입자 버퍼 (필라멘트는 그대로 두고 연기만 다시 뿌림) ──
    void make_tracers(int N) {
        gpu_release();
        S.alloc(N);
        if (!vbo) glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(F4) * std::max(1, S.n), nullptr, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(F4), nullptr);
        gpu_register();
        S.seed(S.tr.data(), S.n, 0, S.F.loops(), 4242u + (uint32_t)S.simt);
        push(0, S.n);
        psize = S.n >= 700000 ? 1.f : (S.n >= 200000 ? 1.5f : 2.f);
    }

    // 호스트에서 새로 만든 입자 [off, off+cnt)를 지금 원본이 있는 곳으로
    void push(int off, int cnt) {
        if (cnt <= 0) return;
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, sizeof(F4) * off, sizeof(F4) * cnt, S.tr.data() + off);
#ifdef USE_HIP
        if (use_gpu && !interop)
            HIPCHECK(hipMemcpy(S.dPosOwn + off, S.tr.data() + off, sizeof(F4) * cnt, hipMemcpyHostToDevice));
#endif
    }

    void set_scene(int sc, bool cam) {
        scene = sc;
        S.scene(sc);
        push(0, S.n);
        S.F.centroid(target);
        expo = -1.f;
        if (!cam) return;
        if (sc == 1) { yaw = 0.75f; pitch = 0.28f; zoom = 1.25f; }
        if (sc == 2) { yaw = 0.95f; pitch = 0.35f; zoom = 0.6f; }
        if (sc == 3) { yaw = 1.15f; pitch = 0.40f; zoom = 1.0f; }
        if (sc == 4) { yaw = 1.05f; pitch = 0.30f; zoom = 1.1f; }
        if (sc == 5) { yaw = 0.60f; pitch = 0.25f; zoom = 1.0f; }
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
        S.ref = target;
        double t0 = glfwGetTime();
        if (use_gpu) {
#ifdef USE_HIP
            if (interop) {
                F4* ptr = nullptr;
                size_t sz = 0;
                HIPCHECK(hipGraphicsMapResources(1, &res, 0));
                HIPCHECK(hipGraphicsResourceGetMappedPointer((void**)&ptr, &sz, res));
                S.frame(true, ptr);
                HIPCHECK(hipGraphicsUnmapResources(1, &res, 0));
            } else {
                S.frame(true, S.dPosOwn);
                HIPCHECK(hipMemcpy(S.tr.data(), S.dPosOwn, sizeof(F4) * S.n, hipMemcpyDeviceToHost));
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4) * S.n, S.tr.data());
            }
#endif
        } else {
            S.frame(false, nullptr);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4) * S.n, S.tr.data());
        }
        sim_ms = 0.9 * sim_ms + 0.1 * (glfwGetTime() - t0) * 1000;
        V3 c;
        if (follow && S.F.centroid(c)) target = target + (c - target) * 0.06f;
    }

    void set_gpu(bool on) {
        if (on && !S.gpu_ok) return;
        if (on == use_gpu) return;
#ifdef USE_HIP
        if (!on && interop) {  // GPU -> CPU: VBO에 있는 최신 상태를 호스트로
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(F4) * S.n, S.tr.data());
        }
        if (on && !interop) HIPCHECK(hipMemcpy(S.dPosOwn, S.tr.data(), sizeof(F4) * S.n, hipMemcpyHostToDevice));
#endif
        use_gpu = on;
    }

    // 화면 좌표(-1..1)의 광선 방향으로, 카메라와 장면 중심의 중간쯤에서 고리 발사
    void fire_ndc(float x, float y) {
        V3 dir = norm(camRight * (x * tanX) + camUp * (y * tanY) - camBack);
        int off = 0, cnt = 0;
        if (S.fire(camPos + dir * (dist * 0.5f), dir, off, cnt)) push(off, cnt);
    }

    // ── 렌더 타깃 ──
    void make_targets() {
        auto mk = [&](GLuint& tex, GLuint& fbo, GLint ifmt, GLenum type, bool mip) {
            if (tex) glDeleteTextures(1, &tex);
            if (fbo) glDeleteFramebuffers(1, &fbo);
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, ifmt, fbw, fbh, 0, GL_RGBA, type, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mip ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            if (mip) glGenerateMipmap(GL_TEXTURE_2D);
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                std::fprintf(stderr, "프레임버퍼 생성 실패\n");
                std::exit(1);
            }
        };
        mk(accTex, accFbo, GL_RGBA32F, GL_FLOAT, true);
        mk(statTex, statFbo, GL_RGBA32F, GL_FLOAT, true);
        if (offscreen) mk(outTex, outFbo, GL_RGBA8, GL_UNSIGNED_BYTE, false);
    }

    void init_gl() {
        pPts = program(VS_PTS, FS_PTS);
        pStats = program(VS_FULL, FS_STATS);
        pComp = program(VS_FULL, FS_COMP);
        pLine = program(VS_LINE, FS_LINE);
        glGenVertexArrays(1, &vao);
        glGenVertexArrays(1, &emptyVao);
        glGenVertexArrays(1, &lineVao);
        glGenBuffers(1, &lineVbo);
        glBindVertexArray(lineVao);
        glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(F4), nullptr);
        if (offscreen) {
            fbw = ww;
            fbh = wh;
        } else {
            glfwGetFramebufferSize(win, &fbw, &fbh);
        }
        make_targets();
        glEnable(GL_PROGRAM_POINT_SIZE);
    }

    // ── 렌더링 ──
    void render() {
        int w = ww, h = wh;
        if (!offscreen) glfwGetFramebufferSize(win, &w, &h);
        if (w <= 0 || h <= 0) return;
        if (w != fbw || h != fbh) {
            fbw = w;
            fbh = h;
            make_targets();
        }
        const float fovy = 0.8f, aspect = (float)fbw / fbh;
        tanY = std::tan(fovy * 0.5f);
        tanX = tanY * aspect;
        dist = 7.5f / zoom;
        M4 R = mul(rotX(pitch), rotY(-yaw));
        camRight = {R.m[0], R.m[4], R.m[8]};
        camUp = {R.m[1], R.m[5], R.m[9]};
        camBack = {R.m[2], R.m[6], R.m[10]};
        camPos = target + camBack * dist;
        M4 Vw = mul(trans(0, 0, -dist), mul(R, trans(-target.x, -target.y, -target.z)));
        M4 Pj = persp(fovy, aspect, 0.05f, 400.f);
        M4 MVP = mul(Pj, Vw);
        auto U = [](GLuint p, const char* n) { return glGetUniformLocation(p, n); };

        // ① 연기 입자 가산 누적
        glBindFramebuffer(GL_FRAMEBUFFER, accFbo);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE);
        glUseProgram(pPts);
        glUniformMatrix4fv(U(pPts, "uView"), 1, GL_FALSE, Vw.m);
        glUniformMatrix4fv(U(pPts, "uProj"), 1, GL_FALSE, Pj.m);
        glUniform1f(U(pPts, "uPointSize"), psize);
        glUniform1f(U(pPts, "uDist"), dist);
        glUniform1f(U(pPts, "uSat"), rmode ? 0.8f : 0.55f);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, S.n);
        glDisable(GL_BLEND);

        // ② 글로우 / 조명 / 잡음 제거용 밉맵
        glBindTexture(GL_TEXTURE_2D, accTex);
        glGenerateMipmap(GL_TEXTURE_2D);

        // ③ 자동 노출: (밀도, 밀도², 켜짐) -> 밉맵 꼭대기 1픽셀 = 전체 평균
        glBindFramebuffer(GL_FRAMEBUFFER, statFbo);
        glBindVertexArray(emptyVao);
        glUseProgram(pStats);
        glUniform1i(U(pStats, "uAcc"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, accTex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindTexture(GL_TEXTURE_2D, statTex);
        glGenerateMipmap(GL_TEXTURE_2D);
        int top = (int)std::floor(std::log2((double)std::max(fbw, fbh)));
        float st[4] = {0, 0, 0, 0};
        glGetTexImage(GL_TEXTURE_2D, top, GL_RGBA, GL_FLOAT, st);
        if (st[0] > 1e-12f && st[2] > 1e-12f) {
            // 밀도 가중 평균(Σl²/Σl)과 켜진 픽셀 평균(Σl/켜진 수)의 기하평균 기준
            float ref = std::sqrt((st[1] / st[0]) * (st[0] / st[2]));
            float tgt = (rmode ? 1.2f : 1.6f) / ref;
            expo = expo <= 0 ? tgt : 0.85f * expo + 0.15f * tgt;
        }

        // ④ 합성 -> 화면
        glBindFramebuffer(GL_FRAMEBUFFER, outFbo);
        glViewport(0, 0, fbw, fbh);
        glUseProgram(pComp);
        glUniform1i(U(pComp, "uAcc"), 0);
        glBindTexture(GL_TEXTURE_2D, accTex);
        glUniform3f(U(pComp, "uCam"), camPos.x, camPos.y, camPos.z);
        glUniform3f(U(pComp, "uRight"), camRight.x, camRight.y, camRight.z);
        glUniform3f(U(pComp, "uUp"), camUp.x, camUp.y, camUp.z);
        glUniform3f(U(pComp, "uBack"), camBack.x, camBack.y, camBack.z);
        glUniform2f(U(pComp, "uTan"), tanX, tanY);
        glUniform1f(U(pComp, "uExpo"), std::max(expo, 1e-6f) * gain);
        glUniform1f(U(pComp, "uFloorY"), FLOOR_Y);
        glUniform1i(U(pComp, "uMode"), rmode);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // ⑤ 와류 심 (필라멘트)
        if (lines && !S.F.x.empty()) {
            const Fil& F = S.F;
            lineBuf.resize(F.x.size() * 2);
            for (size_t i = 0; i < F.x.size(); ++i) {
                lineBuf[2 * i] = F.x[i];
                lineBuf[2 * i + 1] = F.x[F.nx[i]];
            }
            glBindVertexArray(lineVao);
            glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(F4) * lineBuf.size(), lineBuf.data(), GL_STREAM_DRAW);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glUseProgram(pLine);
            glUniformMatrix4fv(U(pLine, "uMVP"), 1, GL_FALSE, MVP.m);
            glUniform1f(U(pLine, "uAlpha"), 0.9f);
            glDrawArrays(GL_LINES, 0, (GLsizei)lineBuf.size());
            glDisable(GL_BLEND);
        }
    }

    void update_title(double fps) {
        char nstr[32], t[512];
        if (S.n >= 1000000) std::snprintf(nstr, sizeof nstr, "%.2fM", S.n / 1e6);
        else std::snprintf(nstr, sizeof nstr, "%dK", S.n / 1000);
        std::string be = use_gpu ? std::string(interop ? "HIP-GL interop " : "HIP+copy ") + S.gpu_name
                                 : "CPU OpenMP " + std::to_string(omp_get_max_threads()) + "T";
        std::snprintf(t, sizeof t,
                      "Vortex 3D | %s | smoke %s | %d rings, %d nodes | reconnect %s %ld | sim %.1f ms (filament %.1f) | "
                      "%.0f fps | t=%.1f x%.2f | %s%s%s%s",
                      be.c_str(), nstr, S.F.loops(), (int)S.F.x.size(), S.recon_on ? "on" : "OFF", S.F.recon, sim_ms,
                      S.fil_ms, fps, S.simt, S.h / 0.05f, rmode ? "glow" : "smoke", follow ? " | follow" : "",
                      autofire ? " | autofire" : "", paused ? " | PAUSED" : "");
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
    switch (key) {
        case GLFW_KEY_ESCAPE: case GLFW_KEY_Q: glfwSetWindowShouldClose(w, 1); break;
        case GLFW_KEY_P: case GLFW_KEY_SPACE: A.paused = !A.paused; break;
        case GLFW_KEY_A: A.autorot = !A.autorot; break;
        case GLFW_KEY_M: A.rmode ^= 1; A.expo = -1.f; break;
        case GLFW_KEY_L: A.lines = !A.lines; break;
        case GLFW_KEY_K: A.S.recon_on = !A.S.recon_on; break;
        case GLFW_KEY_C: A.follow = !A.follow; break;
        case GLFW_KEY_S: A.autofire = !A.autofire; break;
        case GLFW_KEY_E: A.fire_ndc(0.f, 0.f); break;
        case GLFW_KEY_G: A.set_gpu(!A.use_gpu); break;
        case GLFW_KEY_V: A.vsync = !A.vsync; glfwSwapInterval(A.vsync ? 1 : 0); break;
        case GLFW_KEY_F: A.toggle_fullscreen(); break;
        case GLFW_KEY_1: case GLFW_KEY_2: case GLFW_KEY_3: case GLFW_KEY_4: case GLFW_KEY_5:
            A.set_scene(key - GLFW_KEY_0, true);
            break;
        case GLFW_KEY_R: A.set_scene(A.scene, false); break;
        case GLFW_KEY_MINUS: A.S.h = std::max(0.0125f, A.S.h * 0.8f); break;
        case GLFW_KEY_EQUAL: A.S.h = std::min(0.25f, A.S.h * 1.25f); break;
        case GLFW_KEY_COMMA: A.psize = std::max(1.f, A.psize - 0.5f); break;
        case GLFW_KEY_PERIOD: A.psize = std::min(4.f, A.psize + 0.5f); break;
        case GLFW_KEY_B: A.gain *= 0.8f; break;
        case GLFW_KEY_N: A.gain *= 1.25f; break;
        case GLFW_KEY_LEFT_BRACKET:
        case GLFW_KEY_RIGHT_BRACKET: {
            int nn = key == GLFW_KEY_RIGHT_BRACKET ? A.S.n * 2 : A.S.n / 2;
            nn = std::clamp(nn, 16384, A.S.gpu_ok ? 16777216 : 4194304);
            bool g = A.use_gpu;
            A.set_gpu(false);
            A.make_tracers(nn);
            A.set_gpu(g);
            break;
        }
    }
}

static void on_button(GLFWwindow* w, int button, int action, int) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    double x, y;
    glfwGetCursorPos(w, &x, &y);
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        A.ldrag = action == GLFW_PRESS;
        if (A.ldrag) A.autorot = false;
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        int ww, wh;
        glfwGetWindowSize(w, &ww, &wh);
        A.fire_ndc((float)(x / std::max(1, ww)) * 2.f - 1.f, 1.f - (float)(y / std::max(1, wh)) * 2.f);
    }
    A.lastx = x;
    A.lasty = y;
}

static void on_cursor(GLFWwindow* w, double x, double y) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    if (A.ldrag) {
        A.yaw += (float)(x - A.lastx) * 0.006f;
        A.pitch = std::clamp(A.pitch + (float)(y - A.lasty) * 0.006f, -1.45f, 1.45f);
    }
    A.lastx = x;
    A.lasty = y;
}

static void on_scroll(GLFWwindow* w, double, double dy) {
    App& A = *(App*)glfwGetWindowUserPointer(w);
    A.zoom = std::clamp(A.zoom * (float)std::pow(1.1, dy), 0.2f, 8.f);
}

// ─────────────────────────── 벤치마크 / 진단 (창 없이) ───────────────────────────
static int run_bench(int N) {
    Sim S;
    S.probe();
    S.alloc(N);
    S.scene(1);
    S.frame(false, nullptr);  // T, msub 계산 + 예열
    std::printf("Vortex 3D 벤치마크: 연기 입자 %d개, 프레임당 필라멘트 RK4 %d번 + 연기 RK4 %d번 (입자마다 조각 합 4번)\n", S.n,
                2 * S.T * S.msub, S.T);
    std::printf("\n[1] 장면 1 처음 상태: 고리 %d개, 노드 %d개 -> 입자-조각 쌍 %.1f억 개/프레임\n", S.F.loops(), (int)S.F.x.size(),
                4.0 * S.T * S.n * S.F.x.size() / 1e8);
    const int FR = 3;
    int maxT = omp_get_max_threads(), procs = omp_get_num_procs();  // 기본 = 물리 코어 수, 표에는 SMT까지
    std::vector<int> ts{1};
    for (int t = 2; t <= procs; t *= 2) ts.push_back(t);
    if (ts.back() != procs) ts.push_back(procs);
    if (std::getenv("VORTEX_GPU_ONLY")) ts.assign(1, maxT);
    double base = 0;
    for (int t : ts) {
        omp_set_num_threads(t);
        S.scene(1);
        S.frame(false, nullptr);
        double t0 = omp_get_wtime();
        for (int f = 0; f < FR; ++f) S.frame(false, nullptr);
        double ms = (omp_get_wtime() - t0) / FR * 1000;
        if (t == 1) base = ms;
        std::printf("  CPU %2d threads: %9.2f ms/frame  (%6.1f fps)   speedup x%5.2f   (필라멘트 %.2f ms)\n", t, ms,
                    1000 / ms, base / ms, S.fil_ms);
    }
    omp_set_num_threads(maxT);
#ifdef USE_HIP
    auto gpu_time = [&](int GF) {
        HIPCHECK(hipMemcpy(S.dPosOwn, S.tr.data(), sizeof(F4) * S.n, hipMemcpyHostToDevice));
        for (int f = 0; f < 2; ++f) S.frame(true, S.dPosOwn);
        double t0 = omp_get_wtime();
        for (int f = 0; f < GF; ++f) S.frame(true, S.dPosOwn);
        return (omp_get_wtime() - t0) / GF * 1000;
    };
    if (S.gpu_ok) {
        S.scene(1);
        double ms = gpu_time(20);
        std::printf("  GPU %s: %9.2f ms/frame  (%6.1f fps)   speedup x%5.2f (vs 1 thread)\n", S.gpu_name.c_str(), ms,
                    1000 / ms, base / ms);
    } else
#endif
        std::printf("  GPU: 사용 불가 (USE_HIP 없이 빌드됐거나 장치 없음)\n");

    // [2] 난류 상태: 매듭(장면 4)을 연기 없이 400프레임 돌려 고리가 잘게 부서진 상태에서 시작
    S.alloc(0);
    S.scene(4);
    for (int f = 0; f < 400; ++f) {
        V3 c;
        if (S.F.centroid(c)) S.ref = c;
        S.frame(false, nullptr);
    }
    const Fil saved = S.F;
    S.alloc(N);
    auto restart = [&]() {
        S.F = saved;
        S.seed(S.tr.data(), S.n, 0, S.F.loops(), 77u);
    };
    std::printf("\n[2] 장면 4를 400프레임 돌린 난류 상태: 고리 %d개, 노드 %d개 -> 근사 없이 입자-조각 쌍 %.1f억 개/프레임\n",
                S.F.loops(), (int)S.F.x.size(), 4.0 * S.T * S.n * S.F.x.size() / 1e8);
    // 한 프레임 결과: 쌍극자 근사 vs 정확한 합, CPU vs GPU
    restart();
    S.far = 1e9f;
    S.frame(false, nullptr);
    const std::vector<F4> exact = S.tr;
    restart();
    S.far = FAR_TR;
    S.frame(false, nullptr);
    const std::vector<F4> cull = S.tr;
    auto maxdiff = [&](const std::vector<F4>& a, const std::vector<F4>& b, const std::vector<F4>& from) {
        double md = 0, mstep = 0;
        for (int i = 0; i < S.n; ++i) {
            md = std::max(md, (double)len(v3(a[i]) - v3(b[i])));
            mstep = std::max(mstep, (double)len(v3(b[i]) - v3(from[i])));
        }
        std::printf("최대 %.2e (한 프레임 최대 이동 %.2e의 %.2f%%)\n", md, mstep, 100.0 * md / std::max(mstep, 1e-30));
    };
    restart();
    const std::vector<F4> start = S.tr;
    std::printf("  쌍극자 근사 오차 (CPU, 한 프레임): ");
    maxdiff(cull, exact, start);
    for (int pass = 0; pass < 2; ++pass) {
        S.far = pass ? FAR_TR : 1e9f;
        restart();
        S.frame(false, nullptr);
        double t0 = omp_get_wtime();
        for (int f = 0; f < FR; ++f) S.frame(false, nullptr);
        double ms = (omp_get_wtime() - t0) / FR * 1000;
        std::printf("  CPU %2d threads, %s: %9.2f ms/frame  (%6.1f fps)\n", maxT, pass ? "고리 쌍극자 근사" : "정확한 합      ", ms,
                    1000 / ms);
    }
#ifdef USE_HIP
    if (S.gpu_ok) {
        restart();
        S.far = FAR_TR;
        HIPCHECK(hipMemcpy(S.dPosOwn, S.tr.data(), sizeof(F4) * S.n, hipMemcpyHostToDevice));
        S.frame(true, S.dPosOwn);
        std::vector<F4> g(S.n);
        HIPCHECK(hipMemcpy(g.data(), S.dPosOwn, sizeof(F4) * S.n, hipMemcpyDeviceToHost));
        std::printf("  CPU/GPU 결과 차이 (둘 다 근사, GPU는 블록 단위로 판정): ");
        maxdiff(g, cull, start);
        for (int pass = 0; pass < 2; ++pass) {
            S.far = pass ? FAR_TR : 1e9f;
            restart();
            double ms = gpu_time(10);
            std::printf("  GPU %s: %9.2f ms/frame  (%6.1f fps)\n", pass ? "고리 쌍극자 근사" : "정확한 합      ", ms, 1000 / ms);
        }
    }
#endif
    return 0;
}

static int run_diag(int sc, int frames, bool rec) {
    Sim S;
    S.alloc(0);
    S.recon_on = rec;
    S.scene(sc);
    std::printf("장면 %d: 고리 %d개, 노드 %d개, 재연결 %s\n", sc, S.F.loops(), (int)S.F.x.size(), rec ? "on" : "off");
    double tl = omp_get_wtime();
    for (int f = 0; f <= frames; ++f) {
        if (f % 25 == 0) {
            double now = omp_get_wtime();
            std::printf("f=%4d t=%6.2f 고리=%3d 노드=%4d 재연결=%5ld %6.2f ms/f |", f, S.simt, S.F.loops(), (int)S.F.x.size(),
                        S.F.recon, f ? (now - tl) / 25 * 1000 : 0.0);
            std::printf(" [rk %.2f topo %.2f rec %.2f]", S.rk_acc / 25, S.topo_acc / 25, S.F.rec_ms / 25);
            S.rk_acc = S.topo_acc = S.F.rec_ms = 0;
            tl = now;
            for (int k = 0; k < std::min(S.F.loops(), 5); ++k) {
                V3 c{0, 0, 0};
                int s = S.F.ls[k], m = S.F.ln[k];
                for (int t = 0; t < m; ++t) c = c + v3(S.F.x[s + t]);
                c = c * (1.f / m);
                float r = 0;
                for (int t = 0; t < m; ++t) r += len(v3(S.F.x[s + t]) - c);
                std::printf(" (%5.2f %5.2f %5.2f) r=%.2f n=%d", c.x, c.y, c.z, r / m, m);
            }
            std::printf("\n");
        }
        V3 c;
        if (S.F.centroid(c)) S.ref = c;
        S.frame(false, nullptr);
    }
    return 0;
}

// 물리 코어 수 (SMT 형제 중 번호가 가장 작은 논리 CPU만 셈). 모르면 논리 CPU 수
static int physical_cores() {
    int n = 0, total = omp_get_num_procs();
    for (int c = 0; c < total; ++c) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE* f = std::fopen(path, "r");
        if (!f) return total;
        int first = -1;
        if (std::fscanf(f, "%d", &first) != 1) first = c;
        std::fclose(f);
        if (first == c) ++n;
    }
    return std::max(1, n);
}

// ─────────────────────────── 메인 ───────────────────────────
int main(int argc, char** argv) {
    // OpenMP 기본값 (환경 변수로 이미 정했으면 그대로 둠. 런타임은 첫 omp 호출 때 이 값을 읽음):
    //  - 스레드 수 = 물리 코어 수. 이 계산은 코어의 SIMD 유닛을 꽉 채워서 SMT 짝끼리는 서로 방해만 함
    //  - 쉬는 스레드는 바쁜 대기 대신 잠듦 -> 그동안 GL/HIP 드라이버 스레드가 CPU를 씀 (창 모드에서 2~3배 차이)
    //  - 코어 고정(OMP_PROC_BIND)은 안 함: 다른 일이 도는 코어에 묶이면 그 스레드가 전체를 붙잡음
    setenv("OMP_WAIT_POLICY", "passive", 0);
    if (!std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(physical_cores());
    int N = 0, scene = 1, shot_frames = 0;
    const char* shot = nullptr;
    bool bench = false, diag = false, shot_glow = false, shot_lines = false, shot_fire = false, norec = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench")) bench = true;
        else if (!std::strcmp(argv[i], "--diag")) diag = true;
        else if (!std::strcmp(argv[i], "--shot") && i + 2 < argc) {  // 렌더링 확인용: --shot 파일.ppm 프레임수 (창 안 띄움)
            shot = argv[++i];
            shot_frames = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--glow")) shot_glow = true;
        else if (!std::strcmp(argv[i], "--lines")) shot_lines = true;
        else if (!std::strcmp(argv[i], "--autofire")) shot_fire = true;
        else if (!std::strcmp(argv[i], "--norecon")) norec = true;
        else pos.push_back(argv[i]);
    }
    if (diag) {
        int sc = pos.size() > 0 ? std::clamp(std::atoi(pos[0].c_str()), 1, 5) : 1;
        int fr = pos.size() > 1 ? std::atoi(pos[1].c_str()) : 600;
        return run_diag(sc, fr, !norec);
    }
    if (pos.size() > 0) N = std::atoi(pos[0].c_str());
    if (pos.size() > 1) scene = std::clamp(std::atoi(pos[1].c_str()), 1, 5);
    if (bench) return run_bench(N > 0 ? N : 262144);

    if (!glfwInit()) {
        std::fprintf(stderr, "GLFW 초기화 실패 (디스플레이가 없나요?)\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    if (shot) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    App A;
    A.offscreen = shot != nullptr;
    A.win = glfwCreateWindow(A.ww, A.wh, "Vortex 3D", nullptr, nullptr);
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
    if (N <= 0) N = A.S.gpu_ok ? 524288 : 262144;
    A.rmode = shot_glow ? 1 : 0;
    A.lines = shot_lines;
    A.autofire = shot_fire;
    A.S.recon_on = !norec;
    A.init_gl();
    A.make_tracers(N);
    A.set_scene(scene, true);
    A.set_gpu(A.S.gpu_ok);
    if (A.S.gpu_ok) std::printf("HIP: %s, interop %s\n", A.S.gpu_name.c_str(), A.interop ? "OK" : "실패 -> 복사 방식");
    std::printf("연기 입자 %d개, 필라멘트 고리 %d개 / 노드 %d개\n", A.S.n, A.S.F.loops(), (int)A.S.F.x.size());

    double last_title = 0, last_t = glfwGetTime(), fps = 60;
    long frame = 0;
    uint32_t rs = 99u;
    while (!glfwWindowShouldClose(A.win)) {
        glfwPollEvents();
        if (A.autorot) A.yaw += 0.003f;
        double now = glfwGetTime();
        bool fire_now = shot ? (frame % 40 == 20) : (now - A.last_fire > 1.3);
        if (A.autofire && !A.paused && fire_now) {  // 화면 중앙 근처로 조금씩 흩뿌리며 연속 발사
            A.fire_ndc(0.25f * (u01(rs) - 0.5f), 0.25f * (u01(rs) - 0.5f));
            A.last_fire = now;
        }
        A.simulate();
        A.render();
        if (!shot) glfwSwapBuffers(A.win);

        now = glfwGetTime();
        fps = 0.9 * fps + 0.1 / std::max(1e-6, now - last_t);
        last_t = now;
        if (now - last_title > 0.4) {
            A.update_title(fps);
            last_title = now;
        }
        ++frame;
        if (shot && frame >= shot_frames) {
            std::vector<unsigned char> px((size_t)A.fbw * A.fbh * 3);
            glBindFramebuffer(GL_FRAMEBUFFER, A.outFbo);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, A.fbw, A.fbh, GL_RGB, GL_UNSIGNED_BYTE, px.data());
            FILE* f = std::fopen(shot, "wb");
            std::fprintf(f, "P6 %d %d 255\n", A.fbw, A.fbh);
            for (int y = A.fbh - 1; y >= 0; --y) std::fwrite(&px[(size_t)y * A.fbw * 3], 1, (size_t)A.fbw * 3, f);
            std::fclose(f);
            std::printf("saved %s (frame %ld, t=%.2f, 고리 %d, 노드 %d, 재연결 %ld, sim %.1f ms, 필라멘트 %.2f ms)\n", shot, frame,
                        A.S.simt, A.S.F.loops(), (int)A.S.F.x.size(), A.S.F.recon, A.sim_ms, A.S.fil_ms);
            break;
        }
    }
    A.gpu_release();
    glfwTerminate();
    return 0;
}
