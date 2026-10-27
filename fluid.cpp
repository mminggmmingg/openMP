// fluid.cpp — OpenMP로 돌리는 터미널 인터랙티브 유체 시뮬레이션 (Stable Fluids)
//
// 빌드: g++ -O3 -march=native -std=c++17 -fopenmp fluid.cpp -o fluid
// 실행: ./fluid           (해상도 배율 2: 터미널 한 칸에 2x2 시뮬레이션 셀)
//       ./fluid 3         (배율 3: 더 고해상도, 더 무거움 -> 멀티스레드 효과 체감)
//       ./fluid --bench   (화면 없이 스레드 수별 속도 측정)
//       OMP_NUM_THREADS=1 ./fluid 3
//
// 조작: 좌클릭 드래그 = 염료 + 힘   우클릭 드래그 = 힘만
//       휠 / + - = 붓 크기   space = 제트 on/off   v = 와류 강화 on/off
//       p = 일시정지   c = 초기화   q = 종료
//
// 마우스 입력은 xterm SGR 마우스 모드(1006)를 씁니다. GNOME Terminal, Konsole,
// kitty, Alacritty, WezTerm, tmux(set -g mouse on) 등에서 동작합니다.

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

// ───────────────────────── 격자 ─────────────────────────
static int W, H;  // 내부 셀 수 (경계 포함 시 (W+2) x (H+2))
static inline int IX(int i, int j) { return i + (W + 2) * j; }
using Field = std::vector<float>;
static Field u, v, u0, v0, p, dv, cu, dye[3], dye0[3];

static const int PRESSURE_ITERS = 30;

static void alloc_fields() {
    size_t n = (size_t)(W + 2) * (H + 2);
    for (Field* f : {&u, &v, &u0, &v0, &p, &dv, &cu}) f->assign(n, 0.f);
    for (int c = 0; c < 3; ++c) { dye[c].assign(n, 0.f); dye0[c].assign(n, 0.f); }
}

static void clear_fields() {
    for (Field* f : {&u, &v, &u0, &v0, &p, &dv, &cu}) std::fill(f->begin(), f->end(), 0.f);
    for (int c = 0; c < 3; ++c) std::fill(dye[c].begin(), dye[c].end(), 0.f);
}

// b=1: 좌우 벽에서 u 반사, b=2: 상하 벽에서 v 반사, b=0: 스칼라 복사
static void set_bnd(int b, Field& x) {
    for (int i = 1; i <= W; ++i) {
        x[IX(i, 0)]     = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, H + 1)] = b == 2 ? -x[IX(i, H)] : x[IX(i, H)];
    }
    for (int j = 1; j <= H; ++j) {
        x[IX(0, j)]     = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
        x[IX(W + 1, j)] = b == 1 ? -x[IX(W, j)] : x[IX(W, j)];
    }
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, H + 1)]     = 0.5f * (x[IX(1, H + 1)] + x[IX(0, H)]);
    x[IX(W + 1, 0)]     = 0.5f * (x[IX(W, 0)] + x[IX(W + 1, 1)]);
    x[IX(W + 1, H + 1)] = 0.5f * (x[IX(W, H + 1)] + x[IX(W + 1, H)]);
}

// ─────────────────── 시뮬레이션 단계 (전부 OpenMP) ───────────────────

// 와류 강화(vorticity confinement): 수치 소산으로 사라지는 소용돌이를 되살림
static void vorticity(float eps) {
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= H; ++j)
        for (int i = 1; i <= W; ++i)
            cu[IX(i, j)] = 0.5f * ((v[IX(i + 1, j)] - v[IX(i - 1, j)]) -
                                   (u[IX(i, j + 1)] - u[IX(i, j - 1)]));

    #pragma omp parallel for schedule(static)
    for (int j = 2; j < H; ++j)
        for (int i = 2; i < W; ++i) {
            float nx = 0.5f * (std::fabs(cu[IX(i + 1, j)]) - std::fabs(cu[IX(i - 1, j)]));
            float ny = 0.5f * (std::fabs(cu[IX(i, j + 1)]) - std::fabs(cu[IX(i, j - 1)]));
            float len = std::sqrt(nx * nx + ny * ny) + 1e-5f;
            float w = cu[IX(i, j)];
            u[IX(i, j)] += eps * (ny / len) * w;
            v[IX(i, j)] -= eps * (nx / len) * w;
        }
}

// 반-라그랑지안 이류: 각 셀에서 속도를 거꾸로 따라가 이전 값을 보간
static inline float sample(const Field& d, float x, float y) {
    x = std::clamp(x, 0.5f, W + 0.5f);
    y = std::clamp(y, 0.5f, H + 0.5f);
    int i0 = (int)x, j0 = (int)y;
    float s1 = x - i0, s0 = 1.f - s1, t1 = y - j0, t0 = 1.f - t1;
    return s0 * (t0 * d[IX(i0, j0)] + t1 * d[IX(i0, j0 + 1)]) +
           s1 * (t0 * d[IX(i0 + 1, j0)] + t1 * d[IX(i0 + 1, j0 + 1)]);
}

static void advect_velocity(float damp) {
    std::swap(u, u0);
    std::swap(v, v0);
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= H; ++j)
        for (int i = 1; i <= W; ++i) {
            float x = i - u0[IX(i, j)], y = j - v0[IX(i, j)];
            u[IX(i, j)] = sample(u0, x, y) * damp;
            v[IX(i, j)] = sample(v0, x, y) * damp;
        }
    set_bnd(1, u);
    set_bnd(2, v);
}

static void advect_dye(float decay) {
    for (int c = 0; c < 3; ++c) std::swap(dye[c], dye0[c]);
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= H; ++j)
        for (int i = 1; i <= W; ++i) {
            float x = i - u[IX(i, j)], y = j - v[IX(i, j)];
            for (int c = 0; c < 3; ++c) dye[c][IX(i, j)] = sample(dye0[c], x, y) * decay;
        }
    for (int c = 0; c < 3; ++c) set_bnd(0, dye[c]);
}

// 비압축성 투영: 압력 포아송 방정식을 red-black Gauss-Seidel로 풀고 속도에서 기울기를 뺌.
// 일반 Gauss-Seidel은 순서 의존성 때문에 병렬화가 안 되지만, 체스판처럼 빨강/검정으로
// 나누면 같은 색끼리는 서로 의존하지 않아서 omp for로 나눌 수 있음.
static void project() {
    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= H; ++j)
        for (int i = 1; i <= W; ++i)
            dv[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                    v[IX(i, j + 1)] - v[IX(i, j - 1)]);
    set_bnd(0, dv);

    // p는 이전 프레임 값을 그대로 시작값으로 사용 (warm start -> 적은 반복으로 수렴)
    #pragma omp parallel
    {
        for (int k = 0; k < PRESSURE_ITERS; ++k) {
            for (int color = 0; color < 2; ++color) {
                #pragma omp for schedule(static)
                for (int j = 1; j <= H; ++j)
                    for (int i = 1 + ((j + color) & 1); i <= W; i += 2)
                        p[IX(i, j)] = 0.25f * (dv[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                                               p[IX(i, j - 1)] + p[IX(i, j + 1)]);
            }
            #pragma omp single
            set_bnd(0, p);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int j = 1; j <= H; ++j)
        for (int i = 1; i <= W; ++i) {
            u[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    set_bnd(1, u);
    set_bnd(2, v);
}

static bool g_vort = true;

static void step() {
    if (g_vort) vorticity(0.12f);
    project();
    advect_velocity(0.9995f);
    project();
    advect_dye(0.995f);
}

// ─────────────────────── 외력 주입 ───────────────────────
// lerp=false: 속도에 더함 (마우스),  lerp=true: 목표 속도로 끌어당김 (제트, 폭주 방지)
static void splat(float x, float y, float fx, float fy, float rad,
                  const float col[3], float amt, bool lerp = false) {
    int i0 = std::max(1, (int)(x - rad * 2.5f)), i1 = std::min(W, (int)(x + rad * 2.5f) + 1);
    int j0 = std::max(1, (int)(y - rad * 2.5f)), j1 = std::min(H, (int)(y + rad * 2.5f) + 1);
    float inv = 1.f / (rad * rad);
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i) {
            float dx = i - x, dy = j - y;
            float w = std::exp(-(dx * dx + dy * dy) * inv);
            if (lerp) {
                u[IX(i, j)] += (fx - u[IX(i, j)]) * w * 0.5f;
                v[IX(i, j)] += (fy - v[IX(i, j)]) * w * 0.5f;
            } else {
                u[IX(i, j)] += fx * w;
                v[IX(i, j)] += fy * w;
            }
            if (amt > 0)
                for (int c = 0; c < 3; ++c) dye[c][IX(i, j)] += col[c] * w * amt;
        }
}

static void hsv2rgb(float h, float out[3]) {
    h = std::fmod(h, 1.f) * 6.f;
    int k = (int)h;
    float f = h - k, q = 1.f - f;
    float r, g, b;
    switch (k) {
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = q; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = q; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        default: r = 1; g = 0; b = q; break;
    }
    out[0] = r; out[1] = g; out[2] = b;
}

static bool g_jets = true;

static void apply_jets(long frame, int sc) {
    if (!g_jets) return;
    float t = (float)frame;
    float rad = std::max(1.5f, H * 0.045f);
    float speed = 1.2f * sc;
    float ca[3], cb[3];
    hsv2rgb(0.58f + 0.05f * std::sin(t * 0.01f), ca);  // 청록 계열
    hsv2rgb(0.02f + 0.05f * std::sin(t * 0.013f), cb); // 주황 계열
    float a1 = 0.55f * std::sin(t * 0.021f);
    float a2 = 0.55f * std::sin(t * 0.017f + 1.3f);
    splat(W * 0.06f, H * (0.5f + 0.18f * std::sin(t * 0.006f)),
          std::cos(a1) * speed, std::sin(a1) * speed, rad, ca, 0.25f, true);
    splat(W * 0.94f, H * (0.5f + 0.18f * std::cos(t * 0.005f)),
          -std::cos(a2) * speed, std::sin(a2) * speed, rad, cb, 0.25f, true);
}

// ─────────────────────── 터미널 ───────────────────────
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
    raw.c_cc[VMIN] = 0;   // read()가 기다리지 않고 바로 반환
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    g_raw = true;
    std::atexit(restore_term);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    // 대체 화면, 커서 숨김, 마우스 버튼+드래그 추적, SGR 좌표 형식
    std::printf("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[?1000h\x1b[?1002h\x1b[?1006h");
    std::fflush(stdout);
}

struct Mouse {
    float x = 0, y = 0, px = 0, py = 0;
    bool left = false, right = false;
};

// ─────────────────────── 렌더링 ───────────────────────
// 터미널 한 줄 = 픽셀 두 줄 (▀: 글자색=위 픽셀, 배경색=아래 픽셀)
// 줄마다 문자열을 따로 만들어서 렌더링도 omp로 병렬화
static inline unsigned shade(int c, int pr, int sc) {
    float acc[3] = {0, 0, 0};
    for (int dy = 0; dy < sc; ++dy)
        for (int dx = 0; dx < sc; ++dx) {
            int id = IX(c * sc + dx + 1, pr * sc + dy + 1);
            for (int k = 0; k < 3; ++k) acc[k] += dye[k][id];
        }
    float inv = 1.f / (sc * sc);
    static const int base[3] = {5, 7, 14};
    unsigned rgb[3];
    for (int k = 0; k < 3; ++k) {
        float t = 1.f - std::exp(-acc[k] * inv * 1.4f);  // 톤매핑: 진할수록 포화
        int val = base[k] + (int)((255 - base[k]) * t);
        rgb[k] = (unsigned)std::min(255, val) & ~3u;     // 살짝 양자화 -> 같은 색 연속 시 escape 생략
    }
    return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
}

static void render(std::vector<std::string>& lines, int TW, int TR, int sc) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < TR; ++row) {
        std::string& s = lines[row];
        s.clear();
        char tmp[48];
        std::snprintf(tmp, sizeof tmp, "\x1b[%d;1H", row + 1);
        s += tmp;
        unsigned pf = ~0u, pb = ~0u;
        for (int c = 0; c < TW; ++c) {
            unsigned f = shade(c, 2 * row, sc);
            unsigned b = shade(c, 2 * row + 1, sc);
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

// ─────────────────────── 벤치마크 ───────────────────────
static int run_bench() {
    W = 640; H = 320;
    alloc_fields();
    const int STEPS = 150;
    int maxT = omp_get_max_threads();
    std::vector<int> ts{1};
    for (int t = 2; t <= maxT; t *= 2) ts.push_back(t);
    if (ts.back() != maxT) ts.push_back(maxT);

    std::printf("벤치마크: %dx%d 격자, %d스텝, 압력 반복 %dx2\n", W, H, STEPS, PRESSURE_ITERS);
    double t1 = 0;
    for (int t : ts) {
        omp_set_num_threads(t);
        clear_fields();
        for (int f = 0; f < 10; ++f) { apply_jets(f, 2); step(); }  // 워밍업
        double s = omp_get_wtime();
        for (int f = 10; f < 10 + STEPS; ++f) { apply_jets(f, 2); step(); }
        double ms = (omp_get_wtime() - s) / STEPS * 1000;
        if (t == 1) t1 = ms;
        std::printf("  %2d threads: %7.2f ms/step  (%5.1f steps/s)   speedup x%.2f\n",
                    t, ms, 1000 / ms, t1 / ms);
    }
    return 0;
}

// ─────────────────────── 메인 ───────────────────────
int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return run_bench();
    int sc = argc > 1 ? std::clamp(std::atoi(argv[1]), 1, 6) : 2;

    int TW = 120, rows = 40;
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        TW = ws.ws_col;
        rows = ws.ws_row;
    }
    int TR = std::max(4, rows - 1);  // 마지막 줄은 상태 표시
    W = TW * sc;
    H = TR * 2 * sc;
    alloc_fields();
    setup_term();

    std::vector<std::string> lines(TR);
    std::string out, inbuf;
    Mouse m;
    int brush = 3;  // 터미널 칸 단위
    float hue = 0.f;
    bool paused = false;
    long frame = 0;
    double sim_ms = 0, ren_ms = 0, fps = 0;
    const int nthreads = omp_get_max_threads();

    while (!g_quit) {
        double t_frame = omp_get_wtime();

        // ── 입력 처리 ──
        char buf[4096];
        ssize_t n;
        while (g_raw && (n = read(STDIN_FILENO, buf, sizeof buf)) > 0) inbuf.append(buf, (size_t)n);
        size_t pos = 0;
        while (pos < inbuf.size()) {
            if (inbuf[pos] == '\x1b') {
                if (pos + 2 >= inbuf.size()) break;  // 시퀀스가 덜 들어옴
                if (inbuf[pos + 1] == '[' && inbuf[pos + 2] == '<') {
                    size_t end = inbuf.find_first_of("Mm", pos + 3);
                    if (end == std::string::npos) break;
                    int b, x, y;
                    if (std::sscanf(inbuf.c_str() + pos + 3, "%d;%d;%d", &b, &x, &y) == 3) {
                        bool press = inbuf[end] == 'M';
                        m.x = (x - 1) * sc + (sc + 1) * 0.5f;
                        m.y = 2 * (y - 1) * sc + sc + 0.5f;
                        if (b == 64 || b == 65) {
                            if (press) brush = std::clamp(brush + (b == 64 ? 1 : -1), 1, 20);
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
                ++pos;  // 그 밖의 escape(방향키 등)는 무시
                continue;
            }
            switch (inbuf[pos]) {
                case 'q': case 'Q': g_quit = 1; break;
                case 'c': case 'C': clear_fields(); break;
                case ' ': g_jets = !g_jets; break;
                case 'v': case 'V': g_vort = !g_vort; break;
                case 'p': case 'P': paused = !paused; break;
                case '+': case '=': brush = std::min(20, brush + 1); break;
                case '-': case '_': brush = std::max(1, brush - 1); break;
            }
            ++pos;
        }
        inbuf.erase(0, pos);

        // ── 마우스 드래그 → 힘/염료 ──
        if (m.left || m.right) {
            float col[3];
            hsv2rgb(hue, col);
            float dx = m.x - m.px, dy = m.y - m.py;
            float rad = (float)(brush * sc);
            float dist = std::sqrt(dx * dx + dy * dy);
            int steps = std::max(1, (int)std::ceil(dist / (rad * 0.5f)));
            for (int s = 1; s <= steps; ++s) {
                float t = (float)s / steps;
                splat(m.px + dx * t, m.py + dy * t, dx * 0.35f, dy * 0.35f, rad, col,
                      m.left ? 0.5f : 0.f);
            }
            m.px = m.x;
            m.py = m.y;
        }
        hue += 0.004f;

        // ── 시뮬레이션 ──
        double t0 = omp_get_wtime();
        if (!paused) {
            apply_jets(frame, sc);
            step();
            ++frame;
        }
        double t1 = omp_get_wtime();
        render(lines, TW, TR, sc);
        double t2 = omp_get_wtime();
        sim_ms = 0.9 * sim_ms + 0.1 * (t1 - t0) * 1000;
        ren_ms = 0.9 * ren_ms + 0.1 * (t2 - t1) * 1000;

        // ── 출력 ──
        out.clear();
        for (auto& l : lines) out += l;
        char status[256];
        int len = std::snprintf(status, sizeof status,
            " %dx%d | threads %d | sim %.1fms draw %.1fms | %.0f fps | brush %d | jets %s vort %s%s"
            " | L:dye R:push wheel:brush spc:jets v p c q",
            W, H, nthreads, sim_ms, ren_ms, fps, brush, g_jets ? "on" : "off",
            g_vort ? "on" : "off", paused ? " PAUSED" : "");
        len = std::min(len, TW);
        out += "\x1b[" + std::to_string(TR + 1) + ";1H\x1b[0m\x1b[7m";
        out.append(status, (size_t)len);
        out += "\x1b[0m\x1b[K";
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);

        // ── 60fps 맞추기 ──
        double el = omp_get_wtime() - t_frame;
        if (el < 1.0 / 60) usleep((useconds_t)((1.0 / 60 - el) * 1e6));
        fps = 0.9 * fps + 0.1 / (omp_get_wtime() - t_frame);

        if (!g_raw && frame >= 120) break;  // 터미널이 아닐 때(파이프 등)는 잠깐 돌고 종료
    }
    restore_term();
    return 0;
}
