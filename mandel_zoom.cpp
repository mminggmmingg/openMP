// mandel_zoom.cpp — OpenMP로 돌리는 터미널 Mandelbrot 줌인 + 스레드 벤치마크
// 빌드: g++ -O3 -std=c++17 -fopenmp mandel_zoom.cpp -o mandel_zoom
// 실행: ./mandel_zoom            (기본 300프레임)
//       ./mandel_zoom 600        (프레임 수 지정)
//       OMP_NUM_THREADS=2 ./mandel_zoom
#include <omp.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// 줌 목표점 (Seahorse valley 근처, 무한히 들어가도 모양이 계속 나옴)
static const double CX = -0.743643887037151;
static const double CY =  0.131825904205330;

// 한 점의 탈출 반복 수 (부드러운 색을 위해 실수로 반환)
static inline double mandel(double x0, double y0, int max_iter) {
    double x = 0, y = 0, x2 = 0, y2 = 0;
    int i = 0;
    while (x2 + y2 <= 256.0 && i < max_iter) {
        y = 2 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        ++i;
    }
    if (i == max_iter) return -1;  // 집합 내부
    return i + 1 - std::log2(std::log2(x2 + y2) * 0.5);
}

// 반복 수 -> RGB (사인파 팔레트)
static inline void palette(double t, int& r, int& g, int& b) {
    if (t < 0) { r = g = b = 0; return; }
    double s = t * 0.08;
    r = (int)(127.5 + 127.5 * std::sin(s + 0.0));
    g = (int)(127.5 + 127.5 * std::sin(s + 2.1));
    b = (int)(127.5 + 127.5 * std::sin(s + 4.2));
}

// 화면 하나 계산. 위/아래 픽셀 두 개를 문자 하나(▀)에 담아서 세로 해상도를 2배로.
static void render(std::vector<double>& buf, int W, int H, double scale, int max_iter) {
    double aspect = 1.0;  // ▀ 트릭 덕분에 픽셀이 대략 정사각형
    #pragma omp parallel for schedule(dynamic, 1)
    for (int py = 0; py < H; ++py) {
        double y0 = CY + (py - H / 2.0) * scale;
        for (int px = 0; px < W; ++px) {
            double x0 = CX + (px - W / 2.0) * scale * aspect;
            buf[py * W + px] = mandel(x0, y0, max_iter);
        }
    }
}

static void draw(const std::vector<double>& buf, int W, int H, std::string& out) {
    out.clear();
    out += "\x1b[H";  // 커서를 맨 위로
    char tmp[64];
    for (int row = 0; row + 1 < H; row += 2) {
        for (int px = 0; px < W; ++px) {
            int r1, g1, b1, r2, g2, b2;
            palette(buf[row * W + px], r1, g1, b1);
            palette(buf[(row + 1) * W + px], r2, g2, b2);
            std::snprintf(tmp, sizeof tmp, "\x1b[38;2;%d;%d;%dm\x1b[48;2;%d;%d;%dm\u2580",
                          r1, g1, b1, r2, g2, b2);
            out += tmp;
        }
        out += "\x1b[0m\n";
    }
}

static double bench(int threads, int N, int max_iter) {
    omp_set_num_threads(threads);
    std::vector<double> buf((size_t)N * N);
    double scale = 3e-5 / N;
    double t0 = omp_get_wtime();
    render(buf, N, N, scale, max_iter);
    return omp_get_wtime() - t0;
}

int main(int argc, char** argv) {
    int frames = argc > 1 ? std::atoi(argv[1]) : 300;
    int max_threads = omp_get_max_threads();

    winsize ws{};
    int cols = 120, rows = 40;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        cols = ws.ws_col;
        rows = ws.ws_row;
    }
    int W = cols;
    int H = (rows - 2) * 2;  // 마지막 두 줄은 상태 표시용

    std::vector<double> buf((size_t)W * H);
    std::string out;
    out.reserve((size_t)W * H * 40);

    std::printf("\x1b[2J\x1b[?25l");  // 화면 지우고 커서 숨김
    double scale = 3.5 / W;
    double total = 0;

    for (int f = 0; f < frames; ++f) {
        int max_iter = 100 + (int)(40 * std::log2(3.5 / W / scale + 1));  // 깊어질수록 반복 늘림
        double t0 = omp_get_wtime();
        render(buf, W, H, scale, max_iter);
        double dt = omp_get_wtime() - t0;
        total += dt;

        draw(buf, W, H, out);
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::printf("\x1b[0m threads=%d  frame %d/%d  zoom x%.2e  iter=%d  calc %.1f ms  \x1b[K",
                    max_threads, f + 1, frames, 3.5 / W / scale, max_iter, dt * 1000);
        std::fflush(stdout);

        scale *= 0.96;
        usleep(15000);
    }
    std::printf("\x1b[?25h\n\n");  // 커서 복구

    std::printf("평균 프레임 계산 시간: %.2f ms (%d threads)\n\n", total / frames * 1000, max_threads);

    // 벤치마크: 같은 이미지를 스레드 수만 바꿔서 계산
    const int N = 1200, IT = 3000;
    std::printf("벤치마크: %dx%d, max_iter=%d\n", N, N, IT);
    double t1 = bench(1, N, IT);
    std::printf("  %2d thread : %7.3f s\n", 1, t1);
    for (int t = 2; t <= max_threads; t *= 2) {
        double tt = bench(t, N, IT);
        std::printf("  %2d threads: %7.3f s   speedup x%.2f\n", t, tt, t1 / tt);
    }
    if ((max_threads & (max_threads - 1)) != 0) {  // 2의 거듭제곱이 아니면 최대값도 측정
        double tt = bench(max_threads, N, IT);
        std::printf("  %2d threads: %7.3f s   speedup x%.2f\n", max_threads, tt, t1 / tt);
    }
    return 0;
}
