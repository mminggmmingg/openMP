# openMP

OpenMP로 만든 시뮬레이션 / 시각화 토이 프로젝트. 대부분 CPU(OpenMP) 경로와 GPU(ROCm/HIP) 경로가
같은 계산 함수를 공유하고, `--bench`로 스레드 수별 속도와 GPU 속도를 비교할 수 있습니다.

| 파일 | 내용 | 화면 |
|---|---|---|
| `openMP_hello.cpp`, `openMP_hello2.cpp` | OpenMP 첫걸음 (스레드마다 인사) | 터미널 |
| `mandel_zoom.cpp` | 망델브로 집합 줌인 + 스레드 벤치마크 | 터미널 |
| `fluid.cpp` | 마우스로 휘젓는 2D 유체 (Stable Fluids) | 터미널 |
| `sph.cpp`, `sph_more_particle.cpp` | 2D SPH 물 시뮬레이션 (CPU + GPU) | 터미널 |
| `lorenz.cpp` | 로렌츠 끌개, 입자 수백만 개 (CPU + GPU) | 터미널 |
| `lorenz_gl.cpp` | 로렌츠 끌개, HIP 계산 + OpenGL 렌더링 | 창 |
| `sph3d.cpp` | 3D SPH 물 + 화면 공간 물 표면 렌더링 | 창 |
| `vortex3d.cpp` | 3D 소용돌이: 와류 필라멘트법 + 연기 입자 (립프로깅, 충돌, 재연결, 고리 발사) | 창 |

## 빌드

각 `.cpp` 맨 위 주석에 빌드 / 실행 / 조작법이 있습니다. 예 (`vortex3d`):

```bash
# CPU 전용 (OpenMP)
g++ -O3 -march=native -std=c++17 -fopenmp vortex3d.cpp -o vortex3d -lglfw -lGL

# GPU 포함 (ROCm, Strix Halo iGPU = gfx1151)
HIP_CLANG_PATH=/opt/rocm/lib/llvm/bin hipcc -O3 -march=native -std=c++17 -fopenmp -DUSE_HIP \
    --offload-arch=gfx1151 -x hip vortex3d.cpp -o vortex3d -lglfw -lGL
```

- 창이 뜨는 프로그램은 `sudo apt install libglfw3-dev libgl-dev` 필요
- `hipcc`가 시스템 clang을 잡아서 `omp.h`를 못 찾으면 위처럼 `HIP_CLANG_PATH`로 ROCm clang 지정
- 다른 GPU면 `--offload-arch`를 바꾸면 됨 (`/opt/rocm/bin/offload-arch`로 확인)

## 실행

```bash
./vortex3d            # 창 모드
./vortex3d --bench    # 창 없이 CPU 스레드별 / GPU 속도 비교
```

## 라이선스

GPL-3.0 — [LICENSE](LICENSE). Copyright (C) 2026 mminggmmingg. 고쳐서 배포할 때도 같은 GPL-3.0 으로 소스를 공개해야 한다.
