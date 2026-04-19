#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <windows.h>

struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3 operator+(const Vec3& b) const { return { x + b.x, y + b.y, z + b.z }; }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return { x - b.x, y - b.y, z - b.z }; }
    __host__ __device__ Vec3 operator*(float t)        const { return { x * t,   y * t,   z * t }; }
    __host__ __device__ Vec3 operator*(const Vec3& b)  const { return { x * b.x, y * b.y, z * b.z }; }
    __host__ __device__ float dot(const Vec3& b)       const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3  cross(const Vec3& b)     const {
        return { y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x };
    }
    __host__ __device__ float len()  const { return sqrtf(x * x + y * y + z * z); }
    __host__ __device__ Vec3  norm() const { float l = len(); return { x / l, y / l, z / l }; }
};

struct Ray {
    Vec3 origin, dir;
    __host__ __device__ Vec3 at(float t) const { return origin + dir * t; }
};

struct Sphere {
    Vec3  center;
    float radius;
    Vec3  color;
    float specular;
    float reflective;
};

struct Light {
    Vec3  position;
    Vec3  color;
    float intensity;
};

__device__ float hitSphere(const Ray& ray, const Sphere& s, float tMin, float tMax) {
    Vec3  oc = ray.origin - s.center;
    float a = ray.dir.dot(ray.dir);
    float b = 2.0f * oc.dot(ray.dir);
    float c = oc.dot(oc) - s.radius * s.radius;
    float D = b * b - 4 * a * c;
    if (D < 0) return -1.0f;
    float t = (-b - sqrtf(D)) / (2 * a);
    if (t < tMin || t > tMax) {
        t = (-b + sqrtf(D)) / (2 * a);
        if (t < tMin || t > tMax) return -1.0f;
    }
    return t;
}

__device__ Vec3 computeLighting(Vec3 point, Vec3 normal, Vec3 viewDir, float specular,
    const Sphere* spheres, int numSpheres,
    const Light* lights, int numLights)
{
    float ambient = 0.1f;
    Vec3  result(ambient, ambient, ambient);

    for (int l = 0; l < numLights; l++) {
        Vec3  lightDir = (lights[l].position - point).norm();
        float dist = (lights[l].position - point).len();

        Ray shadowRay = { point + lightDir * 0.001f, lightDir };
        bool inShadow = false;
        for (int i = 0; i < numSpheres; i++) {
            float t = hitSphere(shadowRay, spheres[i], 0.001f, dist);
            if (t > 0) { inShadow = true; break; }
        }
        if (inShadow) continue;

        float diff = fmaxf(0.0f, normal.dot(lightDir));
        float spec = 0.0f;
        if (specular > 0 && diff > 0) {
            Vec3  R = normal * (2.0f * normal.dot(lightDir)) - lightDir;
            float RdV = fmaxf(0.0f, R.dot(viewDir));
            spec = powf(RdV, specular) * 0.7f;
        }

        float i = lights[l].intensity;
        result = result + lights[l].color * (diff * 0.85f * i + spec * i);
    }

    return result;
}

__device__ Vec3 traceRay(Ray ray,
    const Sphere* spheres, int numSpheres,
    const Light* lights, int numLights)
{
    Vec3  color(0, 0, 0);
    float reflectStrength = 1.0f;

    for (int bounce = 0; bounce < 4; bounce++) {
        float tMax = 1e9f;
        int   hitIdx = -1;

        for (int i = 0; i < numSpheres; i++) {
            float t = hitSphere(ray, spheres[i], 1e-3f, tMax);
            if (t > 0) { tMax = t; hitIdx = i; }
        }

        if (hitIdx < 0) {
            float t = 0.5f * (ray.dir.norm().y + 1.0f);
            Vec3  sky = Vec3(1, 1, 1) * (1 - t) + Vec3(0.5f, 0.7f, 1.0f) * t;
            color = color + sky * reflectStrength;
            break;
        }

        const Sphere& s = spheres[hitIdx];
        Vec3          point = ray.at(tMax);
        Vec3          normal = (point - s.center).norm();
        Vec3          viewDir = (ray.origin - point).norm();

        Vec3 light = computeLighting(point, normal, viewDir, s.specular,
            spheres, numSpheres, lights, numLights);
        Vec3 local = s.color * light;

        color = color + local * reflectStrength * (1.0f - s.reflective);
        reflectStrength *= s.reflective;
        if (reflectStrength < 0.01f) break;

        Vec3 reflDir = ray.dir - normal * (2.0f * ray.dir.dot(normal));
        ray = { point + reflDir * 0.001f, reflDir };
    }
    return color;
}

__global__ void renderKernel(unsigned char* fb, int width, int height,
    const Sphere* spheres, int numSpheres,
    const Light* lights, int numLights)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    Vec3  camPos(0, 7, 7);
    Vec3  lookAt(0, -1, 1);
    Vec3  up(0, 1, 0);
    float fov = 60.0f * 3.14159f / 180.0f;
    float aspect = (float)width / height;

    Vec3  forward = (lookAt - camPos).norm();
    Vec3  right = forward.cross(up).norm();
    Vec3  upReal = right.cross(forward);

    float halfH = tanf(fov / 2);
    float halfW = halfH * aspect;

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * halfW;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * halfH;

    Vec3 dir = (forward + right * u + upReal * v).norm();
    Ray  ray = { camPos, dir };

    Vec3 col = traceRay(ray, spheres, numSpheres, lights, numLights);

    col.x = sqrtf(fminf(col.x, 1.0f));
    col.y = sqrtf(fminf(col.y, 1.0f));
    col.z = sqrtf(fminf(col.z, 1.0f));

    int idx = (py * width + px) * 3;
    fb[idx + 0] = (unsigned char)(col.x * 255);
    fb[idx + 1] = (unsigned char)(col.y * 255);
    fb[idx + 2] = (unsigned char)(col.z * 255);
}

void saveBMP(const char* filename, unsigned char* data, int width, int height) {
    int padding = (4 - (width * 3) % 4) % 4;
    int rowSize = width * 3 + padding;
    int fileSize = 54 + rowSize * height;

    unsigned char header[54] = {};
    header[0] = 'B'; header[1] = 'M';
    header[2] = (unsigned char)(fileSize);
    header[3] = (unsigned char)(fileSize >> 8);
    header[4] = (unsigned char)(fileSize >> 16);
    header[5] = (unsigned char)(fileSize >> 24);
    header[10] = 54;
    header[14] = 40;
    header[18] = (unsigned char)(width);
    header[19] = (unsigned char)(width >> 8);
    header[20] = (unsigned char)(width >> 16);
    header[21] = (unsigned char)(width >> 24);
    header[22] = (unsigned char)(height);
    header[23] = (unsigned char)(height >> 8);
    header[24] = (unsigned char)(height >> 16);
    header[25] = (unsigned char)(height >> 24);
    header[26] = 1;
    header[28] = 24;

    FILE* f = fopen(filename, "wb");
    if (!f) { printf("Error: cannot open file %s\n", filename); return; }
    fwrite(header, 1, 54, f);

    unsigned char* row = new unsigned char[rowSize]();
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            int src = (y * width + x) * 3;
            int dst = x * 3;
            row[dst + 0] = data[src + 2];
            row[dst + 1] = data[src + 1];
            row[dst + 2] = data[src + 0];
        }
        fwrite(row, 1, rowSize, f);
    }
    delete[] row;
    fclose(f);
}

int readInt(const char* prompt, int lo, int hi) {
    int v;
    do { printf("%s (%d-%d): ", prompt, lo, hi); scanf_s("%d", &v); } while (v < lo || v > hi);
    return v;
}

static const Vec3  SPHERE_COLORS[10] = {
    Vec3(0.8f, 0.2f, 0.2f), Vec3(0.2f, 0.6f, 0.9f), Vec3(0.2f, 0.8f, 0.3f),
    Vec3(0.9f, 0.8f, 0.1f), Vec3(0.9f, 0.4f, 0.8f), Vec3(0.4f, 0.9f, 0.8f),
    Vec3(1.0f, 0.5f, 0.1f), Vec3(0.6f, 0.2f, 0.9f), Vec3(0.9f, 0.9f, 0.9f),
    Vec3(0.3f, 0.9f, 0.5f),
};
static const float SPHERE_SPECULAR[10] = { 200, 500,  50, 300, 150, 100, 400, 250, 600, 120 };
static const float SPHERE_REFLECTIVE[10] = { 0.3f, 0.6f, 0.1f, 0.5f, 0.4f, 0.2f, 0.5f, 0.35f, 0.8f, 0.25f };

int main() {
    SetConsoleOutputCP(65001);

    printf("CUDA Ray Tracer\n\n");

    int numSpheres = readInt("Number of spheres", 1, 10);
    int numLights = readInt("Number of light sources", 1, 2);
    int W = readInt("Image width  (800-1920)", 800, 1920);
    int H = readInt("Image height (600-1080)", 600, 1080);

    char filename[256];
    printf("Output filename (without extension): ");
    scanf_s("%255s", filename, (unsigned)sizeof(filename));
    char bmpFilename[260];
    sprintf_s(bmpFilename, "%s.bmp", filename);

    printf("\n");

    const int   totalSpheres = numSpheres + 1;
    Sphere* sceneSpheres = new Sphere[totalSpheres];
    const float PI = 3.14159265f;
    const float SPHERE_RAD = 1.0f;
    const float MIN_GAP = 0.4f;
    const float MIN_RADIUS = (numSpheres > 1)
        ? (SPHERE_RAD + MIN_GAP / 2.0f) / sinf(PI / numSpheres)
        : 0.0f;
    const float RING_RADIUS = (numSpheres == 1) ? 0.0f : fmaxf(3.0f, MIN_RADIUS);

    for (int i = 0; i < numSpheres; i++) {
        float angle = 2.0f * PI * i / numSpheres;
        sceneSpheres[i].center = Vec3(RING_RADIUS * cosf(angle), 0.0f, RING_RADIUS * sinf(angle));
        sceneSpheres[i].radius = SPHERE_RAD;
        sceneSpheres[i].color = SPHERE_COLORS[i % 10];
        sceneSpheres[i].specular = SPHERE_SPECULAR[i % 10];
        sceneSpheres[i].reflective = SPHERE_REFLECTIVE[i % 10];
    }

    sceneSpheres[numSpheres] = { Vec3(0.0f, -51.0f, 0.0f), 50.0f, Vec3(0.8f, 0.8f, 0.8f), 10, 0.05f };

    Light allLights[2] = {
        { Vec3(2.0f,  4.0f, -3.0f), Vec3(1.0f, 1.0f, 1.0f), 1.0f },
        { Vec3(-3.0f, 3.0f,  2.0f), Vec3(0.8f, 0.8f, 1.0f), 0.7f },
    };

    Sphere* d_spheres;
    cudaMalloc(&d_spheres, totalSpheres * sizeof(Sphere));
    cudaMemcpy(d_spheres, sceneSpheres, totalSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    Light* d_lights;
    cudaMalloc(&d_lights, numLights * sizeof(Light));
    cudaMemcpy(d_lights, allLights, numLights * sizeof(Light), cudaMemcpyHostToDevice);

    unsigned char* d_fb;
    cudaMalloc(&d_fb, W * H * 3);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    printf("Rendering %dx%d, %d spheres, %d light(s)...\n", W, H, numSpheres, numLights);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    renderKernel << <grid, block >> > (d_fb, W, H, d_spheres, totalSpheres, d_lights, numLights);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU render time: %.2f ms\n", ms);

    unsigned char* h_fb = new unsigned char[W * H * 3];
    cudaMemcpy(h_fb, d_fb, W * H * 3, cudaMemcpyDeviceToHost);

    saveBMP(bmpFilename, h_fb, W, H);
    printf("Saved: %s\n", bmpFilename);

    cudaFree(d_fb);
    cudaFree(d_spheres);
    cudaFree(d_lights);
    delete[] h_fb;
    delete[] sceneSpheres;

    return 0;
}
