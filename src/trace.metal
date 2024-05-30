#include <metal_stdlib>
using namespace metal;

#define PI 3.14159265359


struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_direction;
    float3 color;
    float importance;
    int hit_light;
    int i;
    int j;
};

struct Path {
    Ray rays[16];
    int32_t length;
    int32_t from_camera;
    int32_t pad[2];
};

struct Box {
    float3 min;
    float3 max;
    int32_t left;
    int32_t right;
    int32_t pad[2];
};


struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    float3 normal;
    int32_t material;
    int32_t is_light;
    int32_t pad[2];
};


struct Material {
    float3 color;
    float3 emission;
    int32_t type;
    int32_t pad[3];
};


void ray_box_intersect(const thread Ray &ray, const thread Box &box, thread bool &hit, thread float &t) {
    float3 inv_direction = 1.0 / ray.direction;
    float3 t0s = (box.min - ray.origin) * inv_direction;
    float3 t1s = (box.max - ray.origin) * inv_direction;
    float3 tsmaller = min(t0s, t1s);
    float3 tbigger = max(t0s, t1s);
    float tmin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tmax = min(min(min(tbigger.x, tbigger.y), tbigger.z), t);
    hit = tmin <= tmax;
    t = tmin;
}


void ray_triangle_intersect(const thread Ray &ray, const thread Triangle &triangle, thread bool &hit, thread float &t_out) {

    if (dot(ray.direction, triangle.normal) > 0) {
        hit = false;
        return;
    }

    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);
    if (a <= 0) {
        hit = false;
        return;
    }
    float f = 1.0 / a;
    float3 s = ray.origin - triangle.v0;
    float u = f * dot(s, h);
    if (u < 0 || u > 1) {
        hit = false;
        return;
    }
    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);
    if (v < 0 || u + v > 1) {
        hit = false;
        return;
    }
    float t = f * dot(edge2, q);
    if (t > 0) {
        hit = true;
        t_out = t;
    } else {
        hit = false;
    }
}


void traverse_bvh(const thread Ray &ray, const device Box *boxes, const device Triangle *triangles, thread int &best_i, thread float &best_t) {
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0 && stack_ptr < 64) {
        int box_id = stack[--stack_ptr];
        Box box = boxes[box_id];
        bool hit = false;
        float t = INFINITY;
        ray_box_intersect(ray, box, hit, t);
        if (hit) {
            if (box.right == 0) {
                stack[stack_ptr++] = box.left;
                stack[stack_ptr++] = box.left + 1;
            } else {
                for (int i = box.left; i < box.right; i++) {
                    Triangle triangle = triangles[i];
                    bool hit = false;
                    float t = INFINITY;
                    ray_triangle_intersect(ray, triangle, hit, t);
                    if (hit && t < best_t) {
                        best_i = i;
                        best_t = t;
                    }
                }
            }
        }
    }
}


bool visibility_test(const thread Ray &ray, const thread float3 target, const device Box *boxes, const device Triangle *triangles) {
    float3 direction = target - ray.origin;
    float t_max = length(direction);
    int best_i = -1;
    float best_t = INFINITY;
    traverse_bvh(ray, boxes, triangles, best_i, best_t);
    return best_t >= t_max;
}


void local_orthonormal_basis(const thread float3 &n, thread float3 &x, thread float3 &y) {
    if (abs(n.x) > abs(n.y)) {
        x = float3(-n.z, 0, n.x) / sqrt(n.x * n.x + n.z * n.z);
    } else {
        x = float3(0, n.z, -n.y) / sqrt(n.y * n.y + n.z * n.z);
    }
    y = cross(n, x);
}


float3 random_hemisphere_cosine_weighted(const thread float3 &x_axis, const thread float3 &y_axis, const thread float3 &z_axis, const thread float2 &rand) {
    float r = sqrt(rand.x);
    float theta = 2 * PI * rand.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    return x * x_axis + y * y_axis + sqrt(max(0., 1 - rand.x)) * z_axis;
}


float3 random_hemisphere_uniform(const thread float3 &x_axis, const thread float3 &y_axis, const thread float3 &z_axis, const thread float2 &rand) {
    float r = sqrt(1 - rand.x * rand.x);
    float theta = 2 * PI * rand.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1 - r * r);
    return x * x_axis + y * y_axis + z * z_axis;
}


float3 specular_reflection(const thread float3 &direction, const thread float3 &normal) {
    return direction - 2 * dot(direction, normal) * normal;
}


kernel void generate_paths(const device Ray *rays [[ buffer(0) ]],
                   const device Box *boxes [[ buffer(1) ]],
                   const device Triangle *triangles [[ buffer(2) ]],
                   const device Material *materials [[ buffer(3) ]],
                   const device float2 *random [[ buffer(4) ]],
                   device float4 *out [[ buffer(5) ]],
                   device Path *output_paths [[ buffer(6) ]],
                   device int *debug [[ buffer(7) ]],
                   uint id [[ thread_position_in_grid ]]) {
    Path path;
    path.rays[0] = rays[id];
    path.length = 1;
    if (path.rays[0].i >= 0) {
        path.from_camera = 1;
    } else {
        path.from_camera = 0;
    }

    for (int i = 0; i < 8; i++) {
        int best_i = -1;
        float best_t = INFINITY;
        Ray ray = path.rays[i];
        traverse_bvh(ray, boxes, triangles, best_i, best_t);

        if (best_i == -1) {
            break;
        }

        Triangle triangle = triangles[best_i];
        Material material = materials[triangle.material];
        float2 rand = random[id + i];

        Ray new_ray;
        new_ray.origin = ray.origin + ray.direction * best_t;
        new_ray.i = ray.i;
        new_ray.j = ray.j;

        float3 x, y;
        local_orthonormal_basis(triangle.normal, x, y);

        float f, p;
        if (material.type == 0) {
            if (path.from_camera) {
                    new_ray.direction = random_hemisphere_uniform(x, y, triangle.normal, rand);
                    f = 1.0f;
                    p = 1.0f;
                }
            else {
                new_ray.direction = random_hemisphere_uniform(x, y, triangle.normal, rand);
                f = dot(triangle.normal, -ray.direction);
                p = 1.0f / (2 * PI);
            }
        } else {
            new_ray.direction = specular_reflection(ray.direction, triangle.normal);
            f = 1.0f;
            p = 1.0f;
        }

        if (f == 0) {
            break;
        }

        new_ray.color = material.color * f * ray.color;
        new_ray.importance = ray.importance * p;

        if (triangle.is_light) {
            new_ray.hit_light = best_i;
            new_ray.color = ray.color * material.emission;
            new_ray.importance = ray.importance;
        } else {
            new_ray.hit_light = -1;
        }

        path.rays[i + 1] = new_ray;
        path.length = i + 2;

        if (new_ray.hit_light >= 0) {
            break;
        }
    }
    output_paths[id] = path;
    Ray final_ray = path.rays[path.length - 1];
    if (final_ray.hit_light >= 0) {
        out[id] = float4(final_ray.color / final_ray.importance, 1);
    }
    else {
        out[id] = float4(0, 0, 0, 1);
    }
}


kernel void connect_paths(const device Path *camera_paths [[ buffer(0) ]],
                          const device Path *light_paths [[ buffer(1) ]],
                          const device Triangle *triangles [[ buffer(2) ]],
                          const device Material *materials [[ buffer(3) ]],
                          const device Box *boxes [[ buffer(4) ]],
                          device float4 *out [[ buffer(5) ]],
                          device int *debug [[ buffer(6) ]],
                          uint id [[ thread_position_in_grid ]]) {
    Path camera_path = camera_paths[id];
    Path light_path = light_paths[id % 1024];
    float3 sample = float3(0, 0, 0);
    int samples = 0;
    for (int t = 1; t < camera_path.length; t++) {
        Ray camera_ray = camera_path.rays[t];
        Ray light_ray = light_path.rays[0];
        if (visibility_test(camera_ray, light_ray.origin, boxes, triangles)) {
            float3 color = camera_ray.color * light_ray.color;
            sample = sample + color / (camera_ray.importance * light_ray.importance);
            samples++;
        }
    }
    if (samples > 0){
        out[id] = float4(sample, 1);
    }
    else {
        out[id] = float4(0, 0, 0, 1);
    }
}