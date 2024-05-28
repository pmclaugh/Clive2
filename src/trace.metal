#include <metal_stdlib>
using namespace metal;

struct Ray {
    float3 origin;
    float3 direction;
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
};


void ray_box_intersect(const thread Ray &ray, const thread Box &box, thread bool &hit, thread float &t) {
    float3 inv_direction = 1.0 / ray.direction;
    float3 t0s = (box.min - ray.origin) * inv_direction;
    float3 t1s = (box.max - ray.origin) * inv_direction;
    float3 tsmaller = min(t0s, t1s);
    float3 tbigger = max(t0s, t1s);
    float tmin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);
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


kernel void bounce(const device Ray *rays [[ buffer(0) ]],
                   const device Box *boxes [[ buffer(1) ]],
                   const device Triangle *triangles [[ buffer(2) ]],
                   device float4 *out [[ buffer(3) ]],
                   device int *debug [[ buffer(4) ]],
                   uint id [[ thread_position_in_grid ]]) {
    Ray ray = rays[id];

    debug[0] = sizeof(Box);
    debug[1] = boxes[0].pad[0];
    debug[2] = boxes[0].pad[1];

    float best_t = INFINITY;
    int best_i = -1;

    traverse_bvh(ray, boxes, triangles, best_i, best_t);

    if (best_i == -1) {
        out[id] = float4(0, 0, 0, 1);
    }
    else {
        out[id] = float4(triangles[best_i].normal * 0.5f + 0.5f, 1);
    }
}
