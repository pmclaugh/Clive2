#include <metal_stdlib>
using namespace metal;

#define PI 3.14159265359
#define DELTA 0.0001

struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_direction;
    float3 color;
    float3 normal;
    int32_t material;
    float c_importance;
    float l_importance;
    float tot_importance;
    int32_t hit_light;
    int32_t from_camera;
    int32_t pad[2];
};

struct Path {
    Ray rays[8];
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
    float3 t0s = (box.min - ray.origin) * ray.inv_direction;
    float3 t1s = (box.max - ray.origin) * ray.inv_direction;
    float3 tsmaller = min(t0s, t1s);
    float3 tbigger = max(t0s, t1s);
    float tmin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tmax = min(min(min(tbigger.x, tbigger.y), tbigger.z), t);
    hit = tmin <= tmax;
    t = tmin;
}


void ray_triangle_intersect(const thread Ray &ray, const thread Triangle &triangle, thread bool &hit, thread float &t_out) {

    if (dot(ray.direction, triangle.normal) >= 0) {
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
        if (hit && (t < best_t)) {
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


bool visibility_test(const thread float3 origin, const thread float3 target, const device Box *boxes, const device Triangle *triangles) {
    Ray test_ray;
    test_ray.origin = origin;
    float3 direction = target - origin;
    float t_max = length(direction);
    direction = direction / t_max;
    test_ray.direction = direction;
    test_ray.inv_direction = 1.0 / direction;

    int best_i = -1;
    float best_t = t_max;
    traverse_bvh(test_ray, boxes, triangles, best_i, best_t);
    return best_t >= t_max;
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
    path.length = 0;
    Ray ray = rays[id];
    path.from_camera = ray.from_camera;
    debug[id] = 0;

    for (int i = 0; i < 8; i++) {
        path.rays[i] = ray;
        path.length = i + 1;

        if (ray.hit_light >= 0 && path.from_camera == 1) {
            break;
        }

        int best_i = -1;
        float best_t = INFINITY;
        traverse_bvh(ray, boxes, triangles, best_i, best_t);

        if (best_i == -1) {
            break;
        }

        Triangle triangle = triangles[best_i];
        Material material = materials[triangle.material];
        float2 rand = random[id * 8 + i];

        Ray new_ray;
        new_ray.origin = ray.origin + ray.direction * best_t + DELTA * triangle.normal;
        new_ray.normal = triangle.normal;
        new_ray.material = triangle.material;

        float3 x, y;
        local_orthonormal_basis(triangle.normal, x, y);

        float f, c_p, l_p;
        if (material.type == 0) {
            if (path.from_camera) {
                    new_ray.direction = random_hemisphere_uniform(x, y, triangle.normal, rand);
                    f = dot(triangle.normal, new_ray.direction);
                    c_p = dot(triangle.normal, new_ray.direction);
                    l_p = 1.0f / (2 * PI);
                }
            else {
                new_ray.direction = random_hemisphere_uniform(x, y, triangle.normal, rand);
                f = dot(triangle.normal, -ray.direction);
                c_p = dot(triangle.normal, -ray.direction);
                l_p = 1.0f / (2 * PI);
            }
        } else {
            new_ray.direction = specular_reflection(ray.direction, triangle.normal);
            f = 1.0f;
            c_p = 1.0f;
            l_p = 1.0f;
        }

        new_ray.inv_direction = 1.0 / new_ray.direction;
        new_ray.color = material.color * f * ray.color;
        new_ray.c_importance = c_p;
        new_ray.l_importance = l_p;
        if (path.from_camera) {
            new_ray.tot_importance = ray.tot_importance * c_p;
        } else {
            new_ray.tot_importance = ray.tot_importance * l_p;
        }

        if (triangle.is_light) {
            new_ray.hit_light = best_i;
        } else {
            new_ray.hit_light = -1;
        }
        ray = new_ray;
    }
    output_paths[id] = path;
    Ray final_ray = path.rays[path.length - 1];
    if (final_ray.hit_light >= 0) {
        out[id] = float4(final_ray.color / final_ray.tot_importance, 1);
    }
    else {
        out[id] = float4(0, 0, 0, 1);
    }

    debug[id] = path.length;
}


float geometry_term(const thread Ray &a, const thread Ray &b){
    float3 delta = b.origin - a.origin;
    float dist = length(delta);
    delta = delta / dist;

    float camera_cos = dot(a.normal, delta);
    float light_cos = dot(b.normal, -delta);

    return (camera_cos * light_cos) / (dist * dist);
}


Ray get_ray(const thread Path &camera_path, const thread Path &light_path, const thread int t, const thread int s, const thread int i){
    if (i < t){
        return camera_path.rays[i];
    }
    else {
        return light_path.rays[t + s - i - 1];
    }
}


kernel void connect_paths(const device Path *camera_paths [[ buffer(0) ]],
                          const device Path *light_paths [[ buffer(1) ]],
                          const device Triangle *triangles [[ buffer(2) ]],
                          const device Material *materials [[ buffer(3) ]],
                          const device Box *boxes [[ buffer(4) ]],
                          device float4 *out [[ buffer(5) ]],
                          device int *debug [[ buffer(6) ]],
                          device float* float_debug [[ buffer(7) ]],
                          uint id [[ thread_position_in_grid ]]) {
    Path camera_path = camera_paths[id];
    Path light_path = light_paths[id];
    float3 sample = float3(0.0f);
    int sample_count = 0;
    float p_ratios[16];

    debug[id] = light_path.length;

    for (int t = 0; t < camera_path.length; t++){
        for (int s = 0; s < light_path.length; s++){

            Ray light_ray;
            Ray camera_ray;

            if (t == 0){
                // this is where a light ray hits the camera plane. not yet supported.
                continue;
            }
            else if (s == 0){
                // this is where a camera ray hits the light source. not yet supported.
                continue;
            }
            else if (t == 1){
                // this is visibility from camera plane to light ray. not yet supported.
                continue;
            }
            else {
                light_ray = light_path.rays[s - 1];
                camera_ray = camera_path.rays[t - 1];

                float3 dir_l_to_c = camera_ray.origin - light_ray.origin;
                float dist_l_to_c = length(dir_l_to_c);
                dir_l_to_c = dir_l_to_c / dist_l_to_c;

                if (dot(light_ray.normal, dir_l_to_c) <= 0){
                    continue;
                }
                if (dot(camera_ray.normal, -dir_l_to_c) <= 0){
                    continue;
                }
                if (not visibility_test(light_ray.origin, camera_ray.origin, boxes, triangles)){
                    continue;
                }
            }

            for (int i = 0; i < 16; i++){
                p_ratios[i] = 1.0f;
            }

            for (int i = 0; i < s + t; i++){
                float num, denom;
                if (i == 0){
                    Ray a, b;
                    a = get_ray(camera_path, light_path, t, s, i);
                    b = get_ray(camera_path, light_path, t, s, i + 1);
                    num = a.c_importance;
                    denom = b.l_importance * geometry_term(a, b);
                }
                else if (i == s + t - 1) {
                    Ray a, b;
                    a = get_ray(camera_path, light_path, t, s, i - 1);
                    b = get_ray(camera_path, light_path, t, s, i);
                    num = a.c_importance * geometry_term(a, b);
                    denom = b.l_importance;
                }
                else {
                    Ray a, b, c;
                    a = get_ray(camera_path, light_path, t, s, i - 1);
                    b = get_ray(camera_path, light_path, t, s, i);
                    c = get_ray(camera_path, light_path, t, s, i + 1);
                    num = geometry_term(a, b) * a.c_importance;
                    denom = geometry_term(b, c) * c.l_importance;
                }
                p_ratios[i] = num / denom;
            }

            for (int i = 1; i < s + t; i++){
                p_ratios[i] = p_ratios[i] * p_ratios[i - 1];
            }

            float w = 0.0f;
            for (int i = 0; i < s + t; i++){
                w += (p_ratios[s - 1] * p_ratios[s - 1]) / (p_ratios[i] * p_ratios[i]);
            }

            float3 dir_l_to_c = camera_ray.origin - light_ray.origin;
            float dist_l_to_c = length(dir_l_to_c);
            dir_l_to_c = dir_l_to_c / dist_l_to_c;

            float3 prior_camera_color = t > 1 ? camera_path.rays[t - 2].color : float3(1.0f);
            Material camera_material = materials[camera_ray.material];
            float new_camera_f = dot(camera_ray.normal, -dir_l_to_c);
            float3 camera_color = prior_camera_color * new_camera_f * camera_material.color;

            float prior_camera_importance = t > 1 ? camera_path.rays[t - 2].tot_importance : 1.0f;
            float prior_light_importance = s > 1 ? light_path.rays[s - 2].tot_importance : 1.0f;

            sample += (geometry_term(light_ray, camera_ray) * camera_color * light_ray.color) / (w * prior_camera_importance * prior_light_importance);
        }
    }
    out[id] = float4(sample, 1.0f);
}
