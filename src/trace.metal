#include <metal_stdlib>
using namespace metal;

#define PI 3.14159265359f
#define DELTA 0.0001

struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_direction;
    float3 color;
    float3 normal;
    int32_t material;
    int32_t triangle;
    float c_importance;
    float l_importance;
    float tot_importance;
    int32_t hit_light;
    int32_t from_camera;
    int32_t hit_camera;
};

struct WeightAggregator {
    float3 weights[3];
    float3 total_contribution;
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
    float3 n0;
    float3 n1;
    float3 n2;
    float3 normal;
    int32_t material;
    int32_t is_light;
    int32_t is_camera;
    int32_t pad;
};


struct Material {
    float3 color;
    float3 emission;
    int32_t type;
    float alpha;
    float ior;
    int32_t pad;
};


struct Camera {
    float3 center;
    float3 focal_point;
    float3 direction;
    float3 dx;
    float3 dy;
    int32_t pixel_width;
    int32_t pixel_height;
    float phys_width;
    float phys_height;
    float h_fov;
    float v_fov;
    int32_t pad[2];
};

float xorshift_random(thread unsigned int &seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;

    return (float)(seed) / (float)0xFFFFFFFF;
}

float pcg_random(thread unsigned int &state, thread unsigned int &inc) {
    // PCG step (32-bit)
    state = (state * 747796405u + inc) & 0xFFFFFFFF;
    uint xorshifted = (((state >> 22u) ^ state) >> 11u) & 0xFFFFFFFF;
    uint rot = (state >> 28u) & 0xFFFFFFFF;
    uint result = ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF;

    // Convert the random uint to a float in the range [0, 1)
    return (float)(result) / (float)0xFFFFFFFF;
}

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


void ray_triangle_intersect(const thread Ray &ray, const thread Triangle &triangle, thread bool &hit, thread float &t_out, thread float& u, thread float& v) {
    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);
    float f = 1.0 / a;
    float3 s = ray.origin - triangle.v0;
    u = f * dot(s, h);
    if (u < 0 || u > 1) {
        hit = false;
        return;
    }
    float3 q = cross(s, edge1);
    v = f * dot(ray.direction, q);
    if (v < 0 || u + v > 1) {
        hit = false;
        return;
    }
    float t = f * dot(edge2, q);
    if (t > 0.0f) {
        hit = true;
        t_out = t;
    } else {
        hit = false;
    }
}

void traverse_bvh(const thread Ray &ray, const device Box *boxes, const device Triangle *triangles, thread int &best_i, thread float &best_t, thread float &u_out, thread float &v_out) {
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0 && stack_ptr < 64) {
        int box_id = stack[--stack_ptr];
        Box box = boxes[box_id];
        bool hit = false;
        float t = INFINITY;
        ray_box_intersect(ray, box, hit, t);
        if (hit && t < best_t) {
            if (box.right == 0) {
                stack[stack_ptr++] = box.left;
                stack[stack_ptr++] = box.left + 1;
            } else {
                for (int i = box.left; i < box.right; i++) {
                    if (i == ray.triangle) {continue;}
                    Triangle triangle = triangles[i];
                    bool hit = false;
                    t = INFINITY;
                    float u, v;
                    ray_triangle_intersect(ray, triangle, hit, t, u, v);
                    if (hit && t < best_t) {
                        best_i = i;
                        best_t = t;
                        u_out = u;
                        v_out = v;
                    }
                }
            }
        }
    }
}

bool visibility_test(const thread Ray a, const thread Ray b, const device Box *boxes, const device Triangle *triangles) {
    Ray test_ray;
    test_ray.origin = a.origin + a.normal * 0.0001f;
    float3 direction = (b.origin + b.normal * 0.0001f) - (a.origin + a.normal * 0.0001f);
    float t_max = length(direction);
    direction = direction / t_max;
    test_ray.direction = direction;
    test_ray.inv_direction = 1.0 / direction;
    test_ray.triangle = a.triangle;

    int best_i = -1;
    float best_t = t_max;
    float u, v;
    traverse_bvh(test_ray, boxes, triangles, best_i, best_t, u, v);
    return best_t >= t_max;
}


void orthonormal(const thread float3 &n, thread float3 &x, thread float3 &y){
    float3 v;
    if (abs(n.x) <= abs(n.y) && abs(n.x) <= abs(n.z)) {
        v = float3(1, 0, 0);
    }
    else if (abs(n.y) <= abs(n.z)) {
        v = float3(0, 1, 0);
    }
    else {
        v = float3(0, 0, 1);
    }

    x = normalize(v - dot(v, n) * n);
    y = normalize(cross(n, x));
}

float2 sample_disk_concentric(const thread float2 &rand) {
    float2 offset = 2.0 * rand - float2(1.0, 1.0);
    if (offset.x == 0 && offset.y == 0) {
        return float2(0, 0);
    }
    float theta, r;
    if (abs(offset.x) > abs(offset.y)) {
        r = offset.x;
        theta = PI / 4 * (offset.y / offset.x);
    } else {
        r = offset.y;
        theta = PI / 2 - PI / 4 * (offset.x / offset.y);
    }
    return r * float2(cos(theta), sin(theta));
}

float3 random_hemisphere_cosine(const thread float3 &x_axis, const thread float3 &y_axis, const thread float3 &z_axis, const thread float2 &rand) {
    float2 d = sample_disk_concentric(rand);
    float z = sqrt(max(0., 1 - d.x * d.x - d.y * d.y));
    return normalize(d.x * x_axis + d.y * y_axis + z * z_axis);
}

float3 random_hemisphere_uniform(const thread float3 &x_axis, const thread float3 &y_axis, const thread float3 &z_axis, const thread float2 &rand) {
    float z = rand.x;
    float r = sqrt(max(0., 1. - z * z));
    float phi = 2 * PI * rand.y;
    return normalize(r * cos(phi) * x_axis + r * sin(phi) * y_axis + z * z_axis);
}

float3 GGX_sample(const thread float3 &x_axis, const thread float3 &y_axis, const thread float3 &z_axis, const thread float2 &rand, const thread float alpha) {
    float theta = 2 * PI * rand.x;
    float phi = atan(alpha * sqrt(rand.y) / sqrt(1.0f - rand.y));
    return normalize(sin(phi) * cos(theta) * x_axis + sin(phi) * sin(theta) * y_axis + cos(phi) * z_axis);
}

float3 specular_reflection(const thread float3 &i, const thread float3 &m) {
    return normalize(2 * dot(i, m) * m - i);
}

float3 specular_reflect_half_direction(const thread float3 &i, const thread float3 &o) {
    return normalize(i + o);
}

float3 GGX_transmit(const thread float3 &i, const thread float3 &m, const thread float ni, const thread float no) {
    float cosTheta_i = dot(i, m);
    float eta = ni / no;
    float cosTheta_t = sqrt(1 + eta * (cosTheta_i * cosTheta_i - 1));
    return normalize((eta * cosTheta_i - cosTheta_t) * m - eta * i);
}

float3 specular_transmit_half_direction(const thread float3 &i, const thread float3 &o, const thread float ni, const thread float no) {
    return normalize(-(no * o + ni * i));
}

float degreve_fresnel(const thread float3 &i, const thread float3 &m, const thread float ni, const thread float nt) {
    float cosTheta_i = abs(dot(i, m));
    float eta = ni / nt;
    float sinTheta_t2 = eta * eta * (1.0f - cosTheta_i * cosTheta_i);
    if (sinTheta_t2 > 1.0f) {
        return 1.0f;
    }
    float cosTheta_t = sqrt(1.0f - sinTheta_t2);
    float r_parallel = (nt * cosTheta_i - ni * cosTheta_t) / (nt * cosTheta_i + ni * cosTheta_t);
    float r_perpendicular = (ni * cosTheta_i - nt * cosTheta_t) / (ni * cosTheta_i + nt * cosTheta_t);
    return 0.5f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
}

float GGX_G1(const thread float3 &v, const thread float3 &m, const thread float3 &n, const thread float alpha) {
    float mv = dot(m, v);
    float sin2 = 1.0f - mv * mv;
    float tan2 = sin2 / (mv * mv);
    return 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * tan2));
}

float GGX_G(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float alpha) {
    return GGX_G1(i, m, n, alpha) * GGX_G1(o, m, n, alpha);
}

float GGX_D(const thread float3 &m, const thread float3 &n, const thread float alpha) {
    if (alpha == 0) {return 1.0f;}

    float alpha2 = alpha * alpha;
    float cosTheta = dot(m, n);
    float cosTheta2 = cosTheta * cosTheta;
    float tan2 = (1 - cosTheta2) / cosTheta2;

    float denom =  alpha2 + tan2;

    return alpha2 / (cosTheta2 * cosTheta2 * PI * denom * denom);
}

float reflect_jacobian(const thread float3 &m, const thread float3 &o) {
    return 1.0f / (4.0f * abs(dot(m, o)));
}

float transmit_jacobian(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float ni, const thread float no) {
    float cosTheta_i = dot(i, m);
    float cosTheta_o = dot(o, m);
    float numerator = no * no * abs(cosTheta_o);
    float denominator = (ni * cosTheta_i + no * cosTheta_o) * (ni * cosTheta_i + no * cosTheta_o);
    return numerator / denominator;
}

float GGX_BRDF_reflect(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float ni, const thread float no, const thread float alpha) {
    float D = GGX_D(m, n, alpha);
    float G = GGX_G(i, o, m, n, alpha);
    float F = degreve_fresnel(i, m, ni, no);

    return (D * G * F) / (4 * abs(dot(i, n)));
}

float GGX_BRDF_transmit(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float ni, const thread float no, const thread float alpha) {
    float D = GGX_D(m, n, alpha);
    float G = GGX_G(i, o, m, n, alpha);
    float F = degreve_fresnel(i, m, ni, no);

    float im = dot(i, m);
    float om = dot(o, m);
    float in = dot(i, n);
    float on = dot(o, n);

    float coeff = abs((im * om) / (in * on));
    float num = no * no * D * G * (1 - F);
    float denom = (ni * im + no * om) * (ni * im + no * om);

    return coeff * num / denom;
}

float BRDF(const thread float3 &i, const thread float3 &o, const thread float3 &n, const thread float3 &geom_n, const thread Material material) {
    if (material.type == 0) {
        return abs(dot(o, n));
    }
    else {
        float ni, no, alpha;
        alpha = material.alpha;
        if (dot(i, geom_n) > 0) {
            ni = 1.0f;
            no = material.ior;
        }
        else {
            ni = material.ior;
            no = 1.0f;
        }
        if (dot(i, geom_n) * dot(o, geom_n) > 0 && dot(i, n) * dot(o, n) > 0) {
            float3 m = specular_reflect_half_direction(i, o);
            return GGX_BRDF_reflect(i, o, m, n, ni, no, alpha) * abs(dot(o, m));
        }
        else if (dot(i, geom_n) * dot(o, geom_n) < 0 && dot(i, n) * dot(o, n) < 0) {
            float3 m = specular_transmit_half_direction(i, o, ni, no);
            return GGX_BRDF_transmit(i, o, m, n, ni, no, alpha) * abs(dot(o, m));
        }
        else {
            return 0.0f;
        }
    }
}

float3 sample_normal(const thread Triangle &triangle, const thread float u, const thread float v) {
    return normalize(triangle.n0 * (1 - u - v) + triangle.n1 * u + triangle.n2 * v);
}

kernel void generate_paths(const device Ray *rays [[ buffer(0) ]],
                   const device Box *boxes [[ buffer(1) ]],
                   const device Triangle *triangles [[ buffer(2) ]],
                   const device Material *materials [[ buffer(3) ]],
                   device unsigned int *random_buffer [[ buffer(4) ]],
                   device float4 *out [[ buffer(5) ]],
                   device Path *output_paths [[ buffer(6) ]],
                   device float4 *float_debug [[ buffer(7) ]],
                   uint id [[ thread_position_in_grid ]]) {
    Path path;
    path.length = 0;
    Ray ray, new_ray, next_ray;
    ray = rays[id];
    path.from_camera = ray.from_camera;
    out[id] = float4(0, 0, 0, 1);

    unsigned int seed0 = random_buffer[2 * id];
    unsigned int seed1 = random_buffer[2 * id + 1];

    if (path.from_camera == 0) {
        new_ray.l_importance = 1.0f / (2 * PI);
    }
    else {
        new_ray.c_importance = 1.0f;
    }

    for (int i = 0; i < 8; i++) {

        int best_i = -1;
        float best_t = INFINITY;
        float u, v;
        traverse_bvh(ray, boxes, triangles, best_i, best_t, u, v);

        Triangle triangle = triangles[best_i];
        Material material = materials[triangle.material];

        float3 n;
        float ni, no;
        float alpha = material.alpha;
        float3 sampled_normal = sample_normal(triangle, u, v);
        float3 signed_normal;
        if (dot(-ray.direction, triangle.normal) > DELTA) {
            signed_normal = triangle.normal;
            n = sampled_normal;
            ni = 1.0f;
            no = material.ior;
        } else if (dot(-ray.direction, triangle.normal) < -DELTA){
            signed_normal = -triangle.normal;
            n = -sampled_normal;
            ni = material.ior;
            no = 1.0f;
        }
        else {
            break;
        }

        new_ray.origin = ray.origin + ray.direction * best_t;
        new_ray.normal = sampled_normal;
        new_ray.material = triangle.material;
        new_ray.triangle = best_i;
        if (triangle.is_light && dot(ray.direction, triangle.normal) < 0.0f) {new_ray.hit_light = best_i;}
        else {new_ray.hit_light = -1;}
        if (triangle.is_camera) {new_ray.hit_camera = best_i;}
        else {new_ray.hit_camera = -1;}

        float3 x, y;
        orthonormal(n, x, y);

        float3 wi, wo;
        wi = -ray.direction;

        float rand_x_a = pcg_random(seed0, seed1);
        float rand_y_a = pcg_random(seed0, seed1);
        float2 random_roll_a = float2(rand_x_a, rand_y_a);

        float rand_x_b = pcg_random(seed0, seed1);
        float rand_y_b = pcg_random(seed0, seed1);
        float2 random_roll_b = float2(rand_x_b, rand_y_b);

        float f, c_p, l_p;
        if (material.type == 0) {
            if (path.from_camera) {
                wo = random_hemisphere_cosine(x, y, n, random_roll_a);
                f = dot(n, wo) / PI;
                c_p = dot(n, wo) / PI;
                l_p = 1.0f / (2 * PI);
            } else {
                wo = random_hemisphere_uniform(x, y, n, random_roll_a);
                f = dot(n, wo) / PI;
                c_p = dot(n, wi) / PI;
                l_p = 1.0f / (2 * PI);
            }
        } else {
            float3 m = GGX_sample(x, y, n, random_roll_a, alpha);
            float fresnel = degreve_fresnel(wi, m, ni, no);
            float pf = 1.0f;

            if (random_roll_b.x <= fresnel) {
                wo = specular_reflection(wi, m);
                f = GGX_BRDF_reflect(wi, wo, m, sampled_normal, ni, no, alpha);
                pf = fresnel;
            } else {
                wo = GGX_transmit(wi, m, ni, no);
                f = GGX_BRDF_transmit(wi, wo, m, sampled_normal, ni, no, alpha);
                pf = 1.0f - fresnel;
            }

            float pm = abs(dot(m, n)) * GGX_D(m, n, alpha);

            if (dot(wo, n) > 0.0f) {
                c_p = pf * pm * reflect_jacobian(m, wo) / abs(dot(wo, m));
                l_p = pf * pm * reflect_jacobian(m, wi) / abs(dot(wi, m));
            } else {
                if (path.from_camera) {
                    c_p = pf * pm * transmit_jacobian(wi, wo, m, ni, no) / abs(dot(wo, m));
                    l_p = pf * pm * transmit_jacobian(wo, wi, -m, no, ni) / abs(dot(wi, m));
                } else {
                    c_p = pf * pm * transmit_jacobian(wo, wi, -m, ni, no) / abs(dot(wi, m));
                    l_p = pf * pm * transmit_jacobian(wi, wo, m, no, ni) / abs(dot(wo, m));
                }
            }
        }

        if (dot(wi, triangle.normal) > 0.0f) {
            new_ray.color = f * ray.color * material.color;
        } else {
            new_ray.color = f * ray.color;
        }

        if (f == 0.0f) {break;}

        new_ray.direction = wo;
        new_ray.inv_direction = 1.0f / wo;

        if (path.from_camera) {
            next_ray.c_importance = c_p;
            ray.l_importance = l_p;
            new_ray.tot_importance = ray.tot_importance * new_ray.c_importance;
        } else {
            next_ray.l_importance = l_p;
            ray.c_importance = c_p;
            new_ray.tot_importance = ray.tot_importance * new_ray.l_importance;
        }

        path.rays[i] = ray;
        path.length = i + 1;

        ray = new_ray;
        new_ray = next_ray;
    }
    output_paths[id] = path;

    for (int i = 0; i < path.length; i++){
        if (path.rays[i].hit_light >= 0){
            out[id] = float4(path.rays[i - 1].color / path.rays[i].tot_importance, 1);
            break;
        }
    }
    random_buffer[2 * id] = seed0;
    random_buffer[2 * id + 1] = seed1;
}

float geometry_term(const thread Ray &a, const thread Ray &b){
    float3 delta = b.origin - a.origin;
    float dist = length(delta);
    delta = normalize(delta);

    float camera_cos, light_cos;
    camera_cos = abs(dot(a.normal, delta));
    light_cos = abs(dot(b.normal, -delta));

    return (camera_cos * light_cos) / (dist * dist);
}

Ray get_ray(const thread Path &camera_path, const thread Path &light_path, const thread int t, const thread int s, const thread int i){
    if (i < s) {return light_path.rays[i];}
    else {return camera_path.rays[t + s - i - 1];}
}

int get_sample_index(const thread float3 &point, const thread Camera &camera) {
    // Compute the direction from the camera's origin to the point
    float3 dir = point - camera.center;

    // Project the direction onto the camera's basis vectors
    float x = dot(dir, camera.dx);
    float y = dot(dir, camera.dy);

    // Calculate the normalized pixel coordinates with FOV and aspect ratio
    float normalizedX = x / (tan(camera.h_fov / 2.0));
    float normalizedY = y / (tan(camera.v_fov / 2.0));

    // Map normalized coordinates to pixel indices
    int x_index = int((normalizedX + 0.5) * camera.pixel_width);
    int y_index = int((normalizedY + 0.5) * camera.pixel_height);

    // Clamp the indices to the texture dimensions
    if (x_index < 0 || x_index >= camera.pixel_width || y_index < 0 || y_index >= camera.pixel_height) {
        return -1; // Return -1 for out-of-bounds
    }

    // Return the linear index in the texture
    return y_index * camera.pixel_width + x_index;
}

int map_camera_pixel(const thread Ray &source, const device Camera &camera, const device Triangle *triangles, const device Box *boxes, thread Ray &hit_ray){
    float3 dir = normalize(camera.focal_point - source.origin);
    Ray test_ray;
    test_ray.origin = source.origin;
    test_ray.direction = dir;
    test_ray.inv_direction = normalize(1.0f / dir);
    test_ray.triangle = source.triangle;
    test_ray.normal = source.normal;

    int best_i = -1;
    float best_t = INFINITY;
    float u, v;
    traverse_bvh(test_ray, boxes, triangles, best_i, best_t, u, v);

    if (best_i == -1) {return -1;}
    if (triangles[best_i].is_camera == 0) {return -1;}
    if (dot(test_ray.direction, triangles[best_i].normal) > 0.0f) {return -1;}

    hit_ray.origin = test_ray.origin + test_ray.direction * best_t;
    hit_ray.color = float3(1.0f);
    hit_ray.direction = -dir;
    hit_ray.normal = camera.direction;
    hit_ray.material = triangles[best_i].material;
    hit_ray.triangle = best_i;
    hit_ray.hit_camera = 1;
    hit_ray.hit_light = -1;
    hit_ray.c_importance = 1.0f / (camera.phys_width * camera.phys_height);
    hit_ray.l_importance = 1.0f / (2.0f * PI);
    hit_ray.tot_importance = hit_ray.c_importance;

    return 1;
}

float3 pixel_center(const thread Camera &camera, const thread int x, const thread int y){

    float x_normalized = (x - 0.5 * camera.pixel_width) / (float)camera.pixel_width;
    float y_normalized = (y - 0.5 * camera.pixel_height) / (float)camera.pixel_height;

    float3 x_vector = x_normalized * tan(camera.h_fov / 2.0) * camera.dx;
    float3 y_vector = y_normalized * tan(camera.v_fov / 2.0) * camera.dy;

    float3 origin = camera.center + x_vector + y_vector;

    return origin;
}

float gaussian_weight(const thread float3 &p, const thread float3 &q, const thread float sigma){
    float dist = length(p - q);
    return exp(-dist * dist / (2.0f * sigma * sigma));
}

kernel void connect_paths(const device Path *camera_paths [[ buffer(0) ]],
                          const device Path *light_paths [[ buffer(1) ]],
                          const device Triangle *triangles [[ buffer(2) ]],
                          const device Material *materials [[ buffer(3) ]],
                          const device Box *boxes [[ buffer(4) ]],
                          const device Camera *camera [[ buffer(5) ]],
                          device WeightAggregator *weight_aggregators [[ buffer(6) ]],
                          device float4 *out [[ buffer(7) ]],
                          device float4 *light_image [[ buffer(8) ]],
                          uint id [[ thread_position_in_grid ]]) {

    Path camera_path = camera_paths[id];
    Path light_path = light_paths[id];

    int sample_index = id;
    out[id] = float4(0.0f);
    light_image[id] = float4(0.0f);
    WeightAggregator aggregator = weight_aggregators[id];
    aggregator.total_contribution = float3(0.0f);
    Camera c = camera[0];

    for (int t = 0; t < camera_path.length + 1; t++){
        for (int s = 0; s < light_path.length + 1; s++){

            if (s + t < 2) {continue;}

            Ray light_ray;
            light_ray.triangle = -1;
            Ray camera_ray;
            camera_ray.triangle = -1;
            sample_index = id;

            // these need to be on if t==1 is on, but otherwise they're just wasted bandwidth. nearly doubles join time.
            //camera_path = camera_paths[id];
            //light_path = light_paths[id];


            if (t == 0){
                continue;
                // light ray hits the camera plane. disabled due to pinhole camera model
                light_ray = light_path.rays[s - 1];
                if (light_ray.hit_camera < 0) {continue;}
                sample_index = get_sample_index(light_ray.origin, c);
                if (sample_index == -1) {continue;}
            }
            else if (t == 1) {
                continue;
                // light visibility to camera plane. WIP. Disabled for now.
                light_ray = light_path.rays[s - 1];
                if (materials[light_ray.material].type == 1) {continue;}
                int hit = map_camera_pixel(light_ray, camera[0], triangles, boxes, camera_ray);
                if (hit == -1) {continue;}
                sample_index = get_sample_index(camera_ray.origin, c);
                if (sample_index == -1) {continue;}
                camera_path.rays[0] = camera_ray;
            }
            else if (s == 0) {
                //continue;
                // camera ray hits a light source
                camera_ray = camera_path.rays[t - 1];
                if (camera_ray.hit_light < 0) {continue;}
            }
            else {
                //continue;
                // regular join
                light_ray = light_path.rays[s - 1];
                camera_ray = camera_path.rays[t - 1];

                // skip specular joins
                if (materials[light_ray.material].type == 1) {continue;}
                if (materials[camera_ray.material].type == 1) {continue;}

                float3 dir_l_to_c = normalize(camera_ray.origin - light_ray.origin);

                // backface culling for joins
                if (dot(light_ray.normal, dir_l_to_c) < DELTA) {continue;}
                if (dot(camera_ray.normal, -dir_l_to_c) < DELTA) {continue;}

                if (not visibility_test(light_ray, camera_ray, boxes, triangles)) {continue;}
            }
            if (light_ray.triangle == camera_ray.triangle) {continue;}

            float p_ratios[32];
            float p_values[32];

            // set up p_ratios like p1/p0, p2/p1, p3/p2, ... out to pk+1/pk, where k = s + t - 1
            for (int i = 0; i < s + t; i++) {
                float num, denom;
                if (i == 0) {
                    Ray a = get_ray(camera_path, light_path, t, s, i);
                    Ray b = get_ray(camera_path, light_path, t, s, i + 1);
                    if (s == 0) {num = light_path.rays[0].l_importance;}
                    else {num = a.l_importance;}
                    denom = a.c_importance * geometry_term(a, b);
                }
                else if (i == s + t - 1) {
                    Ray a = get_ray(camera_path, light_path, t, s, i);
                    Ray b = get_ray(camera_path, light_path, t, s, i - 1);
                    num = a.l_importance * geometry_term(a, b);
                    if (t == 0) {denom = camera_path.rays[0].c_importance;}
                    else {denom = a.c_importance;}
                }
                else {
                    Ray a, b, c;
                    a = get_ray(camera_path, light_path, t, s, i - 1);
                    b = get_ray(camera_path, light_path, t, s, i);
                    c = get_ray(camera_path, light_path, t, s, i + 1);
                    num = b.l_importance * geometry_term(a, b);
                    denom = b.c_importance * geometry_term(b, c);
                }
                p_ratios[i] = num / denom;
            }

            float prior_camera_importance;
            if (t == 0) {prior_camera_importance = 1.0f;}
            else {prior_camera_importance = camera_path.rays[t - 1].tot_importance;}

            float prior_light_importance;
            if (s == 0) {prior_light_importance = 1.0f;}
            else {prior_light_importance = light_path.rays[s - 1].tot_importance;}

            float p_s = prior_camera_importance * prior_light_importance;

            float p_i = 1.0f;

            for (int i = s; i < s + t; i++) {
                p_values[i + 1] = p_ratios[i] * p_i;
                p_i = p_values[i + 1];
            }

            p_i = 1.0f;

            for (int i = s - 1; i >= 0; i--) {
                p_values[i] = p_i / p_ratios[i];
                p_i = p_values[i];
            }

            p_values[s] = 1.0f;

            for (int i = 0; i < s + t + 1; i++) {
                if (materials[get_ray(camera_path, light_path, t, s, i).material].type == 1) {
                    p_values[i] = 0.0f;
                    p_values[i + 1] = 0.0f;
                }
            }

            // this is because t=0 and t=1 are disabled. greatly enhances caustics by giving s==0 much more weight.
            p_values[s + t] = 0.0f;
            p_values[s + t - 1] = 0.0f;

            float sum = 0.0f;
            for (int i = 0; i < s + t + 1; i++) {sum += p_values[i];}

            float w;
            if (sum > 0.0f && p_values[s] > 0.0f) {w = p_values[s] / sum;}
            else {continue;}

            float3 color = float3(1.0f);
            float g = 1.0f;

            if (s == 0) {color = camera_path.rays[t - 2].color;}
            else if (t == 0) {color = light_path.rays[s - 2].color;}
            else if (t == 1) {
                float3 dir_l_to_c = normalize(camera_ray.origin - light_ray.origin);
                if (s == 1) {color = materials[light_ray.material].emission;}
                else {
                    float3 prior_light_color = light_path.rays[s - 2].color;
                    float3 prior_light_direction = light_path.rays[s - 2].direction;

                    Material light_material = materials[light_ray.material];
                    float3 light_geom_normal = triangles[light_ray.triangle].normal;
                    float new_light_f = BRDF(-prior_light_direction, dir_l_to_c, light_ray.normal, light_geom_normal, light_material);
                    color = prior_light_color * light_material.color * new_light_f;
               }
               g = geometry_term(camera_ray, light_ray);
            }
            else {
                float3 dir_l_to_c = normalize(camera_ray.origin - light_ray.origin);
                float3 camera_color = float3(1.0f);

                float3 prior_camera_color = camera_path.rays[t - 2].color;
                float3 prior_camera_direction = camera_path.rays[t - 2].direction;

                Material camera_material = materials[camera_ray.material];
                float3 camera_geom_normal = triangles[camera_ray.triangle].normal;
                float new_camera_f = BRDF(-prior_camera_direction, -dir_l_to_c, camera_ray.normal, camera_geom_normal, camera_material);
                camera_color = prior_camera_color * new_camera_f * camera_material.color;

                float3 light_color = float3(1.0f);
                if (s == 1) {
                    light_color = materials[light_ray.material].emission;
                }
                else {
                    float3 prior_light_color = light_path.rays[s - 2].color;
                    float3 prior_light_direction = light_path.rays[s - 2].direction;

                    Material light_material = materials[light_ray.material];
                    float3 light_geom_normal = triangles[light_ray.triangle].normal;
                    float new_light_f = BRDF(-prior_light_direction, dir_l_to_c, light_ray.normal, light_geom_normal, light_material);
                    light_color = prior_light_color * new_light_f * light_material.color;
                }
                color = camera_color * light_color;
                g = geometry_term(camera_ray, light_ray);
            }
            float3 sample = w * g * color / p_s;
            aggregator.total_contribution += sample;
        }
    }

    // prep weights for final gather

    float weight_sum = 0.0f;
    float pixel_phys_width = c.phys_width / c.pixel_width;
    float pixel_phys_height = c.phys_height / c.pixel_height;
    float sigma = 0.5f * sqrt(pixel_phys_width * pixel_phys_width + pixel_phys_height * pixel_phys_height);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            aggregator.weights[i][j] = 0.0f;
        }
    }
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int new_sample_x = (sample_index % c.pixel_width) + i;
            int new_sample_y = (sample_index / c.pixel_width) + j;

            if (new_sample_x < 0 || new_sample_x >= c.pixel_width ||
                new_sample_y < 0 || new_sample_y >= c.pixel_height) {
                continue;
            }

            int new_sample_index = new_sample_y * c.pixel_width + new_sample_x;
            if (new_sample_index < 0 || new_sample_index >= c.pixel_width * c.pixel_height) {continue;}

            float weight = gaussian_weight(pixel_center(c, new_sample_x, new_sample_y), camera_path.rays[0].origin, sigma);
            aggregator.weights[i + 1][j + 1] = weight;
            weight_sum += weight;
        }
    }
    if (weight_sum != 0.0f) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                aggregator.weights[i][j] = aggregator.weights[i][j] / weight_sum;
            }
        }
    }
    out[id] = float4(aggregator.total_contribution, 1.0f);
    weight_aggregators[id] = aggregator;
}


kernel void finalize_samples(const device WeightAggregator *weight_aggregators [[ buffer(0) ]],
                             const device Camera *camera [[ buffer(1) ]],
                             device float4 *out [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
    // final gather. in separate kernel to globally sync
    float3 total_sample = float3(0.0f);
    Camera c = camera[0];
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int new_sample_x = (id % c.pixel_width) + i;
            int new_sample_y = (id / c.pixel_width) + j;

            if (new_sample_x < 0 || new_sample_x >= c.pixel_width ||
                new_sample_y < 0 || new_sample_y >= c.pixel_height) {
                continue;
            }

            int new_sample_index = new_sample_y * c.pixel_width + new_sample_x;
            if (new_sample_index < 0 || new_sample_index >= c.pixel_width * c.pixel_height) {continue;}

            // flip indices around center
            int x_idx, y_idx;
            if (i < 0) {x_idx = 2;}
            else if (i == 0) {x_idx = 1;}
            else {x_idx = 0;}
            if (j < 0) {y_idx = 2;}
            else if (j == 0) {y_idx = 1;}
            else {y_idx = 0;}

            float weight = weight_aggregators[new_sample_index].weights[x_idx][y_idx];
            float3 sample = weight_aggregators[new_sample_index].total_contribution * weight;
            total_sample += sample;
        }
    }
    out[id] = float4(total_sample, 1.0f);
}


kernel void generate_camera_rays(const device Camera *camera [[ buffer(0) ]],
                                 device unsigned int *random_buffer [[ buffer(1) ]],
                                 device Ray *out [[ buffer(2) ]],
                                 uint id [[ thread_position_in_grid ]]) {
    Camera c = camera[0];
    Ray ray;

    uint seed0 = random_buffer[2 * id];
    uint seed1 = random_buffer[2 * id + 1];

    float x_offset = pcg_random(seed0, seed1);
    float y_offset = pcg_random(seed0, seed1);

    int pixel_x = id % c.pixel_width;
    int pixel_y = id / c.pixel_width;

    float x_normalized = (pixel_x + x_offset - 0.5 * c.pixel_width) / (float)c.pixel_width;
    float y_normalized = (pixel_y + y_offset - 0.5 * c.pixel_height) / (float)c.pixel_height;

    float3 x_vector = x_normalized * tan(c.h_fov / 2.0) * c.dx;
    float3 y_vector = y_normalized * tan(c.v_fov / 2.0) * c.dy;

    float3 origin = c.center + x_vector + y_vector;
    float3 direction = normalize(c.focal_point - origin);
    ray.origin = origin;
    ray.direction = direction;
    ray.normal = c.direction;
    ray.inv_direction = 1.0f / direction;
    ray.color = float3(1.0f);

    ray.material = 7;
    ray.triangle = -1;
    ray.hit_light = -1;
    ray.hit_camera = -1;
    ray.from_camera = 1;
    ray.c_importance = 1.0f / (c.phys_width * c.phys_height);
    ray.l_importance = 1.0f; // filled in later
    ray.tot_importance = ray.c_importance;

    out[id] = ray;
    random_buffer[2 * id] = seed0;
    random_buffer[2 * id + 1] = seed1;
}


kernel void generate_light_rays(const device Triangle *light_triangles [[buffer(0) ]],
                                const device int *counts [[buffer(1) ]],
                                const device float *surface_areas [[buffer(2) ]],
                                const device int *light_triangle_indices [[buffer(3) ]],
                                device unsigned int *random_buffer [[buffer(4) ]],
                                device Ray *out [[buffer(5) ]],
                                uint id [[thread_position_in_grid]]) {
    Ray ray;
    ray.from_camera = 0;
    ray.hit_light = -1;
    ray.hit_camera = -1;

    unsigned int seed0 = random_buffer[2 * id];
    unsigned int seed1 = random_buffer[2 * id + 1];

    int light_count = counts[0];
    int light_index = (int)(pcg_random(seed0, seed1) * light_count);
    Triangle light_triangle = light_triangles[light_index];
    float surface_area = surface_areas[light_index];

    float u = pcg_random(seed0, seed1);
    float v = pcg_random(seed0, seed1);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float w = 1.0f - u - v;

    ray.normal = light_triangle.normal;
    ray.origin = light_triangle.v0 * u + light_triangle.v1 * v + light_triangle.v2 * w + DELTA * ray.normal;
    float3 x, y;
    orthonormal(ray.normal, x, y);

    float rand_x = pcg_random(seed0, seed1);
    float rand_y = pcg_random(seed0, seed1);
    float2 random_roll = float2(rand_x, rand_y);
    ray.direction = random_hemisphere_uniform(x, y, ray.normal, random_roll);
    ray.inv_direction = 1.0f / ray.direction;
    ray.color = float3(1.0f);

    ray.material = light_triangle.material;

    // todo: for some reason this introduces some visual glitches. investigate eventually.
    // ray.triangle = light_triangle_indices[light_index];
    ray.triangle = -1;

    ray.c_importance = 1.0f; // filled in later
    ray.l_importance = 1.0f / (light_count * surface_area);
    ray.tot_importance = ray.l_importance;

    out[id] = ray;
    random_buffer[2 * id] = seed0;
    random_buffer[2 * id + 1] = seed1;
}
