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
    int32_t pixel_idx;
    int32_t pad[3];
};

struct WeightAggregator {
    float weights[3][3];
    float3 total_contribution;
    float contrib_weight_sum;
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
    int32_t transmissive;
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
    if (t > DELTA) {
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
        float u, v;
        ray_box_intersect(ray, box, hit, t);
        if (hit && t < best_t) {
            if (box.right == 0) {
                stack[stack_ptr++] = box.left;
                stack[stack_ptr++] = box.left + 1;
            } else {
                for (int i = box.left; i < box.right; i++) {
                    Triangle triangle = triangles[i];
                    hit = false;
                    t = INFINITY;
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
    test_ray.origin = a.origin;
    float3 direction = b.origin - a.origin;
    direction = normalize(direction);
    test_ray.direction = direction;
    test_ray.inv_direction = 1.0 / direction;
    test_ray.triangle = a.triangle;

    int best_i = -1;
    float best_t = INFINITY;
    float u, v;

    traverse_bvh(test_ray, boxes, triangles, best_i, best_t, u, v);

    if (best_i == -1) {return false;}
    if (best_i == a.triangle) {return false;}
    if (best_i == b.triangle) {return true;}
    return false;
}


void orthonormal(const thread float3 &n, thread float3 &x, thread float3 &y){
    float3 v;
    if (abs(n.x) <= abs(n.y) && abs(n.x) <= abs(n.z))
        v = float3(1, 0, 0);
    else if (abs(n.y) <= abs(n.z))
        v = float3(0, 1, 0);
    else
        v = float3(0, 0, 1);

    x = normalize(v - dot(v, n) * n);
    y = normalize(cross(n, x));
}

float3 random_hemisphere_cosine(const thread float3 &x_axis, const thread float3 &y_axis, const thread float3 &z_axis, const thread float2 &rand) {
    float theta = acos(sqrt(rand.x));
    float phi = 2 * PI * rand.y;
    return normalize(sin(theta) * cos(phi) * x_axis + sin(theta) * sin(phi) * y_axis + cos(theta) * z_axis);
}

float3 random_hemisphere_uniform(const thread float3 &x_axis, const thread float3 &y_axis, const thread float3 &z_axis, const thread float2 &rand) {
    float z = rand.x;
    float r = sqrt(max(0., 1. - z * z));
    float phi = 2 * PI * rand.y;
    return normalize(r * cos(phi) * x_axis + r * sin(phi) * y_axis + z * z_axis);
}

float3 GGX_sample(const thread float3 &n, const thread float2 &rand, const thread float alpha) {
    float3 x, y;
    orthonormal(n, x, y);

    float theta = 2 * PI * rand.x;
    float phi = atan(alpha * sqrt(rand.y) / sqrt(1.0 - rand.y));
    return normalize(sin(phi) * cos(theta) * x + sin(phi) * sin(theta) * y + cos(phi) * n);
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
    float sinTheta_t2 = eta * eta * (1.0 - cosTheta_i * cosTheta_i);
    if (sinTheta_t2 >= 1.0)
        return 1.0;
    float cosTheta_t = sqrt(1.0 - sinTheta_t2);
    float r_parallel = (nt * cosTheta_i - ni * cosTheta_t) / (nt * cosTheta_i + ni * cosTheta_t);
    float r_perpendicular = (ni * cosTheta_i - nt * cosTheta_t) / (ni * cosTheta_i + nt * cosTheta_t);
    return 0.5f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
}

float GGX_G1(const thread float3 &v, const thread float3 &m, const thread float3 &n, const thread float alpha) {
    float mv = dot(m, v);
    float sin2 = 1.0 - mv * mv;
    float tan2 = sin2 / (mv * mv);
    return 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * tan2));
}

float GGX_G(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float alpha) {
    if (dot(i, m) * dot(i, n) <= 0.0) {return 0.0;}
    if (dot(o, m) * dot(o, n) <= 0.0) {return 0.0;}
    return GGX_G1(i, m, n, alpha) * GGX_G1(o, m, n, alpha);
}

float GGX_D(const thread float3 &m, const thread float3 &n, const thread float alpha) {
    if (alpha == 0.0) {return 1.0;}

    float alpha2 = alpha * alpha;
    float cosTheta = dot(m, n);
    float cosTheta2 = cosTheta * cosTheta;
//    float tan2 = (1 - cosTheta2) / cosTheta2;

    float denom = cosTheta2 * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

float reflect_jacobian(const thread float3 &m, const thread float3 &o) {
    return 1.0 / (4.0 * abs(dot(m, o)));
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

    return (D * G * F) / (4.0 * abs(dot(i, m)));
}

float GGX_BRDF_transmit(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float ni, const thread float no, const thread float alpha) {
    float D = GGX_D(m, n, alpha);
    float G = GGX_G(i, o, m, n, alpha);
    float F = degreve_fresnel(i, m, ni, no);

    float im = dot(i, m);
    float om = dot(o, m);
    float in = dot(i, n);
    float on = dot(o, n);

    float coeff = (im * om) / (in * on);
    float num = no * no * D * G * (1.0 - F);
    float denom = (ni * im + no * om) * (ni * im + no * om);

    return coeff * num / denom;
}

float3 sample_normal(const thread Triangle &triangle, const thread float u, const thread float v) {
    return normalize(triangle.n0 * (1 - u - v) + triangle.n1 * u + triangle.n2 * v);
}

void diffuse_bounce(const thread float3 wi, const thread float3 n, thread bool from_camera, thread float2 random_roll, thread float3 &wo, thread float &f, thread float &c_p, thread float &l_p) {
    float3 x, y;
    orthonormal(n, x, y);
    wo = random_hemisphere_cosine(x, y, n, random_roll);
    f = dot(n, wo) / PI;
    if (from_camera) {
        c_p = dot(n, wo) / PI;
        l_p = dot(n, wi) / PI;
    } else {
        c_p = dot(n, wi) / PI;
        l_p = dot(n, wo) / PI;
    }
}

void reflect_bounce(const thread float3 &wi, const thread float3 &n, const thread float3 &m, const thread float ni, const thread float no, const thread float alpha, thread bool from_camera, thread float3 &wo, thread float &f, thread float &c_p, thread float &l_p) {
    wo = specular_reflection(wi, m);
    f = GGX_BRDF_reflect(wi, wo, m, n, ni, no, alpha);

    float pf = degreve_fresnel(wi, m, ni, no);
    float pm = abs(dot(m, n)) * GGX_D(m, n, alpha);

    if (from_camera) {
        c_p = pf * pm * reflect_jacobian(m, wo);
        l_p = pf * pm * reflect_jacobian(m, wi);
    } else {
        c_p = pf * pm * reflect_jacobian(m, wi);
        l_p = pf * pm * reflect_jacobian(m, wo);
    }
}

void transmit_bounce(const thread float3 &wi, const thread float3 &n, const thread float3 &m, const thread float ni, const thread float no, const thread float alpha, thread bool from_camera, thread float3 &wo, thread float &f, thread float &c_p, thread float &l_p) {
    wo = GGX_transmit(wi, m, ni, no);
    f = GGX_BRDF_transmit(wi, wo, m, n, ni, no, alpha);

    float pf = 1.0 - degreve_fresnel(wi, m, ni, no);
    float pm = abs(dot(m, n)) * GGX_D(m, n, alpha);

    if (from_camera) {
        c_p = pf * pm * transmit_jacobian(wi, wo, m, ni, no);
        l_p = pf * pm * transmit_jacobian(wo, wi, -m, no, ni);
    }
    else {
        c_p = pf * pm * transmit_jacobian(wo, wi, -m, no, ni);
        l_p = pf * pm * transmit_jacobian(wi, wo, m, ni, no);
    }
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
    out[id] = float4(0, 0, 0, 0);

    unsigned int seed0 = random_buffer[2 * id];
    unsigned int seed1 = random_buffer[2 * id + 1];

    if (path.from_camera == 0)
        new_ray.l_importance = 1.0 / (2.0 * PI);
    else {
        // this seems reasonable to me, but unsure.
        new_ray.c_importance = ray.c_importance;
    }

    for (int i = 0; i < 8; i++) {

        int best_i = -1;
        float best_t = INFINITY;
        float u, v;
        traverse_bvh(ray, boxes, triangles, best_i, best_t, u, v);

        if (best_i == -1)
            break;

        Triangle triangle = triangles[best_i];
        Material material = materials[triangle.material];

        float3 n;
        float ni, no;
        float alpha = material.alpha;
        float3 sampled_normal = sample_normal(triangle, u, v);
        if (dot(-ray.direction, triangle.normal) > 0) {
            n = sampled_normal;
            ni = 1.0;
            no = material.ior;
        } else if (dot(-ray.direction, triangle.normal) < 0) {
            n = -sampled_normal;
            ni = material.ior;
            no = 1.0;
        }
        else {
            break;
        }

        new_ray.origin = ray.origin + ray.direction * best_t;
        new_ray.material = triangle.material;
        new_ray.triangle = best_i;

        if (triangle.is_light && dot(ray.direction, triangle.normal) < 0.0)
            new_ray.hit_light = best_i;
        else
            new_ray.hit_light = -1;

        if (triangle.is_camera)
            new_ray.hit_camera = best_i;
        else
            new_ray.hit_camera = -1;

        float3 wi = -ray.direction;

        float rand_x_a = xorshift_random(seed0);
        float rand_y_a = xorshift_random(seed1);
        float2 random_roll_a = float2(rand_x_a, rand_y_a);

        float rand_x_b = xorshift_random(seed0);
        float rand_y_b = xorshift_random(seed1);
        float2 random_roll_b = float2(rand_x_b, rand_y_b);

        float3 wo;
        float f = 1.0;
        float c_p = 1.0;
        float l_p = 1.0;

        float3 m = GGX_sample(n, random_roll_a, alpha);
        if (dot(wi, m) < 0.0)
            m = specular_reflection(m, n);
        if (dot(m, n) < 0.0)
            break;
        new_ray.normal = n;

        float fresnel = degreve_fresnel(wi, m, ni, no);
        if (material.type == 0)
            diffuse_bounce(wi, n, path.from_camera, random_roll_b, wo, f, c_p, l_p);
        else if (material.type == 1)
            if (random_roll_b.x <= fresnel)
                reflect_bounce(wi, n, m, ni, no, alpha, path.from_camera, wo, f, c_p, l_p);
            else
                transmit_bounce(wi, n, m, ni, no, alpha, path.from_camera, wo, f, c_p, l_p);
        else if (material.type == 2)
            if (random_roll_b.x <= fresnel)
                reflect_bounce(wi, n, m, ni, no, alpha, path.from_camera, wo, f, c_p, l_p);
            else
                diffuse_bounce(wi, n, path.from_camera, random_roll_b, wo, f, c_p, l_p);
        else
            reflect_bounce(wi, n, m, ni, no, alpha, path.from_camera, wo, f, c_p, l_p);

        if (dot(wi, triangle.normal) > 0.0 && dot(wo, triangle.normal) > 0.0)
            new_ray.color = f * ray.color * material.color; // external reflection
        else if (dot(wi, triangle.normal) < 0.0 && dot(wo, triangle.normal) > 0.0)
            new_ray.color = f * ray.color * material.color; // egress
        else
            new_ray.color = f * ray.color; // internal reflection, ingress

        new_ray.direction = wo;
        new_ray.inv_direction = 1.0 / wo;

        if (path.from_camera) {
            next_ray.c_importance = c_p;
            ray.l_importance = l_p;
            new_ray.tot_importance = ray.tot_importance * new_ray.c_importance;
        } else {
            next_ray.l_importance = l_p;
            ray.c_importance = c_p;
            new_ray.tot_importance = ray.tot_importance * new_ray.l_importance;
        }

        if (f == 0.0)
            break;

        path.rays[i] = ray;
        path.length = i + 1;

        ray = new_ray;
        new_ray = next_ray;
    }

    output_paths[id] = path;

    for (int i = 0; i < path.length; i++) {
        if (path.rays[i].hit_light >= 0) {
            out[id] = float4(path.rays[i - 1].color / path.rays[i].tot_importance, 1);
            break;
        }
    }

    random_buffer[2 * id] = seed0;
    random_buffer[2 * id + 1] = seed1;
}

float geometry_term(const thread Ray &a, const thread Ray &b){
    float dist = length(b.origin - a.origin);
    // Veach's geometry term has cosines in the numerator, but I include those in f, so just 1 here.
    return 1.0 / (dist * dist);
}

Ray get_ray(const thread Path &camera_path, const thread Path &light_path, const thread int t, const thread int s, const thread int i){
    if (i < s) {return light_path.rays[i];}
    else {return camera_path.rays[t + s - i - 1];}
}

float3 pixel_center(const thread Camera camera, const thread int x, const thread int y){

    float x_normalized = (x - 0.5 * camera.pixel_width) / (float)camera.pixel_width;
    float y_normalized = (y - 0.5 * camera.pixel_height) / (float)camera.pixel_height;

    float3 x_vector = x_normalized * camera.phys_width * camera.dx;
    float3 y_vector = y_normalized * camera.phys_height * camera.dy;

    float3 origin = camera.center + x_vector + y_vector;

    return origin;
}

float gaussian_weight(const thread float3 p, const thread float3 q, const thread float sigma){
    float dist = length(p - q);
    return exp(-dist * dist / (2.0 * sigma * sigma));
}

void world_ray_to_camera_ray(const device Box *boxes,
                            const device Triangle *triangles,
                            const device Material *materials,
                            const thread Camera camera,
                            const thread Ray world_ray,
                            thread int &pixel_idx,
                            thread Ray &camera_ray) {

    if (materials[triangles[world_ray.triangle].material].type > 0)
        return;

    Ray test_ray;
    test_ray.origin = world_ray.origin;
    test_ray.direction = normalize(camera.focal_point - world_ray.origin);
    if (dot(test_ray.direction, camera.direction) > 0.0)
        return;
    test_ray.inv_direction = 1.0 / test_ray.direction;
    test_ray.triangle = world_ray.triangle;
    test_ray.normal = world_ray.normal;

    int best_i = -1;
    float best_t = INFINITY;
    float u, v;
    traverse_bvh(test_ray, boxes, triangles, best_i, best_t, u, v);
    if (best_i == -1)
        return;
    if (!triangles[best_i].is_camera)
        return;

    // get pixel coordinates of point of intersection
    float3 camera_point = test_ray.origin + best_t * test_ray.direction;
    float x = dot(camera_point - camera.center, camera.dx);
    float y = dot(camera_point - camera.center, camera.dy);
    int pixel_x = (int)round((x / camera.phys_width + 0.5) * camera.pixel_width);
    int pixel_y = (int)round((y / camera.phys_height + 0.5) * camera.pixel_height);

    pixel_idx = pixel_y * camera.pixel_width + pixel_x;

    camera_ray.origin = camera_point;
    camera_ray.direction = normalize(camera.focal_point - camera_point);
    camera_ray.inv_direction = 1.0 / camera_ray.direction;
    camera_ray.normal = camera.direction;
    camera_ray.material = 7;
    camera_ray.color = float3(1.0);
    camera_ray.triangle = best_i;
    camera_ray.tot_importance = 1.0;
//    camera_ray.c_importance = 1.0;
//    camera_ray.l_importance = 1.0;
    camera_ray.hit_light = -1;
    camera_ray.hit_camera = best_i;
}


kernel void connect_paths(const device Path *camera_paths [[ buffer(0) ]],
                          const device Path *light_paths [[ buffer(1) ]],
                          const device Triangle *triangles [[ buffer(2) ]],
                          const device Material *materials [[ buffer(3) ]],
                          const device Box *boxes [[ buffer(4) ]],
                          const device Camera *camera [[ buffer(5) ]],
                          device WeightAggregator *weight_aggregators [[ buffer(6) ]],
                          device float4 *out [[ buffer(7) ]],
                          device int *light_pixel_indices [[ buffer(8) ]],
                          device int *light_path_indices [[ buffer(9) ]],
                          device int *light_ray_indices [[ buffer(10) ]],
                          device float *light_weights [[ buffer(11) ]],
                          device float *light_shade [[ buffer(12) ]],
                          uint id [[ thread_position_in_grid ]]) {

    Path camera_path = camera_paths[id];
    Path light_path = light_paths[id];

    out[id] = float4(0.0);
    Camera c = camera[0];

    WeightAggregator aggregator;
    aggregator.total_contribution = float3(0.0);
    int pixel_idx = camera_path.rays[0].pixel_idx;
    int light_pixel_idx = -1;
    int total_pixels = c.pixel_width * c.pixel_height;
    for (int i = 0; i < 8; i++) {
        light_pixel_indices[id + i * total_pixels] = -1;
        light_path_indices[id + i * total_pixels] = -1;
        light_ray_indices[id + i * total_pixels] = -1;
        light_weights[id + i * total_pixels] = 0.0;
        light_shade[id + i * total_pixels] = 0.0;
    }
    float contrib_weight_sum = 0.0;

    for (int t = 1; t < camera_path.length + 1; t++) {
        for (int s = 0; s < light_path.length + 1; s++) {

            if (t + s < 2) {continue;}

            // reset
            Ray light_ray;
            light_ray.triangle = -1;
            Ray camera_ray;
            camera_ray.triangle = -1;
            Path camera_path = camera_paths[id];
            Path light_path = light_paths[id];
            float3 dir_l_to_c;
            light_pixel_idx = -1;

            if (s == 0) {
                // camera ray hits a light source
                camera_ray = camera_path.rays[t - 1];
                if (camera_ray.hit_light < 0) {continue;}
            }
            else if (t == 1) {
                // projection onto camera plane
                light_ray = light_path.rays[s - 1];
                world_ray_to_camera_ray(boxes, triangles, materials, c, light_ray, light_pixel_idx, camera_path.rays[0]);
                if (light_pixel_idx == -1) {continue;}
                camera_ray = camera_path.rays[0];
                dir_l_to_c = normalize(camera_ray.origin - light_ray.origin);
            }
            else {
                // regular join
                camera_ray = camera_path.rays[t - 1];
                light_ray = light_path.rays[s - 1];

                // skip specular joins
                if (materials[light_ray.material].type > 0) {continue;}
                if (materials[camera_ray.material].type > 0) {continue;}

                dir_l_to_c = normalize(camera_ray.origin - light_ray.origin);

                // backface culling
                if (dot(light_ray.normal, dir_l_to_c) < DELTA) {continue;}
                if (dot(camera_ray.normal, -dir_l_to_c) < DELTA) {continue;}

                if (not visibility_test(light_ray, camera_ray, boxes, triangles)) {continue;}
                if (light_ray.triangle == camera_ray.triangle) {continue;}
            }

            float p_ratios[32];
            float p_values[32];

            // populate missing values, these will be reset next loop so it's fine
            if (s == 0) {
                camera_path.rays[t - 1].l_importance = light_path.rays[0].l_importance;
            } else if (t == 1) {
                light_path.rays[s - 1].c_importance = camera_path.rays[0].c_importance;
                camera_path.rays[t - 1].l_importance = abs(dot(light_ray.normal, dir_l_to_c)) / PI;
            } else {
                camera_path.rays[t - 1].l_importance = abs(dot(light_ray.normal, dir_l_to_c)) / PI;
                light_path.rays[s - 1].c_importance = abs(dot(camera_ray.normal, -dir_l_to_c)) / PI;
            }

            // set up p_ratios like p1/p0, p2/p1, p3/p2, ... out to pk+1/pk, where k = s + t - 1
            for (int i = 0; i < s + t; i++) {
                float num, denom;
                if (i == 0) {
                    Ray a = get_ray(camera_path, light_path, t, s, 0);
                    Ray b = get_ray(camera_path, light_path, t, s, 1);

                    num = a.l_importance;
                    denom = a.c_importance * geometry_term(a, b);
                }
                else if (i == s + t - 1) {
                    Ray a = get_ray(camera_path, light_path, t, s, s + t - 1);
                    Ray b = get_ray(camera_path, light_path, t, s, s + t - 2);

                    num = a.l_importance * geometry_term(a, b);
                    denom = a.c_importance;
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

            float prior_camera_importance = camera_ray.tot_importance;

            float prior_light_importance;
            if (s == 0) {prior_light_importance = 1.0;}
            else {prior_light_importance = light_ray.tot_importance;}

            float p_s = prior_camera_importance * prior_light_importance;

            float p_i = p_s;
            for (int i = s; i < s + t + 1; i++) {
                p_values[i + 1] = p_ratios[i] * p_i;
                p_i = p_values[i + 1];
            }

            p_i = p_s;
            for (int i = s - 1; i >= 0; i--) {
                p_values[i] = p_i / p_ratios[i];
                p_i = p_values[i];
            }

            p_values[s] = p_s;

            for (int i = 0; i < s + t; i++) {
                if (materials[get_ray(camera_path, light_path, t, s, i).material].type > 0) {
                    p_values[i] = 0.0;
                    p_values[i + 1] = 0.0;
                }
            }

            float sum = 0.0;
            for (int i = 0; i < s + t + 1; i++)
                sum += p_values[i];

            float w;
            if (p_values[s] > 0.0 && sum > 0.0)
                w = p_values[s] / sum;
            else
                continue;

            float3 color = float3(1.0);
            float g = 1.0;
            float new_light_f = 1.0;
            float new_camera_f = 1.0;

            if (s == 0) {
                float3 prior_color = camera_path.rays[t - 2].color;
                float3 emission = materials[camera_ray.material].emission;
                color = prior_color * emission;
            } else if (t == 1) {
                int prior_light_ind = max(0, s - 2);
                float3 prior_color = light_path.rays[prior_light_ind].color;
                if (s != 1)
                    new_light_f = abs(dot(dir_l_to_c, light_ray.normal)) / PI;
                color = prior_color * new_light_f * materials[light_ray.material].color;
                g = geometry_term(light_ray, camera_ray);
            } else {
                float3 prior_camera_color = camera_path.rays[t - 2].color;
                Material camera_material = materials[camera_ray.material];
                new_camera_f = abs(dot(-dir_l_to_c, camera_ray.normal)) / PI;
                float3 camera_color = prior_camera_color * new_camera_f * camera_material.color;

                float3 light_color;
                if (s == 1) {
                    light_color = materials[light_ray.material].emission;
                }
                else {
                    float3 prior_light_color = light_path.rays[s - 2].color;
                    Material light_material = materials[light_ray.material];
                    new_light_f = abs(dot(dir_l_to_c, light_ray.normal)) / PI;
                    light_color = prior_light_color * new_light_f * light_material.color;
                }
                color = camera_color * light_color;
                g = geometry_term(camera_ray, light_ray);
            }
            if (t != 1)
                aggregator.total_contribution += w * g * color / p_s;
            else {
                light_pixel_indices[id + s * total_pixels] = light_pixel_idx;
                light_path_indices[id + s * total_pixels] = id;
                light_ray_indices[id + s * total_pixels] = s;
                light_weights[id + s * total_pixels] = w;
                light_shade[id + s * total_pixels] = g / p_s;
            }
            contrib_weight_sum += w;
        }
    }

    // prep weights for final gather based on exact ray position relative to nearby pixels
    float weight_sum = 0.0;
    float pixel_phys_width = c.phys_width / c.pixel_width;
    float pixel_phys_height = c.phys_height / c.pixel_height;
    float sigma = 0.5f * sqrt(pixel_phys_width * pixel_phys_width + pixel_phys_height * pixel_phys_height);

    // zero weights
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            aggregator.weights[i][j] = 0.0;

    // calculate and sum weights
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int new_sample_x = (pixel_idx % c.pixel_width) + i;
            int new_sample_y = (pixel_idx / c.pixel_width) + j;

            if (new_sample_x < 0 || new_sample_x >= c.pixel_width ||
                new_sample_y < 0 || new_sample_y >= c.pixel_height) {
                continue;
            }

            int new_sample_index = new_sample_y * c.pixel_width + new_sample_x;
            if (new_sample_index < 0 || new_sample_index >= c.pixel_width * c.pixel_height) {continue;}

            float weight = gaussian_weight(pixel_center(c, new_sample_x, new_sample_y), camera_paths[id].rays[0].origin, sigma);
            aggregator.weights[i + 1][j + 1] = weight;
            weight_sum += weight;
        }
    }

    // normalize weights
    if (weight_sum != 0.0)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                aggregator.weights[i][j] = aggregator.weights[i][j] / weight_sum;

    aggregator.contrib_weight_sum = contrib_weight_sum;

    // write outputs
    out[id] = float4(aggregator.total_contribution, 1.0);
    weight_aggregators[id] = aggregator;
}


kernel void light_image_gather(const device Path *light_paths [[ buffer(0) ]],
                               const device Material *materials [[ buffer(1) ]],
                               const device int32_t *path_indices [[ buffer(2) ]],
                               const device int32_t *ray_indices [[ buffer(3) ]],
                               const device int32_t *bins [[ buffer(4) ]],
                               const device float *weights [[ buffer(5) ]],
                               const device float *shades [[ buffer(6) ]],
                               device float4 *light_image [[ buffer(7) ]],
                               device float *sum_weights [[ buffer(8) ]],
                               uint id [[ thread_position_in_grid ]]) {
    int start_idx = bins[id];
    int end_idx = bins[id + 1];
    float3 total_contribution = float3(0.0);
    float weight_sum = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        int path_idx = path_indices[i];
        int ray_idx = ray_indices[i];
        Path path = light_paths[path_idx];
        Ray ray = path.rays[ray_idx];
        Ray prior_ray = path.rays[max(0, ray_idx - 1)];
        Material mat = materials[ray.material];
        total_contribution += weights[i] * shades[i] * prior_ray.color * mat.color;
        weight_sum += weights[i];
    }
    light_image[id] = float4(total_contribution, 1.0);
    sum_weights[id] += weight_sum;
}


kernel void adaptive_finalize_samples(const device WeightAggregator *weight_aggregators [[ buffer(0) ]],
                             const device Camera *camera_buffer [[ buffer(1) ]],
                             device float4 *out [[ buffer(2) ]],
                             device uint32_t *sample_counts [[ buffer(3) ]],
                             const device uint32_t *sample_bin_offsets [[ buffer(4) ]],
                             device float *sample_weights [[ buffer(5) ]],
                             uint id [[ thread_position_in_grid ]]) {
    Camera camera = camera_buffer[0];
    out[id] = float4(0.0);
    float3 total_sample = float3(0.0);
    sample_counts[id] = 0;
    sample_weights[id] = 0.0;
    float weight_sum = 0.0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int sample_x = (id % camera.pixel_width) + i;
            int sample_y = (id / camera.pixel_width) + j;

            if (sample_x < 0 || sample_x >= camera.pixel_width ||
                sample_y < 0 || sample_y >= camera.pixel_height) {
                continue;
            }

            int sample_index = sample_y * camera.pixel_width + sample_x;
            if (sample_index < 0 || sample_index >= camera.pixel_width * camera.pixel_height) {continue;}

            for (uint32_t k = sample_bin_offsets[sample_index]; k < sample_bin_offsets[sample_index + 1]; k++) {
                float weight = weight_aggregators[k].weights[1 - i][1 - j];
                total_sample += weight * weight_aggregators[k].total_contribution;
                weight_sum += weight * weight_aggregators[k].contrib_weight_sum;
            }
        }
    }
    sample_counts[id] = sample_bin_offsets[id + 1] - sample_bin_offsets[id];
    out[id] = float4(total_sample, 1.0);
    sample_weights[id] = weight_sum;
 }

kernel void generate_camera_rays(const device Camera *camera [[ buffer(0) ]],
                                 device unsigned int *random_buffer [[ buffer(1) ]],
                                 device uint32_t *indices [[ buffer(2) ]],
                                 device Ray *out [[ buffer(3) ]],
                                 uint id [[ thread_position_in_grid ]]) {
    Camera c = camera[0];
    Ray ray;

    uint seed0 = random_buffer[2 * id];
    uint seed1 = random_buffer[2 * id + 1];

    float x_offset = xorshift_random(seed0);
    float y_offset = xorshift_random(seed1);

    int pixel_idx = indices[id];

    int pixel_x = pixel_idx % c.pixel_width;
    int pixel_y = pixel_idx / c.pixel_width;

    float x_normalized = (pixel_x + x_offset - 0.5 * c.pixel_width) / (float)c.pixel_width;
    float y_normalized = (pixel_y + y_offset - 0.5 * c.pixel_height) / (float)c.pixel_height;

    float3 x_vector = x_normalized * c.dx * c.phys_width;
    float3 y_vector = y_normalized * c.dy * c.phys_height;

    float3 origin = c.center + x_vector + y_vector;
    float3 direction = normalize(c.focal_point - origin);
    ray.origin = origin;
    ray.direction = direction;
    ray.normal = c.direction;
    ray.inv_direction = 1.0 / direction;
    ray.color = float3(1.0);

    ray.material = 7;
    ray.triangle = -1;
    ray.hit_light = -1;
    ray.hit_camera = -1;
    ray.from_camera = 1;
    ray.c_importance = 1.0 / (c.phys_width * c.phys_height);
    ray.l_importance = 1.0; // filled in later
    ray.tot_importance = ray.c_importance;

    ray.pixel_idx = pixel_idx;

    out[id] = ray;
    random_buffer[2 * id] = seed0;
    random_buffer[2 * id + 1] = seed1;
}


kernel void generate_light_rays(const device Triangle *light_triangles [[buffer(0) ]],
                                const device float *surface_areas [[buffer(1) ]],
                                const device int32_t *light_triangle_indices [[buffer(2) ]],
                                const device Material *materials[[buffer(3) ]],
                                device unsigned int *random_buffer [[buffer(4) ]],
                                device Ray *out [[buffer(5) ]],
                                const device int32_t *counts [[buffer(6) ]],
                                uint id [[thread_position_in_grid]]) {
    Ray ray;
    ray.from_camera = 0;
    ray.hit_light = -1;
    ray.hit_camera = -1;

    unsigned int seed0 = random_buffer[2 * id];
    unsigned int seed1 = random_buffer[2 * id + 1];

    int light_count = counts[0];
    int light_index = (int)(xorshift_random(seed0) * light_count);
    Triangle light_triangle = light_triangles[light_index];
    float surface_area = surface_areas[light_index];

    float u = xorshift_random(seed0);
    float v = xorshift_random(seed1);
    if (u + v > 1.0) {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    float w = 1.0 - u - v;

    ray.normal = light_triangle.normal;
    ray.origin = light_triangle.v0 * u + light_triangle.v1 * v + light_triangle.v2 * w + DELTA * ray.normal;

    float3 x, y;
    orthonormal(ray.normal, x, y);

    float rand_x = xorshift_random(seed0);
    float rand_y = xorshift_random(seed1);
    float2 random_roll = float2(rand_x, rand_y);
    ray.direction = random_hemisphere_uniform(x, y, ray.normal, random_roll);
    ray.inv_direction = 1.0 / ray.direction;

    ray.material = light_triangle.material;
    Material material = materials[ray.material];
    ray.color = material.emission;

    ray.triangle = light_triangle_indices[light_index];

    ray.c_importance = 1.0; // filled in later
    ray.l_importance = 1.0 / (light_count * surface_area);
    ray.tot_importance = ray.l_importance;

    out[id] = ray;
    random_buffer[2 * id] = seed0;
    random_buffer[2 * id + 1] = seed1;
}
