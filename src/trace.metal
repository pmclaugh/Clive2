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
    int32_t pad;
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
    int32_t pad[2];
};


struct Material {
    float3 color;
    float3 emission;
    int32_t type;
    float alpha;
    float ior;
    int32_t pad;
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


void ray_triangle_intersect(const thread Ray &ray, const thread Triangle &triangle, thread bool &hit, thread float &t_out, thread float& u, thread float& v) {
    float3 edge1 = triangle.v1 - triangle.v0;
    float3 edge2 = triangle.v2 - triangle.v0;
    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);
    if (abs(a) < 0.000001f) {
        hit = false;
        return;
    }
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
    float3 direction = (b.origin + b.normal + 0.0001f) - (a.origin + a.normal * 0.0001f);
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
    float phi = 2 * PI * rand.x;
    float theta = atan(alpha * sqrt(rand.y) / sqrt(1.0f - rand.y));
    return normalize(cos(phi) * sin(theta) * x_axis + sin(phi) * sin(theta) * y_axis + cos(theta) * z_axis);
}

float3 specular_reflection(const thread float3 &i, const thread float3 &m) {
    // in this function, i is incident
    return normalize(i - 2 * dot(i, m) * m);
}

float3 specular_reflect_half_direction(const thread float3 &i, const thread float3 &o) {
    // in this function, i is wi
    return normalize(i + o);
}

float3 GGX_transmit(const thread float3 &i, const thread float3 &m, const thread float ni, const thread float no) {
    // in this function, i is incident
    float cosTheta_i = dot(i, m);
    float eta = ni / no;
    float cosTheta_t = sqrt(1 + eta * eta * (cosTheta_i * cosTheta_i - 1));
    return normalize((eta * cosTheta_i - cosTheta_t) * m - eta * i);
}

float3 specular_transmit_half_direction(const thread float3 &i, const thread float3 &o, const thread float ni, const thread float no) {
    // in this function, i is incident
    return normalize(-(no * o + ni * i));
}

float degreve_fresnel(const thread float3 &i, const thread float3 &m, const thread float ni, const thread float nt) {
    // this function is agnostic about i being incident or wi
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
    // this function is agnostic about i being incident or wi
    float mv = dot(m, v);
    float sin2 = 1.0f - mv * mv;
    float tan2 = sin2 / (mv * mv);
    return 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * tan2));
}

float GGX_G(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float alpha) {
    // this function is agnostic about i being incident or wi
    return GGX_G1(i, m, n, alpha) * GGX_G1(o, m, n, alpha);
}

float GGX_D(const thread float3 &m, const thread float3 &n, const thread float alpha) {
    // this function is agnostic about i being incident or wi
    float cosTheta = dot(m, n);
    float cosTheta2 = cosTheta * cosTheta;
    float alpha2 = alpha * alpha;
    float denom =  1 + cosTheta2 * (alpha2 - 1);

    return alpha2 / (PI * denom * denom);
}

float reflect_jacobian(const thread float3 &m, const thread float3 &o) {
    return 1.0f / (4.0f * abs(dot(m, o)));
}

float transmit_jacobian(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float ni, const thread float no) {
    float cosTheta_i = abs(dot(i, m));
    float cosTheta_o = abs(dot(o, m));
    float numerator = no * no * cosTheta_o;
    float denominator = (ni * cosTheta_i - no * cosTheta_o) * (ni * cosTheta_i - no * cosTheta_o);
    return numerator / denominator;
}

float GGX_BRDF_reflect(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float ni, const thread float no, const thread float alpha) {
    float D = GGX_D(m, n, alpha);
    float G = GGX_G(i, o, m, n, alpha);
    float F = degreve_fresnel(i, m, ni, no);

    return abs(dot(o, m)) * (D * G * F) / (4 * abs(dot(i, n)) * abs(dot(o, n)));
}

float GGX_BRDF_transmit(const thread float3 &i, const thread float3 &o, const thread float3 &m, const thread float3 &n, const thread float ni, const thread float no, const thread float alpha) {
    float D = GGX_D(m, n, alpha);
    float G = GGX_G(i, o, m, n, alpha);
    float F = degreve_fresnel(i, m, ni, no);

    float im = abs(dot(i, m));
    float om = abs(dot(o, m));
    float in = abs(dot(i, n));
    float on = abs(dot(o, n));

    float coeff = (im * om) / (in * on);
    float num = no * no * D * G * (1 - F);
    float denom = (ni * im - no * om) * (ni * im - no * om);

    return abs(dot(o, m)) * coeff * num / denom;
}

float BRDF(const thread float3 &i, const thread float3 &o, const thread float3 &n, const thread float3 &geom_n, const thread Material material) {
    if (material.type == 0) {
        return max(0.0f, dot(o, n));
    }
    else {
        float ni, no, alpha;
        alpha = material.alpha;
        if (dot(i, n) > 0) {
            ni = 1.0f;
            no = material.ior;
        }
        else {
            ni = material.ior;
            no = 1.0f;
        }
        if (dot(i, n) * dot(o, n) > 0 && dot(i, geom_n) * dot(o, geom_n) > 0) {
            float3 m = specular_reflect_half_direction(i, o);
            if (dot(m, n) <= 0.0f || dot(m, geom_n) <= 0.0f) {return 0.0f;}
            if (dot(i, m) * dot(o, m) <= 0.0f) {return 0.0f;}
            return GGX_BRDF_reflect(i, o, m, n, ni, no, alpha);
        }
        else if (dot(i, n) * dot(o, n) < 0 && dot(i, geom_n) * dot(o, geom_n) < 0) {
            float3 m = specular_transmit_half_direction(-i, o, ni, no);
            if (dot(-i, o) <= 0.0f) {return 0.0f;}
            if (dot(m, n) <= 0.0f || dot(m, geom_n) <= 0.0f) {return 0.0f;}
            if (dot(i, m) <= 0.0f || dot(o, m) >= 0.0f) {return 0.0f;}
            return GGX_BRDF_transmit(i, o, m, n, ni, no, alpha);
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
                   const device float2 *random_buffer [[ buffer(4) ]],
                   device float4 *out [[ buffer(5) ]],
                   device Path *output_paths [[ buffer(6) ]],
                   device float4 *float_debug [[ buffer(7) ]],
                   uint id [[ thread_position_in_grid ]]) {
    Path path;
    path.length = 0;
    Ray ray, new_ray, next_ray;
    new_ray.l_importance = 1.0f;
    ray = rays[id];
    path.from_camera = ray.from_camera;
    out[id] = float4(0, 0, 0, 1);

    if (path.from_camera == 0) {
        new_ray.l_importance = 1.0f / (2.0f * PI);
        float3 x, y;
        orthonormal(ray.direction, x, y);
        float2 random_roll = random_buffer[id * 16 + 1];
        ray.normal = ray.direction;
        ray.direction = random_hemisphere_uniform(x, y, ray.direction, random_roll);
    }
    else {
        new_ray.c_importance = 1.0f;
    }

    for (int i = 0; i < 8; i++) {
        int best_i = -1;
        float best_t = INFINITY;
        float u, v;
        traverse_bvh(ray, boxes, triangles, best_i, best_t, u, v);

        if (best_i == -1) {
            break;
        }

        Triangle triangle = triangles[best_i];
        Material material = materials[triangle.material];
        float2 random_roll_a = random_buffer[id * 16 + 2 * i];
        float2 random_roll_b = random_buffer[id * 16 + 2 * i + 1];

        float3 n;
        float ni, no;
        float alpha = material.alpha;
        float3 sampled_normal = sample_normal(triangle, u, v);
        float3 signed_normal;
        if (dot(-ray.direction, triangle.normal) > 0.0f) {
            signed_normal = triangle.normal;
            n = sampled_normal;
            ni = 1.0f;
            no = material.ior;
        }
        else {
            signed_normal = -triangle.normal;
            n = -sampled_normal;
            ni = material.ior;
            no = 1.0f;
        }

        new_ray.origin = ray.origin + ray.direction * best_t;
        new_ray.normal = sampled_normal;
        new_ray.material = triangle.material;
        new_ray.triangle = best_i;
        if (triangle.is_light) {new_ray.hit_light = best_i;}
        else {new_ray.hit_light = -1;}

        float3 x, y;
        orthonormal(n, x, y);

        float3 wi, wo;
        wi = -ray.direction;

        float f, c_p, l_p;
        if (material.type == 0) {
            if (path.from_camera) {
                wo = random_hemisphere_cosine(x, y, n, random_roll_a);
                if (dot(n, wo) <= 0.0f || dot(signed_normal, wo) <= 0.0f) {break;}
                f = dot(n, wo) / PI;
                c_p = dot(n, wo) / PI;
                l_p = 1.0f / (2 * PI);
            }
            else {
                wo = random_hemisphere_uniform(x, y, n, random_roll_a);
                if (dot(n, wo) <= 0.0f || dot(signed_normal, wo) <= 0.0f) {break;}
                f = dot(n, wo) / PI;
                c_p = dot(n, wi) / PI;
                l_p = 1.0f / (2 * PI);
            }
        } else {
            float3 m = GGX_sample(x, y, n, random_roll_a, alpha);
            if (dot(m, n) <= 0.0f || dot(m, signed_normal) <= 0.0f) {break;}

            float fresnel = degreve_fresnel(wi, m, ni, no);
            float pf = 1.0f;

            if (random_roll_b.x <= fresnel) {
                wo = specular_reflection(-wi, m);
                f = GGX_BRDF_reflect(wi, wo, m, sampled_normal, ni, no, alpha);
                pf = fresnel;
                if (dot(wo, n) <= 0.0f || dot(wo, signed_normal) <= 0.0f) {break;}
            } else {
                wo = GGX_transmit(-wi, m, ni, no);
                f = GGX_BRDF_transmit(wi, wo, m, sampled_normal, ni, no, alpha);
                pf = 1.0 - fresnel;
                if (dot(wo, n) >= 0.0f || dot(wo, signed_normal) >= 0.0f) {break;}
            }
            float pm = abs(dot(m, n)) * GGX_D(m, n, alpha);
            float po;
            if (dot(wo, n) > 0.0f) {
                po = pf * pm * reflect_jacobian(m, wo);
            } else {
                po = pf * pm * transmit_jacobian(wi, wo, m, ni, no);
            }
            c_p = po;
            l_p = po;
        }

        new_ray.color = f * ray.color * material.color;

        if (f == 0.0f) {break;}

        new_ray.direction = wo;
        new_ray.inv_direction = 1.0 / wo;

        if (path.from_camera) {
            next_ray.c_importance = c_p;
            ray.l_importance = l_p;
            new_ray.tot_importance = ray.tot_importance * c_p;
        }
        else {
            next_ray.l_importance = l_p;
            ray.c_importance = c_p;
            new_ray.tot_importance = ray.tot_importance * l_p;
        }

        path.rays[i] = ray;
        path.length = i + 1;
        ray = new_ray;
        new_ray = next_ray;
    }
    output_paths[id] = path;

    for (int i = path.length - 1; i >= 0; i--){
        if (path.rays[i].hit_light >= 0){
            out[id] = float4(path.rays[i - 1].color / path.rays[i - 1].tot_importance, 1);
        }
    }
}


float geometry_term(const thread Ray &a, const thread Ray &b){
    float3 delta = b.origin - a.origin;
    float dist = length(delta);
    delta = delta / dist;

    float camera_cos, light_cos;
    camera_cos = abs(dot(a.normal, delta));
    light_cos = abs(dot(b.normal, delta));

    float g = camera_cos * light_cos / (dist * dist);
    return max(DELTA, g);
}


Ray get_ray(const thread Path &camera_path, const thread Path &light_path, const thread int t, const thread int s, const thread int i){
    if (i < s) {return light_path.rays[i];}
    else {return camera_path.rays[t + s - i - 1];}
}


kernel void connect_paths(const device Path *camera_paths [[ buffer(0) ]],
                          const device Path *light_paths [[ buffer(1) ]],
                          const device Triangle *triangles [[ buffer(2) ]],
                          const device Material *materials [[ buffer(3) ]],
                          const device Box *boxes [[ buffer(4) ]],
                          device float4 *out [[ buffer(5) ]],
                          device float *float_debug [[ buffer(6) ]],
                          device Path *output_paths [[ buffer(7) ]],
                          uint id [[ thread_position_in_grid ]]) {
    Path camera_path = camera_paths[id];
    Path light_path = light_paths[id];
    float3 sample = float3(0.0f);
    float p_ratios[32];
    int sample_count = 0;

    for (int t = 0; t < camera_path.length + 1; t++){
        for (int s = 0; s < light_path.length + 1; s++){
            Ray light_ray;
            Ray camera_ray;

            if (t == 0){
                // this is where a light ray hits the camera plane. not yet supported.
                continue;
            }
            else if (s == 0) {
                // this is where a camera ray hits the light source.
                camera_ray = camera_path.rays[t - 1];
                if (camera_ray.hit_light < 0){
                    continue;
                }
            }
            else if (t == 1) {
                // light ray visibility to camera plane. not yet supported.
                continue;
            }
            else {
                light_ray = light_path.rays[s - 1];
                camera_ray = camera_path.rays[t - 1];

                float3 dir_l_to_c = camera_ray.origin - light_ray.origin;
                float dist_l_to_c = length(dir_l_to_c);
                dir_l_to_c = dir_l_to_c / dist_l_to_c;

                if (abs(dot(light_ray.normal, dir_l_to_c)) < DELTA){continue;}
                if (abs(dot(camera_ray.normal, -dir_l_to_c)) < DELTA){continue;}
                if (not visibility_test(light_ray, camera_ray, boxes, triangles)){continue;}

                sample_count++;
            }

            for (int i = 0; i < 32; i++){
                p_ratios[i] = 1.0f;
            }

            // set up p_ratios like p1/p0, p2/p1, p3/p2, ... out to pk+1/pk, where k = s + t - 1
            for (int i = 0; i < s + t; i++){
                float num, denom;
                if (i == 0){
                    Ray a = get_ray(camera_path, light_path, t, s, i);
                    Ray b = get_ray(camera_path, light_path, t, s, i + 1);
                    num = a.l_importance;
                    denom = a.c_importance * geometry_term(a, b);
                }
                else if (i == s + t - 1) {
                    Ray a = get_ray(camera_path, light_path, t, s, i);
                    Ray b = get_ray(camera_path, light_path, t, s, i - 1);
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

            float prior_camera_importance = t > 1 ? camera_path.rays[t - 2].tot_importance : camera_path.rays[0].c_importance;
            float prior_light_importance = s > 1 ? light_path.rays[s - 2].tot_importance : light_path.rays[0].l_importance;
            float p_s = prior_camera_importance * prior_light_importance;
            float p_i = p_s;
            for (int i = s; i < s + t; i++){
                p_ratios[i] = p_ratios[i] * p_i;
                p_i = p_ratios[i];
            }
            p_i = p_s;
            for (int i = s - 1; i >= 0; i--){
                p_ratios[i] = p_i / p_ratios[i];
                p_i = p_ratios[i];
            }

            float sum = p_s;
            for (int i = 0; i < s + t; i++) {sum += p_ratios[i];}

            float w = p_s / sum;

            if (s == 0) {
                Ray prior_camera_ray = camera_path.rays[t - 2];
                sample += w * (prior_camera_ray.color) / (prior_camera_ray.tot_importance);
            }
            else {
                float3 dir_l_to_c = normalize(camera_ray.origin - light_ray.origin);

                float3 prior_camera_color = t > 1 ? camera_path.rays[t - 2].color : float3(1.0f);
                float3 prior_light_color = s > 1 ? light_path.rays[s - 2].color : float3(1.0f);

                float3 prior_camera_direction = t > 1 ? camera_path.rays[t - 2].direction : camera_path.rays[0].direction;
                float3 prior_light_direction = s > 1 ? light_path.rays[s - 2].direction : light_path.rays[0].direction;

                Material camera_material = materials[camera_ray.material];
                float3 camera_geom_normal = triangles[camera_ray.triangle].normal;
                float new_camera_f = BRDF(-prior_camera_direction, -dir_l_to_c, camera_ray.normal, camera_geom_normal, camera_material);
                float3 camera_color = prior_camera_color * new_camera_f * camera_material.color;

                Material light_material = materials[light_ray.material];
                float3 light_geom_normal = triangles[light_ray.triangle].normal;
                float new_light_f = BRDF(-prior_light_direction, dir_l_to_c, light_ray.normal, light_geom_normal, light_material);
                float3 light_color = prior_light_color * new_light_f * light_material.color;

                // float prior_camera_importance = t > 1 ? camera_path.rays[t - 2].tot_importance : camera_path.rays[0].c_importance;
                // float prior_light_importance = s > 1 ? light_path.rays[s - 2].tot_importance : light_path.rays[0].l_importance;

                sample += w * (geometry_term(light_ray, camera_ray) * camera_color * light_color) / (prior_camera_importance * prior_light_importance);
            }
        }
    }
    out[id] = float4(100.0f * sample, 1.0f);
}
