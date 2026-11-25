#[compute]
#version 450
#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 0, binding = 0, std140) uniform Scene {
	mat4 projection, inverse_projection, previous_clip, previous_view, rotation;
	vec4 sun_direction_radius, sun;
	uint node_count, frame, history;
	float intensity;
}
scene;
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;
#if defined(TRACE) || defined(RESOLVE)
layout(set = 0, binding = 16) uniform sampler2D ambient_occlusion;
layout(set = 0, binding = 2) uniform sampler2D normals;
vec3 position_at(ivec2 pixel, float depth) {
	vec2 uv = (vec2(pixel) + 0.5) / vec2(textureSize(depth_buffer, 0));
	vec4 h = scene.inverse_projection * vec4(uv * 2.0 - 1.0, depth, 1.0);
	return h.xyz / h.w;
}
#endif

float pack_direction(vec3 n) {
	n /= dot(abs(n), vec3(1.0));
	vec2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * mix(vec2(-1), vec2(1), greaterThanEqual(n.xy, vec2(0)));
	vec2 q = round((e * 0.5 + 0.5) * 31.0);
	return q.x + q.y * 32.0;
}
vec3 unpack_direction(float p) {
	p = mod(p, 1024.0);
	vec2 e = vec2(mod(p, 32.0), floor(p / 32.0)) / 31.0 * 2.0 - 1.0;
	vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
	if (n.z < 0.0) {
		n.xy = (1.0 - abs(n.yx)) * mix(vec2(-1), vec2(1), greaterThanEqual(n.xy, vec2(0)));
	}
	return normalize(n);
}
#if defined(TRACE) || defined(FILTER)
layout(set = 0, binding = 6) uniform sampler2D prev_geom;
bool previous_uv(vec3 p, out vec2 uv) {
	vec4 clip = scene.previous_clip * vec4(p, 1);
	uv = clip.xy / clip.w * 0.5 + 0.5;
	if (scene.history == 0u || clip.w <= 0.0 || any(lessThan(uv, vec2(0))) || any(greaterThanEqual(uv, vec2(1)))) {
		return false;
	}
	vec4 prior = textureLod(prev_geom, uv, 0.0);
	float z = (scene.previous_view * vec4(p, 1)).z;
	return prior.z < 0.0 && abs(prior.z - z) < max(0.12, abs(z) * 0.012);
}
#endif
#ifdef TRACE
struct Node {
	vec3 minimum;
	uint end;
	vec3 maximum;
	uint leaf;
	vec4 to_box[3];
};
struct Material {
	vec4 diffuse;
	vec4 emission;
	uvec4 geometry;
};
layout(set = 0, binding = 3, std430) readonly buffer Nodes {
	Node nodes[];
};
layout(set = 0, binding = 4, std430) readonly buffer Materials {
	Material materials[];
};
layout(set = 0, binding = 5) uniform sampler2D previous_color;
layout(set = 0, binding = 7, rgba16f) uniform writeonly image2D samples_out;
layout(set = 0, binding = 8, rgba32f) uniform writeonly image2D geometry_out;

vec3 inverse_direction(vec3 d) {
	return mix(vec3(-1), vec3(1), greaterThanEqual(d, vec3(0))) / max(abs(d), vec3(1e-7));
}
vec2 bounds(vec3 a, vec3 b) {
	vec3 lo = min(a, b), hi = max(a, b);
	return vec2(max(lo.x, max(lo.y, lo.z)), min(hi.x, min(hi.y, hi.z)));
}
struct TriangleNode {
	vec3 a;
	uint end;
	vec3 b;
	uint triangle;
	vec4 c;
};
layout(set = 0, binding = 17, std430) readonly buffer TriangleNodes {
	TriangleNode triangle_nodes[];
};

bool trace_mesh(vec3 origin, vec3 direction, uvec2 range, bool any_hit, inout float nearest, out vec3 normal) {
	vec3 inv = inverse_direction(direction);
	bool found = false;
	for (uint i = 0u; i < range.y;) {
		TriangleNode node = triangle_nodes[range.x + i];
		i++;
		if (node.triangle == 0u) {
			vec2 span = bounds((node.a - origin) * inv, (node.b - origin) * inv);
			if (span.x >= nearest || span.y < max(span.x, 0.0)) {
				i = node.end;
			}
			continue;
		}
		vec3 e1 = node.b - node.a, e2 = node.c.xyz - node.a;
		vec3 p = cross(direction, e2);
		float det = dot(e1, p);
		if (abs(det) <= 1e-8 * length(e1) * length(p)) {
			continue;
		}
		vec3 offset = origin - node.a;
		float u = dot(offset, p) / det;
		vec3 q = cross(offset, e1);
		float v = dot(direction, q) / det;
		float t = dot(e2, q) / det;
		if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.005 && t < nearest) {
			if (any_hit) {
				return true;
			}
			nearest = t;
			normal = cross(e1, e2);
			found = true;
		}
	}
	return found;
}

bool trace_proxies(vec3 origin, vec3 direction, bool any_hit, inout float nearest, out uint material, out vec3 normal) {
	vec3 inv = inverse_direction(direction);
	bool found = false;
	for (uint i = 0u; i < scene.node_count;) {
		Node node = nodes[i];
		vec2 span = bounds((node.minimum - origin) * inv, (node.maximum - origin) * inv);
		if (span.x >= nearest || span.y < max(span.x, 0.0)) {
			i = node.end;
			continue;
		}
		i++;
		if (node.leaf == 0u) {
			continue;
		}
		vec3 o, d;
		for (int k = 0; k < 3; k++) {
			o[k] = dot(node.to_box[k], vec4(origin, 1));
			d[k] = dot(node.to_box[k].xyz, direction);
		}
		vec3 face_normal;
		uint candidate = node.leaf - 1u;
		if (!trace_mesh(o, d, materials[candidate].geometry.xy, any_hit, nearest, face_normal)) {
			continue;
		}
		if (any_hit) {
			return true;
		}
		// Use the inverse transpose for nonuniformly scaled instances.
		normal = normalize(node.to_box[0].xyz * face_normal.x + node.to_box[1].xyz * face_normal.y + node.to_box[2].xyz * face_normal.z);
		normal = faceforward(normal, direction, normal);
		material = candidate;
		found = true;
	}
	return found;
}

bool proxies_occluded(vec3 origin, vec3 direction, float distance) {
	uint material;
	vec3 normal;
	return trace_proxies(origin, direction, true, distance, material, normal);
}

bool screen_trace(vec3 origin, vec3 dir, out vec3 hit) {
	float last = 0.0;
	for (int i = 1; i <= 12; i++) {
		float t = scene.sun_direction_radius.w * pow(float(i) / 12.0, 1.5);
		vec3 point = origin + dir * t;
		vec4 clip = scene.projection * vec4(point, 1);
		vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
		if (clip.w <= 0.0 || any(lessThan(uv, vec2(0))) || any(greaterThanEqual(uv, vec2(1)))) {
			break;
		}
		ivec2 pixel = ivec2(uv * vec2(textureSize(depth_buffer, 0)));
		float z = texelFetch(depth_buffer, pixel, 0).r;
		if (z > 0.0) {
			hit = position_at(pixel, z);
			float gap = hit.z - point.z;
			if (gap >= 0.0) {
				vec3 n = normalize(texelFetch(normals, pixel, 0).xyz * 2.0 - 1.0);
				return gap < 0.05 + (t - last) * abs(dir.z) && dot(n, dir) < -0.03 && dot(hit - origin, dir) > 0.03;
			}
		}
		last = t;
	}
	return false;
}
uint hash(uint x) {
	x ^= x >> 16u;
	x *= 0x7feb352du;
	x ^= x >> 15u;
	x *= 0x846ca68bu;
	return x ^ (x >> 16u);
}
void main() {
	ivec2 q = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(q, imageSize(samples_out)))) {
		return;
	}
	ivec2 pixel = min(q * 4 + 2, textureSize(depth_buffer, 0) - 1);
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	if (depth <= 0.0) {
		imageStore(geometry_out, q, vec4(0));
		imageStore(samples_out, q, vec4(0));
		return;
	}
	vec3 p = position_at(pixel, depth);
	vec3 n = normalize(texelFetch(normals, pixel, 0).xyz * 2.0 - 1.0);
	// A per-texel rotation of the R2 sequence spreads consecutive samples over
	// the hemisphere, even over the short window retained by the history filter.
	uint seed = hash(uint(q.x + q.y * imageSize(samples_out).x));
	uvec2 sequence = uvec2(seed, hash(seed + 1u)) + scene.frame * uvec2(0xc13fa9a9u, 0x91e10da5u);
	vec2 sample_uv = vec2(sequence >> 8u) * (1.0 / 16777216.0);
	float z = sample_uv.x;
	float phi = sample_uv.y * 6.2831853;
	vec3 tangent = normalize(cross(abs(n.z) < 0.99 ? vec3(0, 0, 1) : vec3(0, 1, 0), n));
	vec3 dir = z * n + sqrt(1.0 - z * z) * (cos(phi) * tangent + sin(phi) * cross(n, tangent));
	vec3 origin = p + n * max(0.03, -p.z * 0.001), hit, hit_normal;
	vec2 uv;
	vec3 light = vec3(0);
	float dist = scene.sun_direction_radius.w;
	uint material;
	if (screen_trace(origin, dir, hit) && previous_uv(hit, uv)) {
		light = textureLod(previous_color, uv, 0.0).rgb;
	} else if (trace_proxies(origin, dir, false, dist, material, hit_normal)) {
		hit = origin + dir * dist;
		vec4 clip = scene.projection * vec4(hit, 1);
		vec2 curr_uv = clip.xy / clip.w * 0.5 + 0.5;
		bool visible = false;
		if (clip.w > 0.0 && all(greaterThanEqual(curr_uv, vec2(0))) && all(lessThan(curr_uv, vec2(1)))) {
			ivec2 at = ivec2(curr_uv * vec2(textureSize(depth_buffer, 0)));
			float d = texelFetch(depth_buffer, at, 0).r;
			visible = d > 0.0 && abs(position_at(at, d).z - hit.z) < max(0.08, -hit.z * 0.008);
		}
		if (visible && previous_uv(hit, uv)) {
			light = textureLod(previous_color, uv, 0.0).rgb;
		} else {
			Material m = materials[material];
			float cosine = max(dot(hit_normal, scene.sun_direction_radius.xyz), 0.0);
			if (cosine > 0.0 && proxies_occluded(hit + hit_normal * 0.03, scene.sun_direction_radius.xyz, scene.sun_direction_radius.w)) {
				cosine *= 1.0 - scene.sun.a;
			}
			light = m.emission.rgb + m.diffuse.rgb * scene.sun.rgb * cosine;
		}
	}
	imageStore(samples_out, q, vec4(clamp(light, vec3(0), vec3(16)), pack_direction(mat3(scene.rotation) * dir)));
	vec3 world_normal = mat3(scene.rotation) * n;
	float ao = textureLod(ambient_occlusion, (vec2(pixel) + 0.5) / vec2(textureSize(depth_buffer, 0)), 0.0).r;
	imageStore(geometry_out, q, vec4(p, pack_direction(world_normal) + 1024.0 * round(31.0 * ao)));
}
#endif
#if defined(FILTER) || defined(RESOLVE)
layout(set = 0, binding = 8) uniform sampler2D geometry;
float surface_weight(vec4 center, vec4 neighbor) {
	if (neighbor.z >= 0.0) {
		return 0.0;
	}
	float normal_weight = smoothstep(0.65, 0.98, dot(unpack_direction(center.w), unpack_direction(neighbor.w)));
	float ao_delta = abs(mod(floor(center.w / 1024.0), 32.0) - mod(floor(neighbor.w / 1024.0), 32.0)) / 31.0;
	return exp2(-6.0 * ao_delta) * normal_weight * exp2(-abs(center.z - neighbor.z) / max(0.08, -center.z * 0.015));
}
#endif
#ifdef FILTER
layout(set = 0, binding = 18) uniform sampler2D old_statistics;
layout(set = 0, binding = 19, rg16f) uniform writeonly image2D new_statistics;
layout(set = 0, binding = 7) uniform sampler2D samples_in;
layout(set = 0, binding = 9) uniform sampler2D old_r;
layout(set = 0, binding = 10) uniform sampler2D old_g;
layout(set = 0, binding = 11) uniform sampler2D old_b;
layout(set = 0, binding = 12, rgba16f) uniform writeonly image2D new_r;
layout(set = 0, binding = 13, rgba16f) uniform writeonly image2D new_g;
layout(set = 0, binding = 14, rgba16f) uniform writeonly image2D new_b;
// Select a single compatible receiver so repeated reprojections do not blur SH detail.
bool find_history(vec4 center, out ivec2 previous_texel, out float offset) {
	if (scene.history == 0u || center.z >= 0.0) {
		return false;
	}
	vec4 clip = scene.previous_clip * vec4(center.xyz, 1.0);
	if (clip.w <= 0.0) {
		return false;
	}
	vec2 uv = 0.5 * (clip.xy / clip.w + 1.0);
	if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) {
		return false;
	}
	ivec2 full_size = textureSize(depth_buffer, 0);
	ivec2 coarse_size = textureSize(prev_geom, 0);
	// TRACE samples full-resolution pixel 4*q + 2 (pixel centers have a 0.5 offset).
	vec2 pixel = uv * vec2(full_size) - 0.5;
	ivec2 base = ivec2(floor((pixel - 2.0) * 0.25));
	vec3 receiver = (scene.previous_view * vec4(center.xyz, 1.0)).xyz;
	vec3 normal = unpack_direction(center.w);
	vec3 previous_normal = mat3(scene.previous_view) * transpose(mat3(scene.rotation)) * normal;
	float depth_limit = max(0.12, -receiver.z * 0.012);
	float plane_limit = max(0.02, -receiver.z * 0.005);
	float best = -1.0;
	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			ivec2 at = base + ivec2(x, y);
			if (any(lessThan(at, ivec2(0))) || any(greaterThanEqual(at, coarse_size))) {
				continue;
			}
			vec4 prior = texelFetch(prev_geom, at, 0);
			vec3 delta = prior.xyz - receiver;
			float alignment = dot(normal, unpack_direction(prior.w));
			float plane_distance = abs(dot(previous_normal, delta));
			if (prior.z >= 0.0 || abs(delta.z) >= depth_limit || plane_distance >= plane_limit || alignment <= 0.9) {
				continue;
			}
			vec2 d = (vec2(min(at * 4 + 2, full_size - 1)) - pixel) * 0.25;
			float distance_squared = dot(d, d);
			float score = distance_squared + (1.0 - alignment) + plane_distance / plane_limit;
			if (best < 0.0 || score < best) {
				best = score;
				previous_texel = at;
				offset = distance_squared;
			}
		}
	}
	return best >= 0.0;
}
void main() {
	ivec2 q = ivec2(gl_GlobalInvocationID.xy), size = textureSize(geometry, 0);
	if (any(greaterThanEqual(q, size))) {
		return;
	}
	vec4 center = texelFetch(geometry, q, 0);
	vec4 coeff[3] = vec4[3](vec4(0), vec4(0), vec4(0));
	float sum = 0.0;
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			ivec2 at = clamp(q + ivec2(x, y), ivec2(0), size - 1);
			float weight = surface_weight(center, texelFetch(geometry, at, 0));
			vec4 s = texelFetch(samples_in, at, 0);
			vec4 basis = vec4(0.5, unpack_direction(s.a));
			for (int c = 0; c < 3; c++) {
				coeff[c] += s[c] * basis * weight;
			}
			sum += weight;
		}
	}
	for (int c = 0; c < 3; c++) {
		coeff[c] /= max(sum, 0.001);
	}
	float variance = 0.0;
	float age = 1.0;
	ivec2 previous_texel;
	float offset;
	if (find_history(center, previous_texel, offset)) {
		vec4 prior_r = texelFetch(old_r, previous_texel, 0);
		vec4 prior_g = texelFetch(old_g, previous_texel, 0);
		vec4 prior_b = texelFetch(old_b, previous_texel, 0);
		vec2 statistics = texelFetch(old_statistics, previous_texel, 0).rg;
		float prior_variance = statistics.x;
		// A distant fallback carries less history into a newly exposed edge.
		float age_limit = mix(32.0, 8.0, smoothstep(0.5, 1.0, offset));
		age = min(statistics.y + 1.0, age_limit);
		// The constant SH coefficient already carries the temporal mean (basis = 0.5).
		const vec3 luma = vec3(0.2126, 0.7152, 0.0722);
		float prior_mean = dot(vec3(prior_r.x, prior_g.x, prior_b.x), luma);
		float current = dot(vec3(coeff[0].x, coeff[1].x, coeff[2].x), luma);
		float delta = current - prior_mean;
		float sigma = sqrt(prior_variance) + 0.02 + 0.1 * max(current, prior_mean);
		// Allow the variance estimate to settle before accelerating a lighting change.
		float change = smoothstep(3.0 * sigma, 6.0 * sigma, abs(delta)) * smoothstep(4.0, 12.0, age);
		age = mix(age, min(age, 4.0), change);
		float alpha = 1.0 / age;
		variance = (1.0 - alpha) * (prior_variance + alpha * delta * delta);
		coeff[0] = mix(prior_r, coeff[0], alpha);
		coeff[1] = mix(prior_g, coeff[1], alpha);
		coeff[2] = mix(prior_b, coeff[2], alpha);
	}
	imageStore(new_statistics, q, vec4(variance, age, 0.0, 0.0));
	imageStore(new_r, q, coeff[0]);
	imageStore(new_g, q, coeff[1]);
	imageStore(new_b, q, coeff[2]);
}
#endif
#ifdef RESOLVE
layout(set = 0, binding = 12) uniform sampler2D sh_r;
layout(set = 0, binding = 13) uniform sampler2D sh_g;
layout(set = 0, binding = 14) uniform sampler2D sh_b;
layout(set = 0, binding = 15, rgba16f) uniform writeonly image2D lighting;
void main() {
	ivec2 p = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(p, imageSize(lighting)))) {
		return;
	}
	float depth = texelFetch(depth_buffer, p, 0).r;
	vec3 n = normalize(texelFetch(normals, p, 0).xyz * 2.0 - 1.0);
	vec4 basis = vec4(1, mat3(scene.rotation) * n);
	float ao = textureLod(ambient_occlusion, (vec2(p) + 0.5) / vec2(imageSize(lighting)), 0.0).r;
	vec4 center = vec4(position_at(p, depth), pack_direction(basis.yzw) + 1024.0 * round(31.0 * ao));
	vec2 grid = (vec2(p) - 2.0) / 4.0;
	ivec2 base = ivec2(floor(grid)), size = textureSize(geometry, 0);
	vec2 f = fract(grid);
	vec3 light = vec3(0);
	float sum = 0.0;
	for (int y = 0; y <= 1; y++) {
		for (int x = 0; x <= 1; x++) {
			ivec2 at = clamp(base + ivec2(x, y), ivec2(0), size - 1);
			float w = (x == 1 ? f.x : 1.0 - f.x) * (y == 1 ? f.y : 1.0 - f.y) * surface_weight(center, texelFetch(geometry, at, 0));
			light += w * vec3(dot(texelFetch(sh_r, at, 0), basis), dot(texelFetch(sh_g, at, 0), basis), dot(texelFetch(sh_b, at, 0), basis));
			sum += w;
		}
	}
	imageStore(lighting, p, vec4(depth > 0.0 ? max(light / max(sum, 0.001), vec3(0)) * scene.intensity : vec3(0), 0));
}
#endif
