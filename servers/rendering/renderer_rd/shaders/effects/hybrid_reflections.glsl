#[compute]

#version 460

#VERSION_DEFINES

#ifdef HARDWARE_TRACE
#extension GL_EXT_ray_query : require
#endif

#ifdef SCREEN_TRACE
layout(local_size_x = 8, local_size_y = 8) in;
#else
layout(local_size_x = 64) in;
#endif

layout(set = 0, binding = 0) uniform sampler2D opaque_depth;
layout(set = 0, binding = 1) uniform sampler2D opaque_color;
layout(set = 0, binding = 2) uniform sampler2D surface_depth;
layout(set = 0, binding = 3) uniform sampler2D surface_normal;
layout(rgba16f, set = 0, binding = 4) uniform restrict image2D reflection_output;

layout(rg32f, set = 0, binding = 9) uniform restrict writeonly image2D surface_hit;

layout(set = 0, binding = 5, std140) uniform SceneData {
	mat4 projection[2];
	mat4 inv_projection[2];
	vec4 eye_offset[2];
}
scene;

#ifdef SCREEN_TRACE
#define QUEUE_ACCESS writeonly
#else
#define QUEUE_ACCESS readonly
#endif

layout(set = 0, binding = 6, std430) restrict QUEUE_ACCESS buffer Misses {
	uint pixels[];
}
misses;

#ifdef SCREEN_TRACE
layout(set = 0, binding = 7, std430) restrict buffer Dispatch {
#else
layout(set = 0, binding = 7, std430) restrict readonly buffer Dispatch {
#endif
	uint groups_x;
	uint groups_y;
	uint groups_z;
	uint count;
}
dispatch;

layout(push_constant, std430) uniform Params {
	uint view;
	uint node_count;
	bool orthogonal;
	uint pad;
}
params;

const float MAX_TRAVEL = 48.0;
const uint GROUPS_PER_ROW = 1024;

struct Surface {
	vec3 position;
	vec3 normal;
	float roughness;
};

vec3 view_position(vec2 uv, float depth) {
	vec4 point = scene.inv_projection[params.view] * vec4(uv * 2.0 - 1.0, depth, 1.0);
	return point.xyz / point.w;
}

bool read_surface(ivec2 pixel, ivec2 size, out Surface surface) {
	if (any(greaterThanEqual(pixel, size))) {
		return false;
	}
	float depth = texelFetch(surface_depth, pixel, 0).r;
	if (depth <= 0.0 || depth < texelFetch(opaque_depth, pixel, 0).r) {
		return false;
	}
	vec4 data = texelFetch(surface_normal, pixel, 0);
	surface.roughness = min(data.a, 1.0 - data.a) * (255.0 / 127.0);
	if (surface.roughness >= 0.7) {
		return false;
	}
	surface.position = view_position((vec2(pixel) + 0.5) / vec2(size), depth);
	surface.normal = normalize(data.xyz * 2.0 - 1.0);
	return true;
}

bool matching_surface(Surface a, Surface b) {
	return dot(a.normal, b.normal) > 0.85 &&
			abs(dot(a.position - b.position, a.normal)) < max(0.04, abs(a.position.z) * 0.004);
}

vec3 reflection_direction(Surface surface) {
	vec3 incident = params.orthogonal ? vec3(0.0, 0.0, -1.0) : normalize(surface.position - scene.eye_offset[params.view].xyz);
	return reflect(incident, surface.normal);
}

bool project_point(vec3 point, out vec2 uv) {
	vec4 clip = scene.projection[params.view] * vec4(point, 1.0);
	uv = clip.xy / max(clip.w, 0.00001) * 0.5 + 0.5;
	return clip.w > 0.0 && clip.z > 0.0 && clip.z <= clip.w &&
			all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)));
}

float surface_gap(vec2 uv, vec3 point) {
	float depth = textureLod(opaque_depth, uv, 0.0).r;
	return depth > 0.0 ? view_position(uv, depth).z - point.z : -1.0;
}

float edge_weight(vec2 uv) {
	vec2 edge = min(uv, 1.0 - uv);
	// Leave room for the probe to take over before a reflected source exits.
	return smoothstep(0.0, 0.28, min(edge.x, edge.y));
}

vec4 sample_reflection(vec2 uv, float roughness, float weight) {
	return vec4(textureLod(opaque_color, uv, 12.0 * roughness).rgb, 1.0) * weight;
}

#ifdef SCREEN_TRACE

shared vec4 quad_hits[64];
shared uint unresolved[64];

vec4 march(Surface surface, vec3 ray) {
	vec3 origin = surface.position + surface.normal * 0.025;
	vec4 clip = scene.projection[params.view] * vec4(origin, 1.0);
	vec4 clip_ray = scene.projection[params.view] * vec4(ray, 0.0);

	// Clip the ray against the viewport in homogeneous space before spacing the steps in screen space.
	vec4 distances = vec4(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);
	vec4 direction = -vec4(clip_ray.w + clip_ray.x, clip_ray.w - clip_ray.x, clip_ray.w + clip_ray.y, clip_ray.w - clip_ray.y);
	vec4 limits = mix(vec4(MAX_TRAVEL), distances / max(direction, vec4(0.00001)), greaterThan(direction, vec4(0.0)));
	float limit = min(MAX_TRAVEL, min(min(limits.x, limits.y), min(limits.z, limits.w)));
	if (clip_ray.z > clip_ray.w) {
		limit = min(limit, (clip.w - clip.z) / (clip_ray.z - clip_ray.w));
	}
	if (clip_ray.z < 0.0) {
		limit = min(limit, -clip.z / clip_ray.z);
	}
	if (limit <= 0.03) {
		return vec4(0.0);
	}

	float end_w = clip.w + clip_ray.w * limit;
	float low = 0.03;
	for (int step = 1; step <= 8; step++) {
		float s = float(step) * (0.999 / 8.0);
		float high = limit * (s * clip.w) / max(mix(end_w, clip.w, s), 0.00001);
		if (high <= low) {
			continue;
		}
		vec3 point = origin + ray * high;
		vec2 uv;
		if (!project_point(point, uv)) {
			break;
		}
		float gap = surface_gap(uv, point);
		if (gap >= 0.0) {
			for (int refine = 0; refine < 5; refine++) {
				float middle = 0.5 * (low + high);
				point = origin + ray * middle;
				vec2 middle_uv;
				if (project_point(point, middle_uv) && surface_gap(middle_uv, point) >= 0.0) {
					high = middle;
					uv = middle_uv;
				} else {
					low = middle;
				}
			}
			point = origin + ray * high;
			gap = surface_gap(uv, point);
			float weight = edge_weight(uv) * (1.0 - smoothstep(0.05, 0.24 + 0.02 * high, gap));
			if (weight > 0.0 && dot(point - surface.position, surface.normal) > 0.0) {
				return vec4(uv, high, weight);
			}
			break;
		}
		low = high;
	}
	return vec4(0.0);
}

void main() {
	ivec2 size = imageSize(reflection_output);
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	uint lane = gl_LocalInvocationIndex;
	Surface surface;
	bool active_surface = read_surface(pixel, size, surface);
	vec3 ray = active_surface ? reflection_direction(surface) : vec3(0.0);
	vec4 hit = vec4(0.0);
	if (active_surface) {
		hit = march(surface, ray);
	}
	quad_hits[lane] = hit;
	barrier();

	ivec2 quad = pixel & ~ivec2(1);
	uint base_lane = lane - uint((pixel.x & 1) + 8 * (pixel.y & 1));
	vec4 best_hit = vec4(0.0);
	int best_lane = -1;
	for (int i = 0; i < 4; i++) {
		vec4 candidate = quad_hits[base_lane + uint((i & 1) + 8 * (i >> 1))];
		if (candidate.w > best_hit.w) {
			best_hit = candidate;
			best_lane = i;
		}
	}

	if (active_surface && hit.w <= 0.0 && best_lane >= 0) {
		Surface neighbor;
		vec2 uv;
		vec3 point = surface.position + surface.normal * 0.025 + ray * best_hit.z;
		if (read_surface(quad + ivec2(best_lane & 1, best_lane >> 1), size, neighbor) &&
				matching_surface(surface, neighbor) && project_point(point, uv)) {
			float gap = surface_gap(uv, point);
			float weight = best_hit.w * edge_weight(uv) * (1.0 - smoothstep(0.15, 0.60, abs(gap)));
			hit = vec4(uv, best_hit.z, weight);
		}
	}

	if (all(lessThan(pixel, size))) {
		imageStore(surface_hit, pixel, vec4(active_surface ? texelFetch(surface_depth, pixel, 0).r : 0.0, 0.0, 0.0, 0.0));
		imageStore(reflection_output, pixel, hit.w > 0.0 ? sample_reflection(hit.xy, surface.roughness, hit.w) : vec4(0.0));
	}

	if (params.node_count == 0) {
		return;
	}
	unresolved[lane] = uint(active_surface && hit.w < 0.995);
	barrier();
	if (lane == base_lane) {
		uint mask = unresolved[lane] | (unresolved[lane + 1] << 1) | (unresolved[lane + 8] << 2) | (unresolved[lane + 9] << 3);
		if (mask == 0u) {
			return;
		}
		uint index = atomicAdd(dispatch.count, 1);
		misses.pixels[index] = (uint(quad.x / 2 + quad.y / 2 * ((size.x + 1) / 2)) << 4) | mask;
		uint group = index / 64;
		atomicMax(dispatch.groups_x, min(group + 1, GROUPS_PER_ROW));
		atomicMax(dispatch.groups_y, group / GROUPS_PER_ROW + 1);
	}
}

#else

#ifdef HARDWARE_TRACE
layout(set = 0, binding = 8) uniform accelerationStructureEXT reflection_scene;

float trace_scene(vec3 origin, vec3 direction, out bool displaced) {
	displaced = false;
	rayQueryEXT query;
	rayQueryInitializeEXT(query, reflection_scene, gl_RayFlagsOpaqueEXT, 0xff, origin, 0.10, direction, MAX_TRAVEL);
	while (rayQueryProceedEXT(query)) {
	}
	return rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT ? rayQueryGetIntersectionTEXT(query, true) : MAX_TRAVEL;
}
#else
struct Node {
	vec3 minimum;
	uint end;
	vec3 maximum;
	uint leaf;
	vec4 to_box[3];
};

layout(set = 0, binding = 8, std430) restrict readonly buffer Nodes {
	Node nodes[];
}
bvh;

vec2 box_interval(vec3 origin, vec3 ray, vec3 minimum, vec3 maximum) {
	vec3 safe_ray = mix(ray, mix(vec3(-1e-7), vec3(1e-7), greaterThanEqual(ray, vec3(0.0))), lessThan(abs(ray), vec3(1e-7)));
	vec3 t0 = (minimum - origin) / safe_ray;
	vec3 t1 = (maximum - origin) / safe_ray;
	vec3 entry = min(t0, t1);
	vec3 exit = max(t0, t1);
	return vec2(max(entry.x, max(entry.y, entry.z)), min(exit.x, min(exit.y, exit.z)));
}

float trace_scene(vec3 origin, vec3 ray, out bool displaced) {
	displaced = false;
	float nearest = MAX_TRAVEL;
	uint index = 0;
	while (index < params.node_count) {
		Node node = bvh.nodes[index];
		vec2 interval = box_interval(origin, ray, node.minimum, node.maximum);
		if (interval.x >= nearest || interval.y < max(interval.x, 0.0)) {
			index = node.end;
			continue;
		}
		if (node.leaf != 0) {
			vec3 box_origin;
			vec3 box_ray;
			for (int axis = 0; axis < 3; axis++) {
				box_origin[axis] = dot(node.to_box[axis], vec4(origin, 1.0));
				box_ray[axis] = dot(node.to_box[axis].xyz, ray);
			}
			interval = box_interval(box_origin, box_ray, vec3(0.0), vec3(1.0));
			// Ignore boxes already containing the source, and prevent self-intersection.
			if (interval.x > 0.10 && interval.y >= interval.x) {
				// A conservative animation bound encloses the leaves. Its front face can be empty.
				float hit = node.leaf == 2u ? (interval.x + interval.y) * 0.5 : interval.x;
				if (hit < nearest) {
					nearest = hit;
					displaced = node.leaf == 2u;
				}
			}
		}
		index++;
	}
	return nearest;
}

#endif

void resolve_reflection(ivec2 pixel, ivec2 size) {
	Surface surface;
	if (!read_surface(pixel, size, surface)) {
		return;
	}
	vec3 ray = reflection_direction(surface);
	vec3 origin = surface.position + surface.normal * 0.025;
	bool displaced;
	float travel = trace_scene(origin, ray, displaced);
	if (travel >= MAX_TRAVEL) {
		return;
	}

	// A proxy outside the visible scene contributes occlusion. Use opaque radiance
	// only when the proxy intersection agrees with the visible geometry.
	vec3 point = origin + ray * travel;
	vec2 uv;
	float visible = 0.0;
	bool on_screen = project_point(point, uv);
	if (on_screen && !displaced) {
		float depth = textureLod(opaque_depth, uv, 0.0).r;
		if (depth > 0.0) {
			float gap = abs(view_position(uv, depth).z - point.z);
			visible = edge_weight(uv) * (1.0 - smoothstep(0.08, 0.45 + 0.02 * travel, gap));
		}
	}
	vec4 screen = imageLoad(reflection_output, pixel);
	// A displaced bound encloses empty gaps. Its midpoint gives probe
	// parallax, but it cannot confirm a second screen-space color sample.
	if (displaced) {
		screen *= screen.a;
	}
	vec4 proxy = visible > 0.0 ? sample_reflection(uv, surface.roughness, visible) : vec4(0.0);
	imageStore(reflection_output, pixel, screen + (1.0 - screen.a) * proxy);
	// Preserve the hit distance and tag rays supplied entirely by the offscreen probe.
	imageStore(surface_hit, pixel, vec4(texelFetch(surface_depth, pixel, 0).r, on_screen ? travel : -travel, 0.0, 0.0));
}

void main() {
	uint index = (gl_WorkGroupID.y * GROUPS_PER_ROW + gl_WorkGroupID.x) * 64 + gl_LocalInvocationIndex;
	if (index >= dispatch.count) {
		return;
	}
	ivec2 size = imageSize(reflection_output);
	uint entry = misses.pixels[index];
	uint quad_index = entry >> 4;
	uint quad_width = uint((size.x + 1) / 2);
	ivec2 quad = 2 * ivec2(quad_index % quad_width, quad_index / quad_width);
	uint mask = entry & 15u;
	for (int i = 0; i < 4; i++) {
		if ((mask & (1u << i)) != 0u) {
			resolve_reflection(quad + ivec2(i & 1, i >> 1), size);
		}
	}
}
#endif
