#[compute]

#version 460

#VERSION_DEFINES

#if defined(MODE_TEMPORAL) || defined(MODE_SPATIAL)
#extension GL_EXT_ray_query : require
#extension GL_EXT_buffer_reference_uvec2 : require
#endif

layout(local_size_x = 8, local_size_y = 8) in;

// ReSTIR GI (Ouyang et al., 2021), using one diffuse bounce and bounded reservoirs.
const float PI = 3.14159265359;
const uint MAX_HISTORY_AGE = 8u;
const int MAX_LIGHTS = 4;

struct DirectionalLight {
	vec4 direction_shadow;
	vec4 radiance;
	uvec4 layers;
};

layout(set = 0, binding = 0, std140) uniform Scene {
	mat4 inverse_projection;
	mat4 camera_basis;
	mat4 previous_world_to_clip;
	vec4 camera_offset;
	DirectionalLight lights[MAX_LIGHTS];
	ivec2 size;
	ivec2 half_size;
	float radius;
	float intensity;
	float normal_rejection;
	float sharpness;
	uint frame;
	uint history_valid;
	uint light_count;
	uint view;
}
scene;

#ifndef MODE_SPATIAL
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;
layout(set = 0, binding = 2) uniform sampler2D normal_buffer;
#endif

struct Reservoir {
	vec4 receiver; // Camera-relative position and candidate count.
	vec4 hit; // Camera-relative position and normalized reservoir weight.
	vec4 radiance; // Incident radiance and selected sample's target density.
	uvec4 attributes; // Octahedral receiver and hit normals, then the sample age.
};

#if defined(MODE_TEMPORAL) || defined(MODE_SPATIAL)
struct Surface {
	uvec2 vertex_address;
	uvec2 index_address;
	uvec4 layout_data; // Vertex stride, index stride, layer mask.
	vec4 diffuse;
	vec4 emission;
};

layout(set = 0, binding = 3) uniform accelerationStructureEXT acceleration_structure;

layout(set = 0, binding = 4, std430) restrict readonly buffer Surfaces {
	Surface surfaces[];
};

layout(buffer_reference, std430, buffer_reference_align = 4) restrict readonly buffer VertexData {
	uint values[];
};

#ifdef MONTE_CARLO
layout(set = 0, binding = 5, std430) restrict writeonly buffer CurrentReceivers {
	uvec4 current_receivers[];
};
layout(rgba16f, set = 0, binding = 6) uniform restrict writeonly image2D half_lighting;
#else
layout(set = 0, binding = 5, std430) restrict readonly buffer SourceReservoirs {
	Reservoir source_reservoirs[];
};

#ifdef MODE_TEMPORAL
layout(set = 0, binding = 6, std430) restrict writeonly buffer DestinationReservoirs {
	// Save temporal reservoirs before spatial reuse to avoid propagating spatial correlations.
	Reservoir destination_reservoirs[];
};
#else
layout(rgba16f, set = 0, binding = 6) uniform restrict writeonly image2D half_lighting;
#endif
#endif
#endif

#ifdef MODE_RESOLVE
layout(set = 0, binding = 3) uniform sampler2D half_lighting;
layout(rgba16f, set = 0, binding = 4) uniform restrict writeonly image2D lighting;
layout(set = 0, binding = 5, std430) restrict readonly buffer PreviousReceivers {
	#ifdef MONTE_CARLO
	uvec4 previous_receivers[];
#else
	Reservoir previous_receivers[];
#endif
};
layout(set = 0, binding = 6) uniform sampler2D previous_lighting;
layout(rgba16f, set = 0, binding = 7) uniform restrict writeonly image2D filtered_history;
#endif

bool in_bounds(ivec2 pixel, ivec2 size) {
	return all(greaterThanEqual(pixel, ivec2(0))) && all(lessThan(pixel, size));
}

uint reservoir_index(ivec2 pixel) {
	return uint(pixel.x + scene.half_size.x * pixel.y);
}

#ifndef MODE_SPATIAL
ivec2 receiver_pixel(ivec2 half_pixel) {
	return min(half_pixel * 2, scene.size - 1);
}

bool load_receiver(ivec2 pixel, out vec3 position, out vec3 normal) {
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	if (depth <= 0.0) {
		return false;
	}
	vec3 encoded_normal = texelFetch(normal_buffer, pixel, 0).xyz * 2.0 - 1.0;
	if (dot(encoded_normal, encoded_normal) < 0.25) {
		return false;
	}
	vec2 ndc = (vec2(pixel) + 0.5) / vec2(scene.size) * 2.0 - 1.0;
	vec4 view = scene.inverse_projection * vec4(ndc, depth, 1.0);
	position = mat3(scene.camera_basis) * (view.xyz / view.w);
	normal = normalize(mat3(scene.camera_basis) * encoded_normal);
	return true;
}
#endif

vec2 octahedral_wrap(vec2 v) {
	return (1.0 - abs(v.yx)) * mix(vec2(-1.0), vec2(1.0), greaterThanEqual(v, vec2(0.0)));
}

uint encode_normal(vec3 normal) {
	normal /= dot(abs(normal), vec3(1.0));
	return packSnorm2x16(normal.z >= 0.0 ? normal.xy : octahedral_wrap(normal.xy));
}

vec3 decode_normal(uint packed) {
	vec2 encoded = unpackSnorm2x16(packed);
	vec3 normal = vec3(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
	if (normal.z < 0.0) {
		normal.xy = octahedral_wrap(normal.xy);
	}
	return normalize(normal);
}

#if defined(MODE_TEMPORAL) || defined(MODE_SPATIAL)

uint rng;

float random_float() {
	rng = rng * 747796405u + 2891336453u;
	uint bits = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
	return float(((bits >> 22u) ^ bits) >> 8u) * (1.0 / 16777216.0);
}

uint load_index(Surface surface, uint index) {
	uint stride = surface.layout_data.y;
	if (stride == 0u) {
		return index;
	}
	VertexData indices = VertexData(surface.index_address);
	if (stride == 2u) {
		return (indices.values[index >> 1u] >> ((index & 1u) * 16u)) & 65535u;
	}
	return indices.values[index];
}

vec3 load_position(Surface surface, uint index) {
	VertexData vertices = VertexData(surface.vertex_address);
	if (surface.layout_data.x == 8u) {
		return vec3(unpackUnorm2x16(vertices.values[index * 2u]), unpackUnorm2x16(vertices.values[index * 2u + 1u]).x);
	}
	uint first = index * 3u;
	return uintBitsToFloat(uvec3(vertices.values[first], vertices.values[first + 1u], vertices.values[first + 2u]));
}

struct Hit {
	vec3 position;
	vec3 normal;
	uint surface;
};

bool trace_surface(vec3 origin, vec3 direction, float distance, out Hit hit) {
	rayQueryEXT query;
	rayQueryInitializeEXT(query, acceleration_structure, gl_RayFlagsOpaqueEXT, 0xff, origin, 0.001, direction, distance);
	while (rayQueryProceedEXT(query)) {
	}
	if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
		return false;
	}

	hit.surface = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true);
	Surface surface = surfaces[hit.surface];
	uint first = 3u * rayQueryGetIntersectionPrimitiveIndexEXT(query, true);
	vec3 a = load_position(surface, load_index(surface, first));
	vec3 b = load_position(surface, load_index(surface, first + 1u));
	vec3 c = load_position(surface, load_index(surface, first + 2u));
	mat3 object_to_world = mat3(rayQueryGetIntersectionObjectToWorldEXT(query, true));
	vec3 normal = normalize(cross(object_to_world * (b - a), object_to_world * (c - a)));
	hit.normal = faceforward(normal, direction, normal);
	hit.position = origin + direction * rayQueryGetIntersectionTEXT(query, true);
	return true;
}

#ifdef MODE_TEMPORAL
bool occluded(vec3 origin, vec3 direction, float distance) {
	rayQueryEXT query;
	rayQueryInitializeEXT(query, acceleration_structure, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
			0xff, origin, 0.001, direction, distance);
	while (rayQueryProceedEXT(query)) {
	}
	return rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

vec3 evaluate_radiance(Hit hit) {
	Surface surface = surfaces[hit.surface];
	vec3 result = surface.emission.rgb;
	for (uint i = 0u; i < scene.light_count; i++) {
		DirectionalLight light = scene.lights[i];
		float cosine = dot(hit.normal, light.direction_shadow.xyz);
		if (cosine <= 0.0 || (surface.layout_data.z & light.layers.x) == 0u) {
			continue;
		}
		float visibility = 1.0;
		if (light.direction_shadow.w > 0.0 &&
				occluded(hit.position + hit.normal * 0.005, light.direction_shadow.xyz, light.radiance.a)) {
			visibility -= light.direction_shadow.w;
		}
		result += surface.diffuse.rgb * light.radiance.rgb * (visibility * cosine / PI);
	}
	return clamp(result, vec3(0.0), vec3(128.0));
}
#endif

vec3 integrand(vec3 receiver, vec3 normal, vec3 hit, vec3 incident_radiance) {
	vec3 offset = hit - receiver;
	float distance = length(offset);
	float cosine = max(dot(normal, offset) / max(distance, 0.0001), 0.0);
	float attenuation = 1.0 - smoothstep(scene.radius * 0.75, scene.radius, distance);
	return incident_radiance * (cosine * attenuation / PI);
}

float target_density(vec3 contribution) {
	return dot(contribution, vec3(0.2126, 0.7152, 0.0722));
}

void resample(inout Reservoir reservoir, inout float weight_sum, Hit hit, vec3 radiance, float target, float weight, float count, uint age) {
	reservoir.receiver.w += count;
	weight_sum += weight;
	if (weight > 0.0 && random_float() * weight_sum < weight) {
		reservoir.hit.xyz = hit.position;
		reservoir.radiance = vec4(radiance, target);
		reservoir.attributes.y = encode_normal(hit.normal);
		reservoir.attributes.z = age;
	}
}

float normalized_weight(Reservoir reservoir, float weight_sum) {
	float denominator = reservoir.receiver.w * reservoir.radiance.w;
	return denominator > 0.0 ? min(weight_sum / denominator, 64.0) : 0.0;
}

void reuse_sample(inout Reservoir reservoir, inout float weight_sum, vec3 normal, Reservoir source, float max_count, uint age) {
	float count = min(source.receiver.w, max_count);
	if (source.hit.w <= 0.0) {
		reservoir.receiver.w += count;
		return;
	}

	vec3 receiver = reservoir.receiver.xyz;
	vec3 offset = source.hit.xyz - receiver;
	float distance = length(offset);
	if (distance < 0.015 || distance >= scene.radius || dot(normal, offset) <= 0.0) {
		return;
	}

	// Reconnect the endpoints in the current TLAS. This also rejects removed or moved hit surfaces.
	vec3 origin = receiver + normal * 0.005;
	vec3 ray = source.hit.xyz - origin;
	float ray_length = length(ray);
	float tolerance = clamp(ray_length * 0.005, 0.008, 0.05);
	Hit hit;
	if (!trace_surface(origin, ray / ray_length, ray_length + tolerance, hit) ||
			length(hit.position - source.hit.xyz) > tolerance) {
		return;
	}

	vec3 old_normal = decode_normal(source.attributes.y);
	if (dot(hit.normal, old_normal) < 0.8) {
		return;
	}
	vec3 old_offset = source.receiver.xyz - source.hit.xyz;
	float old_distance = length(old_offset);
	float old_cosine = dot(old_normal, old_offset) / max(old_distance, 0.0001);
	float new_cosine = dot(hit.normal, -offset) / distance;
	if (old_cosine <= 0.001 || new_cosine <= 0.001) {
		return;
	}

	// Ratio of the solid-angle measures at the new and old receivers.
	float jacobian = new_cosine * old_distance * old_distance / (old_cosine * distance * distance);
	if (jacobian > 10.0) {
		return;
	}
#ifdef MODE_TEMPORAL
	vec3 radiance = evaluate_radiance(hit);
#else
	vec3 radiance = source.radiance.rgb;
#endif
	float target = target_density(integrand(receiver, normal, hit.position, radiance));
	resample(reservoir, weight_sum, hit, radiance, target, target * source.hit.w * count * jacobian, count, age);
}

bool compatible(Reservoir source, vec3 position, vec3 normal) {
	return source.receiver.w > 0.0 &&
			dot(normal, decode_normal(source.attributes.x)) >= mix(0.55, 0.9, scene.normal_rejection) &&
			abs(dot(normal, source.receiver.xyz - position)) < max(0.02, length(position) * 0.0075);
}
#endif

#ifdef MODE_TEMPORAL
vec3 sample_cosine_hemisphere(vec3 normal) {
	float phi = 2.0 * PI * random_float();
	float r2 = random_float();
	vec3 tangent = normalize(cross(abs(normal.z) < 0.999 ? vec3(0, 0, 1) : vec3(0, 1, 0), normal));
	vec3 bitangent = cross(normal, tangent);
	return tangent * (cos(phi) * sqrt(r2)) + bitangent * (sin(phi) * sqrt(r2)) + normal * sqrt(1.0 - r2);
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (!in_bounds(pixel, scene.half_size)) {
		return;
	}
	uint index = reservoir_index(pixel);
	rng = index ^ (scene.frame * 19349663u + scene.view * 73856093u);
#ifdef MONTE_CARLO
	vec3 receiver, normal;
	vec3 light = vec3(0.0);
	uvec4 geometry = uvec4(0u);
	if (load_receiver(receiver_pixel(pixel), receiver, normal)) {
		geometry = uvec4(floatBitsToUint(receiver), encode_normal(normal));
		vec3 direction = sample_cosine_hemisphere(normal);
		Hit hit;
		if (trace_surface(receiver + normal * 0.005, direction, scene.radius, hit)) {
			// The Lambertian cosine and 1/pi cancel against the sampling PDF.
			float distance = length(hit.position - receiver);
			light = evaluate_radiance(hit) * (1.0 - smoothstep(scene.radius * 0.75, scene.radius, distance));
		}
	}
	current_receivers[index] = geometry;
	imageStore(half_lighting, pixel, vec4(light, 0.0));
#else
	Reservoir reservoir = Reservoir(vec4(0.0), vec4(0.0), vec4(0.0), uvec4(0u));
	vec3 receiver, normal;
	if (!load_receiver(receiver_pixel(pixel), receiver, normal)) {
		destination_reservoirs[index] = reservoir;
		return;
	}

	reservoir.receiver.xyz = receiver;
	reservoir.attributes.x = encode_normal(normal);
	float weight_sum = 0.0;
	vec3 direction = sample_cosine_hemisphere(normal);
	Hit hit;
	if (trace_surface(receiver + normal * 0.005, direction, scene.radius, hit)) {
		vec3 radiance = evaluate_radiance(hit);
		float pdf = max(dot(normal, direction) / PI, 0.00001);
		float target = target_density(integrand(receiver, normal, hit.position, radiance));
		resample(reservoir, weight_sum, hit, radiance, target, target / pdf, 1.0, 0u);
	} else {
		reservoir.receiver.w = 1.0;
	}

	if (scene.history_valid != 0u) {
		vec3 previous_position = receiver + scene.camera_offset.xyz;
		vec4 clip = scene.previous_world_to_clip * vec4(previous_position, 1.0);
		if (clip.w > 0.0 && clip.z > 0.0 && clip.z <= clip.w) {
			vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
			ivec2 previous_pixel = ivec2(floor(uv * vec2(scene.size) * 0.5));
			if (in_bounds(previous_pixel, scene.half_size) && all(lessThan(uv, vec2(1.0)))) {
				Reservoir previous = source_reservoirs[reservoir_index(previous_pixel)];
				previous.receiver.xyz -= scene.camera_offset.xyz;
				previous.hit.xyz -= scene.camera_offset.xyz;
				if (previous.attributes.z < MAX_HISTORY_AGE && compatible(previous, receiver, normal) &&
						length(previous.receiver.xyz - receiver) < max(0.05, length(receiver) * 0.025)) {
					reuse_sample(reservoir, weight_sum, normal, previous, 12.0, previous.attributes.z + 1u);
				}
			}
		}
	}

	reservoir.hit.w = normalized_weight(reservoir, weight_sum);
	destination_reservoirs[index] = reservoir;
#endif
}
#endif

#ifdef MODE_SPATIAL
void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (!in_bounds(pixel, scene.half_size)) {
		return;
	}

	uint index = reservoir_index(pixel);
	rng = index ^ (scene.frame * 83492791u + scene.view * 73856093u + 371u);
	Reservoir reservoir = source_reservoirs[index];
	if (reservoir.receiver.w <= 0.0) {
		imageStore(half_lighting, pixel, vec4(0.0));
		return;
	}

	float weight_sum = reservoir.hit.w * reservoir.receiver.w * reservoir.radiance.w;
	vec3 normal = decode_normal(reservoir.attributes.x);
	float rotation = 2.0 * PI * random_float();
	for (int i = 0; i < 4; i++) {
		float angle = rotation + float(i) * (0.5 * PI);
		ivec2 offset = ivec2(round(vec2(cos(angle), sin(angle)) * (1.5 + 2.0 * random_float())));
		ivec2 neighbor_pixel = pixel + offset;
		if (!in_bounds(neighbor_pixel, scene.half_size)) {
			continue;
		}
		Reservoir neighbor = source_reservoirs[reservoir_index(neighbor_pixel)];
		if (compatible(neighbor, reservoir.receiver.xyz, normal)) {
			reuse_sample(reservoir, weight_sum, normal, neighbor, 4.0, neighbor.attributes.z);
		}
	}

	vec3 light = integrand(reservoir.receiver.xyz, normal, reservoir.hit.xyz, reservoir.radiance.rgb);
	imageStore(half_lighting, pixel, vec4(light * normalized_weight(reservoir, weight_sum), 0.0));
}
#endif

#ifdef MODE_RESOLVE
// Validate each bilinear tap before mixing colors across a silhouette.
vec4 gather_history(vec2 uv, vec3 receiver, vec3 normal) {
	vec2 grid = (uv * vec2(scene.size) - 0.5) * 0.5;
	ivec2 base = ivec2(floor(grid));
	vec2 fraction = fract(grid);
	float max_distance = max(0.05, length(receiver) * 0.025);
	float max_plane_distance = max(0.02, length(receiver) * 0.005);
	vec4 history = vec4(0.0);
	float support = 0.0;
	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			ivec2 pixel = base + ivec2(x, y);
			float weight = (x == 0 ? 1.0 - fraction.x : fraction.x) * (y == 0 ? 1.0 - fraction.y : fraction.y);
			if (weight < 0.0001 || !in_bounds(pixel, scene.half_size)) {
				continue;
			}
#ifdef MONTE_CARLO
			uvec4 previous = previous_receivers[reservoir_index(pixel)];
			vec3 position = uintBitsToFloat(previous.xyz);
			uint encoded_normal = previous.w;
			bool valid = any(notEqual(previous.xyz, uvec3(0u)));
#else
			Reservoir previous = previous_receivers[reservoir_index(pixel)];
			vec3 position = previous.receiver.xyz;
			uint encoded_normal = previous.attributes.x;
			bool valid = previous.receiver.w > 0.0;
#endif
			vec3 delta = position - (receiver + scene.camera_offset.xyz);
			if (!valid || dot(delta, delta) > max_distance * max_distance || abs(dot(normal, delta)) > max_plane_distance) {
				continue;
			}
			weight *= smoothstep(0.88, 0.97, dot(normal, decode_normal(encoded_normal)));
			history += weight * texelFetch(previous_lighting, pixel, 0);
			support += weight;
		}
	}
	if (support < 0.25) {
		return vec4(0.0);
	}
	history.rgb /= support;
	// Partial coverage shortens the effective history at disocclusions.
	return history;
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (!in_bounds(pixel, scene.size)) {
		return;
	}
	vec3 receiver, normal;
	vec3 light = vec3(0.0);
	float total_weight = 0.0;
	vec3 second_moment = vec3(0.0);
	if (load_receiver(pixel, receiver, normal)) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				ivec2 sample_pixel = pixel / 2 + ivec2(x, y);
				if (!in_bounds(sample_pixel, scene.half_size)) {
					continue;
				}
				vec3 sample_position, sample_normal;
				if (!load_receiver(receiver_pixel(sample_pixel), sample_position, sample_normal)) {
					continue;
				}
				float normal_agreement = max(dot(normal, sample_normal), 0.0);
				float plane_distance = abs(dot(normal, sample_position - receiver));
				float depth_weight = exp2(-plane_distance * mix(15.0, 80.0, scene.sharpness) / max(1.0, length(receiver) * 0.1));
				float normal_weight = pow(normal_agreement, mix(4.0, 24.0, scene.sharpness));
				vec2 offset = vec2(receiver_pixel(sample_pixel) - pixel);
				float weight = depth_weight * normal_weight * exp2(-0.2 * dot(offset, offset));
				vec3 sample_light = texelFetch(half_lighting, sample_pixel, 0).rgb;
				light += sample_light * weight;
				second_moment += sample_light * sample_light * weight;
				total_weight += weight;
			}
		}
	}
	light /= max(total_weight, 0.001);
	float frames = total_weight > 0.001 ? 1.0 : 0.0;
	if (scene.history_valid != 0u && total_weight > 0.001) {
		vec3 previous_position = receiver + scene.camera_offset.xyz;
		vec4 clip = scene.previous_world_to_clip * vec4(previous_position, 1.0);
		if (clip.w > 0.0 && clip.z > 0.0 && clip.z <= clip.w) {
			vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
			if (all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)))) {
				vec4 history = gather_history(uv, receiver, normal);
				vec3 deviation = sqrt(max(second_moment / total_weight - light * light, vec3(0.0)));
				vec3 clipped = clamp(history.rgb, light - deviation * 1.5, light + deviation * 1.5);
				// Reduce persistence when lighting moves outside the current neighborhood.
				float change = length(history.rgb - clipped) / max(length(deviation) + length(light) * 0.1, 0.001);
				float retained = min(history.a, 11.0) / (1.0 + change * change);
				frames += retained;
				light = mix(light, clipped, retained / frames);
			}
		}
	}
	if (all(equal(pixel, receiver_pixel(pixel / 2)))) {
		imageStore(filtered_history, pixel / 2, vec4(light, frames));
	}
	imageStore(lighting, pixel, vec4(light * scene.intensity, 0.0));
}
#endif
