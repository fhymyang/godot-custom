#[compute]

#version 450

#VERSION_DEFINES

#include "../scene_data_inc.glsl"
#include "../light_data_inc.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform SceneDataBlock {
	SceneData current;
	SceneData previous;
}
scene;

layout(set = 0, binding = 1, std140) uniform DirectionalLights {
	DirectionalLightData data[MAX_DIRECTIONAL_LIGHTS];
}
lights;

layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
layout(set = 0, binding = 3) uniform sampler2D normal_buffer;
layout(set = 0, binding = 4) uniform sampler2DShadow shadow_atlas;
layout(set = 0, binding = 5) uniform sampler2D shadow_depth_atlas;
layout(set = 0, binding = 6) uniform sampler2D history;
layout(rgba16f, set = 0, binding = 7) uniform restrict writeonly image2D output_buffer;

layout(push_constant, std430) uniform Params {
	vec4 atlas_rects[4];
	uint light_index;
	uint filter_samples;
	uint blocker_samples;
	float history_weight;
}
params;

mat2 sample_rotation(ivec2 pixel) {
	vec2 seed = vec2(pixel) + scene.current.taa_frame_count * 5.588238;
	float angle = fract(52.9829189 * fract(dot(seed, vec2(0.06711056, 0.00583715)))) * 6.283185307;
	float s = sin(angle);
	float c = cos(angle);
	return mat2(c, -s, s, c);
}

// The first two moments come from the shadow-map kernel, so their scale tracks
// the cascade and penumbra rather than the screen pixels used by color TAA.
vec3 shadow_moments(DirectionalLightData light, mat4 shadow_matrix, int split, vec3 receiver, vec3 normal, mat2 rotation) {
	vec3 normal_bias = normal * (1.0 - max(0.0, dot(light.direction, normal)));
	normal_bias -= light.direction * dot(light.direction, normal_bias);
	vec3 biased = receiver + light.direction * light.shadow_bias[split] + normal_bias * light.shadow_normal_bias[split];
	vec4 projected = shadow_matrix * vec4(biased, 1.0);
	vec3 coord = projected.xyz / projected.w;
	vec2 half_texel = scene.current.directional_shadow_pixel_size * 0.5;
	vec2 uv_min = params.atlas_rects[split].xy + half_texel;
	vec2 uv_max = params.atlas_rects[split].zw - half_texel;
	vec2 uv_scales[4] = vec2[4](light.uv_scale1, light.uv_scale2, light.uv_scale3, light.uv_scale4);

	bool pcss = light.softshadow_angle > 0.0 && light.soft_shadow_scale > 0.0;
	uint samples = params.filter_samples;
	float split_scale = light.blend_splits ? 1.0 : light.shadow_split_offsets.x / light.shadow_split_offsets[split];
	vec2 footprint = scene.current.directional_shadow_pixel_size * light.soft_shadow_scale * split_scale;
	float softness = 0.0;
	if (pcss) {
		float search = max(0.0, dot(light.direction, biased) - light.shadow_range_begin[split]);
		footprint = uv_scales[split] * search * light.softshadow_angle * light.soft_shadow_scale;
		float blocker_depth = 0.0;
		uint blockers = 0;
		for (uint i = 0; i < params.blocker_samples; i++) {
			vec2 uv = clamp(coord.xy + footprint * (rotation * scene.current.directional_penumbra_shadow_kernel[i].xy), uv_min, uv_max);
			float depth = textureLod(shadow_depth_atlas, uv, 0.0).r;
			if (depth > coord.z) {
				blocker_depth += depth;
				blockers++;
			}
		}
		if (blockers == 0u) {
			// Still sample the center, which the rotated blocker search can miss on thin features.
			float center = textureLod(shadow_atlas, vec3(clamp(coord.xy, uv_min, uv_max), coord.z), 0.0);
			return vec3(center, center * center, 0.0);
		}
		blocker_depth /= float(blockers);
		float penumbra = (blocker_depth - coord.z) / max(1.0 - blocker_depth, 0.0001);
		footprint *= penumbra;
		softness = smoothstep(1.0, 7.0, length(footprint / scene.current.directional_shadow_pixel_size));
		samples = params.blocker_samples;
	}

	if (samples == 0u || light.soft_shadow_scale <= 0.0) {
		float center = textureLod(shadow_atlas, vec3(clamp(coord.xy, uv_min, uv_max), coord.z), 0.0);
		return vec3(center, center * center, 0.0);
	}
	vec2 moments = vec2(0.0);
	for (uint i = 0; i < samples; i++) {
		vec2 tap = pcss ? scene.current.directional_penumbra_shadow_kernel[i].xy : scene.current.directional_soft_shadow_kernel[i].xy;
		vec2 uv = clamp(coord.xy + footprint * (rotation * tap), uv_min, uv_max);
		float visibility = textureLod(shadow_atlas, vec3(uv, coord.z), 0.0);
		moments += vec2(visibility, visibility * visibility);
	}
	return vec3(moments / float(samples), softness);
}

vec3 view_position(vec2 uv, float depth) {
	vec4 position = scene.current.inv_projection_matrix * vec4(uv * 2.0 - 1.0, depth, 1.0);
	return position.xyz / position.w;
}

float contact_visibility(vec3 receiver, vec3 normal, vec3 light_direction, ivec2 size) {
	float facing = smoothstep(0.05, 0.30, dot(normal, light_direction));
	if (facing <= 0.0) {
		return 1.0;
	}
	float visibility = 1.0;
	const float ray_length = 0.60;
	for (int step = 1; step <= 6; step++) {
		float distance = float(step) * ray_length / 6.0;
		vec3 ray = receiver + 0.025 * normal + distance * light_direction;
		vec4 clip = scene.current.projection_matrix * vec4(ray, 1.0);
		if (clip.w <= 0.0) {
			break;
		}
		vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
		vec2 pixel = uv * vec2(size) - 0.5;
		ivec2 base = ivec2(floor(pixel));
		if (any(lessThan(base, ivec2(0))) || any(greaterThanEqual(base + 1, size))) {
			break;
		}
		float nearest_depth = texelFetch(depth_buffer, ivec2(floor(pixel + 0.5)), 0).r;
		if (nearest_depth <= 0.0) {
			continue;
		}
		vec2 f = fract(pixel);
		float top = mix(texelFetch(depth_buffer, base, 0).r, texelFetch(depth_buffer, base + ivec2(1, 0), 0).r, f.x);
		float bottom = mix(texelFetch(depth_buffer, base + ivec2(0, 1), 0).r, texelFetch(depth_buffer, base + 1, 0).r, f.x);
		float linear_depth = mix(top, bottom, f.y);
		// Reverse Z: test against the farther surface, and use the nearer one for thickness.
		float intersection = view_position(uv, min(nearest_depth, linear_depth)).z - ray.z;
		float penetration = view_position(uv, max(nearest_depth, linear_depth)).z - ray.z;
		float hit = smoothstep(0.008, 0.032, intersection) * (1.0 - smoothstep(0.055, 0.16, penetration));
		visibility = min(visibility, 1.0 - hit * (1.0 - smoothstep(0.18, ray_length, distance)) * facing * 0.65);
	}
	return visibility;
}

float reproject_history(vec3 receiver, vec3 moments, ivec2 size) {
	if (params.history_weight <= 0.0) {
		return moments.x;
	}
	vec3 world = vec4(receiver, 1.0) * scene.current.inv_view_matrix;
	vec3 previous_receiver = vec4(world, 1.0) * scene.previous.view_matrix;
	vec4 clip = scene.previous.projection_matrix * vec4(previous_receiver, 1.0);
	if (clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w) {
		return moments.x;
	}
	vec2 pixel = (clip.xy / clip.w * 0.5 + 0.5) * vec2(size) - 0.5;
	if (any(lessThan(pixel, vec2(0.0))) || any(greaterThan(pixel, vec2(size - 1)))) {
		return moments.x;
	}
	ivec2 base = ivec2(floor(pixel));
	vec2 fraction = fract(pixel);
	float depth = -previous_receiver.z;
	float depth_tolerance = max(0.045, depth * 0.0015);
	float history_sum = 0.0;
	float weight_sum = 0.0;
	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			vec4 tap = texelFetch(history, min(base + ivec2(x, y), size - 1), 0);
			if (tap.z == float(params.light_index + 1u) && abs(tap.y - depth) < depth_tolerance) {
				float weight = (x == 0 ? 1.0 - fraction.x : fraction.x) * (y == 0 ? 1.0 - fraction.y : fraction.y);
				weight *= 1.0 - smoothstep(0.5 * depth_tolerance, depth_tolerance, abs(tap.y - depth));
				history_sum += tap.x * weight;
				weight_sum += weight;
			}
		}
	}
	if (weight_sum < 0.05) {
		return moments.x;
	}
	float sigma = sqrt(max(0.0, moments.y - moments.x * moments.x));
	float extent = sigma * mix(1.65, 0.70, moments.z) + 0.04;
	float clipped = clamp(history_sum / weight_sum, moments.x - extent, moments.x + extent);
	return clamp(mix(moments.x, clipped, params.history_weight * weight_sum), 0.0, 1.0);
}

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(output_buffer);
	if (any(greaterThanEqual(pixel, size))) {
		return;
	}
	float depth = texelFetch(depth_buffer, pixel, 0).r;
	if (depth <= 0.0) {
		imageStore(output_buffer, pixel, vec4(0.0));
		return;
	}
	vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
	vec3 receiver = view_position(uv, depth);
	vec3 normal = normalize(texelFetch(normal_buffer, pixel, 0).xyz * 2.0 - 1.0);
	DirectionalLightData light = lights.data[params.light_index];
	float receiver_depth = -receiver.z;
	if (receiver_depth >= min(light.shadow_split_offsets.w, 65000.0)) {
		imageStore(output_buffer, pixel, vec4(0.0));
		return;
	}
	int split = receiver_depth < light.shadow_split_offsets.x ? 0 : (receiver_depth < light.shadow_split_offsets.y ? 1 : (receiver_depth < light.shadow_split_offsets.z ? 2 : 3));
	mat4 matrices[4] = mat4[4](light.shadow_matrix1, light.shadow_matrix2, light.shadow_matrix3, light.shadow_matrix4);
	mat2 rotation = sample_rotation(pixel);
	vec3 moments = shadow_moments(light, matrices[split], split, receiver, normal, rotation);
	if (light.blend_splits && split < 3 && light.shadow_split_offsets[split + 1] > light.shadow_split_offsets[split]) {
		float split_depth = light.shadow_split_offsets[split];
		float blend = smoothstep(split_depth * 0.9, split_depth, receiver_depth);
		if (blend > 0.0) {
			moments = mix(moments, shadow_moments(light, matrices[split + 1], split + 1, receiver, normal, rotation), blend);
		}
	}

	float visibility = reproject_history(receiver, moments, size);
	// Keep screen-space contacts out of history so transient occluders do not leave trails.
	float contact = visibility > 0.01 ? contact_visibility(receiver, normal, light.direction, size) : 1.0;
	imageStore(output_buffer, pixel, vec4(visibility, receiver_depth, float(params.light_index + 1u), contact));
}
