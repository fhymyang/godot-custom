vec2 octahedron_wrap(vec2 v) {
	vec2 signVal;
	signVal.x = v.x >= 0.0 ? 1.0 : -1.0;
	signVal.y = v.y >= 0.0 ? 1.0 : -1.0;
	return (1.0 - abs(v.yx)) * signVal;
}

vec2 octahedron_encode(vec3 n) {
	// https://twitter.com/Stubbesaurus/status/937994790553227264
	n /= (abs(n.x) + abs(n.y) + abs(n.z));
	n.xy = n.z >= 0.0 ? n.xy : octahedron_wrap(n.xy);
	n.xy = n.xy * 0.5 + 0.5;
	return n.xy;
}

void sdfgi_process(uint cascade, vec3 cascade_pos, vec3 cam_pos, vec3 cam_normal, vec3 cam_specular_normal, bool use_specular, float roughness, out vec3 diffuse_light, out vec3 specular_light, out float blend) {
	cascade_pos += cam_normal * sdfgi.normal_bias;

	vec3 base_pos = floor(cascade_pos);
	//cascade_pos += mix(vec3(0.0),vec3(0.01),lessThan(abs(cascade_pos-base_pos),vec3(0.01))) * cam_normal;
	ivec3 probe_base_pos = ivec3(base_pos);

	vec4 diffuse_accum = vec4(0.0);
	vec3 specular_accum;

	ivec3 tex_pos = ivec3(probe_base_pos.xy, int(cascade));
	tex_pos.x += probe_base_pos.z * sdfgi.probe_axis_size;
	tex_pos.xy = tex_pos.xy * (SDFGI_OCT_SIZE + 2) + ivec2(1);

	vec3 diffuse_posf = (vec3(tex_pos) + vec3(octahedron_encode(cam_normal) * float(SDFGI_OCT_SIZE), 0.0)) * sdfgi.lightprobe_tex_pixel_size;

	vec3 specular_posf;

	if (use_specular) {
		specular_accum = vec3(0.0);
		specular_posf = (vec3(tex_pos) + vec3(octahedron_encode(cam_specular_normal) * float(SDFGI_OCT_SIZE), 0.0)) * sdfgi.lightprobe_tex_pixel_size;
	}

	vec4 light_accum = vec4(0.0);
	float weight_accum = 0.0;

	for (uint j = 0; j < 8; j++) {
		ivec3 offset = (ivec3(j) >> ivec3(0, 1, 2)) & ivec3(1, 1, 1);
		ivec3 probe_posi = probe_base_pos;
		probe_posi += offset;

		// Compute weight

		vec3 probe_pos = vec3(probe_posi);
		vec3 probe_to_pos = cascade_pos - probe_pos;
		vec3 probe_dir = normalize(-probe_to_pos);

		vec3 trilinear = vec3(1.0) - abs(probe_to_pos);
		float weight = trilinear.x * trilinear.y * trilinear.z * max(0.005, dot(cam_normal, probe_dir));

		// Compute lightprobe occlusion

		if (sdfgi.use_occlusion) {
			ivec3 occ_indexv = abs((sdfgi.cascades[cascade].probe_world_offset + probe_posi) & ivec3(1, 1, 1)) * ivec3(1, 2, 4);
			vec4 occ_mask = mix(vec4(0.0), vec4(1.0), equal(ivec4(occ_indexv.x | occ_indexv.y), ivec4(0, 1, 2, 3)));

			vec3 occ_pos = clamp(cascade_pos, probe_pos - sdfgi.occlusion_clamp, probe_pos + sdfgi.occlusion_clamp) * sdfgi.probe_to_uvw;
			occ_pos.z += float(cascade);
			if (occ_indexv.z != 0) { //z bit is on, means index is >=4, so make it switch to the other half of textures
				occ_pos.x += 1.0;
			}

			occ_pos *= sdfgi.occlusion_renormalize;
			float occlusion = dot(textureLod(sampler3D(sdfgi_occlusion_cascades, SAMPLER_LINEAR_CLAMP), occ_pos, 0.0), occ_mask);

			weight *= max(occlusion, 0.01);
		}

		// Compute lightprobe texture position

		vec3 diffuse;
		vec3 pos_uvw = diffuse_posf;
		pos_uvw.xy += vec2(offset.xy) * sdfgi.lightprobe_uv_offset.xy;
		pos_uvw.x += float(offset.z) * sdfgi.lightprobe_uv_offset.z;
		diffuse = textureLod(sampler2DArray(sdfgi_lightprobe_texture, SAMPLER_LINEAR_CLAMP), pos_uvw, 0.0).rgb;

		diffuse_accum += vec4(diffuse * weight * sdfgi.cascades[cascade].exposure_normalization, weight);

		if (use_specular) {
			vec3 specular = vec3(0.0);
			vec3 pos_uvw = specular_posf;
			pos_uvw.xy += vec2(offset.xy) * sdfgi.lightprobe_uv_offset.xy;
			pos_uvw.x += float(offset.z) * sdfgi.lightprobe_uv_offset.z;
			if (roughness < 0.99) {
				specular = textureLod(sampler2DArray(sdfgi_lightprobe_texture, SAMPLER_LINEAR_CLAMP), pos_uvw + vec3(0, 0, float(sdfgi.max_cascades)), 0.0).rgb;
			}
			if (roughness > 0.5) {
				specular = mix(specular, textureLod(sampler2DArray(sdfgi_lightprobe_texture, SAMPLER_LINEAR_CLAMP), pos_uvw, 0.0).rgb, (roughness - 0.5) * 2.0);
			}

			specular_accum += specular * weight * sdfgi.cascades[cascade].exposure_normalization;
		}
	}

	if (diffuse_accum.a > 0.0) {
		diffuse_accum.rgb /= diffuse_accum.a;
	}

	diffuse_light = diffuse_accum.rgb;

	if (use_specular) {
		if (diffuse_accum.a > 0.0) {
			specular_accum /= diffuse_accum.a;
		}

		specular_light = specular_accum;
	}

	{
		//process blend
		float blend_from = (float(sdfgi.probe_axis_size - 1) / 2.0) - 2.5;
		float blend_to = blend_from + 2.0;

		vec3 inner_pos = cam_pos * sdfgi.cascades[cascade].to_probe;

		float len = length(inner_pos);

		inner_pos = abs(normalize(inner_pos));
		len *= max(inner_pos.x, max(inner_pos.y, inner_pos.z));

		if (len >= blend_from) {
			blend = smoothstep(blend_from, blend_to, len);
		} else {
			blend = 0.0;
		}
	}
}
