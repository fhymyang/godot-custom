/**************************************************************************/
/*  restir_gi.cpp                                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "restir_gi.h"

#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_data_rd.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

#define RB_RESTIR_GI_HALF_LIGHTING SNAME("half_lighting")

void RestirGI::RenderBuffers::clear() {
	RD *rd = RD::get_singleton();
	for (RID buffer : uniforms) {
		if (buffer.is_valid()) {
			rd->free_rid(buffer);
		}
	}
	for (const auto &view_reservoirs : reservoirs) {
		for (RID buffer : view_reservoirs) {
			if (buffer.is_valid()) {
				rd->free_rid(buffer);
			}
		}
	}
	if (surfaces.is_valid()) {
		rd->free_rid(surfaces);
	}
	if (tlas.is_valid()) {
		rd->free_rid(tlas);
	}
	*this = RenderBuffers();
}

RestirGI::RestirGI() {
	const Vector<String> modes = { "\n#define MODE_TEMPORAL\n", "\n#define MODE_SPATIAL\n", "\n#define MODE_RESOLVE\n", "\n#define MODE_TEMPORAL\n#define MONTE_CARLO\n", "\n#define MODE_RESOLVE\n#define MONTE_CARLO\n" };
	shader.initialize(modes);
	shader_version = shader.version_create();
	for (int i = 0; i < PASS_COUNT; i++) {
		pipelines[i].create_compute_pipeline(shader.version_get_shader(shader_version, i));
	}
}

RestirGI::~RestirGI() {
	for (PipelineDeferredRD &pipeline : pipelines) {
		pipeline.free();
	}
	shader.version_free(shader_version);
}

bool RestirGI::render(const Ref<RenderSceneBuffersRD> &p_buffers, RenderBuffers &r_buffers, const RenderSceneDataRD &p_scene, RID p_environment,
		uint64_t p_frame, const RID *p_normals, const LocalVector<RD::AccelerationStructureInstance> &p_instances, const LocalVector<Surface> &p_surfaces,
		const HashSet<RID> &p_geometry_buffers, const LocalVector<DirectionalLight> &p_lights, const Settings &p_settings, bool p_monte_carlo) {
	static_assert(sizeof(Surface) == 64);
	static_assert(sizeof(DirectionalLight) == 48);
	static_assert(sizeof(SceneUniforms) == 448);
	static constexpr uint32_t reservoir_size = 64; // Four vec4/uvec4 fields in restir_gi.glsl.

	RD *rd = RD::get_singleton();
	UniformSetCacheRD *uniform_sets = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL_V(uniform_sets, false);
	ERR_FAIL_COND_V(p_instances.size() != p_surfaces.size() || p_lights.size() > MAX_LIGHTS, false);

	const uint32_t view_count = p_buffers->get_view_count();
	ERR_FAIL_COND_V(view_count > RendererSceneRender::MAX_RENDER_VIEWS, false);
	const Size2i size = p_buffers->get_internal_size();
	const Size2i half_size = (size + Size2i(1, 1)) / 2;
	const uint64_t reservoir_bytes = uint64_t(half_size.x) * half_size.y * (p_monte_carlo ? 16 : reservoir_size);
	if (p_instances.is_empty() || reservoir_bytes == 0 || reservoir_bytes > UINT32_MAX) {
		return false;
	}

	const Pass first_pass = p_monte_carlo ? PASS_MONTE_CARLO : PASS_TEMPORAL;
	const Pass resolve = p_monte_carlo ? PASS_MONTE_CARLO_RESOLVE : PASS_RESOLVE;
	RID shader_rids[PASS_COUNT];
	RID pipeline_rids[PASS_COUNT];
	for (int i = first_pass; i <= resolve; i++) {
		shader_rids[i] = shader.version_get_shader(shader_version, i);
		pipeline_rids[i] = pipelines[i].get_rid();
		if (shader_rids[i].is_null() || pipeline_rids[i].is_null()) {
			return false;
		}
	}

	if (r_buffers.size != size || r_buffers.view_count != view_count || r_buffers.monte_carlo != p_monte_carlo) {
		r_buffers.clear();
		p_buffers->clear_context(RB_SCOPE_RESTIR_GI);
		r_buffers.size = size;
		r_buffers.monte_carlo = p_monte_carlo;
		r_buffers.view_count = view_count;
	}

	for (uint32_t view = 0; view < view_count; view++) {
		for (RID &buffer : r_buffers.reservoirs[view]) {
			if (buffer.is_null()) {
				buffer = rd->storage_buffer_create(uint32_t(reservoir_bytes));
				if (buffer.is_null()) {
					return false;
				}
			}
		}
		if (r_buffers.uniforms[view].is_null()) {
			r_buffers.uniforms[view] = rd->uniform_buffer_create(sizeof(SceneUniforms));
			if (r_buffers.uniforms[view].is_null()) {
				return false;
			}
		}
	}

	if (p_instances.size() > r_buffers.scene_capacity) {
		const uint32_t capacity = Math::next_power_of_2(p_instances.size());
		const RID tlas = rd->tlas_create(capacity, RD::ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT);
		if (tlas.is_null()) {
			return false;
		}
		const RID surfaces = rd->storage_buffer_create(capacity * sizeof(Surface));
		if (surfaces.is_null()) {
			rd->free_rid(tlas);
			return false;
		}
		if (r_buffers.tlas.is_valid()) {
			rd->free_rid(r_buffers.tlas);
			rd->free_rid(r_buffers.surfaces);
		}
		r_buffers.tlas = tlas;
		r_buffers.surfaces = surfaces;
		r_buffers.scene_capacity = capacity;
	}

	if (rd->buffer_update(r_buffers.surfaces, 0, p_surfaces.size() * sizeof(Surface), p_surfaces.ptr()) != OK ||
			rd->tlas_build(r_buffers.tlas, Span(p_instances.ptr(), p_instances.size())) != OK) {
		return false;
	}

	const uint32_t texture_usage = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
	const StringName half_name = p_monte_carlo ? SNAME("monte_carlo_half_lighting") : RB_RESTIR_GI_HALF_LIGHTING;
	const RID half_texture = p_buffers->create_texture(RB_SCOPE_RESTIR_GI, half_name, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, texture_usage, RD::TEXTURE_SAMPLES_1, half_size);
	const RID final_texture = p_buffers->create_texture(RB_SCOPE_RESTIR_GI, RB_RESTIR_GI_LIGHTING, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, texture_usage);
	const StringName history_names[2] = { SNAME("filtered_history_0"), SNAME("filtered_history_1") };
	for (const StringName &name : history_names) {
		if (p_buffers->create_texture(RB_SCOPE_RESTIR_GI, name, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, texture_usage, RD::TEXTURE_SAMPLES_1, half_size).is_null()) {
			return false;
		}
	}
	if (half_texture.is_null() || final_texture.is_null()) {
		return false;
	}

	const bool history_valid = r_buffers.environment == p_environment && r_buffers.last_frame + 1 == p_frame;
	const uint32_t current = r_buffers.reservoir_index ^ 1;
	const Vector3 camera_offset = p_scene.cam_transform.origin - r_buffers.previous_camera.origin;
	const Projection camera_rotation(Transform3D(p_scene.cam_transform.basis, Vector3()));
	const Projection previous_rotation_inverse(Transform3D(r_buffers.previous_camera.basis.inverse(), Vector3()));
	const RID sampler = MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	for (uint32_t view = 0; view < view_count; view++) {
		const Projection projection = view_count > 1 ? p_scene.get_view_projection(view) : p_scene.get_cam_projection();
		SceneUniforms uniforms = {};
		MaterialStorage::store_camera(projection.inverse(), uniforms.inverse_projection);
		MaterialStorage::store_camera(camera_rotation, uniforms.camera_basis);
		MaterialStorage::store_camera(r_buffers.previous_projections[view] * previous_rotation_inverse, uniforms.previous_world_to_clip);
		uniforms.camera_offset[0] = camera_offset.x;
		uniforms.camera_offset[1] = camera_offset.y;
		uniforms.camera_offset[2] = camera_offset.z;
		for (uint32_t i = 0; i < p_lights.size(); i++) {
			uniforms.lights[i] = p_lights[i];
		}
		uniforms.size[0] = size.x;
		uniforms.size[1] = size.y;
		uniforms.half_size[0] = half_size.x;
		uniforms.half_size[1] = half_size.y;
		uniforms.settings = p_settings;
		uniforms.frame = uint32_t(p_frame);
		uniforms.history_valid = history_valid;
		uniforms.light_count = p_lights.size();
		uniforms.view = view;

		if (rd->buffer_update(r_buffers.uniforms[view], 0, sizeof(SceneUniforms), &uniforms) != OK) {
			return false;
		}

		const RID depth = p_buffers->get_depth_texture(view);
		const RID half_light = p_buffers->get_texture_slice(RB_SCOPE_RESTIR_GI, half_name, view, 0);
		const RID final_light = p_buffers->get_texture_slice(RB_SCOPE_RESTIR_GI, RB_RESTIR_GI_LIGHTING, view, 0);
		const RD::Uniform scene_uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, r_buffers.uniforms[view]);
		const RD::Uniform depth_uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, Vector<RID>{ sampler, depth });
		const RD::Uniform normal_uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, Vector<RID>{ sampler, p_normals[view] });
		const RD::Uniform tlas_uniform(RD::UNIFORM_TYPE_ACCELERATION_STRUCTURE, 3, r_buffers.tlas);
		const RD::Uniform surfaces_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, r_buffers.surfaces);
		const RD::Uniform history_uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, r_buffers.reservoirs[view][r_buffers.reservoir_index]);
		const RD::Uniform temporal_output(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, r_buffers.reservoirs[view][current]);
		const RD::Uniform current_geometry(RD::UNIFORM_TYPE_STORAGE_BUFFER, 5, r_buffers.reservoirs[view][current]);
		const RD::Uniform spatial_output(RD::UNIFORM_TYPE_IMAGE, 6, half_light);
		const RD::Uniform half_light_uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, Vector<RID>{ sampler, half_light });
		const RD::Uniform resolve_output(RD::UNIFORM_TYPE_IMAGE, 4, final_light);

		const RID previous_light = p_buffers->get_texture_slice(RB_SCOPE_RESTIR_GI, history_names[r_buffers.reservoir_index], view, 0);
		const RID current_light = p_buffers->get_texture_slice(RB_SCOPE_RESTIR_GI, history_names[current], view, 0);
		const RD::Uniform previous_light_uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, Vector<RID>{ sampler, previous_light });
		const RD::Uniform current_light_uniform(RD::UNIFORM_TYPE_IMAGE, 7, current_light);

		RID sets[PASS_COUNT];
		if (p_monte_carlo) {
			sets[PASS_MONTE_CARLO] = uniform_sets->get_cache(shader_rids[PASS_MONTE_CARLO], 0, scene_uniform, depth_uniform, normal_uniform, tlas_uniform, surfaces_uniform, current_geometry, spatial_output);
		} else {
			sets[PASS_TEMPORAL] = uniform_sets->get_cache(shader_rids[PASS_TEMPORAL], 0, scene_uniform, depth_uniform, normal_uniform, tlas_uniform, surfaces_uniform, history_uniform, temporal_output);
			sets[PASS_SPATIAL] = uniform_sets->get_cache(shader_rids[PASS_SPATIAL], 0, scene_uniform, tlas_uniform, surfaces_uniform, current_geometry, spatial_output);
		}
		sets[resolve] = uniform_sets->get_cache(shader_rids[resolve], 0, scene_uniform, depth_uniform, normal_uniform, half_light_uniform, resolve_output, history_uniform, previous_light_uniform, current_light_uniform);
		for (int i = first_pass; i <= resolve; i++) {
			if (sets[i].is_null()) {
				return false;
			}
			const RD::ComputeListID compute_list = rd->compute_list_begin();
			rd->compute_list_bind_compute_pipeline(compute_list, pipeline_rids[i]);
			rd->compute_list_bind_uniform_set(compute_list, sets[i], 0);
			if (i != resolve) {
				// Physical buffer references must also be registered with the render graph.
				for (RID buffer : p_geometry_buffers) {
					rd->compute_list_add_buffer_read(compute_list, buffer);
				}
			}

			const Size2i dispatch_size = i == resolve ? size : half_size;
			rd->compute_list_dispatch_threads(compute_list, dispatch_size.x, dispatch_size.y, 1);
			rd->compute_list_end();
		}
		r_buffers.previous_projections[view] = projection;
	}

	r_buffers.previous_camera = p_scene.cam_transform;
	r_buffers.environment = p_environment;
	r_buffers.last_frame = p_frame;
	r_buffers.reservoir_index = current;
	return true;
}
