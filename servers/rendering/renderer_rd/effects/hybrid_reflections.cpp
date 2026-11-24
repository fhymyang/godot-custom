/**************************************************************************/
/*  hybrid_reflections.cpp                                               */
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

#include "hybrid_reflections.h"

#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_data_rd.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

#include <algorithm>

using namespace RendererRD;

void HybridReflections::RenderBuffers::clear() {
	RD *rd = RD::get_singleton();
	for (RID buffer : { queue, dispatch, nodes, projections, tlas }) {
		if (buffer.is_valid()) {
			rd->free_rid(buffer);
		}
	}
	tlas = RID();
	tlas_capacity = 0;
	queue = RID();
	dispatch = RID();
	nodes = RID();
	projections = RID();
	queue_capacity = 0;
	node_capacity = 0;
}

HybridReflections::RenderBuffers::~RenderBuffers() {
	clear();
}

HybridReflections::HybridReflections() {
	Vector<String> variants = { "\n#define SCREEN_TRACE\n", "\n#define PROXY_TRACE\n" };
	if (RD::get_singleton()->has_feature(RD::SUPPORTS_RAY_QUERY)) {
		variants.push_back("\n#define HARDWARE_TRACE\n");
	}
	shader.initialize(variants);
	version = shader.version_create();
}

HybridReflections::~HybridReflections() {
	shader.version_free(version);
}

void HybridReflections::_build_nodes(LocalVector<Node> &r_nodes, Proxy *p_proxies, uint32_t p_count) {
	AABB bounds = p_proxies[0].transform.xform(p_proxies[0].bounds);
	for (uint32_t i = 1; i < p_count; i++) {
		bounds.merge_with(p_proxies[i].transform.xform(p_proxies[i].bounds));
	}

	const uint32_t index = r_nodes.size();
	Node node = {};
	for (uint32_t axis = 0; axis < 3; axis++) {
		node.minimum[axis] = bounds.position[axis];
		node.maximum[axis] = bounds.get_end()[axis];
	}
	r_nodes.push_back(node);

	if (p_count == 1) {
		const Transform3D to_box = (p_proxies[0].transform * Transform3D(Basis::from_scale(p_proxies[0].bounds.size), p_proxies[0].bounds.position)).affine_inverse();
		for (uint32_t row = 0; row < 3; row++) {
			for (uint32_t col = 0; col < 3; col++) {
				r_nodes[index].to_box[row][col] = to_box.basis[row][col];
			}
			r_nodes[index].to_box[row][3] = to_box.origin[row];
		}
		r_nodes[index].leaf = p_proxies[0].displaced ? 2 : 1;
	} else {
		const int axis = bounds.get_longest_axis_index();
		const uint32_t middle = p_count / 2;
		std::nth_element(p_proxies, p_proxies + middle, p_proxies + p_count, [axis](const Proxy &p_a, const Proxy &p_b) {
			return p_a.transform.xform(p_a.bounds.get_center())[axis] < p_b.transform.xform(p_b.bounds.get_center())[axis];
		});
		_build_nodes(r_nodes, p_proxies, middle);
		_build_nodes(r_nodes, p_proxies + middle, p_count - middle);
	}
	// The escape index allows depth-first traversal without a per-ray stack.
	r_nodes[index].end = r_nodes.size();
}

bool HybridReflections::render(RenderSceneBuffersRD &p_buffers, RenderBuffers &r_data, const RenderSceneDataRD &p_scene, LocalVector<Proxy> &r_proxies, bool p_hardware) {
	RD *rd = RD::get_singleton();
	UniformSetCacheRD *uniforms = UniformSetCacheRD::get_singleton();
	const Size2i size = p_buffers.get_internal_size();
	const uint32_t views = p_buffers.get_view_count();
	const uint32_t capacity = ((size.x + 1) / 2) * ((size.y + 1) / 2);
	static_assert(sizeof(Node) == 80);

	struct SceneData {
		float projection[2][16];
		float inv_projection[2][16];
		float eye_offset[2][4];
	} scene_data = {};

	LocalVector<Node> nodes;
	if (!r_proxies.is_empty()) {
		nodes.reserve(r_proxies.size() * 2 - 1);
		_build_nodes(nodes, r_proxies.ptr(), r_proxies.size());
	}

	if (!r_data.projections.is_valid()) {
		r_data.projections = rd->uniform_buffer_create(sizeof(SceneData));
	}
	if (!r_data.dispatch.is_valid()) {
		r_data.dispatch = rd->storage_buffer_create(sizeof(uint32_t) * 4, Vector<uint8_t>(), RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
	}
	if (r_data.queue_capacity < capacity) {
		if (r_data.queue.is_valid()) {
			rd->free_rid(r_data.queue);
		}
		r_data.queue = rd->storage_buffer_create(capacity * sizeof(uint32_t));
		r_data.queue_capacity = r_data.queue.is_valid() ? capacity : 0;
	}
	const uint32_t node_count = MAX(1u, nodes.size());
	if (!p_hardware && r_data.node_capacity < node_count) {
		if (r_data.nodes.is_valid()) {
			rd->free_rid(r_data.nodes);
		}
		const uint32_t node_capacity = Math::next_power_of_2(node_count);
		r_data.nodes = rd->storage_buffer_create(node_capacity * sizeof(Node));
		r_data.node_capacity = r_data.nodes.is_valid() ? node_capacity : 0;
	}

	const Pass fallback = p_hardware ? PASS_HARDWARE : PASS_FALLBACK;
	for (Pass pass : { PASS_SCREEN_TRACE, fallback }) {
		if (!pipelines[pass].is_valid()) {
			const RID variant = shader.version_get_shader(version, pass);
			if (variant.is_valid()) {
				pipelines[pass] = rd->compute_pipeline_create(variant);
			}
		}
	}

	if (!r_data.projections.is_valid() || !r_data.dispatch.is_valid() || !r_data.queue.is_valid() || (!p_hardware && !r_data.nodes.is_valid()) ||
			!pipelines[PASS_SCREEN_TRACE].is_valid() || !pipelines[fallback].is_valid()) {
		return false;
	}

	for (uint32_t v = 0; v < views; v++) {
		const Projection projection = views > 1 ? p_scene.get_view_projection(v) : p_scene.get_cam_projection();
		MaterialStorage::store_camera(projection, scene_data.projection[v]);
		MaterialStorage::store_camera(projection.inverse(), scene_data.inv_projection[v]);
		const Vector3 eye = p_scene.get_view_eye_offset(v);
		for (int axis = 0; axis < 3; axis++) {
			scene_data.eye_offset[v][axis] = eye[axis];
		}
	}
	if (rd->buffer_update(r_data.projections, 0, sizeof(SceneData), &scene_data) != OK ||
			(!nodes.is_empty() && rd->buffer_update(r_data.nodes, 0, nodes.size() * sizeof(Node), nodes.ptr()) != OK)) {
		return false;
	}

	MaterialStorage *materials = MaterialStorage::get_singleton();
	const RID nearest = materials->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID linear = materials->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	for (uint32_t view = 0; view < views; view++) {
		struct PushConstant {
			uint32_t view;
			uint32_t nodes;
			uint32_t orthogonal;
			uint32_t pad;
		} push = { view, p_hardware ? 1u : nodes.size(), p_scene.cam_orthogonal, 0 };
		const uint32_t dispatch_args[] = { 0, 1, 1, 0 };
		if (rd->buffer_update(r_data.dispatch, 0, sizeof(dispatch_args), dispatch_args) != OK) {
			return false;
		}

		const StringName color_name = p_buffers.has_texture(RB_SCOPE_BUFFERS, RB_TEX_BACK_COLOR) ? RB_TEX_BACK_COLOR : RB_TEX_BLUR_0;
		const RID color = p_buffers.get_texture_slice(RB_SCOPE_BUFFERS, color_name, view, 0, 1, p_buffers.get_texture_format(RB_SCOPE_BUFFERS, color_name).mipmaps);
		const RD::Uniform opaque_depth(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>{ nearest, p_buffers.get_depth_texture(view) });
		const RD::Uniform opaque_color(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, Vector<RID>{ linear, color });
		const RD::Uniform surface_depth(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, Vector<RID>{ nearest, p_buffers.get_texture_slice(RB_SCOPE_HYBRID_REFLECTIONS, RB_HYBRID_REFLECTION_DEPTH, view, 0) });
		const RD::Uniform surface_normal(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, Vector<RID>{ nearest, p_buffers.get_texture_slice(RB_SCOPE_HYBRID_REFLECTIONS, RB_HYBRID_REFLECTION_NORMAL, view, 0) });
		const RD::Uniform reflection(RD::UNIFORM_TYPE_IMAGE, 4, p_buffers.get_texture_slice(RB_SCOPE_HYBRID_REFLECTIONS, RB_HYBRID_REFLECTION, view, 0));
		const RD::Uniform projection_uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 5, r_data.projections);
		const RD::Uniform queue(RD::UNIFORM_TYPE_STORAGE_BUFFER, 6, r_data.queue);
		const RD::Uniform dispatch(RD::UNIFORM_TYPE_STORAGE_BUFFER, 7, r_data.dispatch);
		const RD::Uniform hit(RD::UNIFORM_TYPE_IMAGE, 9, p_buffers.get_texture_slice(RB_SCOPE_HYBRID_REFLECTIONS, RB_HYBRID_REFLECTION_HIT, view, 0));
		const RD::Uniform bvh(p_hardware ? RD::UNIFORM_TYPE_ACCELERATION_STRUCTURE : RD::UNIFORM_TYPE_STORAGE_BUFFER, 8, p_hardware ? r_data.tlas : r_data.nodes);

		const RID screen_set = uniforms->get_cache(shader.version_get_shader(version, PASS_SCREEN_TRACE), 0, opaque_depth, opaque_color, surface_depth, surface_normal, reflection, projection_uniform, queue, dispatch, hit);
		const RID fallback_set = uniforms->get_cache(shader.version_get_shader(version, fallback), 0, opaque_depth, opaque_color, surface_depth, surface_normal, reflection, projection_uniform, queue, dispatch, bvh, hit);
		if (!screen_set.is_valid() || !fallback_set.is_valid()) {
			return false;
		}

		RD::ComputeListID list = rd->compute_list_begin();
		rd->compute_list_bind_compute_pipeline(list, pipelines[PASS_SCREEN_TRACE]);
		rd->compute_list_bind_uniform_set(list, screen_set, 0);
		rd->compute_list_set_push_constant(list, &push, sizeof(push));
		rd->compute_list_dispatch_threads(list, size.x, size.y, 1);
		rd->compute_list_end();

		if (p_hardware || !nodes.is_empty()) {
			list = rd->compute_list_begin();
			rd->compute_list_bind_compute_pipeline(list, pipelines[fallback]);
			rd->compute_list_bind_uniform_set(list, fallback_set, 0);
			rd->compute_list_set_push_constant(list, &push, sizeof(push));
			rd->compute_list_dispatch_indirect(list, r_data.dispatch, 0);
			rd->compute_list_end();
		}
	}

	return true;
}
