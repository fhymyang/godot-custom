/**************************************************************************/
/*  hybrid_diffuse_gi.cpp                                               */
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

#include "hybrid_diffuse_gi.h"

#include "restir_gi.h"
#include "ss_effects.h"

#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/texture_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/storage/utilities.h"

using namespace RendererRD;

void HybridDiffuseGI::Buffers::clear() {
	RD *rd = RD::get_singleton();
	for (RID buffer : uniforms) {
		if (buffer.is_valid()) {
			rd->free_rid(buffer);
		}
	}
	for (RID buffer : { nodes, materials, triangles }) {
		if (buffer.is_valid()) {
			rd->free_rid(buffer);
		}
	}
	*this = Buffers();
}

HybridDiffuseGI::HybridDiffuseGI() {
	shader.initialize({ "\n#define TRACE\n", "\n#define FILTER\n", "\n#define RESOLVE\n" });
	version = shader.version_create();
}

HybridDiffuseGI::~HybridDiffuseGI() {
	shader.version_free(version);
}

bool HybridDiffuseGI::render(const Ref<RenderSceneBuffersRD> &rb, Buffers &data, const RenderSceneDataRD &scene, RID environment,
		const RID *normals, LocalVector<HybridReflections::Proxy> &proxies, const LocalVector<Material> &materials,
		const LocalVector<const MeshStorage::GIProxyGeometry *> &geometry_list, const Vector3 &sun_direction, const Color &sun, bool use_ao, float radius, float intensity) {
	RD *rd = RD::get_singleton();
	RID variants[3];
	for (int i = 0; i < 3; i++) {
		variants[i] = shader.version_get_shader(version, i);
		if (variants[i].is_null()) {
			return false;
		}
		if (pipelines[i].is_null()) {
			pipelines[i] = rd->compute_pipeline_create(variants[i]);
		}
		if (pipelines[i].is_null()) {
			return false;
		}
	}

	const Size2i size = rb->get_internal_size();
	const Size2i coarse = (size + Size2i(3, 3)) / 4;
	LocalVector<HybridReflections::Node> nodes;
	if (!proxies.is_empty()) {
		HybridReflections::build_nodes(nodes, proxies.ptr(), proxies.size());
	}
	if (data.capacity < MAX(1u, nodes.size())) {
		data.clear();
		data.capacity = Math::next_power_of_2(MAX(1u, nodes.size()));
		data.nodes = rd->storage_buffer_create(data.capacity * sizeof(HybridReflections::Node));
		data.materials = rd->storage_buffer_create(data.capacity * sizeof(Material));
	}
	if (data.nodes.is_null() || data.materials.is_null()) {
		data.clear();
		return false;
	}
	if (!nodes.is_empty() && (rd->buffer_update(data.nodes, 0, nodes.size() * sizeof(nodes[0]), nodes.ptr()) != OK || rd->buffer_update(data.materials, 0, materials.size() * sizeof(Material), materials.ptr()) != OK)) {
		return false;
	}

	// Instances share a mesh-local triangle hierarchy. Camera motion only changes the top level.
	static_assert(sizeof(MeshStorage::GIProxyNode) == 48);
	static_assert(sizeof(Material) == 48);
	Vector<uint64_t> versions;
	for (const auto *geometry : geometry_list) {
		versions.push_back(geometry->version);
	}
	if (data.triangles.is_null() || data.geometry_versions != versions) {
		uint32_t node_count = 0;
		for (const auto *geometry : geometry_list) {
			node_count += geometry->nodes.size();
		}
		Vector<MeshStorage::GIProxyNode> atlas;
		atlas.resize(MAX(1u, node_count));
		auto *destination = atlas.ptrw();
		for (const auto *geometry : geometry_list) {
			memcpy(destination, geometry->nodes.ptr(), geometry->nodes.size() * sizeof(*destination));
			destination += geometry->nodes.size();
		}
		uint64_t bytes = uint64_t(atlas.size()) * sizeof(MeshStorage::GIProxyNode);
		ERR_FAIL_COND_V(bytes > UINT32_MAX, false);
		RID buffer = rd->storage_buffer_create(bytes);
		if (buffer.is_null()) {
			return false;
		}
		if (rd->buffer_update(buffer, 0, bytes, atlas.ptr()) != OK) {
			rd->free_rid(buffer);
			return false;
		}
		if (data.triangles.is_valid()) {
			rd->free_rid(data.triangles);
		}
		data.triangles = buffer;
		data.geometry_versions = versions;
	}

	const uint64_t frame = RSG::rasterizer->get_frame_number();
	const uint32_t usage = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
	const StringName scope = SNAME("hybrid_diffuse_gi");
	const bool history = data.size == size && data.frame + 1 == frame && data.environment == environment;
	if (data.size != size) {
		rb->clear_context(scope);
	}
	const StringName raw = SNAME("sample");
	const StringName geometry[2] = { SNAME("geometry_0"), SNAME("geometry_1") };
	const StringName sh[2][3] = { { SNAME("red_0"), SNAME("green_0"), SNAME("blue_0") }, { SNAME("red_1"), SNAME("green_1"), SNAME("blue_1") } };
	const uint32_t current = frame & 1;
	auto allocate = [&](const StringName &name, RD::DataFormat format) {
		return rb->create_texture(scope, name, format, usage, RD::TEXTURE_SAMPLES_1, coarse);
	};
	if (allocate(raw, RD::DATA_FORMAT_R16G16B16A16_SFLOAT).is_null()) {
		return false;
	}
	for (int i = 0; i < 2; i++) {
		if (allocate(geometry[i], RD::DATA_FORMAT_R32G32B32A32_SFLOAT).is_null()) {
			return false;
		}
		for (const StringName &name : sh[i]) {
			if (allocate(name, RD::DATA_FORMAT_R16G16B16A16_SFLOAT).is_null()) {
				return false;
			}
		}
	}
	if (rb->create_texture(RB_SCOPE_RESTIR_GI, RB_RESTIR_GI_LIGHTING, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage).is_null()) {
		return false;
	}
	const RID sampler = MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	for (uint32_t v = 0; v < rb->get_view_count(); v++) {
		struct Uniforms {
			float projection[16], inverse_projection[16], previous_clip[16], previous_view[16], rotation[16];
			float sun_direction[4], sun[4];
			uint32_t count, frame, history;
			float intensity;
		} u = {};
		const Projection projection = rb->get_view_count() > 1 ? scene.get_view_projection(v) : scene.get_cam_projection();
		const Projection previous_view(data.previous_camera.affine_inverse() * scene.cam_transform);
		MaterialStorage::store_camera(projection, u.projection);
		MaterialStorage::store_camera(projection.inverse(), u.inverse_projection);
		MaterialStorage::store_camera(data.previous_projection[v] * previous_view, u.previous_clip);
		MaterialStorage::store_camera(previous_view, u.previous_view);
		MaterialStorage::store_camera(Projection(Transform3D(scene.cam_transform.basis, Vector3())), u.rotation);
		for (int i = 0; i < 3; i++) {
			u.sun_direction[i] = sun_direction[i];
			u.sun[i] = sun[i];
		}
		u.sun[3] = sun.a;
		u.sun_direction[3] = radius;
		u.intensity = intensity;
		u.count = nodes.size();
		u.frame = frame;
		u.history = history;
		if (data.uniforms[v].is_null()) {
			data.uniforms[v] = rd->uniform_buffer_create(sizeof(u));
		}
		if (rd->buffer_update(data.uniforms[v], 0, sizeof(u), &u) != OK) {
			return false;
		}
		auto tex = [&](const StringName &name) { return rb->get_texture_slice(scope, name, v, 0); };
		auto sample = [&](int binding, RID texture) { return RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, binding, Vector<RID>{ sampler, texture }); };
		auto image = [&](int binding, RID texture) { return RD::Uniform(RD::UNIFORM_TYPE_IMAGE, binding, texture); };
		const RD::Uniform scene_uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, data.uniforms[v]);
		const RD::Uniform depth = sample(1, rb->get_depth_texture(v));
		const RD::Uniform normal = sample(2, normals[v]);
		const RD::Uniform old_geometry = sample(6, tex(geometry[current ^ 1]));
		auto *sets = UniformSetCacheRD::get_singleton();
		RID bindings[3];
		const RID ao = use_ao && rb->has_texture(RB_SCOPE_SSAO, RB_FINAL) ? rb->get_texture_slice(RB_SCOPE_SSAO, RB_FINAL, v, 0) : TextureStorage::get_singleton()->texture_rd_get_default(TextureStorage::DEFAULT_RD_TEXTURE_WHITE);
		bindings[0] = sets->get_cache(variants[0], 0, scene_uniform, depth, normal,
				RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 3, data.nodes), RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, data.materials),
				sample(5, rb->get_texture_slice(RB_SCOPE_SSLF, RB_LAST_FRAME, v, 0)), old_geometry,
				image(7, tex(raw)), image(8, tex(geometry[current])), sample(16, ao), RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 17, data.triangles));
		bindings[1] = sets->get_cache(variants[1], 0, scene_uniform, old_geometry, sample(7, tex(raw)), sample(8, tex(geometry[current])),
				sample(9, tex(sh[current ^ 1][0])), sample(10, tex(sh[current ^ 1][1])), sample(11, tex(sh[current ^ 1][2])),
				image(12, tex(sh[current][0])), image(13, tex(sh[current][1])), image(14, tex(sh[current][2])));
		bindings[2] = sets->get_cache(variants[2], 0, scene_uniform, depth, normal, sample(8, tex(geometry[current])),
				sample(12, tex(sh[current][0])), sample(13, tex(sh[current][1])), sample(14, tex(sh[current][2])),
				image(15, rb->get_texture_slice(RB_SCOPE_RESTIR_GI, RB_RESTIR_GI_LIGHTING, v, 0)), sample(16, ao));
		static constexpr const char *pass_names[] = {
			"Hybrid GI Trace",
			"Hybrid GI Temporal Filter",
			"Hybrid GI Resolve",
		};
		for (RID binding : bindings) {
			if (binding.is_null()) {
				return false;
			}
		}
		RENDER_TIMESTAMP("> Hybrid GI Compute Passes");
		for (int pass = 0; pass < 3; pass++) {
			RENDER_TIMESTAMP(pass_names[pass]);
			Size2i groups = pass == 2 ? size : coarse;
			auto list = rd->compute_list_begin();
			rd->compute_list_bind_compute_pipeline(list, pipelines[pass]);
			rd->compute_list_bind_uniform_set(list, bindings[pass], 0);
			rd->compute_list_dispatch_threads(list, groups.x, groups.y, 1);
			rd->compute_list_end();
		}
		RENDER_TIMESTAMP("< Hybrid GI Compute Passes");
		data.previous_projection[v] = projection;
	}
	data.previous_camera = scene.cam_transform;
	data.size = size;
	data.environment = environment;
	data.frame = frame;
	return true;
}
