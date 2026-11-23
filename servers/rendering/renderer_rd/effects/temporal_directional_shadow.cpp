/**************************************************************************/
/*  temporal_directional_shadow.cpp                                      */
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

#include "temporal_directional_shadow.h"

#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/light_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"
#include "servers/rendering/rendering_server_globals.h"

using namespace RendererRD;

TemporalDirectionalShadow::TemporalDirectionalShadow() {
	String defines = "\n#define MAX_VIEWS " + itos(RendererSceneRender::MAX_RENDER_VIEWS) + "\n";
	defines += "#define MAX_DIRECTIONAL_LIGHTS " + itos(RendererSceneRender::MAX_DIRECTIONAL_LIGHTS) + "\n";
#ifdef REAL_T_IS_DOUBLE
	defines += "#define USE_DOUBLE_PRECISION\n";
#endif
	shader.initialize(Vector<String>{ "\n" }, defines);
	version = shader.version_create();
	const RID compiled = shader.version_get_shader(version, 0);
	if (compiled.is_valid()) {
		pipeline = RD::get_singleton()->compute_pipeline_create(compiled);
	}
}

TemporalDirectionalShadow::~TemporalDirectionalShadow() {
	shader.version_free(version);
}

void TemporalDirectionalShadow::render(RenderSceneBuffersRD &p_buffers, RID p_scene_uniforms, RID p_normals, RID p_shadow_sampler, History &r_history) {
	LightStorage *light_storage = LightStorage::get_singleton();
	const LightStorage::TemporalDirectionalLight &light = light_storage->get_temporal_directional_light();
	const RID atlas = light_storage->directional_shadow_get_texture();
	if (light.instance.is_null() || atlas.is_null() || pipeline.is_null()) {
		r_history = History();
		return;
	}

	const uint64_t frame = RSG::rasterizer->get_frame_number();
	const StringName scope = SNAME("temporal_directional_shadow");
	const StringName name = frame % 2 == 0 ? SNAME("even") : SNAME("odd");
	const StringName previous_name = frame % 2 == 0 ? SNAME("odd") : SNAME("even");
	const uint32_t usage = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
	const RID output = p_buffers.create_texture(scope, name, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage);
	// The previous texture must be distinct even when history is invalid.
	const RID previous = p_buffers.create_texture(scope, previous_name, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage);
	if (output.is_null() || previous.is_null()) {
		r_history = History();
		return;
	}

	struct PushConstant {
		float atlas_rects[4][4];
		uint32_t light_index;
		uint32_t filter_samples;
		uint32_t blocker_samples;
		float history_weight;
	} push = {};
	for (int i = 0; i < 4; i++) {
		const Rect2 &rect = light.atlas_rects[i];
		push.atlas_rects[i][0] = rect.position.x;
		push.atlas_rects[i][1] = rect.position.y;
		push.atlas_rects[i][2] = rect.get_end().x;
		push.atlas_rects[i][3] = rect.get_end().y;
	}
	push.light_index = light.index;
	const RendererSceneRenderRD *renderer = RendererSceneRenderRD::get_singleton();
	push.filter_samples = renderer->directional_soft_shadow_samples_get();
	push.blocker_samples = renderer->directional_penumbra_shadow_samples_get();
	if (r_history.texture == previous && r_history.frame + 1 == frame && r_history.light == light.instance) {
		// Reject changes like clock scrubs, while retaining slowly moving sun shadows.
		const float direction_change = (r_history.direction - light.direction).length();
		const float rejection = CLAMP((direction_change - 0.003f) / 0.032f, 0.0f, 1.0f);
		push.history_weight = 0.82f * (1.0f - rejection * rejection * (3.0f - 2.0f * rejection));
	}

	const RID nearest = MaterialStorage::get_singleton()->sampler_rd_get_default(RSE::CANVAS_ITEM_TEXTURE_FILTER_NEAREST, RSE::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);
	const RID compiled = shader.version_get_shader(version, 0);
	const RID uniform_set = UniformSetCacheRD::get_singleton()->get_cache(compiled, 0,
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 0, { p_scene_uniforms }),
			RD::Uniform(RD::UNIFORM_TYPE_UNIFORM_BUFFER, 1, { light_storage->get_directional_light_buffer() }),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 2, { nearest, p_buffers.get_depth_texture(0) }),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 3, { nearest, p_normals }),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 4, { p_shadow_sampler, atlas }),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 5, { nearest, atlas }),
			RD::Uniform(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 6, { nearest, previous }),
			RD::Uniform(RD::UNIFORM_TYPE_IMAGE, 7, { output }));

	if (uniform_set.is_null()) {
		r_history = History();
		return;
	}
	RD *rd = RD::get_singleton();
	RD::ComputeListID list = rd->compute_list_begin();
	rd->compute_list_bind_compute_pipeline(list, pipeline);
	rd->compute_list_bind_uniform_set(list, uniform_set, 0);
	rd->compute_list_set_push_constant(list, &push, sizeof(push));
	const Size2i size = p_buffers.get_internal_size();
	rd->compute_list_dispatch_threads(list, size.x, size.y, 1);
	rd->compute_list_end();
	r_history = { output, light.instance, light.direction, frame };
}
