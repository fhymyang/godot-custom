/**************************************************************************/
/*  restir_gi.h                                                           */
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

#pragma once

#include "servers/rendering/renderer_rd/pipeline_deferred_rd.h"
#include "servers/rendering/renderer_rd/shaders/effects/restir_gi.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"

#define RB_SCOPE_RESTIR_GI SNAME("restir_gi")
#define RB_RESTIR_GI_LIGHTING SNAME("lighting")

class RenderSceneBuffersRD;
class RenderSceneDataRD;

namespace RendererRD {

class RestirGI {
public:
	struct RenderBuffers {
		RID tlas;
		RID sharc[RendererSceneRender::MAX_RENDER_VIEWS];
		float cache_radius = 0.0f;
		RID surfaces;
		RID reservoirs[RendererSceneRender::MAX_RENDER_VIEWS][2];
		RID uniforms[RendererSceneRender::MAX_RENDER_VIEWS];
		uint32_t scene_capacity = 0;
		uint32_t reservoir_index = 0;
		uint32_t view_count = 0;
		Size2i size;
		bool monte_carlo = false;
		bool use_sdfgi = false;

		uint64_t last_frame = 0;
		RID environment;
		Transform3D previous_camera;
		Projection previous_projections[RendererSceneRender::MAX_RENDER_VIEWS];

		void clear();
	};

	struct Surface {
		uint32_t vertex_address[2];
		uint32_t index_address[2];
		uint32_t vertex_stride;
		uint32_t index_stride;
		uint32_t layers;
		uint32_t padding = 0;
		float diffuse[4] = {};
		float emission[4] = {};
	};

	static constexpr uint32_t MAX_LIGHTS = 4;

	struct DirectionalLight {
		float direction[3];
		float shadow_opacity;
		float radiance[3];
		float shadow_distance;
		uint32_t layers[4] = {};
	};

	static constexpr uint32_t MAX_OMNI_LIGHTS = 8;

	struct OmniLight {
		float position[3];
		float inv_range;
		float radiance[3];
		float attenuation;
		uint32_t layers;
		float shadow_opacity;
		uint32_t padding[2] = {};
	};

	struct SDFGIProbes {
		RID parameters;
		RID radiance;
		RID occlusion;
	};

	struct Settings {
		float radius;
		float intensity;
		float normal_rejection;
		float sharpness;
	};

	RestirGI();
	~RestirGI();

	bool render(const Ref<RenderSceneBuffersRD> &p_buffers, RenderBuffers &r_buffers, const RenderSceneDataRD &p_scene, RID p_environment,
			uint64_t p_frame, const RID *p_normals, const LocalVector<RD::AccelerationStructureInstance> &p_instances,
			const LocalVector<Surface> &p_surfaces, const HashSet<RID> &p_geometry_buffers, const LocalVector<DirectionalLight> &p_lights, const LocalVector<OmniLight> &p_omni_lights,
			const Settings &p_settings, bool p_monte_carlo, const SDFGIProbes &p_probes);

private:
	enum Pass {
		PASS_SHARC_UPDATE,
		PASS_SHARC_RESOLVE,
		PASS_TEMPORAL,
		PASS_SPATIAL,
		PASS_RESOLVE,
		PASS_MONTE_CARLO,
		PASS_MONTE_CARLO_RESOLVE,
		PASS_TEMPORAL_SDFGI,
		PASS_COUNT
	};

	RestirGiShaderRD shader;
	RID shader_version;
	PipelineDeferredRD pipelines[PASS_COUNT];

	struct SceneUniforms {
		float inverse_projection[16];
		float camera_basis[16];
		float previous_world_to_clip[16];
		float camera_offset[4];
		float camera_origin[4];
		DirectionalLight lights[MAX_LIGHTS];
		OmniLight omni_lights[MAX_OMNI_LIGHTS];
		int32_t size[2];
		int32_t half_size[2];
		Settings settings;
		uint32_t frame;
		uint32_t history_valid;
		uint32_t light_count;
		uint32_t view;
		uint32_t omni_count;
		uint32_t padding[3] = {};
	};
};

} // namespace RendererRD
