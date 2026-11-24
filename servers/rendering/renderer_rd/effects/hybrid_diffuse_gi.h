/**************************************************************************/
/*  hybrid_diffuse_gi.h                                               */
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

#include "hybrid_reflections.h"

#include "servers/rendering/renderer_rd/shaders/effects/hybrid_diffuse_gi.glsl.gen.h"
#include "servers/rendering/renderer_scene_render.h"

namespace RendererRD {

class HybridDiffuseGI {
public:
	struct Material {
		float diffuse[4];
		float emission[4];
	};
	struct Buffers {
		RID nodes;
		RID materials;
		RID uniforms[RendererSceneRender::MAX_RENDER_VIEWS];
		uint32_t capacity = 0;
		uint64_t frame = 0;
		RID environment;
		Size2i size;
		Transform3D previous_camera;
		Projection previous_projection[RendererSceneRender::MAX_RENDER_VIEWS];
		void clear();
	};

	HybridDiffuseGI();
	~HybridDiffuseGI();
	bool render(const Ref<RenderSceneBuffersRD> &p_buffers, Buffers &r_data, const RenderSceneDataRD &p_scene, RID p_environment,
			const RID *p_normals, LocalVector<HybridReflections::Proxy> &r_proxies, const LocalVector<Material> &p_materials,
			const Vector3 &p_sun_direction, const Color &p_sun, bool p_use_ao, float p_radius, float p_intensity);

private:
	HybridDiffuseGiShaderRD shader;
	RID version;
	RID pipelines[3];
};
} // namespace RendererRD
