/**************************************************************************/
/*  hybrid_reflections.h                                                 */
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

#include "servers/rendering/renderer_rd/shaders/effects/hybrid_reflections.glsl.gen.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"

#define RB_SCOPE_HYBRID_REFLECTIONS SNAME("hybrid_reflections")
#define RB_HYBRID_REFLECTION_DEPTH SNAME("surface_depth")
#define RB_HYBRID_REFLECTION_HIT SNAME("surface_hit")
#define RB_HYBRID_REFLECTION_NORMAL SNAME("surface_normal")
#define RB_HYBRID_REFLECTION SNAME("reflection")

class RenderSceneDataRD;

namespace RendererRD {

class HybridReflections {
public:
	struct Proxy {
		bool displaced;
		AABB bounds;
		Transform3D transform;
	};

	struct RenderBuffers {
		RID queue;
		RID dispatch;
		RID nodes;
		RID projections;
		uint32_t queue_capacity = 0;
		uint32_t node_capacity = 0;
		void clear();
		~RenderBuffers();
	};

	HybridReflections();
	~HybridReflections();

	bool render(RenderSceneBuffersRD &p_buffers, RenderBuffers &r_data, const RenderSceneDataRD &p_scene, LocalVector<Proxy> &r_proxies);

private:
	enum Pass {
		PASS_SCREEN_TRACE,
		PASS_FALLBACK,
		PASS_COUNT,
	};

	struct Node {
		float minimum[3];
		uint32_t end;
		float maximum[3];
		uint32_t leaf;
		float to_box[3][4];
	};

	static void _build_nodes(LocalVector<Node> &r_nodes, Proxy *p_proxies, uint32_t p_count);

	HybridReflectionsShaderRD shader;
	RID version;
	RID pipelines[PASS_COUNT];
};

} // namespace RendererRD
