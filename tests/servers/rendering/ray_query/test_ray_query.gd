extends SceneTree
## Run with Godot --path tests/servers/rendering/ray_query --rendering-driver metal
## --gpu-validation --script res://test_ray_query.gd on a device supporting inline ray queries.
## Use --rendering-driver vulkan to test the Vulkan backend.

const QUERY_SHADER = """
#version 460
#extension GL_EXT_ray_query : require

layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;
layout(set = 0, binding = 1, std430) restrict writeonly buffer Results {
	uint hits[2];
};
layout(push_constant, std430) uniform Params {
	uint index;
} params;

void main() {
	rayQueryEXT query;
	rayQueryInitializeEXT(query, scene, gl_RayFlagsOpaqueEXT, 0xff,
			vec3(0.0, 0.0, 1.0), 0.001, vec3(0.0, 0.0, -1.0), 2.0);
	while (rayQueryProceedEXT(query)) {
	}
	hits[params.index] = rayQueryGetIntersectionTypeEXT(query, true) ==
			gl_RayQueryCommittedIntersectionTriangleEXT ?
			uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)) + 1u : 0u;
}
"""

func _initialize() -> void:
	var rd := RenderingServer.create_local_rendering_device()
	if rd == null:
		push_error("Could not create a local RenderingDevice.")
		quit(1)
		return

	var failure := _test_ray_query(rd)
	rd.free()
	if failure.is_empty():
		print("Ray query hit and miss results matched.")
		quit(0)
	else:
		push_error(failure)
		quit(1)


func _test_ray_query(rd: RenderingDevice) -> String:
	if not rd.has_feature(RenderingDevice.SUPPORTS_RAY_QUERY):
		return "Inline ray queries are unavailable on this device."

	var source := RDShaderSource.new()
	source.source_compute = QUERY_SHADER
	var spirv := rd.shader_compile_spirv_from_source(source, false)
	if not spirv.compile_error_compute.is_empty():
		return spirv.compile_error_compute
	var shader := rd.shader_create_from_spirv(spirv)
	if not shader.is_valid():
		return "Could not create the ray query shader."
	var pipeline := rd.compute_pipeline_create(shader)
	if not pipeline.is_valid():
		return "Could not create the ray query compute pipeline."

	var vertices := PackedFloat32Array([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0]).to_byte_array()
	var buffer_flags := RenderingDevice.BUFFER_CREATION_DEVICE_ADDRESS_BIT | RenderingDevice.BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT
	var vertex_buffer := rd.vertex_buffer_create(vertices.size(), vertices, buffer_flags)
	var geometry := RDAccelerationStructureGeometry.new()
	geometry.vertex_buffer = vertex_buffer
	geometry.vertex_count = 3
	geometry.vertex_stride = 12
	geometry.vertex_format = RenderingDevice.DATA_FORMAT_R32G32B32_SFLOAT
	var blas := rd.blas_create([geometry], 0)
	if not blas.is_valid() or rd.blas_build(blas) != OK:
		return "Could not build the triangle BLAS."

	var tlas := rd.tlas_create(2, 0)
	var output := rd.storage_buffer_create(8, PackedInt32Array([-1, -1]).to_byte_array())
	if not tlas.is_valid() or not output.is_valid():
		return "Could not allocate the TLAS or output buffer."
	var scene_uniform := RDUniform.new()
	scene_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	scene_uniform.binding = 0
	scene_uniform.add_id(tlas)
	var output_uniform := RDUniform.new()
	output_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	output_uniform.binding = 1
	output_uniform.add_id(output)
	var uniforms := rd.uniform_set_create([scene_uniform, output_uniform], shader, 0)
	if not uniforms.is_valid():
		return "Could not bind the TLAS and output buffer."

	# Queue two TLAS builds before submitting so the second build must use a different
	# range of the instance buffer. The instance uses the default zero SBT range.
	var instance := RDAccelerationStructureInstance.new()
	instance.blas = blas
	instance.id = 7
	for index in 2:
		instance.transform = Transform3D(Basis.IDENTITY, Vector3(10.0 * index, 0.0, 0.0))
		if rd.tlas_build(tlas, [instance]) != OK:
			return "Could not build a TLAS for an inline ray query."

		var compute_list := rd.compute_list_begin()
		var params := PackedInt32Array([index]).to_byte_array()
		rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
		rd.compute_list_bind_uniform_set(compute_list, uniforms, 0)
		rd.compute_list_set_push_constant(compute_list, params, params.size())
		rd.compute_list_dispatch(compute_list, 1, 1, 1)
		rd.compute_list_end()

	rd.submit()
	rd.sync()
	var results := rd.buffer_get_data(output).to_int32_array()
	for rid in [uniforms, output, tlas, blas, vertex_buffer, pipeline, shader]:
		rd.free_rid(rid)

	if results != PackedInt32Array([8, 0]):
		return "Expected a hit followed by a miss, got %s." % [results]
	return ""
