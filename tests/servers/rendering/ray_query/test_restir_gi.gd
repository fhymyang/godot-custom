extends SceneTree
## Run with Godot --path tests/servers/rendering/ray_query --rendering-method forward_plus
## --rendering-driver metal --gpu-validation --script res://test_restir_gi.gd.
## Use --rendering-driver vulkan to test the Vulkan backend.
## Pass -- --check-fallback on an RD device that lacks inline ray queries.

const VIEWPORT_SIZE := 96


func _initialize() -> void:
	_run.call_deferred()


func _run() -> void:
	var rd := RenderingServer.get_rendering_device()
	var check_fallback := "--check-fallback" in OS.get_cmdline_user_args()
	if rd == null or rd.has_feature(RenderingDevice.SUPPORTS_RAY_QUERY) == check_fallback:
		push_error("This test requires an RD device with matching ray query support.")
		quit(1)
		return

	var viewport := SubViewport.new()
	viewport.size = Vector2i(VIEWPORT_SIZE, VIEWPORT_SIZE)
	viewport.world_3d = World3D.new()
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	root.add_child(viewport)

	var environment_node := WorldEnvironment.new()
	var environment := Environment.new()
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_DISABLED
	environment.ssil_radius = 5.0
	environment.ssil_use_restir = true
	environment_node.environment = environment
	viewport.add_child(environment_node)

	var floor_material := StandardMaterial3D.new()
	floor_material.albedo_color = Color.WHITE
	_add_box(viewport, Vector3(5.0, 0.1, 5.0), Vector3(0.0, -0.05, 0.0), floor_material)

	# The entire red emitter is outside the orthographic camera frustum.
	var emitter_material := StandardMaterial3D.new()
	emitter_material.albedo_color = Color.BLACK
	emitter_material.emission_enabled = true
	emitter_material.emission = Color.RED
	emitter_material.emission_energy_multiplier = 4.0
	var emitter := _add_box(viewport, Vector3(0.2, 3.0, 3.8), Vector3(1.75, 1.5, 0.0), emitter_material)

	var camera := Camera3D.new()
	camera.position = Vector3(0.0, 6.0, 0.0)
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 3.0
	viewport.add_child(camera)
	camera.look_at(Vector3.ZERO, Vector3.FORWARD)
	camera.make_current()

	await RenderingServer.frame_post_draw
	await RenderingServer.frame_post_draw
	var baseline := _mean_red(viewport)
	environment.ssil_enabled = true
	for frame in 20:
		await RenderingServer.frame_post_draw
	var with_gi := _mean_red(viewport)

	# Moving a rigid mesh must update the TLAS and invalidate reused hit points.
	emitter.position.x = 20.0
	for frame in 10:
		await RenderingServer.frame_post_draw
	var moved := _mean_red(viewport)

	print("Mean red channel before GI, with the offscreen emitter, and after moving it: ", [baseline, with_gi, moved])
	var passed := baseline < 0.02 and moved < 0.02
	passed = passed and (with_gi < 0.02 if check_fallback else with_gi > baseline + 0.025)
	if not passed:
		push_error("Unexpected offscreen indirect lighting.")
	quit(0 if passed else 1)


func _add_box(viewport: SubViewport, size: Vector3, position: Vector3, material: StandardMaterial3D) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = size
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.material_override = material
	instance.position = position
	viewport.add_child(instance)
	return instance


func _mean_red(viewport: SubViewport) -> float:
	var image := viewport.get_texture().get_image()
	var total := 0.0
	for y in range(40, 56):
		for x in range(40, 56):
			total += image.get_pixel(x, y).r
	return total / 256.0
