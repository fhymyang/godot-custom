extends Node3D

const MODES := ["GI off", "Hybrid GI", "Single-bounce Monte Carlo", "ReSTIR + SHaRC", "ReSTIR + SDFGI"]
const SOURCE_POSITION := Vector3(1.75, 1.5, 0.0)
const CAMERA_POSITION := Vector3(0, 5, 4.7)
const TARGET := Vector3(0, 0.01, 0)
@onready var environment: Environment = $WorldEnvironment.environment
@onready var camera: Camera3D = $Camera3D
@onready var emitter: MeshInstance3D = $Emitter
var status := Label.new()
var marker := Label.new()
var mode := 3
var source_here := true
var orbit := false
var phase := 0.0
var ray_query := false


func _ready() -> void:
	_pose_camera()
	var overlay := CanvasLayer.new()
	add_child(overlay)
	overlay.add_child(status)
	status.position = Vector2(14, 10)
	status.add_theme_constant_override("outline_size", 5)
	overlay.add_child(marker)
	marker.text = "+  receiver"
	marker.add_theme_constant_override("outline_size", 4)
	var rd := RenderingServer.get_rendering_device()
	ray_query = rd != null and rd.has_feature(RenderingDevice.SUPPORTS_RAY_QUERY)
	_set_mode(3 if ray_query else 1)
	if "--verify" in OS.get_cmdline_user_args():
		_verify.call_deferred()


func _set_mode(value: int) -> void:
	if value >= 2 and not ray_query:
		return
	mode = value
	ProjectSettings.set_setting("rendering/environment/ssil/use_hybrid_diffuse_gi", value == 1)
	ProjectSettings.set_setting("rendering/environment/ssil/restir_use_sdfgi_probes", value == 4)
	environment.sdfgi_enabled = value == 4
	environment.ssil_enabled = value != 0
	environment.ssil_use_restir = value >= 3
	environment.ssil_use_monte_carlo = value == 2
	status.text = "%s    |    Source %s\n1 Off   2 Hybrid   3 Single bounce   4 ReSTIR + SHaRC   5 ReSTIR + SDFGI\nSpace  Move source     C  Orbit / reset camera" % [MODES[mode], "in place" if source_here else "removed"]
	if not ray_query:
		status.text += "\nModes 3 to 5 require inline ray queries."


func _pose_camera() -> void:
	camera.position = CAMERA_POSITION.rotated(Vector3.UP, 0.24 * sin(phase))
	camera.look_at(Vector3(0, 0.7, 0), Vector3.UP)


func _process(delta: float) -> void:
	if orbit:
		phase += delta
		_pose_camera()
	marker.position = camera.unproject_position(TARGET) - Vector2(4, 12)


func _move_source() -> void:
	source_here = not source_here
	emitter.position = SOURCE_POSITION if source_here else SOURCE_POSITION + Vector3.RIGHT * 20
	_set_mode(mode)


func _unhandled_key_input(event: InputEvent) -> void:
	if "--verify" in OS.get_cmdline_user_args():
		return
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode >= KEY_1 and event.keycode <= KEY_5:
			_set_mode(event.keycode - KEY_1)
		elif event.keycode == KEY_SPACE:
			_move_source()
		elif event.keycode == KEY_C:
			orbit = not orbit
			phase = 0
			_pose_camera()


func _measure() -> float:
	for frame in 64:
		await RenderingServer.frame_post_draw
	var center := Vector2i(camera.unproject_position(TARGET))
	var sums := PackedFloat64Array()
	var squares := PackedFloat64Array()
	sums.resize(100)
	squares.resize(100)
	for frame in 32:
		await RenderingServer.frame_post_draw
		var image := get_viewport().get_texture().get_image()
		for y in range(-5, 5):
			for x in range(-5, 5):
				var value := image.get_pixel(center.x + x, center.y + y).r
				var index := (y + 5) * 10 + x + 5
				sums[index] += value
				squares[index] += value * value
	var mean := 0.0
	var sigma := 0.0
	for i in 100:
		var pixel_mean := sums[i] / 32.0
		mean += pixel_mean
		sigma += sqrt(maxf(0.0, squares[i] / 32.0 - pixel_mean * pixel_mean))
	print("%s: mean red %.4f, temporal SD %.3f / 255" % [MODES[mode], mean / 100.0, sigma * 255.0 / 100.0])
	return mean / 100.0


func _verify() -> void:
	if not ray_query:
		push_error("GI verification requires Forward+ with inline ray queries.")
		get_tree().quit(1)
		return
	marker.hide()
	_set_mode(0)
	var off := await _measure()
	_set_mode(1)
	await _measure()
	_set_mode(2)
	var single := await _measure()
	_set_mode(3)
	var bounced := await _measure()
	_move_source()
	var removed := await _measure()
	_move_source()
	_set_mode(4)
	var probe_bounced := await _measure()
	_move_source()
	var probe_removed := await _measure()
	# The emitter is back in range. Zeroing just probe energy should recover the
	# dark single-bounce reference if the reservoir is actually using probes.
	_move_source()
	environment.sdfgi_energy = 0.0
	var zero_probe_energy := await _measure()
	print("SDFGI receiver red: lit=%f  source moved=%f  zero probe energy=%f" % [probe_bounced, probe_removed, zero_probe_energy])
	print("Receiver mean red: single=%f  SHaRC=%f  source removed=%f" % [single, bounced, removed])
	get_tree().quit(0 if off < 0.02 and single < 0.02 and bounced > single + 0.025 and removed < 0.02 and probe_bounced > single + 0.025 and zero_probe_energy < 0.02 else 1)
