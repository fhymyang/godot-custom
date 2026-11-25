# GI Lab

Open this project with the custom Forward+ engine and run the main scene. The emissive panel is the only light source. A black baffle blocks the direct path from the marked floor patch to the emitter. The white back wall provides a path for diffuse interreflection.

| Key | Action |
| --- | --- |
| 1 | Disable GI for a dark reference |
| 2 | Use hybrid diffuse GI |
| 3 | Use the single-bounce Monte Carlo reference |
| 4 | Use ReSTIR with the spatial radiance cache |
| 5 | Use the same ReSTIR reservoirs with SDFGI diffuse probes at secondary hits |
| Space | Move the emitter out of range or restore it |
| C | Orbit slowly or return to the reference camera |

Modes 3 through 5 require inline ray queries. Compare the marked patch with the camera still, then orbit to inspect disocclusion. Move the emitter to inspect how quickly cached illumination disappears.

Run the automated check from the engine repository on macOS:

```sh
bin/godot.macos.template_release.arm64 --path tests/servers/rendering/gi_lab --rendering-driver metal --gpu-validation -- --verify
```

The check warms each configuration for 64 frames, then reads the central 10 × 10 pixels around the marker for 32 frames with the overlay marker hidden. It reports mean red and mean per-pixel temporal standard deviation in 8-bit red-channel units. It checks the dark GI-off and single-bounce references, the multi-bounce signal, and its disappearance after the emitter moves. Temporal deviation and hybrid lighting are reported for comparison.

For the SHaRC/SDFGI comparison, switch between 4 and 5 at the same camera position. The SDFGI configuration uses 0.1 m cells in its finest cascade, occlusion enabled, and sky injection disabled. The emitter is static GI geometry. Moving it tests the limitations of updating voxelized emissive geometry. The automated check reports the SDFGI residual separately.
