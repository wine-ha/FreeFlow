"""
Render the fluid + solid VTK frame sequence produced by FreeFlow
(`render_data/fluid_frame_*.vtk` and `solid_frame_*.vtk`) into a video.

Usage:
    python scripts/vtk_to_video.py \
        --input output/swimming_forward_lbs/torus/torus/render_data \
        --output output/swimming_forward_lbs/torus/torus/simulation.mp4 \
        --fps 20 \
        --mode slice3

Visualization modes (``--mode``):
    slice       : single axis-aligned slice of |v| (2D look, fast)              [default]
    slice3      : three orthogonal slices (x/y/z) of |v|                         (pseudo-3D)
    isosurface  : |v| iso-surfaces at several levels                             (true 3D shells)
    streamlines : streamlines seeded around the swimmer                          (true 3D)
    volume      : direct volume rendering of |v|                                 (true 3D cloud)

Dependencies:
    pip install pyvista imageio imageio-ffmpeg numpy

Notes:
    * Fluid files are XML ImageData with a 3-component "velocity" array.
    * Solid files are XML UnstructuredGrid (the swimmer mesh).
    * Rendering is done off-screen, so no GUI is required.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError:
    sys.stderr.write(
        "[ERROR] pyvista is required. Install with: pip install pyvista\n")
    raise

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio  # type: ignore


FRAME_RE = re.compile(r"(?:fluid|solid)_frame_(\d+)\.vtk$", re.IGNORECASE)


def collect_frames(input_dir: Path):
    """Return a sorted list of frame indices that have both fluid & solid files."""
    fluid_ids, solid_ids = set(), set()
    for p in input_dir.iterdir():
        m = FRAME_RE.search(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if p.name.lower().startswith("fluid_"):
            fluid_ids.add(idx)
        elif p.name.lower().startswith("solid_"):
            solid_ids.add(idx)
    common = sorted(fluid_ids & solid_ids)
    if not common:
        raise RuntimeError(
            f"No matching fluid/solid VTK pairs found in {input_dir}")
    return common


def _sniff_vtk_kind(path: Path) -> str:
    """Detect the real VTK flavour by inspecting the first bytes of the file.

    The FreeFlow C++ side writes XML VTK files but saves them with a ``.vtk``
    extension, which confuses PyVista (it picks the legacy VTK reader by
    default). We therefore peek at the header ourselves.

    Returns one of: ``"vti"`` (XML ImageData), ``"vtu"`` (XML UnstructuredGrid),
    ``"vtp"`` (XML PolyData), ``"vtr"`` (XML RectilinearGrid),
    ``"vts"`` (XML StructuredGrid), or ``"legacy"``.
    """
    with open(path, "rb") as f:
        head = f.read(2048)
    text = head.decode("utf-8", errors="ignore").lower()
    if "<?xml" in text or "<vtkfile" in text:
        if 'type="imagedata"' in text:
            return "vti"
        if 'type="unstructuredgrid"' in text:
            return "vtu"
        if 'type="polydata"' in text:
            return "vtp"
        if 'type="rectilineargrid"' in text:
            return "vtr"
        if 'type="structuredgrid"' in text:
            return "vts"
        # Unknown XML flavour; fall back to PyVista auto-detection
        return "xml"
    return "legacy"


def _read_vtk_any(path: Path):
    """Read a FreeFlow ``.vtk`` file regardless of whether it is XML or legacy."""
    kind = _sniff_vtk_kind(path)
    if kind == "vti":
        import vtk  # local import so pyvista handles vtk wheels for us
        reader = vtk.vtkXMLImageDataReader()
    elif kind == "vtu":
        import vtk
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif kind == "vtp":
        import vtk
        reader = vtk.vtkXMLPolyDataReader()
    elif kind == "vtr":
        import vtk
        reader = vtk.vtkXMLRectilinearGridReader()
    elif kind == "vts":
        import vtk
        reader = vtk.vtkXMLStructuredGridReader()
    else:
        # Legacy .vtk or unknown -> let PyVista figure it out
        return pv.read(str(path))

    reader.SetFileName(str(path))
    reader.Update()
    return pv.wrap(reader.GetOutput())


def load_fluid(path: Path):
    mesh = _read_vtk_any(path)
    # the C++ side writes a vector field named "velocity"
    if "velocity" in mesh.array_names:
        v = np.asarray(mesh["velocity"])
        mesh["vmag"] = np.linalg.norm(v, axis=1).astype(np.float32)
    return mesh


def load_solid(path: Path):
    return _read_vtk_any(path)


def compute_global_vmag_range(input_dir: Path, frame_ids, sample=8):
    """Scan a few frames to pick a stable color range."""
    picks = np.linspace(0, len(frame_ids) - 1, min(sample, len(frame_ids))).astype(int)
    lo, hi = np.inf, -np.inf
    for i in picks:
        fid = frame_ids[i]
        mesh = load_fluid(input_dir / f"fluid_frame_{fid}.vtk")
        if "vmag" not in mesh.array_names:
            continue
        vmag = mesh["vmag"]
        lo = min(lo, float(np.percentile(vmag, 1)))
        hi = max(hi, float(np.percentile(vmag, 99)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 0.1
    return lo, hi


def make_slice(fluid_mesh, axis: str):
    axis = axis.lower()
    bounds = fluid_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    if axis == "x":
        origin, normal = (cx, cy, cz), (1, 0, 0)
    elif axis == "y":
        origin, normal = (cx, cy, cz), (0, 1, 0)
    else:  # z
        origin, normal = (cx, cy, cz), (0, 0, 1)
    return fluid_mesh.slice(normal=normal, origin=origin)


def make_ortho_slices(fluid_mesh):
    """Return a MultiBlock of three orthogonal slices through the fluid domain centre."""
    bounds = fluid_mesh.bounds
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    return fluid_mesh.slice_orthogonal(x=center[0], y=center[1], z=center[2])


def make_isosurface(fluid_mesh, levels):
    """Extract |v| iso-surfaces at the given levels (returns a PolyData)."""
    if "vmag" not in fluid_mesh.array_names:
        return None
    try:
        return fluid_mesh.contour(isosurfaces=list(levels), scalars="vmag")
    except Exception as e:
        print(f"[WARN] isosurface failed: {e}")
        return None


def make_streamlines(fluid_mesh, solid_mesh, n_points=200, max_time=200.0):
    """Seed streamlines on a sphere around the swimmer centre."""
    if "velocity" not in fluid_mesh.array_names:
        return None
    sb = solid_mesh.bounds
    center = (
        0.5 * (sb[0] + sb[1]),
        0.5 * (sb[2] + sb[3]),
        0.5 * (sb[4] + sb[5]),
    )
    radius = 0.5 * max(sb[1] - sb[0], sb[3] - sb[2], sb[5] - sb[4])
    radius = max(radius, 1e-6) * 1.5
    try:
        return fluid_mesh.streamlines(
            vectors="velocity",
            source_center=center,
            source_radius=radius,
            n_points=n_points,
            max_time=max_time,
            integration_direction="both",
        )
    except Exception as e:
        print(f"[WARN] streamlines failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Directory containing fluid_frame_*.vtk / solid_frame_*.vtk")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output video file (.mp4 / .gif). Defaults to <input>/../simulation.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--size", nargs=2, type=int, default=[1280, 720],
                        metavar=("W", "H"))
    parser.add_argument("--mode", choices=["slice", "slice3", "isosurface",
                                           "streamlines", "volume"],
                        default="slice",
                        help="How to visualize the 3D fluid field")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default="z",
                        help="Axis of the slice plane (only used when --mode=slice)")
    parser.add_argument("--iso-levels", type=float, nargs="+",
                        default=None,
                        help="|v| iso-surface levels (only used when --mode=isosurface). "
                             "Default: 5 levels spanning the estimated range.")
    parser.add_argument("--stream-points", type=int, default=200,
                        help="Number of seed points for streamlines")
    parser.add_argument("--opacity", default="sigmoid",
                        help="Opacity transfer function for volume rendering "
                             "(pyvista preset name or single float)")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--solid-color", default="#c94f7c")
    parser.add_argument("--bg-color", default="white")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only render the first N frames (for quick tests)")
    parser.add_argument("--quality", type=int, default=8,
                        help="Video encoder quality 1-10 (imageio ffmpeg)")
    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    output_path: Path = (
        args.output if args.output is not None
        else input_dir.parent / "simulation.mp4"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_ids = collect_frames(input_dir)
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]
    print(f"[INFO] Found {len(frame_ids)} frames in {input_dir}")
    print(f"[INFO] Visualization mode: {args.mode}")

    # Decide global color range from a few sample frames
    if args.vmin is None or args.vmax is None:
        print("[INFO] Sampling frames to estimate color range...")
        lo, hi = compute_global_vmag_range(input_dir, frame_ids)
        vmin = args.vmin if args.vmin is not None else lo
        vmax = args.vmax if args.vmax is not None else hi
    else:
        vmin, vmax = args.vmin, args.vmax
    print(f"[INFO] Using |v| color range [{vmin:.4g}, {vmax:.4g}]")

    # Default iso-levels if needed
    if args.mode == "isosurface" and args.iso_levels is None:
        # skip the very bottom to avoid a level that fills the whole domain
        args.iso_levels = list(np.linspace(vmin + 0.2 * (vmax - vmin),
                                           vmax * 0.9, 5))
        print(f"[INFO] Auto iso-levels: {['%.4g' % v for v in args.iso_levels]}")

    # Set up off-screen plotter
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, window_size=args.size)
    plotter.set_background(args.bg_color)

    # Initialise writer
    suffix = output_path.suffix.lower()
    writer_kwargs = {"fps": args.fps}
    if suffix == ".mp4":
        writer_kwargs.update({"quality": args.quality, "codec": "libx264",
                              "pixelformat": "yuv420p", "macro_block_size": 1})
    print(f"[INFO] Writing -> {output_path}")
    writer = imageio.get_writer(str(output_path), **writer_kwargs)

    # Build the initial scene
    first_id = frame_ids[0]
    fluid0 = load_fluid(input_dir / f"fluid_frame_{first_id}.vtk")
    solid0 = load_solid(input_dir / f"solid_frame_{first_id}.vtk")

    # Bounding box outline of the fluid domain for spatial reference
    plotter.add_mesh(fluid0.outline(), color="gray", line_width=1)

    fluid_actors = []   # list so we can replace per-frame
    volume_actor = None

    def _build_fluid_geometry(fluid, solid):
        """Compute the geometry to render for the fluid part at this frame."""
        if args.mode == "slice":
            return make_slice(fluid, args.slice_axis)
        if args.mode == "slice3":
            return make_ortho_slices(fluid)
        if args.mode == "isosurface":
            return make_isosurface(fluid, args.iso_levels)
        if args.mode == "streamlines":
            return make_streamlines(fluid, solid, n_points=args.stream_points)
        return None  # "volume" handled separately

    # ---- initial actors for the fluid -----------------------------------
    if args.mode == "volume":
        # Volume rendering: needs ImageData with the scalar field
        volume_actor = plotter.add_volume(
            fluid0, scalars="vmag", cmap=args.cmap,
            clim=(vmin, vmax), opacity=args.opacity,
            shade=False,
        )
    elif args.mode == "slice3":
        geom0 = _build_fluid_geometry(fluid0, solid0)
        a = plotter.add_mesh(
            geom0, scalars="vmag", cmap=args.cmap, clim=(vmin, vmax),
            scalar_bar_args={"title": "|velocity|", "n_labels": 4},
        )
        fluid_actors.append(a)
    elif args.mode == "isosurface":
        geom0 = _build_fluid_geometry(fluid0, solid0)
        if geom0 is not None and geom0.n_points > 0:
            a = plotter.add_mesh(
                geom0, scalars="vmag", cmap=args.cmap, clim=(vmin, vmax),
                opacity=0.35, smooth_shading=True,
                scalar_bar_args={"title": "|velocity|", "n_labels": 4},
            )
            fluid_actors.append(a)
    elif args.mode == "streamlines":
        geom0 = _build_fluid_geometry(fluid0, solid0)
        if geom0 is not None and geom0.n_points > 0:
            # render as thin tubes coloured by |v|
            tubes = geom0.tube(radius=None)
            a = plotter.add_mesh(
                tubes, scalars="vmag", cmap=args.cmap, clim=(vmin, vmax),
                scalar_bar_args={"title": "|velocity|", "n_labels": 4},
            )
            fluid_actors.append(a)
    else:  # slice
        geom0 = _build_fluid_geometry(fluid0, solid0)
        a = plotter.add_mesh(
            geom0, scalars="vmag", cmap=args.cmap, clim=(vmin, vmax),
            scalar_bar_args={"title": "|velocity|", "n_labels": 4},
        )
        fluid_actors.append(a)

    solid_actor = plotter.add_mesh(
        solid0, color=args.solid_color, smooth_shading=True,
        show_edges=False,
    )

    # Fixed isometric camera
    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.camera.zoom(1.1)

    try:
        for step, fid in enumerate(frame_ids):
            fluid = load_fluid(input_dir / f"fluid_frame_{fid}.vtk")
            solid = load_solid(input_dir / f"solid_frame_{fid}.vtk")

            # --- update fluid visualization -----------------------------
            if args.mode == "volume":
                # Replace the volume mapper input with the new ImageData
                if volume_actor is not None:
                    volume_actor.mapper.SetInputData(fluid)
            else:
                # Modes that need re-extracted geometry every frame
                # The simplest (and most robust) way is to remove old actors
                # and add new ones. This is a bit more expensive but avoids
                # issues with MultiBlock inputs / topology changes.
                for a in fluid_actors:
                    plotter.remove_actor(a, render=False)
                fluid_actors.clear()

                geom = _build_fluid_geometry(fluid, solid)
                if geom is not None and getattr(geom, "n_points", 1) > 0:
                    if args.mode == "isosurface":
                        a = plotter.add_mesh(
                            geom, scalars="vmag", cmap=args.cmap,
                            clim=(vmin, vmax), opacity=0.35,
                            smooth_shading=True,
                            show_scalar_bar=False,
                        )
                    elif args.mode == "streamlines":
                        tubes = geom.tube(radius=None)
                        a = plotter.add_mesh(
                            tubes, scalars="vmag", cmap=args.cmap,
                            clim=(vmin, vmax), show_scalar_bar=False,
                        )
                    else:  # slice / slice3
                        a = plotter.add_mesh(
                            geom, scalars="vmag", cmap=args.cmap,
                            clim=(vmin, vmax), show_scalar_bar=False,
                        )
                    fluid_actors.append(a)

            # --- update solid mesh --------------------------------------
            solid_actor.mapper.SetInputData(solid)

            plotter.add_text(
                f"frame {fid}", name="frame_label",
                position="upper_left", font_size=12, color="black",
            )

            plotter.render()
            img = plotter.screenshot(return_img=True)
            writer.append_data(img)

            if (step + 1) % 10 == 0 or step + 1 == len(frame_ids):
                print(f"[INFO]   rendered {step + 1}/{len(frame_ids)}")
    finally:
        writer.close()
        plotter.close()

    print(f"[DONE] Video saved: {output_path}")


if __name__ == "__main__":
    main()
