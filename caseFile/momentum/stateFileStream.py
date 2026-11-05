# Clean ParaView state — streamlines with arrows at z=1.5 (XY plane)
# Rewritten from auto-trace: removed all hidden/unused pipeline objects


import os
from paraview.simple import *
import math
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
paraview.simple._DisableFirstRenderCameraReset()

# ─── Data sources ─────────────────────────────────────────────────────────────

buildingsobj = WavefrontOBJReader(
    registrationName='buildings.obj',
    FileName='constant/triSurface/buildings.obj')

foamfoam = OpenFOAMReader(
    registrationName='foam.foam',
    FileName='foam.foam')
foamfoam.SkipZeroTime = 1
foamfoam.CaseType = 'Reconstructed Case'
foamfoam.LabelSize = '32-bit'
foamfoam.ScalarSize = '64-bit (DP)'
foamfoam.Createcelltopointfiltereddata = 1
foamfoam.MeshRegions = ['internalMesh']
foamfoam.CellArrays = ['U']   # only U needed


# ─── Views & animation scene ──────────────────────────────────────────────────

renderView1 = GetActiveViewOrCreate('RenderView')
animationScene1 = GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()

# ─── Velocity colour map (Rainbow Desaturated, auto-rescale per timestep) ─────

uLUT = GetColorTransferFunction('U')
uLUT.AutomaticRescaleRangeMode = 'Never'
uLUT.VectorMode = 'Magnitude'
uLUT.ScalarRangeInitialized = 0
uLUT.ApplyPreset('Rainbow Desaturated', True)

uPWF = GetOpacityTransferFunction('U')
uPWF.ScalarRangeInitialized = 0

# ─── Buildings ────────────────────────────────────────────────────────────────

buildingsobjDisplay = Show(buildingsobj, renderView1, 'GeometryRepresentation')
buildingsobjDisplay.Representation = 'Surface'
buildingsobjDisplay.ColorArrayName = [None, '']
buildingsobjDisplay.AmbientColor = [0.6, 0.6, 0.6]
buildingsobjDisplay.DiffuseColor = [0.6, 0.6, 0.6]

# ─── Site boundary outline (extract edges, raised to z=2.1) ───────────────────

site_boundary_shiftedobj = WavefrontOBJReader(
    registrationName='site_boundary_shifted.obj',
    FileName='constant/triSurface/site_boundary_shifted.obj')

extractEdges1 = ExtractEdges(
    registrationName='ExtractEdges1',
    Input=site_boundary_shiftedobj)

transform1 = Transform(
    registrationName='Transform1',
    Input=extractEdges1)
transform1.Transform = 'Transform'
transform1.TransformAllInputVectors = 1
transform1.Transform.Translate = [0.0, 0.0, 4.0]

transform1Display = Show(transform1, renderView1, 'GeometryRepresentation')
transform1Display.Representation = 'Surface'
transform1Display.ColorArrayName = [None, '']
transform1Display.AmbientColor = [0.0, 0.0, 0.0]
transform1Display.DiffuseColor = [0.0, 0.0, 0.0]
transform1Display.LineWidth = 2.0

# ─── Streamlines with arrows at z = 1.5 ──────────────────────────────────────

# Force foamfoam pipeline to execute so bounds are valid
# (foamfoam is not shown, so renderView1.Update() alone won't execute it)
UpdatePipeline(proxy=foamfoam)
bounds_info = foamfoam.GetDataInformation().GetBounds()
xmin_d, xmax_d = bounds_info[0], bounds_info[1]
ymin_d, ymax_d = bounds_info[2], bounds_info[3]
zmin_d, zmax_d = bounds_info[4], bounds_info[5]
xcen_d = 0.5 * (xmin_d + xmax_d)
ycen_d = 0.5 * (ymin_d + ymax_d)
zcen_d = 0.5 * (zmin_d + zmax_d)
span_x = xmax_d - xmin_d
span_y = ymax_d - ymin_d

# print(f"Data bounds: x[{xmin_d}, {xmax_d}], y[{ymin_d}, {ymax_d}], z[{zmin_d}, {zmax_d}]")


# ─── Define z-slice ─────────────────────────────────────────────────
from paraview import servermanager

# Get cell centers from OpenFOAM mesh
cellCenters = CellCenters(registrationName='CellCenters_for_z', Input=foamfoam)
UpdatePipeline(proxy=cellCenters)

centers_data = servermanager.Fetch(cellCenters)

pts_list = []

if centers_data.IsA('vtkCompositeDataSet'):
    it = centers_data.NewIterator()
    it.SkipEmptyNodesOn()
    it.InitTraversal()

    while not it.IsDoneWithTraversal():
        block = it.GetCurrentDataObject()
        if block is not None and block.GetNumberOfPoints() > 0:
            pts_list.append(dsa.WrapDataObject(block).Points)
        it.GoToNextItem()
else:
    pts_list.append(dsa.WrapDataObject(centers_data).Points)

centers = np.vstack(pts_list)

# Find cell centers near the lower-left mesh corner.
# xmin_d/ymin_d is a boundary corner, so use a small xy-neighborhood
# around the nearest actual cell center instead of requiring exact x/y match.
xy_dist2 = (centers[:, 0] - xmin_d)**2 + (centers[:, 1] - ymin_d)**2
nearest_i = np.argmin(xy_dist2)
nearest_xy = centers[nearest_i, :2]

search_scale = max(span_x, span_y, 1.0)
radius = 0.005 * search_scale
max_radius = 0.10 * search_scale
z_levels = np.array([])

while radius <= max_radius:
    xy_radius = np.sqrt((centers[:, 0] - nearest_xy[0])**2 +
                        (centers[:, 1] - nearest_xy[1])**2)

    corner_cells = centers[xy_radius <= radius]

    z_levels = np.unique(np.round(corner_cells[:, 2], 12))
    z_levels = np.sort(z_levels[z_levels > 0.0])

    if len(z_levels) >= 2:
        break

    radius *= 2.0

if len(z_levels) < 2:
    raise RuntimeError(
        "Could not find the second vertical cell center near xmin/ymin. "
        "Increase max_radius or check the mesh near the lower-left corner."
    )

# --- Subrat: read user-specified height offset (metres above the second cell centre) ---
HEIGHT_FILE = "slice_height_m.abc"
if os.path.isfile(HEIGHT_FILE):
    with open(HEIGHT_FILE) as fh:
        try:
            user_height = float(fh.read().strip())
        except ValueError:
            raise RuntimeError(f"{HEIGHT_FILE} does not contain a valid number")
else:
    print(f"WARNING: {HEIGHT_FILE} not found, using user_height = 0.0")
    user_height = 0.0
print("Using user_height =", user_height)


streamline_z =  user_height + z_levels[1]   # second vertical cell center above z = 0
print("Using streamline_z =", streamline_z)

# Horizontal slice at z=1.5 (velocity input for streamtracer only — not shown)
slice_z15 = Slice(registrationName='Slice_z1.5', Input=foamfoam)
slice_z15.SliceType = 'Plane'
# slice_z15.SliceType.Origin = [xcen_d, ycen_d, 2.0]
slice_z15.SliceType.Origin = [xcen_d, ycen_d, streamline_z]
slice_z15.SliceType.Normal = [0.0, 0.0, 1.0]
slice_z15.Triangulatetheslice = 1

# Full-domain seed grid at z=1.5 — seeds inside wake zones too
seedGrid = Plane(registrationName='SeedGrid_z1.5')
# seedGrid.Origin = [xmin_d, ymin_d, 2.0]
# seedGrid.Point1 = [xmax_d, ymin_d, 2.0]
# seedGrid.Point2 = [xmin_d, ymax_d, 2.0]

slice_z15.SliceType.Origin = [xcen_d, ycen_d, streamline_z]

seedGrid.Origin = [xmin_d, ymin_d, streamline_z]
seedGrid.Point1 = [xmax_d, ymin_d, streamline_z]
seedGrid.Point2 = [xmin_d, ymax_d, streamline_z]

if int(span_x/7.5) > 20 and int(span_y/7.5) > 20:
    seedGrid.XResolution = 18
    seedGrid.YResolution = 18
    print("Condition satisfied")
else:
    seedGrid.XResolution = int(span_x/7.5)
    seedGrid.YResolution = int(span_y/7.5)
    

# StreamTracer with custom seed grid
streamTracer_z15 = StreamTracerWithCustomSource(
    registrationName='StreamTracer_z1.5',
    Input=slice_z15)
streamTracer_z15.SeedSource = seedGrid
streamTracer_z15.SurfaceStreamlines = 1   # keeps lines on the 2D slice plane
streamTracer_z15.Vectors = ['POINTS', 'U']
streamTracer_z15.MaximumStreamlineLength = max(span_x, span_y)
streamTracer_z15.IntegrationDirection = 'BOTH'
streamTracer_z15.IntegratorType = 'Runge-Kutta 4-5'
streamTracer_z15.InitialStepLength = 0.2
streamTracer_z15.MinimumStepLength = 0.01
streamTracer_z15.MaximumStepLength = 0.5
streamTracer_z15.MaximumSteps = 2000

# Streamlines coloured by U magnitude (auto-rescaled to actual data range)
stDisplay = Show(streamTracer_z15, renderView1, 'GeometryRepresentation')
stDisplay.Representation = 'Surface'
stDisplay.LineWidth = 0.5
ColorBy(stDisplay, ('POINTS', 'U', 'Magnitude'))
stDisplay.LookupTable = uLUT
stDisplay.RescaleTransferFunctionToDataRange(False, True)
stDisplay.SetScalarBarVisibility(renderView1, True)

uLUTColorBar = GetScalarBar(uLUT, renderView1)
uLUTColorBar.Title = 'Wind Speed (m/s)'
uLUTColorBar.ComponentTitle = ''

# Colorbar spans exactly the data Y extent (= the contour frame).
bar_bottom = 0.0
bar_length = 1.0
uLUTColorBar.WindowLocation = 'Any Location'
uLUTColorBar.Position = [0.90, bar_bottom + bar_length * 0.1]
uLUTColorBar.ScalarBarLength = bar_length * 0.8

uLUTColorBar.LabelFormat = '%.2f'
uLUTColorBar.RangeLabelFormat = '%.2f'

# Arrow glyphs every Nth point to show flow direction
glyph_z15 = Glyph(
    registrationName='Glyph_arrows_z1.5',
    Input=streamTracer_z15,
    GlyphType='Arrow')
glyph_z15.OrientationArray = ['POINTS', 'U']
glyph_z15.ScaleArray = ['POINTS', 'No scale array']
glyph_z15.ScaleFactor = span_x * 0.05
glyph_z15.GlyphMode = 'Every Nth Point'
glyph_z15.Stride = 20

glyphDisplay = Show(glyph_z15, renderView1, 'GeometryRepresentation')
glyphDisplay.Representation = 'Surface'
glyphDisplay.ColorArrayName = [None, '']
glyphDisplay.AmbientColor = [0.0, 0.0, 0.0]
glyphDisplay.DiffuseColor = [0.0, 0.0, 0.0]

# get active source.
glyph_arrows_z15 = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# get display properties
glyph_arrows_z15Display = GetDisplayProperties(glyph_arrows_z15, view=renderView1)

# set scalar coloring
ColorBy(glyph_arrows_z15Display, ('POINTS', 'U', 'Magnitude'))

# get color transfer function/color map for 'U'
uLUT = GetColorTransferFunction('U')

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(uLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
glyph_arrows_z15Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
glyph_arrows_z15Display.SetScalarBarVisibility(renderView1, True)

# get opacity transfer function/opacity map for 'U'
uPWF = GetOpacityTransferFunction('U')

# get 2D transfer function for 'U'
uTF2D = GetTransferFunction2D('U')


renderView1.Update()

# ─── Camera: top-down, full domain ────────────────────────────────────────────

# Derive pixel dimensions from domain aspect ratio — no hardcoded values.
_right_pad     = 0.15
_view_span_x   = span_x / (1.0 - _right_pad)
_domain_aspect = _view_span_x / span_y if span_y != 0 else 1.0
_target_px     = 800  # longer-side pixel budget
if _domain_aspect >= 1.0:
    image_width_px  = _target_px
    image_height_px = max(1, int(round(_target_px / _domain_aspect)))
else:
    image_height_px = _target_px
    image_width_px  = max(1, int(round(_target_px * _domain_aspect)))

image_width  = image_width_px
image_height = image_height_px

right_pad = _right_pad

view_xmin = xmin_d
view_xmax = xmax_d + (right_pad / (1.0 - right_pad)) * span_x

view_span_x = view_xmax - view_xmin
view_xcen = 0.5 * (view_xmin + view_xmax)

domain_aspect = view_span_x / span_y if span_y != 0 else 1.0
image_aspect  = image_width / image_height

renderView1.CameraParallelProjection = 0
fov_rad = math.radians(30.0)

if domain_aspect > image_aspect:
    fit_span = view_span_x
    fit_dim  = image_width
else:
    fit_span = span_y
    fit_dim  = image_height

z_required = (fit_span / (2.0 * math.tan(fov_rad / 2.0))) * (image_height / fit_dim)
z_required *= 1.05

renderView1.CameraPosition  = [view_xcen, ycen_d, zcen_d + z_required]
renderView1.CameraFocalPoint = [view_xcen, ycen_d, zcen_d]
renderView1.CameraViewUp     = [0, 1, 0]


# ─── Layout ───────────────────────────────────────────────────────────────────

layout1 = GetLayout()
layout1.SetSize(image_width_px, image_height_px)

# find source
buildingsobj = FindSource('buildings.obj')

# set active source
SetActiveSource(buildingsobj)

# get display properties
buildingsobjDisplay = GetDisplayProperties(buildingsobj, view=renderView1)

# change solid color
buildingsobjDisplay.AmbientColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.DiffuseColor = [1.0, 1.0, 1.0]

renderView1.OrientationAxesVisibility = 0
# ─── Save animation ───────────────────────────────────────────────────────────

SaveAnimation(
    filename='temporal1.jpeg',
    viewOrLayout=renderView1,
    location=16,
    ImageResolution=[image_width_px, image_height_px],
    FontScaling='Scale fonts proportionally',
    OverrideColorPalette='WhiteBackground',
    StereoMode='No change',
    TransparentBackground=0,
    FrameRate=1,
    FrameStride=1,
    FrameWindow=[0, 149],
    Quality=100,
    Progressive=1,
    SuffixFormat='.%04d')
