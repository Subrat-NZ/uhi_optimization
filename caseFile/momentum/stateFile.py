# trace generated using paraview version 5.13.3
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 13

#### import the simple module from the paraview
import os
from paraview.simple import *
from paraview import servermanager
from vtk.numpy_interface import dataset_adapter as dsa
import numpy as np
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


# create a new 'Wavefront OBJ Reader'
buildingsobj = WavefrontOBJReader(registrationName='buildings.obj', FileName='constant/triSurface/buildings.obj')

# create a new 'Open FOAM Reader'
foamfoam = OpenFOAMReader(registrationName='foam.foam', FileName='foam.foam')
foamfoam.SkipZeroTime = 1
foamfoam.CaseType = 'Reconstructed Case'
foamfoam.LabelSize = '32-bit'
foamfoam.ScalarSize = '64-bit (DP)'
foamfoam.Createcelltopointfiltereddata = 1
foamfoam.Adddimensionalunitstoarraynames = 0
foamfoam.MeshRegions = ['internalMesh']
foamfoam.CellArrays = ['U', 'k', 'nut', 'omega', 'p']
foamfoam.PointArrays = []
foamfoam.LagrangianArrays = []
foamfoam.Cachemesh = 1
foamfoam.WeightPointDataByCellSize = 0
foamfoam.ListtimestepsaccordingtocontrolDict = 0
foamfoam.Lagrangianpositionswithoutextradata = 1
foamfoam.Readzones = 0
foamfoam.Copydatatocellzones = 0

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')


# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# show data in view
foamfoamDisplay = Show(foamfoam, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'p'
pTF2D = GetTransferFunction2D('p')
pTF2D.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
pTF2D.Boxes = []
pTF2D.ScalarRangeInitialized = 0
pTF2D.Range = [0.0, 1.0, 0.0, 1.0]
pTF2D.OutputDimensions = [10, 10]

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')
pLUT.InterpretValuesAsCategories = 0
pLUT.AnnotationsInitialized = 0
pLUT.ShowCategoricalColorsinDataRangeOnly = 0
pLUT.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
pLUT.RescaleOnVisibilityChange = 0
pLUT.TransferFunction2D = pTF2D
pLUT.RGBPoints = [101200.5, 0.231373, 0.298039, 0.752941, 101320.125, 0.865003, 0.865003, 0.865003, 101439.75, 0.705882, 0.0156863, 0.14902]
pLUT.UseLogScale = 0
pLUT.UseOpacityControlPointsFreehandDrawing = 0
pLUT.ShowDataHistogram = 0
pLUT.AutomaticDataHistogramComputation = 0
pLUT.DataHistogramNumberOfBins = 10
pLUT.ColorSpace = 'Diverging'
pLUT.UseBelowRangeColor = 0
pLUT.BelowRangeColor = [0.0, 0.0, 0.0]
pLUT.UseAboveRangeColor = 0
pLUT.AboveRangeColor = [0.5, 0.5, 0.5]
pLUT.NanColor = [1.0, 1.0, 0.0]
pLUT.NanOpacity = 1.0
pLUT.Discretize = 1
pLUT.NumberOfTableValues = 256
pLUT.ScalarRangeInitialized = 1.0
pLUT.HSVWrap = 0
pLUT.VectorComponent = 0
pLUT.VectorMode = 'Magnitude'
pLUT.AllowDuplicateScalars = 1
pLUT.Annotations = []
pLUT.ActiveAnnotatedValues = []
pLUT.IndexedColors = []
pLUT.IndexedOpacities = []
pLUT.EnableOpacityMapping = 0

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')
pPWF.Points = [101200.5, 0.0, 0.5, 0.0, 101439.75, 1.0, 0.5, 0.0]
pPWF.AllowDuplicateScalars = 1
pPWF.UseLogScale = 0
pPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
foamfoamDisplay.Selection = None
foamfoamDisplay.Representation = 'Surface'
foamfoamDisplay.ColorArrayName = ['POINTS', 'p']
foamfoamDisplay.LookupTable = pLUT
foamfoamDisplay.MapScalars = 1
foamfoamDisplay.MultiComponentsMapping = 0
foamfoamDisplay.InterpolateScalarsBeforeMapping = 1
foamfoamDisplay.UseNanColorForMissingArrays = 0
foamfoamDisplay.Opacity = 1.0
foamfoamDisplay.PointSize = 2.0
foamfoamDisplay.LineWidth = 1.0
foamfoamDisplay.RenderLinesAsTubes = 0
foamfoamDisplay.RenderPointsAsSpheres = 0
foamfoamDisplay.DisableLighting = 0
foamfoamDisplay.Diffuse = 1.0
foamfoamDisplay.Interpolation = 'Gouraud'
foamfoamDisplay.Specular = 0.0
foamfoamDisplay.SpecularColor = [1.0, 1.0, 1.0]
foamfoamDisplay.SpecularPower = 100.0
foamfoamDisplay.Luminosity = 0.0
foamfoamDisplay.Ambient = 0.0
foamfoamDisplay.Roughness = 0.3
foamfoamDisplay.Metallic = 0.0
foamfoamDisplay.EdgeTint = [1.0, 1.0, 1.0]
foamfoamDisplay.Anisotropy = 0.0
foamfoamDisplay.AnisotropyRotation = 0.0
foamfoamDisplay.BaseIOR = 1.5
foamfoamDisplay.CoatStrength = 0.0
foamfoamDisplay.CoatIOR = 2.0
foamfoamDisplay.CoatRoughness = 0.0
foamfoamDisplay.CoatColor = [1.0, 1.0, 1.0]
foamfoamDisplay.SelectNormalArray = 'None'
foamfoamDisplay.SelectTangentArray = 'None'
foamfoamDisplay.ComputePointNormals = 0
foamfoamDisplay.Splitting = 1
foamfoamDisplay.FeatureAngle = 30.0
foamfoamDisplay.SelectTCoordArray = 'None'
foamfoamDisplay.Texture = None
foamfoamDisplay.RepeatTextures = 1
foamfoamDisplay.InterpolateTextures = 0
foamfoamDisplay.SeamlessU = 0
foamfoamDisplay.SeamlessV = 0
foamfoamDisplay.UseMipmapTextures = 0
foamfoamDisplay.ShowTexturesOnBackface = 1
foamfoamDisplay.BaseColorTexture = None
foamfoamDisplay.NormalTexture = None
foamfoamDisplay.NormalScale = 1.0
foamfoamDisplay.CoatNormalTexture = None
foamfoamDisplay.CoatNormalScale = 1.0
foamfoamDisplay.MaterialTexture = None
foamfoamDisplay.OcclusionStrength = 1.0
foamfoamDisplay.AnisotropyTexture = None
foamfoamDisplay.EmissiveTexture = None
foamfoamDisplay.EmissiveFactor = [1.0, 1.0, 1.0]
foamfoamDisplay.TextureTransform = 'Transform2'
foamfoamDisplay.EdgeOpacity = 1.0
foamfoamDisplay.BackfaceRepresentation = 'Follow Frontface'
foamfoamDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
foamfoamDisplay.BackfaceOpacity = 1.0
foamfoamDisplay.Translation = [0.0, 0.0, 0.0]
foamfoamDisplay.Scale = [1.0, 1.0, 1.0]
foamfoamDisplay.Orientation = [0.0, 0.0, 0.0]
foamfoamDisplay.Origin = [0.0, 0.0, 0.0]
foamfoamDisplay.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
foamfoamDisplay.Pickable = 1
foamfoamDisplay.Triangulate = 0
foamfoamDisplay.UseShaderReplacements = 0
foamfoamDisplay.ShaderReplacements = ''
foamfoamDisplay.NonlinearSubdivisionLevel = 1
foamfoamDisplay.MatchBoundariesIgnoringCellOrder = 0
foamfoamDisplay.UseDataPartitions = 0
foamfoamDisplay.OSPRayUseScaleArray = 'All Approximate'
foamfoamDisplay.OSPRayScaleArray = 'p'
foamfoamDisplay.OSPRayScaleFunction = 'Piecewise Function'
foamfoamDisplay.OSPRayMaterial = 'None'
foamfoamDisplay.Assembly = 'Hierarchy'
foamfoamDisplay.SelectedBlockSelectors = ['']
foamfoamDisplay.BlockSelectors = ['/']
foamfoamDisplay.BlockColors = []
foamfoamDisplay.BlockColorArrayNames = []
foamfoamDisplay.BlockLookupTables = []
foamfoamDisplay.BlockUseSeparateColorMaps = []
foamfoamDisplay.BlockMapScalars = []
foamfoamDisplay.BlockInterpolateScalarsBeforeMappings = []
foamfoamDisplay.BlockOpacities = []
foamfoamDisplay.BlockMapScalarsGUI = 1
foamfoamDisplay.BlockInterpolateScalarsBeforeMappingsGUI = 1
foamfoamDisplay.BlockOpacitiesGUI = 1.0
foamfoamDisplay.Orient = 0
foamfoamDisplay.OrientationMode = 'Direction'
foamfoamDisplay.SelectOrientationVectors = 'U'
foamfoamDisplay.Scaling = 0
foamfoamDisplay.ScaleMode = 'No Data Scaling Off'
foamfoamDisplay.ScaleFactor = 24.811399841308596
foamfoamDisplay.SelectScaleArray = 'p'
foamfoamDisplay.GlyphType = 'Arrow'
foamfoamDisplay.UseGlyphTable = 0
foamfoamDisplay.GlyphTableIndexArray = 'p'
foamfoamDisplay.UseCompositeGlyphTable = 0
foamfoamDisplay.UseGlyphCullingAndLOD = 0
foamfoamDisplay.LODValues = []
foamfoamDisplay.ColorByLODIndex = 0
foamfoamDisplay.GaussianRadius = 1.2405699920654296
foamfoamDisplay.ShaderPreset = 'Sphere'
foamfoamDisplay.CustomTriangleScale = 3
foamfoamDisplay.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
foamfoamDisplay.Emissive = 0
foamfoamDisplay.ScaleByArray = 0
foamfoamDisplay.SetScaleArray = ['POINTS', 'p']
foamfoamDisplay.ScaleArrayComponent = ''
foamfoamDisplay.UseScaleFunction = 1
foamfoamDisplay.ScaleTransferFunction = 'Piecewise Function'
foamfoamDisplay.OpacityByArray = 0
foamfoamDisplay.OpacityArray = ['POINTS', 'p']
foamfoamDisplay.OpacityArrayComponent = ''
foamfoamDisplay.OpacityTransferFunction = 'Piecewise Function'
foamfoamDisplay.DataAxesGrid = 'Grid Axes Representation'
foamfoamDisplay.SelectionCellLabelBold = 0
foamfoamDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
foamfoamDisplay.SelectionCellLabelFontFamily = 'Arial'
foamfoamDisplay.SelectionCellLabelFontFile = ''
foamfoamDisplay.SelectionCellLabelFontSize = 18
foamfoamDisplay.SelectionCellLabelItalic = 0
foamfoamDisplay.SelectionCellLabelJustification = 'Left'
foamfoamDisplay.SelectionCellLabelOpacity = 1.0
foamfoamDisplay.SelectionCellLabelShadow = 0
foamfoamDisplay.SelectionPointLabelBold = 0
foamfoamDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
foamfoamDisplay.SelectionPointLabelFontFamily = 'Arial'
foamfoamDisplay.SelectionPointLabelFontFile = ''
foamfoamDisplay.SelectionPointLabelFontSize = 18
foamfoamDisplay.SelectionPointLabelItalic = 0
foamfoamDisplay.SelectionPointLabelJustification = 'Left'
foamfoamDisplay.SelectionPointLabelOpacity = 1.0
foamfoamDisplay.SelectionPointLabelShadow = 0
foamfoamDisplay.PolarAxes = 'Polar Axes Representation'
foamfoamDisplay.ScalarOpacityFunction = pPWF
foamfoamDisplay.ScalarOpacityUnitDistance = 8.094451459249287
foamfoamDisplay.UseSeparateOpacityArray = 0
foamfoamDisplay.OpacityArrayName = ['POINTS', 'p']
foamfoamDisplay.OpacityComponent = ''
foamfoamDisplay.SelectMapper = 'Projected tetra'
foamfoamDisplay.SamplingDimensions = [128, 128, 128]
foamfoamDisplay.UseFloatingPointFrameBuffer = 1
foamfoamDisplay.SelectInputVectors = ['POINTS', 'U']
foamfoamDisplay.NumberOfSteps = 40
foamfoamDisplay.StepSize = 0.25
foamfoamDisplay.NormalizeVectors = 1
foamfoamDisplay.EnhancedLIC = 1
foamfoamDisplay.ColorMode = 'Blend'
foamfoamDisplay.LICIntensity = 0.8
foamfoamDisplay.MapModeBias = 0.0
foamfoamDisplay.EnhanceContrast = 'Off'
foamfoamDisplay.LowLICContrastEnhancementFactor = 0.0
foamfoamDisplay.HighLICContrastEnhancementFactor = 0.0
foamfoamDisplay.LowColorContrastEnhancementFactor = 0.0
foamfoamDisplay.HighColorContrastEnhancementFactor = 0.0
foamfoamDisplay.AntiAlias = 0
foamfoamDisplay.MaskOnSurface = 1
foamfoamDisplay.MaskThreshold = 0.0
foamfoamDisplay.MaskIntensity = 0.0
foamfoamDisplay.MaskColor = [0.5, 0.5, 0.5]
foamfoamDisplay.GenerateNoiseTexture = 0
foamfoamDisplay.NoiseType = 'Gaussian'
foamfoamDisplay.NoiseTextureSize = 128
foamfoamDisplay.NoiseGrainSize = 2
foamfoamDisplay.MinNoiseValue = 0.0
foamfoamDisplay.MaxNoiseValue = 0.8
foamfoamDisplay.NumberOfNoiseLevels = 1024
foamfoamDisplay.ImpulseNoiseProbability = 1.0
foamfoamDisplay.ImpulseNoiseBackgroundValue = 0.0
foamfoamDisplay.NoiseGeneratorSeed = 1
foamfoamDisplay.CompositeStrategy = 'AUTO'
foamfoamDisplay.UseLICForLOD = 0
foamfoamDisplay.WriteLog = ''

# init the 'Transform2' selected for 'TextureTransform'
foamfoamDisplay.TextureTransform.Translate = [0.0, 0.0, 0.0]
foamfoamDisplay.TextureTransform.Rotate = [0.0, 0.0, 0.0]
foamfoamDisplay.TextureTransform.Scale = [1.0, 1.0, 1.0]

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
foamfoamDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
foamfoamDisplay.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
foamfoamDisplay.GlyphType.TipResolution = 6
foamfoamDisplay.GlyphType.TipRadius = 0.1
foamfoamDisplay.GlyphType.TipLength = 0.35
foamfoamDisplay.GlyphType.ShaftResolution = 6
foamfoamDisplay.GlyphType.ShaftRadius = 0.03
foamfoamDisplay.GlyphType.Invert = 0

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
foamfoamDisplay.ScaleTransferFunction.Points = [101200.5, 0.0, 0.5, 0.0, 101439.75, 1.0, 0.5, 0.0]
foamfoamDisplay.ScaleTransferFunction.UseLogScale = 0

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
foamfoamDisplay.OpacityTransferFunction.Points = [101200.5, 0.0, 0.5, 0.0, 101439.75, 1.0, 0.5, 0.0]
foamfoamDisplay.OpacityTransferFunction.UseLogScale = 0

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
foamfoamDisplay.DataAxesGrid.XTitle = 'X Axis'
foamfoamDisplay.DataAxesGrid.YTitle = 'Y Axis'
foamfoamDisplay.DataAxesGrid.ZTitle = 'Z Axis'
foamfoamDisplay.DataAxesGrid.XTitleFontFamily = 'Arial'
foamfoamDisplay.DataAxesGrid.XTitleFontFile = ''
foamfoamDisplay.DataAxesGrid.XTitleBold = 0
foamfoamDisplay.DataAxesGrid.XTitleItalic = 0
foamfoamDisplay.DataAxesGrid.XTitleFontSize = 12
foamfoamDisplay.DataAxesGrid.XTitleShadow = 0
foamfoamDisplay.DataAxesGrid.XTitleOpacity = 1.0
foamfoamDisplay.DataAxesGrid.YTitleFontFamily = 'Arial'
foamfoamDisplay.DataAxesGrid.YTitleFontFile = ''
foamfoamDisplay.DataAxesGrid.YTitleBold = 0
foamfoamDisplay.DataAxesGrid.YTitleItalic = 0
foamfoamDisplay.DataAxesGrid.YTitleFontSize = 12
foamfoamDisplay.DataAxesGrid.YTitleShadow = 0
foamfoamDisplay.DataAxesGrid.YTitleOpacity = 1.0
foamfoamDisplay.DataAxesGrid.ZTitleFontFamily = 'Arial'
foamfoamDisplay.DataAxesGrid.ZTitleFontFile = ''
foamfoamDisplay.DataAxesGrid.ZTitleBold = 0
foamfoamDisplay.DataAxesGrid.ZTitleItalic = 0
foamfoamDisplay.DataAxesGrid.ZTitleFontSize = 12
foamfoamDisplay.DataAxesGrid.ZTitleShadow = 0
foamfoamDisplay.DataAxesGrid.ZTitleOpacity = 1.0
foamfoamDisplay.DataAxesGrid.FacesToRender = 63
foamfoamDisplay.DataAxesGrid.CullBackface = 0
foamfoamDisplay.DataAxesGrid.CullFrontface = 1
foamfoamDisplay.DataAxesGrid.ShowGrid = 0
foamfoamDisplay.DataAxesGrid.ShowEdges = 1
foamfoamDisplay.DataAxesGrid.ShowTicks = 1
foamfoamDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 1
foamfoamDisplay.DataAxesGrid.AxesToLabel = 63
foamfoamDisplay.DataAxesGrid.XLabelFontFamily = 'Arial'
foamfoamDisplay.DataAxesGrid.XLabelFontFile = ''
foamfoamDisplay.DataAxesGrid.XLabelBold = 0
foamfoamDisplay.DataAxesGrid.XLabelItalic = 0
foamfoamDisplay.DataAxesGrid.XLabelFontSize = 12
foamfoamDisplay.DataAxesGrid.XLabelShadow = 0
foamfoamDisplay.DataAxesGrid.XLabelOpacity = 1.0
foamfoamDisplay.DataAxesGrid.YLabelFontFamily = 'Arial'
foamfoamDisplay.DataAxesGrid.YLabelFontFile = ''
foamfoamDisplay.DataAxesGrid.YLabelBold = 0
foamfoamDisplay.DataAxesGrid.YLabelItalic = 0
foamfoamDisplay.DataAxesGrid.YLabelFontSize = 12
foamfoamDisplay.DataAxesGrid.YLabelShadow = 0
foamfoamDisplay.DataAxesGrid.YLabelOpacity = 1.0
foamfoamDisplay.DataAxesGrid.ZLabelFontFamily = 'Arial'
foamfoamDisplay.DataAxesGrid.ZLabelFontFile = ''
foamfoamDisplay.DataAxesGrid.ZLabelBold = 0
foamfoamDisplay.DataAxesGrid.ZLabelItalic = 0
foamfoamDisplay.DataAxesGrid.ZLabelFontSize = 12
foamfoamDisplay.DataAxesGrid.ZLabelShadow = 0
foamfoamDisplay.DataAxesGrid.ZLabelOpacity = 1.0
foamfoamDisplay.DataAxesGrid.XAxisNotation = 'Mixed'
foamfoamDisplay.DataAxesGrid.XAxisPrecision = 2
foamfoamDisplay.DataAxesGrid.XAxisUseCustomLabels = 0
foamfoamDisplay.DataAxesGrid.XAxisLabels = []
foamfoamDisplay.DataAxesGrid.YAxisNotation = 'Mixed'
foamfoamDisplay.DataAxesGrid.YAxisPrecision = 2
foamfoamDisplay.DataAxesGrid.YAxisUseCustomLabels = 0
foamfoamDisplay.DataAxesGrid.YAxisLabels = []
foamfoamDisplay.DataAxesGrid.ZAxisNotation = 'Mixed'
foamfoamDisplay.DataAxesGrid.ZAxisPrecision = 2
foamfoamDisplay.DataAxesGrid.ZAxisUseCustomLabels = 0
foamfoamDisplay.DataAxesGrid.ZAxisLabels = []
foamfoamDisplay.DataAxesGrid.UseCustomBounds = 0
foamfoamDisplay.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
foamfoamDisplay.PolarAxes.Visibility = 0
foamfoamDisplay.PolarAxes.Translation = [0.0, 0.0, 0.0]
foamfoamDisplay.PolarAxes.Scale = [1.0, 1.0, 1.0]
foamfoamDisplay.PolarAxes.Orientation = [0.0, 0.0, 0.0]
foamfoamDisplay.PolarAxes.EnableCustomBounds = [0, 0, 0]
foamfoamDisplay.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
foamfoamDisplay.PolarAxes.EnableCustomRange = 0
foamfoamDisplay.PolarAxes.CustomRange = [0.0, 1.0]
foamfoamDisplay.PolarAxes.AutoPole = 1
foamfoamDisplay.PolarAxes.PolarAxisVisibility = 1
foamfoamDisplay.PolarAxes.RadialAxesVisibility = 1
foamfoamDisplay.PolarAxes.DrawRadialGridlines = 1
foamfoamDisplay.PolarAxes.PolarArcsVisibility = 1
foamfoamDisplay.PolarAxes.DrawPolarArcsGridlines = 1
foamfoamDisplay.PolarAxes.NumberOfRadialAxes = 0
foamfoamDisplay.PolarAxes.DeltaAngleRadialAxes = 45.0
foamfoamDisplay.PolarAxes.NumberOfPolarAxes = 5
foamfoamDisplay.PolarAxes.DeltaRangePolarAxes = 0.0
foamfoamDisplay.PolarAxes.CustomMinRadius = 1
foamfoamDisplay.PolarAxes.MinimumRadius = 0.0
foamfoamDisplay.PolarAxes.CustomMaxRadius = 0
foamfoamDisplay.PolarAxes.MaximumRadius = 1.0
foamfoamDisplay.PolarAxes.CustomAngles = 1
foamfoamDisplay.PolarAxes.MinimumAngle = 0.0
foamfoamDisplay.PolarAxes.MaximumAngle = 90.0
foamfoamDisplay.PolarAxes.RadialAxesOriginToPolarAxis = 1
foamfoamDisplay.PolarAxes.PolarArcResolutionPerDegree = 0.2
foamfoamDisplay.PolarAxes.Ratio = 1.0
foamfoamDisplay.PolarAxes.EnableOverallColor = 1
foamfoamDisplay.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
foamfoamDisplay.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
foamfoamDisplay.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
foamfoamDisplay.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
foamfoamDisplay.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
foamfoamDisplay.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
foamfoamDisplay.PolarAxes.PolarAxisTitleVisibility = 1
foamfoamDisplay.PolarAxes.PolarAxisTitle = 'Radial Distance'
foamfoamDisplay.PolarAxes.PolarAxisTitleLocation = 'Bottom'
foamfoamDisplay.PolarAxes.PolarTitleOffset = [20.0, 20.0]
foamfoamDisplay.PolarAxes.PolarLabelVisibility = 1
foamfoamDisplay.PolarAxes.PolarLabelFormat = '%-#6.3g'
foamfoamDisplay.PolarAxes.PolarLabelExponentLocation = 'Labels'
foamfoamDisplay.PolarAxes.PolarLabelOffset = 10.0
foamfoamDisplay.PolarAxes.PolarExponentOffset = 5.0
foamfoamDisplay.PolarAxes.RadialTitleVisibility = 1
foamfoamDisplay.PolarAxes.RadialTitleFormat = '%-#3.1f'
foamfoamDisplay.PolarAxes.RadialTitleLocation = 'Bottom'
foamfoamDisplay.PolarAxes.RadialTitleOffset = [20.0, 0.0]
foamfoamDisplay.PolarAxes.RadialUnitsVisibility = 1
foamfoamDisplay.PolarAxes.ScreenSize = 10.0
foamfoamDisplay.PolarAxes.PolarAxisTitleOpacity = 1.0
foamfoamDisplay.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
foamfoamDisplay.PolarAxes.PolarAxisTitleFontFile = ''
foamfoamDisplay.PolarAxes.PolarAxisTitleBold = 0
foamfoamDisplay.PolarAxes.PolarAxisTitleItalic = 0
foamfoamDisplay.PolarAxes.PolarAxisTitleShadow = 0
foamfoamDisplay.PolarAxes.PolarAxisTitleFontSize = 12
foamfoamDisplay.PolarAxes.PolarAxisLabelOpacity = 1.0
foamfoamDisplay.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
foamfoamDisplay.PolarAxes.PolarAxisLabelFontFile = ''
foamfoamDisplay.PolarAxes.PolarAxisLabelBold = 0
foamfoamDisplay.PolarAxes.PolarAxisLabelItalic = 0
foamfoamDisplay.PolarAxes.PolarAxisLabelShadow = 0
foamfoamDisplay.PolarAxes.PolarAxisLabelFontSize = 12
foamfoamDisplay.PolarAxes.LastRadialAxisTextOpacity = 1.0
foamfoamDisplay.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
foamfoamDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
foamfoamDisplay.PolarAxes.LastRadialAxisTextBold = 0
foamfoamDisplay.PolarAxes.LastRadialAxisTextItalic = 0
foamfoamDisplay.PolarAxes.LastRadialAxisTextShadow = 0
foamfoamDisplay.PolarAxes.LastRadialAxisTextFontSize = 12
foamfoamDisplay.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
foamfoamDisplay.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
foamfoamDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
foamfoamDisplay.PolarAxes.SecondaryRadialAxesTextBold = 0
foamfoamDisplay.PolarAxes.SecondaryRadialAxesTextItalic = 0
foamfoamDisplay.PolarAxes.SecondaryRadialAxesTextShadow = 0
foamfoamDisplay.PolarAxes.SecondaryRadialAxesTextFontSize = 12
foamfoamDisplay.PolarAxes.EnableDistanceLOD = 1
foamfoamDisplay.PolarAxes.DistanceLODThreshold = 0.7
foamfoamDisplay.PolarAxes.EnableViewAngleLOD = 1
foamfoamDisplay.PolarAxes.ViewAngleLODThreshold = 0.7
foamfoamDisplay.PolarAxes.SmallestVisiblePolarAngle = 0.5
foamfoamDisplay.PolarAxes.PolarTicksVisibility = 1
foamfoamDisplay.PolarAxes.ArcTicksOriginToPolarAxis = 1
foamfoamDisplay.PolarAxes.TickLocation = 'Both'
foamfoamDisplay.PolarAxes.AxisTickVisibility = 1
foamfoamDisplay.PolarAxes.AxisMinorTickVisibility = 0
foamfoamDisplay.PolarAxes.AxisTickMatchesPolarAxes = 1
foamfoamDisplay.PolarAxes.DeltaRangeMajor = 1.0
foamfoamDisplay.PolarAxes.DeltaRangeMinor = 0.5
foamfoamDisplay.PolarAxes.ArcTickVisibility = 1
foamfoamDisplay.PolarAxes.ArcMinorTickVisibility = 0
foamfoamDisplay.PolarAxes.ArcTickMatchesRadialAxes = 1
foamfoamDisplay.PolarAxes.DeltaAngleMajor = 10.0
foamfoamDisplay.PolarAxes.DeltaAngleMinor = 5.0
foamfoamDisplay.PolarAxes.TickRatioRadiusSize = 0.02
foamfoamDisplay.PolarAxes.PolarAxisMajorTickSize = 0.0
foamfoamDisplay.PolarAxes.PolarAxisTickRatioSize = 0.3
foamfoamDisplay.PolarAxes.PolarAxisMajorTickThickness = 1.0
foamfoamDisplay.PolarAxes.PolarAxisTickRatioThickness = 0.5
foamfoamDisplay.PolarAxes.LastRadialAxisMajorTickSize = 0.0
foamfoamDisplay.PolarAxes.LastRadialAxisTickRatioSize = 0.3
foamfoamDisplay.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
foamfoamDisplay.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
foamfoamDisplay.PolarAxes.ArcMajorTickSize = 0.0
foamfoamDisplay.PolarAxes.ArcTickRatioSize = 0.3
foamfoamDisplay.PolarAxes.ArcMajorTickThickness = 1.0
foamfoamDisplay.PolarAxes.ArcTickRatioThickness = 0.5
foamfoamDisplay.PolarAxes.Use2DMode = 0
foamfoamDisplay.PolarAxes.UseLogAxis = 0

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
foamfoamDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
buildingsobjDisplay = Show(buildingsobj, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
buildingsobjDisplay.Selection = None
buildingsobjDisplay.Representation = 'Surface'
buildingsobjDisplay.ColorArrayName = [None, '']
buildingsobjDisplay.LookupTable = None
buildingsobjDisplay.MapScalars = 1
buildingsobjDisplay.MultiComponentsMapping = 0
buildingsobjDisplay.InterpolateScalarsBeforeMapping = 1
buildingsobjDisplay.UseNanColorForMissingArrays = 0
buildingsobjDisplay.Opacity = 1.0
buildingsobjDisplay.PointSize = 2.0
buildingsobjDisplay.LineWidth = 1.0
buildingsobjDisplay.RenderLinesAsTubes = 0
buildingsobjDisplay.RenderPointsAsSpheres = 0
buildingsobjDisplay.DisableLighting = 0
buildingsobjDisplay.Diffuse = 1.0
buildingsobjDisplay.Interpolation = 'Gouraud'
buildingsobjDisplay.Specular = 0.0
buildingsobjDisplay.SpecularColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.SpecularPower = 100.0
buildingsobjDisplay.Luminosity = 0.0
buildingsobjDisplay.Ambient = 0.0
buildingsobjDisplay.Roughness = 0.3
buildingsobjDisplay.Metallic = 0.0
buildingsobjDisplay.EdgeTint = [1.0, 1.0, 1.0]
buildingsobjDisplay.Anisotropy = 0.0
buildingsobjDisplay.AnisotropyRotation = 0.0
buildingsobjDisplay.BaseIOR = 1.5
buildingsobjDisplay.CoatStrength = 0.0
buildingsobjDisplay.CoatIOR = 2.0
buildingsobjDisplay.CoatRoughness = 0.0
buildingsobjDisplay.CoatColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.SelectNormalArray = 'None'
buildingsobjDisplay.SelectTangentArray = 'None'
buildingsobjDisplay.ComputePointNormals = 0
buildingsobjDisplay.Splitting = 1
buildingsobjDisplay.FeatureAngle = 30.0
buildingsobjDisplay.SelectTCoordArray = 'None'
buildingsobjDisplay.Texture = None
buildingsobjDisplay.RepeatTextures = 1
buildingsobjDisplay.InterpolateTextures = 0
buildingsobjDisplay.SeamlessU = 0
buildingsobjDisplay.SeamlessV = 0
buildingsobjDisplay.UseMipmapTextures = 0
buildingsobjDisplay.ShowTexturesOnBackface = 1
buildingsobjDisplay.BaseColorTexture = None
buildingsobjDisplay.NormalTexture = None
buildingsobjDisplay.NormalScale = 1.0
buildingsobjDisplay.CoatNormalTexture = None
buildingsobjDisplay.CoatNormalScale = 1.0
buildingsobjDisplay.MaterialTexture = None
buildingsobjDisplay.OcclusionStrength = 1.0
buildingsobjDisplay.AnisotropyTexture = None
buildingsobjDisplay.EmissiveTexture = None
buildingsobjDisplay.EmissiveFactor = [1.0, 1.0, 1.0]
buildingsobjDisplay.TextureTransform = 'Transform2'
buildingsobjDisplay.EdgeOpacity = 1.0
buildingsobjDisplay.BackfaceRepresentation = 'Follow Frontface'
buildingsobjDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.BackfaceOpacity = 1.0
buildingsobjDisplay.Translation = [0.0, 0.0, 0.0]
buildingsobjDisplay.Scale = [1.0, 1.0, 1.0]
buildingsobjDisplay.Orientation = [0.0, 0.0, 0.0]
buildingsobjDisplay.Origin = [0.0, 0.0, 0.0]
buildingsobjDisplay.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
buildingsobjDisplay.Pickable = 1
buildingsobjDisplay.Triangulate = 0
buildingsobjDisplay.UseShaderReplacements = 0
buildingsobjDisplay.ShaderReplacements = ''
buildingsobjDisplay.NonlinearSubdivisionLevel = 1
buildingsobjDisplay.MatchBoundariesIgnoringCellOrder = 0
buildingsobjDisplay.UseDataPartitions = 0
buildingsobjDisplay.OSPRayUseScaleArray = 'All Approximate'
buildingsobjDisplay.OSPRayScaleArray = ''
buildingsobjDisplay.OSPRayScaleFunction = 'Piecewise Function'
buildingsobjDisplay.OSPRayMaterial = 'None'
buildingsobjDisplay.Assembly = ''
buildingsobjDisplay.SelectedBlockSelectors = ['']
buildingsobjDisplay.BlockSelectors = ['/']
buildingsobjDisplay.BlockColors = []
buildingsobjDisplay.BlockColorArrayNames = []
buildingsobjDisplay.BlockLookupTables = []
buildingsobjDisplay.BlockUseSeparateColorMaps = []
buildingsobjDisplay.BlockMapScalars = []
buildingsobjDisplay.BlockInterpolateScalarsBeforeMappings = []
buildingsobjDisplay.BlockOpacities = []
buildingsobjDisplay.BlockMapScalarsGUI = 1
buildingsobjDisplay.BlockInterpolateScalarsBeforeMappingsGUI = 1
buildingsobjDisplay.BlockOpacitiesGUI = 1.0
buildingsobjDisplay.Orient = 0
buildingsobjDisplay.OrientationMode = 'Direction'
buildingsobjDisplay.SelectOrientationVectors = 'None'
buildingsobjDisplay.Scaling = 0
buildingsobjDisplay.ScaleMode = 'No Data Scaling Off'
buildingsobjDisplay.ScaleFactor = 12.40545
buildingsobjDisplay.SelectScaleArray = 'None'
buildingsobjDisplay.GlyphType = 'Arrow'
buildingsobjDisplay.UseGlyphTable = 0
buildingsobjDisplay.GlyphTableIndexArray = 'None'
buildingsobjDisplay.UseCompositeGlyphTable = 0
buildingsobjDisplay.UseGlyphCullingAndLOD = 0
buildingsobjDisplay.LODValues = []
buildingsobjDisplay.ColorByLODIndex = 0
buildingsobjDisplay.GaussianRadius = 0.6202725
buildingsobjDisplay.ShaderPreset = 'Sphere'
buildingsobjDisplay.CustomTriangleScale = 3
buildingsobjDisplay.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
buildingsobjDisplay.Emissive = 0
buildingsobjDisplay.ScaleByArray = 0
buildingsobjDisplay.SetScaleArray = [None, '']
buildingsobjDisplay.ScaleArrayComponent = 0
buildingsobjDisplay.UseScaleFunction = 1
buildingsobjDisplay.ScaleTransferFunction = 'Piecewise Function'
buildingsobjDisplay.OpacityByArray = 0
buildingsobjDisplay.OpacityArray = [None, '']
buildingsobjDisplay.OpacityArrayComponent = 0
buildingsobjDisplay.OpacityTransferFunction = 'Piecewise Function'
buildingsobjDisplay.DataAxesGrid = 'Grid Axes Representation'
buildingsobjDisplay.SelectionCellLabelBold = 0
buildingsobjDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
buildingsobjDisplay.SelectionCellLabelFontFamily = 'Arial'
buildingsobjDisplay.SelectionCellLabelFontFile = ''
buildingsobjDisplay.SelectionCellLabelFontSize = 18
buildingsobjDisplay.SelectionCellLabelItalic = 0
buildingsobjDisplay.SelectionCellLabelJustification = 'Left'
buildingsobjDisplay.SelectionCellLabelOpacity = 1.0
buildingsobjDisplay.SelectionCellLabelShadow = 0
buildingsobjDisplay.SelectionPointLabelBold = 0
buildingsobjDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
buildingsobjDisplay.SelectionPointLabelFontFamily = 'Arial'
buildingsobjDisplay.SelectionPointLabelFontFile = ''
buildingsobjDisplay.SelectionPointLabelFontSize = 18
buildingsobjDisplay.SelectionPointLabelItalic = 0
buildingsobjDisplay.SelectionPointLabelJustification = 'Left'
buildingsobjDisplay.SelectionPointLabelOpacity = 1.0
buildingsobjDisplay.SelectionPointLabelShadow = 0
buildingsobjDisplay.PolarAxes = 'Polar Axes Representation'
buildingsobjDisplay.SelectInputVectors = [None, '']
buildingsobjDisplay.NumberOfSteps = 40
buildingsobjDisplay.StepSize = 0.25
buildingsobjDisplay.NormalizeVectors = 1
buildingsobjDisplay.EnhancedLIC = 1
buildingsobjDisplay.ColorMode = 'Blend'
buildingsobjDisplay.LICIntensity = 0.8
buildingsobjDisplay.MapModeBias = 0.0
buildingsobjDisplay.EnhanceContrast = 'Off'
buildingsobjDisplay.LowLICContrastEnhancementFactor = 0.0
buildingsobjDisplay.HighLICContrastEnhancementFactor = 0.0
buildingsobjDisplay.LowColorContrastEnhancementFactor = 0.0
buildingsobjDisplay.HighColorContrastEnhancementFactor = 0.0
buildingsobjDisplay.AntiAlias = 0
buildingsobjDisplay.MaskOnSurface = 1
buildingsobjDisplay.MaskThreshold = 0.0
buildingsobjDisplay.MaskIntensity = 0.0
buildingsobjDisplay.MaskColor = [0.5, 0.5, 0.5]
buildingsobjDisplay.GenerateNoiseTexture = 0
buildingsobjDisplay.NoiseType = 'Gaussian'
buildingsobjDisplay.NoiseTextureSize = 128
buildingsobjDisplay.NoiseGrainSize = 2
buildingsobjDisplay.MinNoiseValue = 0.0
buildingsobjDisplay.MaxNoiseValue = 0.8
buildingsobjDisplay.NumberOfNoiseLevels = 1024
buildingsobjDisplay.ImpulseNoiseProbability = 1.0
buildingsobjDisplay.ImpulseNoiseBackgroundValue = 0.0
buildingsobjDisplay.NoiseGeneratorSeed = 1
buildingsobjDisplay.CompositeStrategy = 'AUTO'
buildingsobjDisplay.UseLICForLOD = 0
buildingsobjDisplay.WriteLog = ''

# init the 'Transform2' selected for 'TextureTransform'
buildingsobjDisplay.TextureTransform.Translate = [0.0, 0.0, 0.0]
buildingsobjDisplay.TextureTransform.Rotate = [0.0, 0.0, 0.0]
buildingsobjDisplay.TextureTransform.Scale = [1.0, 1.0, 1.0]

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
buildingsobjDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
buildingsobjDisplay.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
buildingsobjDisplay.GlyphType.TipResolution = 6
buildingsobjDisplay.GlyphType.TipRadius = 0.1
buildingsobjDisplay.GlyphType.TipLength = 0.35
buildingsobjDisplay.GlyphType.ShaftResolution = 6
buildingsobjDisplay.GlyphType.ShaftRadius = 0.03
buildingsobjDisplay.GlyphType.Invert = 0

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
buildingsobjDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
buildingsobjDisplay.ScaleTransferFunction.UseLogScale = 0

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
buildingsobjDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
buildingsobjDisplay.OpacityTransferFunction.UseLogScale = 0

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
buildingsobjDisplay.DataAxesGrid.XTitle = 'X Axis'
buildingsobjDisplay.DataAxesGrid.YTitle = 'Y Axis'
buildingsobjDisplay.DataAxesGrid.ZTitle = 'Z Axis'
buildingsobjDisplay.DataAxesGrid.XTitleFontFamily = 'Arial'
buildingsobjDisplay.DataAxesGrid.XTitleFontFile = ''
buildingsobjDisplay.DataAxesGrid.XTitleBold = 0
buildingsobjDisplay.DataAxesGrid.XTitleItalic = 0
buildingsobjDisplay.DataAxesGrid.XTitleFontSize = 12
buildingsobjDisplay.DataAxesGrid.XTitleShadow = 0
buildingsobjDisplay.DataAxesGrid.XTitleOpacity = 1.0
buildingsobjDisplay.DataAxesGrid.YTitleFontFamily = 'Arial'
buildingsobjDisplay.DataAxesGrid.YTitleFontFile = ''
buildingsobjDisplay.DataAxesGrid.YTitleBold = 0
buildingsobjDisplay.DataAxesGrid.YTitleItalic = 0
buildingsobjDisplay.DataAxesGrid.YTitleFontSize = 12
buildingsobjDisplay.DataAxesGrid.YTitleShadow = 0
buildingsobjDisplay.DataAxesGrid.YTitleOpacity = 1.0
buildingsobjDisplay.DataAxesGrid.ZTitleFontFamily = 'Arial'
buildingsobjDisplay.DataAxesGrid.ZTitleFontFile = ''
buildingsobjDisplay.DataAxesGrid.ZTitleBold = 0
buildingsobjDisplay.DataAxesGrid.ZTitleItalic = 0
buildingsobjDisplay.DataAxesGrid.ZTitleFontSize = 12
buildingsobjDisplay.DataAxesGrid.ZTitleShadow = 0
buildingsobjDisplay.DataAxesGrid.ZTitleOpacity = 1.0
buildingsobjDisplay.DataAxesGrid.FacesToRender = 63
buildingsobjDisplay.DataAxesGrid.CullBackface = 0
buildingsobjDisplay.DataAxesGrid.CullFrontface = 1
buildingsobjDisplay.DataAxesGrid.ShowGrid = 0
buildingsobjDisplay.DataAxesGrid.ShowEdges = 1
buildingsobjDisplay.DataAxesGrid.ShowTicks = 1
buildingsobjDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 1
buildingsobjDisplay.DataAxesGrid.AxesToLabel = 63
buildingsobjDisplay.DataAxesGrid.XLabelFontFamily = 'Arial'
buildingsobjDisplay.DataAxesGrid.XLabelFontFile = ''
buildingsobjDisplay.DataAxesGrid.XLabelBold = 0
buildingsobjDisplay.DataAxesGrid.XLabelItalic = 0
buildingsobjDisplay.DataAxesGrid.XLabelFontSize = 12
buildingsobjDisplay.DataAxesGrid.XLabelShadow = 0
buildingsobjDisplay.DataAxesGrid.XLabelOpacity = 1.0
buildingsobjDisplay.DataAxesGrid.YLabelFontFamily = 'Arial'
buildingsobjDisplay.DataAxesGrid.YLabelFontFile = ''
buildingsobjDisplay.DataAxesGrid.YLabelBold = 0
buildingsobjDisplay.DataAxesGrid.YLabelItalic = 0
buildingsobjDisplay.DataAxesGrid.YLabelFontSize = 12
buildingsobjDisplay.DataAxesGrid.YLabelShadow = 0
buildingsobjDisplay.DataAxesGrid.YLabelOpacity = 1.0
buildingsobjDisplay.DataAxesGrid.ZLabelFontFamily = 'Arial'
buildingsobjDisplay.DataAxesGrid.ZLabelFontFile = ''
buildingsobjDisplay.DataAxesGrid.ZLabelBold = 0
buildingsobjDisplay.DataAxesGrid.ZLabelItalic = 0
buildingsobjDisplay.DataAxesGrid.ZLabelFontSize = 12
buildingsobjDisplay.DataAxesGrid.ZLabelShadow = 0
buildingsobjDisplay.DataAxesGrid.ZLabelOpacity = 1.0
buildingsobjDisplay.DataAxesGrid.XAxisNotation = 'Mixed'
buildingsobjDisplay.DataAxesGrid.XAxisPrecision = 2
buildingsobjDisplay.DataAxesGrid.XAxisUseCustomLabels = 0
buildingsobjDisplay.DataAxesGrid.XAxisLabels = []
buildingsobjDisplay.DataAxesGrid.YAxisNotation = 'Mixed'
buildingsobjDisplay.DataAxesGrid.YAxisPrecision = 2
buildingsobjDisplay.DataAxesGrid.YAxisUseCustomLabels = 0
buildingsobjDisplay.DataAxesGrid.YAxisLabels = []
buildingsobjDisplay.DataAxesGrid.ZAxisNotation = 'Mixed'
buildingsobjDisplay.DataAxesGrid.ZAxisPrecision = 2
buildingsobjDisplay.DataAxesGrid.ZAxisUseCustomLabels = 0
buildingsobjDisplay.DataAxesGrid.ZAxisLabels = []
buildingsobjDisplay.DataAxesGrid.UseCustomBounds = 0
buildingsobjDisplay.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
buildingsobjDisplay.PolarAxes.Visibility = 0
buildingsobjDisplay.PolarAxes.Translation = [0.0, 0.0, 0.0]
buildingsobjDisplay.PolarAxes.Scale = [1.0, 1.0, 1.0]
buildingsobjDisplay.PolarAxes.Orientation = [0.0, 0.0, 0.0]
buildingsobjDisplay.PolarAxes.EnableCustomBounds = [0, 0, 0]
buildingsobjDisplay.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
buildingsobjDisplay.PolarAxes.EnableCustomRange = 0
buildingsobjDisplay.PolarAxes.CustomRange = [0.0, 1.0]
buildingsobjDisplay.PolarAxes.AutoPole = 1
buildingsobjDisplay.PolarAxes.PolarAxisVisibility = 1
buildingsobjDisplay.PolarAxes.RadialAxesVisibility = 1
buildingsobjDisplay.PolarAxes.DrawRadialGridlines = 1
buildingsobjDisplay.PolarAxes.PolarArcsVisibility = 1
buildingsobjDisplay.PolarAxes.DrawPolarArcsGridlines = 1
buildingsobjDisplay.PolarAxes.NumberOfRadialAxes = 0
buildingsobjDisplay.PolarAxes.DeltaAngleRadialAxes = 45.0
buildingsobjDisplay.PolarAxes.NumberOfPolarAxes = 5
buildingsobjDisplay.PolarAxes.DeltaRangePolarAxes = 0.0
buildingsobjDisplay.PolarAxes.CustomMinRadius = 1
buildingsobjDisplay.PolarAxes.MinimumRadius = 0.0
buildingsobjDisplay.PolarAxes.CustomMaxRadius = 0
buildingsobjDisplay.PolarAxes.MaximumRadius = 1.0
buildingsobjDisplay.PolarAxes.CustomAngles = 1
buildingsobjDisplay.PolarAxes.MinimumAngle = 0.0
buildingsobjDisplay.PolarAxes.MaximumAngle = 90.0
buildingsobjDisplay.PolarAxes.RadialAxesOriginToPolarAxis = 1
buildingsobjDisplay.PolarAxes.PolarArcResolutionPerDegree = 0.2
buildingsobjDisplay.PolarAxes.Ratio = 1.0
buildingsobjDisplay.PolarAxes.EnableOverallColor = 1
buildingsobjDisplay.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
buildingsobjDisplay.PolarAxes.PolarAxisTitleVisibility = 1
buildingsobjDisplay.PolarAxes.PolarAxisTitle = 'Radial Distance'
buildingsobjDisplay.PolarAxes.PolarAxisTitleLocation = 'Bottom'
buildingsobjDisplay.PolarAxes.PolarTitleOffset = [20.0, 20.0]
buildingsobjDisplay.PolarAxes.PolarLabelVisibility = 1
buildingsobjDisplay.PolarAxes.PolarLabelFormat = '%-#6.3g'
buildingsobjDisplay.PolarAxes.PolarLabelExponentLocation = 'Labels'
buildingsobjDisplay.PolarAxes.PolarLabelOffset = 10.0
buildingsobjDisplay.PolarAxes.PolarExponentOffset = 5.0
buildingsobjDisplay.PolarAxes.RadialTitleVisibility = 1
buildingsobjDisplay.PolarAxes.RadialTitleFormat = '%-#3.1f'
buildingsobjDisplay.PolarAxes.RadialTitleLocation = 'Bottom'
buildingsobjDisplay.PolarAxes.RadialTitleOffset = [20.0, 0.0]
buildingsobjDisplay.PolarAxes.RadialUnitsVisibility = 1
buildingsobjDisplay.PolarAxes.ScreenSize = 10.0
buildingsobjDisplay.PolarAxes.PolarAxisTitleOpacity = 1.0
buildingsobjDisplay.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
buildingsobjDisplay.PolarAxes.PolarAxisTitleFontFile = ''
buildingsobjDisplay.PolarAxes.PolarAxisTitleBold = 0
buildingsobjDisplay.PolarAxes.PolarAxisTitleItalic = 0
buildingsobjDisplay.PolarAxes.PolarAxisTitleShadow = 0
buildingsobjDisplay.PolarAxes.PolarAxisTitleFontSize = 12
buildingsobjDisplay.PolarAxes.PolarAxisLabelOpacity = 1.0
buildingsobjDisplay.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
buildingsobjDisplay.PolarAxes.PolarAxisLabelFontFile = ''
buildingsobjDisplay.PolarAxes.PolarAxisLabelBold = 0
buildingsobjDisplay.PolarAxes.PolarAxisLabelItalic = 0
buildingsobjDisplay.PolarAxes.PolarAxisLabelShadow = 0
buildingsobjDisplay.PolarAxes.PolarAxisLabelFontSize = 12
buildingsobjDisplay.PolarAxes.LastRadialAxisTextOpacity = 1.0
buildingsobjDisplay.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
buildingsobjDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
buildingsobjDisplay.PolarAxes.LastRadialAxisTextBold = 0
buildingsobjDisplay.PolarAxes.LastRadialAxisTextItalic = 0
buildingsobjDisplay.PolarAxes.LastRadialAxisTextShadow = 0
buildingsobjDisplay.PolarAxes.LastRadialAxisTextFontSize = 12
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesTextBold = 0
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesTextItalic = 0
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesTextShadow = 0
buildingsobjDisplay.PolarAxes.SecondaryRadialAxesTextFontSize = 12
buildingsobjDisplay.PolarAxes.EnableDistanceLOD = 1
buildingsobjDisplay.PolarAxes.DistanceLODThreshold = 0.7
buildingsobjDisplay.PolarAxes.EnableViewAngleLOD = 1
buildingsobjDisplay.PolarAxes.ViewAngleLODThreshold = 0.7
buildingsobjDisplay.PolarAxes.SmallestVisiblePolarAngle = 0.5
buildingsobjDisplay.PolarAxes.PolarTicksVisibility = 1
buildingsobjDisplay.PolarAxes.ArcTicksOriginToPolarAxis = 1
buildingsobjDisplay.PolarAxes.TickLocation = 'Both'
buildingsobjDisplay.PolarAxes.AxisTickVisibility = 1
buildingsobjDisplay.PolarAxes.AxisMinorTickVisibility = 0
buildingsobjDisplay.PolarAxes.AxisTickMatchesPolarAxes = 1
buildingsobjDisplay.PolarAxes.DeltaRangeMajor = 1.0
buildingsobjDisplay.PolarAxes.DeltaRangeMinor = 0.5
buildingsobjDisplay.PolarAxes.ArcTickVisibility = 1
buildingsobjDisplay.PolarAxes.ArcMinorTickVisibility = 0
buildingsobjDisplay.PolarAxes.ArcTickMatchesRadialAxes = 1
buildingsobjDisplay.PolarAxes.DeltaAngleMajor = 10.0
buildingsobjDisplay.PolarAxes.DeltaAngleMinor = 5.0
buildingsobjDisplay.PolarAxes.TickRatioRadiusSize = 0.02
buildingsobjDisplay.PolarAxes.PolarAxisMajorTickSize = 0.0
buildingsobjDisplay.PolarAxes.PolarAxisTickRatioSize = 0.3
buildingsobjDisplay.PolarAxes.PolarAxisMajorTickThickness = 1.0
buildingsobjDisplay.PolarAxes.PolarAxisTickRatioThickness = 0.5
buildingsobjDisplay.PolarAxes.LastRadialAxisMajorTickSize = 0.0
buildingsobjDisplay.PolarAxes.LastRadialAxisTickRatioSize = 0.3
buildingsobjDisplay.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
buildingsobjDisplay.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
buildingsobjDisplay.PolarAxes.ArcMajorTickSize = 0.0
buildingsobjDisplay.PolarAxes.ArcTickRatioSize = 0.3
buildingsobjDisplay.PolarAxes.ArcMajorTickThickness = 1.0
buildingsobjDisplay.PolarAxes.ArcTickRatioThickness = 0.5
buildingsobjDisplay.PolarAxes.Use2DMode = 0
buildingsobjDisplay.PolarAxes.UseLogAxis = 0

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(foamfoamDisplay, ('POINTS', 'U', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
foamfoamDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
foamfoamDisplay.SetScalarBarVisibility(renderView1, True)

# get 2D transfer function for 'U'
uTF2D = GetTransferFunction2D('U')
uTF2D.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
uTF2D.Boxes = []
uTF2D.ScalarRangeInitialized = 0
uTF2D.Range = [0.0, 1.0, 0.0, 1.0]
uTF2D.OutputDimensions = [10, 10]

# get color transfer function/color map for 'U'
uLUT = GetColorTransferFunction('U')
uLUT.InterpretValuesAsCategories = 0
uLUT.AnnotationsInitialized = 0
uLUT.ShowCategoricalColorsinDataRangeOnly = 0
# uLUT.AutomaticRescaleRangeMode = 'Grow and update every timestep'
uLUT.AutomaticRescaleRangeMode = 'Never'
uLUT.RescaleOnVisibilityChange = 0
uLUT.TransferFunction2D = uTF2D
uLUT.RGBPoints = [0.0, 0.278431372549, 0.278431372549, 0.858823529412, 2.7630616761697433, 0.0, 0.0, 0.360784313725, 5.506801242715921, 0.0, 1.0, 1.0, 8.28918502850923, 0.0, 0.501960784314, 0.0, 11.032924595055409, 1.0, 1.0, 0.0, 13.795986271225152, 1.0, 0.380392156863, 0.0, 16.559047947394895, 0.419607843137, 0.0, 0.0, 19.32210962356464, 0.878431372549, 0.301960784314, 0.301960784314]
uLUT.UseLogScale = 0
uLUT.UseOpacityControlPointsFreehandDrawing = 0
uLUT.ShowDataHistogram = 0
uLUT.AutomaticDataHistogramComputation = 0
uLUT.DataHistogramNumberOfBins = 10
uLUT.ColorSpace = 'RGB'
uLUT.UseBelowRangeColor = 0
uLUT.BelowRangeColor = [0.0, 0.0, 0.0]
uLUT.UseAboveRangeColor = 0
uLUT.AboveRangeColor = [0.5, 0.5, 0.5]
uLUT.NanColor = [1.0, 1.0, 0.0]
uLUT.NanOpacity = 1.0
uLUT.Discretize = 1
uLUT.NumberOfTableValues = 256
uLUT.ScalarRangeInitialized = 1.0
uLUT.HSVWrap = 0
uLUT.VectorComponent = 0
uLUT.VectorMode = 'Magnitude'
uLUT.AllowDuplicateScalars = 1
uLUT.Annotations = []
uLUT.ActiveAnnotatedValues = []
uLUT.IndexedColors = []
uLUT.IndexedOpacities = []
uLUT.EnableOpacityMapping = 0
uLUT.ApplyPreset('Rainbow Desaturated', True)

# get opacity transfer function/opacity map for 'U'
uPWF = GetOpacityTransferFunction('U')
uPWF.Points = [0.0, 0.0, 0.5, 0.0, 19.32210962356464, 1.0, 0.5, 0.0]
uPWF.AllowDuplicateScalars = 1
uPWF.UseLogScale = 0
uPWF.ScalarRangeInitialized = 1

# rescale color and/or opacity maps used to exactly fit the current data range
foamfoamDisplay.RescaleTransferFunctionToDataRange(False, True)

# Find the second vertical cell center above z=0 near the lower-left mesh corner.
UpdatePipeline(proxy=foamfoam)
bounds_info = foamfoam.GetDataInformation().GetBounds()
xmin_d, xmax_d = bounds_info[0], bounds_info[1]
ymin_d, ymax_d = bounds_info[2], bounds_info[3]
span_x = xmax_d - xmin_d
span_y = ymax_d - ymin_d

cellCenters = CellCenters(registrationName='CellCenters_for_slice_z', Input=foamfoam)
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

#slice_z = z_levels[1]
slice_z = user_height + z_levels[1]
print("Using slice_z =", slice_z)

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=foamfoam)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.UseDual = 0
slice1.Crinkleslice = 0
slice1.Triangulatetheslice = 1
slice1.SliceOffsetValues = [0.0]
slice1.PointMergeMethod = 'Uniform Binning'

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [1.8850021362304688, 1.7292518615722656, 24.780000686645508]
slice1.SliceType.Normal = [1.0, 0.0, 0.0]
slice1.SliceType.Offset = 0.0

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [1.8850021362304688, 1.7292518615722656, 24.780000686645508]
slice1.HyperTreeGridSlicer.Normal = [1.0, 0.0, 0.0]
slice1.HyperTreeGridSlicer.Offset = 0.0

# init the 'Uniform Binning' selected for 'PointMergeMethod'
slice1.PointMergeMethod.Divisions = [50, 50, 50]
slice1.PointMergeMethod.Numberofpointsperbucket = 8

# Properties modified on slice1.SliceType
slice1.SliceType.Origin = [1.9, 1.8, slice_z]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Selection = None
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = ['POINTS', 'p']
slice1Display.LookupTable = pLUT
slice1Display.MapScalars = 1
slice1Display.MultiComponentsMapping = 0
slice1Display.InterpolateScalarsBeforeMapping = 1
slice1Display.UseNanColorForMissingArrays = 0
slice1Display.Opacity = 1.0
slice1Display.PointSize = 2.0
slice1Display.LineWidth = 1.0
slice1Display.RenderLinesAsTubes = 0
slice1Display.RenderPointsAsSpheres = 0
slice1Display.DisableLighting = 0
slice1Display.Diffuse = 1.0
slice1Display.Interpolation = 'Gouraud'
slice1Display.Specular = 0.0
slice1Display.SpecularColor = [1.0, 1.0, 1.0]
slice1Display.SpecularPower = 100.0
slice1Display.Luminosity = 0.0
slice1Display.Ambient = 0.0
slice1Display.Roughness = 0.3
slice1Display.Metallic = 0.0
slice1Display.EdgeTint = [1.0, 1.0, 1.0]
slice1Display.Anisotropy = 0.0
slice1Display.AnisotropyRotation = 0.0
slice1Display.BaseIOR = 1.5
slice1Display.CoatStrength = 0.0
slice1Display.CoatIOR = 2.0
slice1Display.CoatRoughness = 0.0
slice1Display.CoatColor = [1.0, 1.0, 1.0]
slice1Display.SelectNormalArray = 'None'
slice1Display.SelectTangentArray = 'None'
slice1Display.ComputePointNormals = 0
slice1Display.Splitting = 1
slice1Display.FeatureAngle = 30.0
slice1Display.SelectTCoordArray = 'None'
slice1Display.Texture = None
slice1Display.RepeatTextures = 1
slice1Display.InterpolateTextures = 0
slice1Display.SeamlessU = 0
slice1Display.SeamlessV = 0
slice1Display.UseMipmapTextures = 0
slice1Display.ShowTexturesOnBackface = 1
slice1Display.BaseColorTexture = None
slice1Display.NormalTexture = None
slice1Display.NormalScale = 1.0
slice1Display.CoatNormalTexture = None
slice1Display.CoatNormalScale = 1.0
slice1Display.MaterialTexture = None
slice1Display.OcclusionStrength = 1.0
slice1Display.AnisotropyTexture = None
slice1Display.EmissiveTexture = None
slice1Display.EmissiveFactor = [1.0, 1.0, 1.0]
slice1Display.TextureTransform = 'Transform2'
slice1Display.EdgeOpacity = 1.0
slice1Display.BackfaceRepresentation = 'Follow Frontface'
slice1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
slice1Display.BackfaceOpacity = 1.0
slice1Display.Translation = [0.0, 0.0, 0.0]
slice1Display.Scale = [1.0, 1.0, 1.0]
slice1Display.Orientation = [0.0, 0.0, 0.0]
slice1Display.Origin = [0.0, 0.0, 0.0]
slice1Display.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
slice1Display.Pickable = 1
slice1Display.Triangulate = 0
slice1Display.UseShaderReplacements = 0
slice1Display.ShaderReplacements = ''
slice1Display.NonlinearSubdivisionLevel = 1
slice1Display.MatchBoundariesIgnoringCellOrder = 0
slice1Display.UseDataPartitions = 0
slice1Display.OSPRayUseScaleArray = 'All Approximate'
slice1Display.OSPRayScaleArray = 'p'
slice1Display.OSPRayScaleFunction = 'Piecewise Function'
slice1Display.OSPRayMaterial = 'None'
slice1Display.Assembly = 'Hierarchy'
slice1Display.SelectedBlockSelectors = ['']
slice1Display.BlockSelectors = ['/']
slice1Display.BlockColors = []
slice1Display.BlockColorArrayNames = []
slice1Display.BlockLookupTables = []
slice1Display.BlockUseSeparateColorMaps = []
slice1Display.BlockMapScalars = []
slice1Display.BlockInterpolateScalarsBeforeMappings = []
slice1Display.BlockOpacities = []
slice1Display.BlockMapScalarsGUI = 1
slice1Display.BlockInterpolateScalarsBeforeMappingsGUI = 1
slice1Display.BlockOpacitiesGUI = 1.0
slice1Display.Orient = 0
slice1Display.OrientationMode = 'Direction'
slice1Display.SelectOrientationVectors = 'U'
slice1Display.Scaling = 0
slice1Display.ScaleMode = 'No Data Scaling Off'
slice1Display.ScaleFactor = 24.8
slice1Display.SelectScaleArray = 'p'
slice1Display.GlyphType = 'Arrow'
slice1Display.UseGlyphTable = 0
slice1Display.GlyphTableIndexArray = 'p'
slice1Display.UseCompositeGlyphTable = 0
slice1Display.UseGlyphCullingAndLOD = 0
slice1Display.LODValues = []
slice1Display.ColorByLODIndex = 0
slice1Display.GaussianRadius = 1.2405699539184571
slice1Display.ShaderPreset = 'Sphere'
slice1Display.CustomTriangleScale = 3
slice1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
slice1Display.Emissive = 0
slice1Display.ScaleByArray = 0
slice1Display.SetScaleArray = ['POINTS', 'p']
slice1Display.ScaleArrayComponent = ''
slice1Display.UseScaleFunction = 1
slice1Display.ScaleTransferFunction = 'Piecewise Function'
slice1Display.OpacityByArray = 0
slice1Display.OpacityArray = ['POINTS', 'p']
slice1Display.OpacityArrayComponent = ''
slice1Display.OpacityTransferFunction = 'Piecewise Function'
slice1Display.DataAxesGrid = 'Grid Axes Representation'
slice1Display.SelectionCellLabelBold = 0
slice1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
slice1Display.SelectionCellLabelFontFamily = 'Arial'
slice1Display.SelectionCellLabelFontFile = ''
slice1Display.SelectionCellLabelFontSize = 18
slice1Display.SelectionCellLabelItalic = 0
slice1Display.SelectionCellLabelJustification = 'Left'
slice1Display.SelectionCellLabelOpacity = 1.0
slice1Display.SelectionCellLabelShadow = 0
slice1Display.SelectionPointLabelBold = 0
slice1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
slice1Display.SelectionPointLabelFontFamily = 'Arial'
slice1Display.SelectionPointLabelFontFile = ''
slice1Display.SelectionPointLabelFontSize = 18
slice1Display.SelectionPointLabelItalic = 0
slice1Display.SelectionPointLabelJustification = 'Left'
slice1Display.SelectionPointLabelOpacity = 1.0
slice1Display.SelectionPointLabelShadow = 0
slice1Display.PolarAxes = 'Polar Axes Representation'
slice1Display.SelectInputVectors = ['POINTS', 'U']
slice1Display.NumberOfSteps = 40
slice1Display.StepSize = 0.25
slice1Display.NormalizeVectors = 1
slice1Display.EnhancedLIC = 1
slice1Display.ColorMode = 'Blend'
slice1Display.LICIntensity = 0.8
slice1Display.MapModeBias = 0.0
slice1Display.EnhanceContrast = 'Off'
slice1Display.LowLICContrastEnhancementFactor = 0.0
slice1Display.HighLICContrastEnhancementFactor = 0.0
slice1Display.LowColorContrastEnhancementFactor = 0.0
slice1Display.HighColorContrastEnhancementFactor = 0.0
slice1Display.AntiAlias = 0
slice1Display.MaskOnSurface = 1
slice1Display.MaskThreshold = 0.0
slice1Display.MaskIntensity = 0.0
slice1Display.MaskColor = [0.5, 0.5, 0.5]
slice1Display.GenerateNoiseTexture = 0
slice1Display.NoiseType = 'Gaussian'
slice1Display.NoiseTextureSize = 128
slice1Display.NoiseGrainSize = 2
slice1Display.MinNoiseValue = 0.0
slice1Display.MaxNoiseValue = 0.8
slice1Display.NumberOfNoiseLevels = 1024
slice1Display.ImpulseNoiseProbability = 1.0
slice1Display.ImpulseNoiseBackgroundValue = 0.0
slice1Display.NoiseGeneratorSeed = 1
slice1Display.CompositeStrategy = 'AUTO'
slice1Display.UseLICForLOD = 0
slice1Display.WriteLog = ''

# init the 'Transform2' selected for 'TextureTransform'
slice1Display.TextureTransform.Translate = [0.0, 0.0, 0.0]
slice1Display.TextureTransform.Rotate = [0.0, 0.0, 0.0]
slice1Display.TextureTransform.Scale = [1.0, 1.0, 1.0]

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
slice1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
slice1Display.GlyphType.TipResolution = 6
slice1Display.GlyphType.TipRadius = 0.1
slice1Display.GlyphType.TipLength = 0.35
slice1Display.GlyphType.ShaftResolution = 6
slice1Display.GlyphType.ShaftRadius = 0.03
slice1Display.GlyphType.Invert = 0

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [101230.0625, 0.0, 0.5, 0.0, 101436.921875, 1.0, 0.5, 0.0]
slice1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [101230.0625, 0.0, 0.5, 0.0, 101436.921875, 1.0, 0.5, 0.0]
slice1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
slice1Display.DataAxesGrid.XTitle = 'X Axis'
slice1Display.DataAxesGrid.YTitle = 'Y Axis'
slice1Display.DataAxesGrid.ZTitle = 'Z Axis'
slice1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
slice1Display.DataAxesGrid.XTitleFontFile = ''
slice1Display.DataAxesGrid.XTitleBold = 0
slice1Display.DataAxesGrid.XTitleItalic = 0
slice1Display.DataAxesGrid.XTitleFontSize = 12
slice1Display.DataAxesGrid.XTitleShadow = 0
slice1Display.DataAxesGrid.XTitleOpacity = 1.0
slice1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
slice1Display.DataAxesGrid.YTitleFontFile = ''
slice1Display.DataAxesGrid.YTitleBold = 0
slice1Display.DataAxesGrid.YTitleItalic = 0
slice1Display.DataAxesGrid.YTitleFontSize = 12
slice1Display.DataAxesGrid.YTitleShadow = 0
slice1Display.DataAxesGrid.YTitleOpacity = 1.0
slice1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
slice1Display.DataAxesGrid.ZTitleFontFile = ''
slice1Display.DataAxesGrid.ZTitleBold = 0
slice1Display.DataAxesGrid.ZTitleItalic = 0
slice1Display.DataAxesGrid.ZTitleFontSize = 12
slice1Display.DataAxesGrid.ZTitleShadow = 0
slice1Display.DataAxesGrid.ZTitleOpacity = 1.0
slice1Display.DataAxesGrid.FacesToRender = 63
slice1Display.DataAxesGrid.CullBackface = 0
slice1Display.DataAxesGrid.CullFrontface = 1
slice1Display.DataAxesGrid.ShowGrid = 0
slice1Display.DataAxesGrid.ShowEdges = 1
slice1Display.DataAxesGrid.ShowTicks = 1
slice1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
slice1Display.DataAxesGrid.AxesToLabel = 63
slice1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
slice1Display.DataAxesGrid.XLabelFontFile = ''
slice1Display.DataAxesGrid.XLabelBold = 0
slice1Display.DataAxesGrid.XLabelItalic = 0
slice1Display.DataAxesGrid.XLabelFontSize = 12
slice1Display.DataAxesGrid.XLabelShadow = 0
slice1Display.DataAxesGrid.XLabelOpacity = 1.0
slice1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
slice1Display.DataAxesGrid.YLabelFontFile = ''
slice1Display.DataAxesGrid.YLabelBold = 0
slice1Display.DataAxesGrid.YLabelItalic = 0
slice1Display.DataAxesGrid.YLabelFontSize = 12
slice1Display.DataAxesGrid.YLabelShadow = 0
slice1Display.DataAxesGrid.YLabelOpacity = 1.0
slice1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
slice1Display.DataAxesGrid.ZLabelFontFile = ''
slice1Display.DataAxesGrid.ZLabelBold = 0
slice1Display.DataAxesGrid.ZLabelItalic = 0
slice1Display.DataAxesGrid.ZLabelFontSize = 12
slice1Display.DataAxesGrid.ZLabelShadow = 0
slice1Display.DataAxesGrid.ZLabelOpacity = 1.0
slice1Display.DataAxesGrid.XAxisNotation = 'Mixed'
slice1Display.DataAxesGrid.XAxisPrecision = 2
slice1Display.DataAxesGrid.XAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.XAxisLabels = []
slice1Display.DataAxesGrid.YAxisNotation = 'Mixed'
slice1Display.DataAxesGrid.YAxisPrecision = 2
slice1Display.DataAxesGrid.YAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.YAxisLabels = []
slice1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
slice1Display.DataAxesGrid.ZAxisPrecision = 2
slice1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.ZAxisLabels = []
slice1Display.DataAxesGrid.UseCustomBounds = 0
slice1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
slice1Display.PolarAxes.Visibility = 0
slice1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
slice1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
slice1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
slice1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
slice1Display.PolarAxes.EnableCustomRange = 0
slice1Display.PolarAxes.CustomRange = [0.0, 1.0]
slice1Display.PolarAxes.AutoPole = 1
slice1Display.PolarAxes.PolarAxisVisibility = 1
slice1Display.PolarAxes.RadialAxesVisibility = 1
slice1Display.PolarAxes.DrawRadialGridlines = 1
slice1Display.PolarAxes.PolarArcsVisibility = 1
slice1Display.PolarAxes.DrawPolarArcsGridlines = 1
slice1Display.PolarAxes.NumberOfRadialAxes = 0
slice1Display.PolarAxes.DeltaAngleRadialAxes = 45.0
slice1Display.PolarAxes.NumberOfPolarAxes = 5
slice1Display.PolarAxes.DeltaRangePolarAxes = 0.0
slice1Display.PolarAxes.CustomMinRadius = 1
slice1Display.PolarAxes.MinimumRadius = 0.0
slice1Display.PolarAxes.CustomMaxRadius = 0
slice1Display.PolarAxes.MaximumRadius = 1.0
slice1Display.PolarAxes.CustomAngles = 1
slice1Display.PolarAxes.MinimumAngle = 0.0
slice1Display.PolarAxes.MaximumAngle = 90.0
slice1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
slice1Display.PolarAxes.PolarArcResolutionPerDegree = 0.2
slice1Display.PolarAxes.Ratio = 1.0
slice1Display.PolarAxes.EnableOverallColor = 1
slice1Display.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.PolarAxisTitleVisibility = 1
slice1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
slice1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
slice1Display.PolarAxes.PolarTitleOffset = [20.0, 20.0]
slice1Display.PolarAxes.PolarLabelVisibility = 1
slice1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
slice1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
slice1Display.PolarAxes.PolarLabelOffset = 10.0
slice1Display.PolarAxes.PolarExponentOffset = 5.0
slice1Display.PolarAxes.RadialTitleVisibility = 1
slice1Display.PolarAxes.RadialTitleFormat = '%-#3.1f'
slice1Display.PolarAxes.RadialTitleLocation = 'Bottom'
slice1Display.PolarAxes.RadialTitleOffset = [20.0, 0.0]
slice1Display.PolarAxes.RadialUnitsVisibility = 1
slice1Display.PolarAxes.ScreenSize = 10.0
slice1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
slice1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
slice1Display.PolarAxes.PolarAxisTitleFontFile = ''
slice1Display.PolarAxes.PolarAxisTitleBold = 0
slice1Display.PolarAxes.PolarAxisTitleItalic = 0
slice1Display.PolarAxes.PolarAxisTitleShadow = 0
slice1Display.PolarAxes.PolarAxisTitleFontSize = 12
slice1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
slice1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
slice1Display.PolarAxes.PolarAxisLabelFontFile = ''
slice1Display.PolarAxes.PolarAxisLabelBold = 0
slice1Display.PolarAxes.PolarAxisLabelItalic = 0
slice1Display.PolarAxes.PolarAxisLabelShadow = 0
slice1Display.PolarAxes.PolarAxisLabelFontSize = 12
slice1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
slice1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
slice1Display.PolarAxes.LastRadialAxisTextFontFile = ''
slice1Display.PolarAxes.LastRadialAxisTextBold = 0
slice1Display.PolarAxes.LastRadialAxisTextItalic = 0
slice1Display.PolarAxes.LastRadialAxisTextShadow = 0
slice1Display.PolarAxes.LastRadialAxisTextFontSize = 12
slice1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
slice1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
slice1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
slice1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
slice1Display.PolarAxes.EnableDistanceLOD = 1
slice1Display.PolarAxes.DistanceLODThreshold = 0.7
slice1Display.PolarAxes.EnableViewAngleLOD = 1
slice1Display.PolarAxes.ViewAngleLODThreshold = 0.7
slice1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
slice1Display.PolarAxes.PolarTicksVisibility = 1
slice1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
slice1Display.PolarAxes.TickLocation = 'Both'
slice1Display.PolarAxes.AxisTickVisibility = 1
slice1Display.PolarAxes.AxisMinorTickVisibility = 0
slice1Display.PolarAxes.AxisTickMatchesPolarAxes = 1
slice1Display.PolarAxes.DeltaRangeMajor = 1.0
slice1Display.PolarAxes.DeltaRangeMinor = 0.5
slice1Display.PolarAxes.ArcTickVisibility = 1
slice1Display.PolarAxes.ArcMinorTickVisibility = 0
slice1Display.PolarAxes.ArcTickMatchesRadialAxes = 1
slice1Display.PolarAxes.DeltaAngleMajor = 10.0
slice1Display.PolarAxes.DeltaAngleMinor = 5.0
slice1Display.PolarAxes.TickRatioRadiusSize = 0.02
slice1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
slice1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
slice1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
slice1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
slice1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
slice1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
slice1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
slice1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
slice1Display.PolarAxes.ArcMajorTickSize = 0.0
slice1Display.PolarAxes.ArcTickRatioSize = 0.3
slice1Display.PolarAxes.ArcMajorTickThickness = 1.0
slice1Display.PolarAxes.ArcTickRatioThickness = 0.5
slice1Display.PolarAxes.Use2DMode = 0
slice1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(foamfoam, renderView1)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# Adjust camera

# current camera placement for renderView1
renderView1.CameraPosition = [1.8850021362304688, 1.7292518615722656, 617.0336769100838]
renderView1.CameraFocalPoint = [1.8850021362304688, 1.7292518615722656, 24.780000686645508]
renderView1.CameraParallelScale = 153.2865309386078

# set scalar coloring
ColorBy(slice1Display, ('POINTS', 'U', 'Magnitude'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)
# Adjust camera

# current camera placement for renderView1
renderView1.CameraPosition = [1.8850021362304688, 1.7292518615722656, 617.0336769100838]
renderView1.CameraFocalPoint = [1.8850021362304688, 1.7292518615722656, 24.780000686645508]
renderView1.CameraParallelScale = 153.2865309386078

# set active source
SetActiveSource(buildingsobj)

# toggle interactive widget visibility (only when running from the GUI)
HideInteractiveWidgets(proxy=slice1.SliceType)

# Adjust camera

# current camera placement for renderView1
renderView1.CameraPosition = [1.8850021362304688, 1.7292518615722656, 359.0917608932871]
renderView1.CameraFocalPoint = [1.8850021362304688, 1.7292518615722656, 24.780000686645508]
renderView1.CameraParallelScale = 153.2865309386078

#START: SUBART

site_boundary_shiftedobj = WavefrontOBJReader(registrationName='site_boundary_shifted.obj', FileName='constant/triSurface/site_boundary_shifted.obj')
UpdatePipeline(proxy=site_boundary_shiftedobj)

bounds = site_boundary_shiftedobj.GetDataInformation().GetBounds()

x_min_site_boundary = bounds[0]
x_max_site_boundary = bounds[1]
y_min_site_boundary = bounds[2]
y_max_site_boundary = bounds[3]

delta_x_site_boundary = x_max_site_boundary - x_min_site_boundary
delta_y_site_boundary = y_max_site_boundary - y_min_site_boundary
center_x_site_boundary = (x_max_site_boundary + x_min_site_boundary) / 2.0
center_y_site_boundary = (y_max_site_boundary + y_min_site_boundary) / 2.0



# SUBRAT: Added the average velocity
# -------------------------------------------------------------------------
# U_average on existing Slice1 inside site_boundary XY limits
# -------------------------------------------------------------------------

UpdatePipeline(proxy=slice1)

# Calculate U magnitude on the same POINTS data used for visualization
uMagCalc = Calculator(
    registrationName='U_Magnitude_on_Slice',
    Input=slice1
)
uMagCalc.AttributeType = 'Point Data'
uMagCalc.ResultArrayName = 'U_Magnitude'
uMagCalc.Function = 'mag(U)'

UpdatePipeline(proxy=uMagCalc)

# Get slice z bounds to define a thin clipping box
slice_bounds = uMagCalc.GetDataInformation().GetBounds()

z_min_slice = slice_bounds[4]
z_max_slice = slice_bounds[5]

eps_z = 1e-3
z0 = z_min_slice - eps_z
z1 = z_max_slice + eps_z

# Clip slice to rectangular XY limits of site_boundary_shifted.obj
clipSiteBoundaryBox = Clip(
    registrationName='Slice_inside_site_boundary_XY_limits',
    Input=uMagCalc
)

clipSiteBoundaryBox.ClipType = 'Box'

clipSiteBoundaryBox.ClipType.Position = [
    x_min_site_boundary,
    y_min_site_boundary,
    z0
]

clipSiteBoundaryBox.ClipType.Length = [
    x_max_site_boundary - x_min_site_boundary,
    y_max_site_boundary - y_min_site_boundary,
    z1 - z0
]

# Usually Invert = 1 keeps the inside of the box.
# The sanity check below verifies it.
clipSiteBoundaryBox.Invert = 1

UpdatePipeline(proxy=clipSiteBoundaryBox)

# Sanity check: clipped bounds should lie inside site-boundary XY limits
clipped_bounds = clipSiteBoundaryBox.GetDataInformation().GetBounds()

if (
    clipped_bounds[0] < x_min_site_boundary - 1e-6 or
    clipped_bounds[1] > x_max_site_boundary + 1e-6 or
    clipped_bounds[2] < y_min_site_boundary - 1e-6 or
    clipped_bounds[3] > y_max_site_boundary + 1e-6
):
    print("Invert=1 did not keep inside box. Switching to Invert=0.")
    clipSiteBoundaryBox.Invert = 0
    UpdatePipeline(proxy=clipSiteBoundaryBox)
    clipped_bounds = clipSiteBoundaryBox.GetDataInformation().GetBounds()

# Integrate U_Magnitude over clipped slice area
integratedU = IntegrateVariables(
    registrationName='Integrated_U_Magnitude_site_boundary',
    Input=clipSiteBoundaryBox
)

UpdatePipeline(proxy=integratedU)

integratedData = servermanager.Fetch(integratedU)


def get_array_value(dataset, array_name):
    for data_attr in [
        dataset.GetPointData(),
        dataset.GetCellData(),
        dataset.GetFieldData()
    ]:
        arr = data_attr.GetArray(array_name)
        if arr is not None:
            return arr.GetTuple1(0)

    raise RuntimeError("Array not found: {}".format(array_name))


U_mag_integral = get_array_value(integratedData, 'U_Magnitude')
area = get_array_value(integratedData, 'Area')

if area <= 0.0:
    raise RuntimeError("Clipped slice area is zero. Check site_boundary bounds and slice position.")

U_average = U_mag_integral / area

output_file = "site_boundary_average_velocity.txt"
with open(output_file, "w") as f:
    f.write("U_average = {:.10f} m/s\n".format(U_average))
        
print("--------------------------------------------------")
print("U_average calculation on Slice1")
print("site x_min =", x_min_site_boundary)
print("site x_max =", x_max_site_boundary)
print("site y_min =", y_min_site_boundary)
print("site y_max =", y_max_site_boundary)
print("clipped bounds =", clipped_bounds)
print("clipped slice area =", area)
print("integral mag(U) dA =", U_mag_integral)
print("U_average =", U_average, "m/s")
print("--------------------------------------------------")

# Subrat: End of average velocity calculation




# find source
buildingsobj = FindSource('buildings.obj')

# find source
foamfoam = FindSource('foam.foam')

# find source
slice1 = FindSource('Slice1')

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
site_boundary_shiftedobjDisplay = Show(site_boundary_shiftedobj, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
site_boundary_shiftedobjDisplay.Selection = None
site_boundary_shiftedobjDisplay.Representation = 'Surface'
site_boundary_shiftedobjDisplay.ColorArrayName = [None, '']
site_boundary_shiftedobjDisplay.LookupTable = None
site_boundary_shiftedobjDisplay.MapScalars = 1
site_boundary_shiftedobjDisplay.MultiComponentsMapping = 0
site_boundary_shiftedobjDisplay.InterpolateScalarsBeforeMapping = 1
site_boundary_shiftedobjDisplay.UseNanColorForMissingArrays = 0
site_boundary_shiftedobjDisplay.Opacity = 1.0
site_boundary_shiftedobjDisplay.PointSize = 2.0
site_boundary_shiftedobjDisplay.LineWidth = 1.0
site_boundary_shiftedobjDisplay.RenderLinesAsTubes = 0
site_boundary_shiftedobjDisplay.RenderPointsAsSpheres = 0
site_boundary_shiftedobjDisplay.DisableLighting = 0
site_boundary_shiftedobjDisplay.Diffuse = 1.0
site_boundary_shiftedobjDisplay.Interpolation = 'Gouraud'
site_boundary_shiftedobjDisplay.Specular = 0.0
site_boundary_shiftedobjDisplay.SpecularColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.SpecularPower = 100.0
site_boundary_shiftedobjDisplay.Luminosity = 0.0
site_boundary_shiftedobjDisplay.Ambient = 0.0
site_boundary_shiftedobjDisplay.Roughness = 0.3
site_boundary_shiftedobjDisplay.Metallic = 0.0
site_boundary_shiftedobjDisplay.EdgeTint = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.Anisotropy = 0.0
site_boundary_shiftedobjDisplay.AnisotropyRotation = 0.0
site_boundary_shiftedobjDisplay.BaseIOR = 1.5
site_boundary_shiftedobjDisplay.CoatStrength = 0.0
site_boundary_shiftedobjDisplay.CoatIOR = 2.0
site_boundary_shiftedobjDisplay.CoatRoughness = 0.0
site_boundary_shiftedobjDisplay.CoatColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.SelectNormalArray = 'None'
site_boundary_shiftedobjDisplay.SelectTangentArray = 'None'
site_boundary_shiftedobjDisplay.ComputePointNormals = 0
site_boundary_shiftedobjDisplay.Splitting = 1
site_boundary_shiftedobjDisplay.FeatureAngle = 30.0
site_boundary_shiftedobjDisplay.SelectTCoordArray = 'None'
site_boundary_shiftedobjDisplay.Texture = None
site_boundary_shiftedobjDisplay.RepeatTextures = 1
site_boundary_shiftedobjDisplay.InterpolateTextures = 0
site_boundary_shiftedobjDisplay.SeamlessU = 0
site_boundary_shiftedobjDisplay.SeamlessV = 0
site_boundary_shiftedobjDisplay.UseMipmapTextures = 0
site_boundary_shiftedobjDisplay.ShowTexturesOnBackface = 1
site_boundary_shiftedobjDisplay.BaseColorTexture = None
site_boundary_shiftedobjDisplay.NormalTexture = None
site_boundary_shiftedobjDisplay.NormalScale = 1.0
site_boundary_shiftedobjDisplay.CoatNormalTexture = None
site_boundary_shiftedobjDisplay.CoatNormalScale = 1.0
site_boundary_shiftedobjDisplay.MaterialTexture = None
site_boundary_shiftedobjDisplay.OcclusionStrength = 1.0
site_boundary_shiftedobjDisplay.AnisotropyTexture = None
site_boundary_shiftedobjDisplay.EmissiveTexture = None
site_boundary_shiftedobjDisplay.EmissiveFactor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.TextureTransform = 'Transform2'
site_boundary_shiftedobjDisplay.EdgeOpacity = 1.0
site_boundary_shiftedobjDisplay.BackfaceRepresentation = 'Follow Frontface'
site_boundary_shiftedobjDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.BackfaceOpacity = 1.0
site_boundary_shiftedobjDisplay.Translation = [0.0, 0.0, 0.0]
site_boundary_shiftedobjDisplay.Scale = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.Orientation = [0.0, 0.0, 0.0]
site_boundary_shiftedobjDisplay.Origin = [0.0, 0.0, 0.0]
site_boundary_shiftedobjDisplay.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
site_boundary_shiftedobjDisplay.Pickable = 1
site_boundary_shiftedobjDisplay.Triangulate = 0
site_boundary_shiftedobjDisplay.UseShaderReplacements = 0
site_boundary_shiftedobjDisplay.ShaderReplacements = ''
site_boundary_shiftedobjDisplay.NonlinearSubdivisionLevel = 1
site_boundary_shiftedobjDisplay.MatchBoundariesIgnoringCellOrder = 0
site_boundary_shiftedobjDisplay.UseDataPartitions = 0
site_boundary_shiftedobjDisplay.OSPRayUseScaleArray = 'All Approximate'
site_boundary_shiftedobjDisplay.OSPRayScaleArray = ''
site_boundary_shiftedobjDisplay.OSPRayScaleFunction = 'Piecewise Function'
site_boundary_shiftedobjDisplay.OSPRayMaterial = 'None'
site_boundary_shiftedobjDisplay.Assembly = ''
site_boundary_shiftedobjDisplay.SelectedBlockSelectors = ['']
site_boundary_shiftedobjDisplay.BlockSelectors = ['/']
site_boundary_shiftedobjDisplay.BlockColors = []
site_boundary_shiftedobjDisplay.BlockColorArrayNames = []
site_boundary_shiftedobjDisplay.BlockLookupTables = []
site_boundary_shiftedobjDisplay.BlockUseSeparateColorMaps = []
site_boundary_shiftedobjDisplay.BlockMapScalars = []
site_boundary_shiftedobjDisplay.BlockInterpolateScalarsBeforeMappings = []
site_boundary_shiftedobjDisplay.BlockOpacities = []
site_boundary_shiftedobjDisplay.BlockMapScalarsGUI = 1
site_boundary_shiftedobjDisplay.BlockInterpolateScalarsBeforeMappingsGUI = 1
site_boundary_shiftedobjDisplay.BlockOpacitiesGUI = 1.0
site_boundary_shiftedobjDisplay.Orient = 0
site_boundary_shiftedobjDisplay.OrientationMode = 'Direction'
site_boundary_shiftedobjDisplay.SelectOrientationVectors = 'None'
site_boundary_shiftedobjDisplay.Scaling = 0
site_boundary_shiftedobjDisplay.ScaleMode = 'No Data Scaling Off'
site_boundary_shiftedobjDisplay.ScaleFactor = 34.0601
site_boundary_shiftedobjDisplay.SelectScaleArray = 'None'
site_boundary_shiftedobjDisplay.GlyphType = 'Arrow'
site_boundary_shiftedobjDisplay.UseGlyphTable = 0
site_boundary_shiftedobjDisplay.GlyphTableIndexArray = 'None'
site_boundary_shiftedobjDisplay.UseCompositeGlyphTable = 0
site_boundary_shiftedobjDisplay.UseGlyphCullingAndLOD = 0
site_boundary_shiftedobjDisplay.LODValues = []
site_boundary_shiftedobjDisplay.ColorByLODIndex = 0
site_boundary_shiftedobjDisplay.GaussianRadius = 1.703005
site_boundary_shiftedobjDisplay.ShaderPreset = 'Sphere'
site_boundary_shiftedobjDisplay.CustomTriangleScale = 3
site_boundary_shiftedobjDisplay.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
site_boundary_shiftedobjDisplay.Emissive = 0
site_boundary_shiftedobjDisplay.ScaleByArray = 0
site_boundary_shiftedobjDisplay.SetScaleArray = [None, '']
site_boundary_shiftedobjDisplay.ScaleArrayComponent = 0
site_boundary_shiftedobjDisplay.UseScaleFunction = 1
site_boundary_shiftedobjDisplay.ScaleTransferFunction = 'Piecewise Function'
site_boundary_shiftedobjDisplay.OpacityByArray = 0
site_boundary_shiftedobjDisplay.OpacityArray = [None, '']
site_boundary_shiftedobjDisplay.OpacityArrayComponent = 0
site_boundary_shiftedobjDisplay.OpacityTransferFunction = 'Piecewise Function'
site_boundary_shiftedobjDisplay.DataAxesGrid = 'Grid Axes Representation'
site_boundary_shiftedobjDisplay.SelectionCellLabelBold = 0
site_boundary_shiftedobjDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
site_boundary_shiftedobjDisplay.SelectionCellLabelFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.SelectionCellLabelFontFile = ''
site_boundary_shiftedobjDisplay.SelectionCellLabelFontSize = 18
site_boundary_shiftedobjDisplay.SelectionCellLabelItalic = 0
site_boundary_shiftedobjDisplay.SelectionCellLabelJustification = 'Left'
site_boundary_shiftedobjDisplay.SelectionCellLabelOpacity = 1.0
site_boundary_shiftedobjDisplay.SelectionCellLabelShadow = 0
site_boundary_shiftedobjDisplay.SelectionPointLabelBold = 0
site_boundary_shiftedobjDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
site_boundary_shiftedobjDisplay.SelectionPointLabelFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.SelectionPointLabelFontFile = ''
site_boundary_shiftedobjDisplay.SelectionPointLabelFontSize = 18
site_boundary_shiftedobjDisplay.SelectionPointLabelItalic = 0
site_boundary_shiftedobjDisplay.SelectionPointLabelJustification = 'Left'
site_boundary_shiftedobjDisplay.SelectionPointLabelOpacity = 1.0
site_boundary_shiftedobjDisplay.SelectionPointLabelShadow = 0
site_boundary_shiftedobjDisplay.PolarAxes = 'Polar Axes Representation'
site_boundary_shiftedobjDisplay.SelectInputVectors = [None, '']
site_boundary_shiftedobjDisplay.NumberOfSteps = 40
site_boundary_shiftedobjDisplay.StepSize = 0.25
site_boundary_shiftedobjDisplay.NormalizeVectors = 1
site_boundary_shiftedobjDisplay.EnhancedLIC = 1
site_boundary_shiftedobjDisplay.ColorMode = 'Blend'
site_boundary_shiftedobjDisplay.LICIntensity = 0.8
site_boundary_shiftedobjDisplay.MapModeBias = 0.0
site_boundary_shiftedobjDisplay.EnhanceContrast = 'Off'
site_boundary_shiftedobjDisplay.LowLICContrastEnhancementFactor = 0.0
site_boundary_shiftedobjDisplay.HighLICContrastEnhancementFactor = 0.0
site_boundary_shiftedobjDisplay.LowColorContrastEnhancementFactor = 0.0
site_boundary_shiftedobjDisplay.HighColorContrastEnhancementFactor = 0.0
site_boundary_shiftedobjDisplay.AntiAlias = 0
site_boundary_shiftedobjDisplay.MaskOnSurface = 1
site_boundary_shiftedobjDisplay.MaskThreshold = 0.0
site_boundary_shiftedobjDisplay.MaskIntensity = 0.0
site_boundary_shiftedobjDisplay.MaskColor = [0.5, 0.5, 0.5]
site_boundary_shiftedobjDisplay.GenerateNoiseTexture = 0
site_boundary_shiftedobjDisplay.NoiseType = 'Gaussian'
site_boundary_shiftedobjDisplay.NoiseTextureSize = 128
site_boundary_shiftedobjDisplay.NoiseGrainSize = 2
site_boundary_shiftedobjDisplay.MinNoiseValue = 0.0
site_boundary_shiftedobjDisplay.MaxNoiseValue = 0.8
site_boundary_shiftedobjDisplay.NumberOfNoiseLevels = 1024
site_boundary_shiftedobjDisplay.ImpulseNoiseProbability = 1.0
site_boundary_shiftedobjDisplay.ImpulseNoiseBackgroundValue = 0.0
site_boundary_shiftedobjDisplay.NoiseGeneratorSeed = 1
site_boundary_shiftedobjDisplay.CompositeStrategy = 'AUTO'
site_boundary_shiftedobjDisplay.UseLICForLOD = 0
site_boundary_shiftedobjDisplay.WriteLog = ''

# init the 'Transform2' selected for 'TextureTransform'
site_boundary_shiftedobjDisplay.TextureTransform.Translate = [0.0, 0.0, 0.0]
site_boundary_shiftedobjDisplay.TextureTransform.Rotate = [0.0, 0.0, 0.0]
site_boundary_shiftedobjDisplay.TextureTransform.Scale = [1.0, 1.0, 1.0]

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
site_boundary_shiftedobjDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
site_boundary_shiftedobjDisplay.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
site_boundary_shiftedobjDisplay.GlyphType.TipResolution = 6
site_boundary_shiftedobjDisplay.GlyphType.TipRadius = 0.1
site_boundary_shiftedobjDisplay.GlyphType.TipLength = 0.35
site_boundary_shiftedobjDisplay.GlyphType.ShaftResolution = 6
site_boundary_shiftedobjDisplay.GlyphType.ShaftRadius = 0.03
site_boundary_shiftedobjDisplay.GlyphType.Invert = 0

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
site_boundary_shiftedobjDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
site_boundary_shiftedobjDisplay.ScaleTransferFunction.UseLogScale = 0

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
site_boundary_shiftedobjDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
site_boundary_shiftedobjDisplay.OpacityTransferFunction.UseLogScale = 0

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitle = 'X Axis'
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitle = 'Y Axis'
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitle = 'Z Axis'
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitleFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitleFontFile = ''
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitleBold = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitleItalic = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitleFontSize = 12
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitleShadow = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.XTitleOpacity = 1.0
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitleFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitleFontFile = ''
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitleBold = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitleItalic = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitleFontSize = 12
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitleShadow = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.YTitleOpacity = 1.0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitleFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitleFontFile = ''
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitleBold = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitleItalic = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitleFontSize = 12
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitleShadow = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZTitleOpacity = 1.0
site_boundary_shiftedobjDisplay.DataAxesGrid.FacesToRender = 63
site_boundary_shiftedobjDisplay.DataAxesGrid.CullBackface = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.CullFrontface = 1
site_boundary_shiftedobjDisplay.DataAxesGrid.ShowGrid = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ShowEdges = 1
site_boundary_shiftedobjDisplay.DataAxesGrid.ShowTicks = 1
site_boundary_shiftedobjDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 1
site_boundary_shiftedobjDisplay.DataAxesGrid.AxesToLabel = 63
site_boundary_shiftedobjDisplay.DataAxesGrid.XLabelFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.DataAxesGrid.XLabelFontFile = ''
site_boundary_shiftedobjDisplay.DataAxesGrid.XLabelBold = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.XLabelItalic = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.XLabelFontSize = 12
site_boundary_shiftedobjDisplay.DataAxesGrid.XLabelShadow = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.XLabelOpacity = 1.0
site_boundary_shiftedobjDisplay.DataAxesGrid.YLabelFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.DataAxesGrid.YLabelFontFile = ''
site_boundary_shiftedobjDisplay.DataAxesGrid.YLabelBold = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.YLabelItalic = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.YLabelFontSize = 12
site_boundary_shiftedobjDisplay.DataAxesGrid.YLabelShadow = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.YLabelOpacity = 1.0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZLabelFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.DataAxesGrid.ZLabelFontFile = ''
site_boundary_shiftedobjDisplay.DataAxesGrid.ZLabelBold = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZLabelItalic = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZLabelFontSize = 12
site_boundary_shiftedobjDisplay.DataAxesGrid.ZLabelShadow = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZLabelOpacity = 1.0
site_boundary_shiftedobjDisplay.DataAxesGrid.XAxisNotation = 'Mixed'
site_boundary_shiftedobjDisplay.DataAxesGrid.XAxisPrecision = 2
site_boundary_shiftedobjDisplay.DataAxesGrid.XAxisUseCustomLabels = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.XAxisLabels = []
site_boundary_shiftedobjDisplay.DataAxesGrid.YAxisNotation = 'Mixed'
site_boundary_shiftedobjDisplay.DataAxesGrid.YAxisPrecision = 2
site_boundary_shiftedobjDisplay.DataAxesGrid.YAxisUseCustomLabels = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.YAxisLabels = []
site_boundary_shiftedobjDisplay.DataAxesGrid.ZAxisNotation = 'Mixed'
site_boundary_shiftedobjDisplay.DataAxesGrid.ZAxisPrecision = 2
site_boundary_shiftedobjDisplay.DataAxesGrid.ZAxisUseCustomLabels = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.ZAxisLabels = []
site_boundary_shiftedobjDisplay.DataAxesGrid.UseCustomBounds = 0
site_boundary_shiftedobjDisplay.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
site_boundary_shiftedobjDisplay.PolarAxes.Visibility = 0
site_boundary_shiftedobjDisplay.PolarAxes.Translation = [0.0, 0.0, 0.0]
site_boundary_shiftedobjDisplay.PolarAxes.Scale = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.Orientation = [0.0, 0.0, 0.0]
site_boundary_shiftedobjDisplay.PolarAxes.EnableCustomBounds = [0, 0, 0]
site_boundary_shiftedobjDisplay.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.EnableCustomRange = 0
site_boundary_shiftedobjDisplay.PolarAxes.CustomRange = [0.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.AutoPole = 1
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.RadialAxesVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.DrawRadialGridlines = 1
site_boundary_shiftedobjDisplay.PolarAxes.PolarArcsVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.DrawPolarArcsGridlines = 1
site_boundary_shiftedobjDisplay.PolarAxes.NumberOfRadialAxes = 0
site_boundary_shiftedobjDisplay.PolarAxes.DeltaAngleRadialAxes = 45.0
site_boundary_shiftedobjDisplay.PolarAxes.NumberOfPolarAxes = 5
site_boundary_shiftedobjDisplay.PolarAxes.DeltaRangePolarAxes = 0.0
site_boundary_shiftedobjDisplay.PolarAxes.CustomMinRadius = 1
site_boundary_shiftedobjDisplay.PolarAxes.MinimumRadius = 0.0
site_boundary_shiftedobjDisplay.PolarAxes.CustomMaxRadius = 0
site_boundary_shiftedobjDisplay.PolarAxes.MaximumRadius = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.CustomAngles = 1
site_boundary_shiftedobjDisplay.PolarAxes.MinimumAngle = 0.0
site_boundary_shiftedobjDisplay.PolarAxes.MaximumAngle = 90.0
site_boundary_shiftedobjDisplay.PolarAxes.RadialAxesOriginToPolarAxis = 1
site_boundary_shiftedobjDisplay.PolarAxes.PolarArcResolutionPerDegree = 0.2
site_boundary_shiftedobjDisplay.PolarAxes.Ratio = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.EnableOverallColor = 1
site_boundary_shiftedobjDisplay.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitle = 'Radial Distance'
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleLocation = 'Bottom'
site_boundary_shiftedobjDisplay.PolarAxes.PolarTitleOffset = [20.0, 20.0]
site_boundary_shiftedobjDisplay.PolarAxes.PolarLabelVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.PolarLabelFormat = '%-#6.3g'
site_boundary_shiftedobjDisplay.PolarAxes.PolarLabelExponentLocation = 'Labels'
site_boundary_shiftedobjDisplay.PolarAxes.PolarLabelOffset = 10.0
site_boundary_shiftedobjDisplay.PolarAxes.PolarExponentOffset = 5.0
site_boundary_shiftedobjDisplay.PolarAxes.RadialTitleVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.RadialTitleFormat = '%-#3.1f'
site_boundary_shiftedobjDisplay.PolarAxes.RadialTitleLocation = 'Bottom'
site_boundary_shiftedobjDisplay.PolarAxes.RadialTitleOffset = [20.0, 0.0]
site_boundary_shiftedobjDisplay.PolarAxes.RadialUnitsVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.ScreenSize = 10.0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleOpacity = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleFontFile = ''
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleBold = 0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleItalic = 0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleShadow = 0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTitleFontSize = 12
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisLabelOpacity = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisLabelFontFile = ''
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisLabelBold = 0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisLabelItalic = 0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisLabelShadow = 0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisLabelFontSize = 12
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTextOpacity = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTextBold = 0
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTextItalic = 0
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTextShadow = 0
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTextFontSize = 12
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesTextBold = 0
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesTextItalic = 0
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesTextShadow = 0
site_boundary_shiftedobjDisplay.PolarAxes.SecondaryRadialAxesTextFontSize = 12
site_boundary_shiftedobjDisplay.PolarAxes.EnableDistanceLOD = 1
site_boundary_shiftedobjDisplay.PolarAxes.DistanceLODThreshold = 0.7
site_boundary_shiftedobjDisplay.PolarAxes.EnableViewAngleLOD = 1
site_boundary_shiftedobjDisplay.PolarAxes.ViewAngleLODThreshold = 0.7
site_boundary_shiftedobjDisplay.PolarAxes.SmallestVisiblePolarAngle = 0.5
site_boundary_shiftedobjDisplay.PolarAxes.PolarTicksVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.ArcTicksOriginToPolarAxis = 1
site_boundary_shiftedobjDisplay.PolarAxes.TickLocation = 'Both'
site_boundary_shiftedobjDisplay.PolarAxes.AxisTickVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.AxisMinorTickVisibility = 0
site_boundary_shiftedobjDisplay.PolarAxes.AxisTickMatchesPolarAxes = 1
site_boundary_shiftedobjDisplay.PolarAxes.DeltaRangeMajor = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.DeltaRangeMinor = 0.5
site_boundary_shiftedobjDisplay.PolarAxes.ArcTickVisibility = 1
site_boundary_shiftedobjDisplay.PolarAxes.ArcMinorTickVisibility = 0
site_boundary_shiftedobjDisplay.PolarAxes.ArcTickMatchesRadialAxes = 1
site_boundary_shiftedobjDisplay.PolarAxes.DeltaAngleMajor = 10.0
site_boundary_shiftedobjDisplay.PolarAxes.DeltaAngleMinor = 5.0
site_boundary_shiftedobjDisplay.PolarAxes.TickRatioRadiusSize = 0.02
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisMajorTickSize = 0.0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTickRatioSize = 0.3
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisMajorTickThickness = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.PolarAxisTickRatioThickness = 0.5
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisMajorTickSize = 0.0
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTickRatioSize = 0.3
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
site_boundary_shiftedobjDisplay.PolarAxes.ArcMajorTickSize = 0.0
site_boundary_shiftedobjDisplay.PolarAxes.ArcTickRatioSize = 0.3
site_boundary_shiftedobjDisplay.PolarAxes.ArcMajorTickThickness = 1.0
site_boundary_shiftedobjDisplay.PolarAxes.ArcTickRatioThickness = 0.5
site_boundary_shiftedobjDisplay.PolarAxes.Use2DMode = 0
site_boundary_shiftedobjDisplay.PolarAxes.UseLogAxis = 0

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Extract Edges'
extractEdges1 = ExtractEdges(registrationName='ExtractEdges1', Input=site_boundary_shiftedobj)

# show data in view
extractEdges1Display = Show(extractEdges1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
extractEdges1Display.Selection = None
extractEdges1Display.Representation = 'Surface'
extractEdges1Display.ColorArrayName = [None, '']
extractEdges1Display.LookupTable = None
extractEdges1Display.MapScalars = 1
extractEdges1Display.MultiComponentsMapping = 0
extractEdges1Display.InterpolateScalarsBeforeMapping = 1
extractEdges1Display.UseNanColorForMissingArrays = 0
extractEdges1Display.Opacity = 1.0
extractEdges1Display.PointSize = 2.0
extractEdges1Display.LineWidth = 1.0
extractEdges1Display.RenderLinesAsTubes = 0
extractEdges1Display.RenderPointsAsSpheres = 0
extractEdges1Display.DisableLighting = 0
extractEdges1Display.Diffuse = 1.0
extractEdges1Display.Interpolation = 'Gouraud'
extractEdges1Display.Specular = 0.0
extractEdges1Display.SpecularColor = [1.0, 1.0, 1.0]
extractEdges1Display.SpecularPower = 100.0
extractEdges1Display.Luminosity = 0.0
extractEdges1Display.Ambient = 0.0
extractEdges1Display.Roughness = 0.3
extractEdges1Display.Metallic = 0.0
extractEdges1Display.EdgeTint = [1.0, 1.0, 1.0]
extractEdges1Display.Anisotropy = 0.0
extractEdges1Display.AnisotropyRotation = 0.0
extractEdges1Display.BaseIOR = 1.5
extractEdges1Display.CoatStrength = 0.0
extractEdges1Display.CoatIOR = 2.0
extractEdges1Display.CoatRoughness = 0.0
extractEdges1Display.CoatColor = [1.0, 1.0, 1.0]
extractEdges1Display.SelectNormalArray = 'None'
extractEdges1Display.SelectTangentArray = 'None'
extractEdges1Display.ComputePointNormals = 0
extractEdges1Display.Splitting = 1
extractEdges1Display.FeatureAngle = 30.0
extractEdges1Display.SelectTCoordArray = 'None'
extractEdges1Display.Texture = None
extractEdges1Display.RepeatTextures = 1
extractEdges1Display.InterpolateTextures = 0
extractEdges1Display.SeamlessU = 0
extractEdges1Display.SeamlessV = 0
extractEdges1Display.UseMipmapTextures = 0
extractEdges1Display.ShowTexturesOnBackface = 1
extractEdges1Display.BaseColorTexture = None
extractEdges1Display.NormalTexture = None
extractEdges1Display.NormalScale = 1.0
extractEdges1Display.CoatNormalTexture = None
extractEdges1Display.CoatNormalScale = 1.0
extractEdges1Display.MaterialTexture = None
extractEdges1Display.OcclusionStrength = 1.0
extractEdges1Display.AnisotropyTexture = None
extractEdges1Display.EmissiveTexture = None
extractEdges1Display.EmissiveFactor = [1.0, 1.0, 1.0]
extractEdges1Display.TextureTransform = 'Transform2'
extractEdges1Display.EdgeOpacity = 1.0
extractEdges1Display.BackfaceRepresentation = 'Follow Frontface'
extractEdges1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
extractEdges1Display.BackfaceOpacity = 1.0
extractEdges1Display.Translation = [0.0, 0.0, 0.0]
extractEdges1Display.Scale = [1.0, 1.0, 1.0]
extractEdges1Display.Orientation = [0.0, 0.0, 0.0]
extractEdges1Display.Origin = [0.0, 0.0, 0.0]
extractEdges1Display.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
extractEdges1Display.Pickable = 1
extractEdges1Display.Triangulate = 0
extractEdges1Display.UseShaderReplacements = 0
extractEdges1Display.ShaderReplacements = ''
extractEdges1Display.NonlinearSubdivisionLevel = 1
extractEdges1Display.MatchBoundariesIgnoringCellOrder = 0
extractEdges1Display.UseDataPartitions = 0
extractEdges1Display.OSPRayUseScaleArray = 'All Approximate'
extractEdges1Display.OSPRayScaleArray = ''
extractEdges1Display.OSPRayScaleFunction = 'Piecewise Function'
extractEdges1Display.OSPRayMaterial = 'None'
extractEdges1Display.Assembly = ''
extractEdges1Display.SelectedBlockSelectors = ['']
extractEdges1Display.BlockSelectors = ['/']
extractEdges1Display.BlockColors = []
extractEdges1Display.BlockColorArrayNames = []
extractEdges1Display.BlockLookupTables = []
extractEdges1Display.BlockUseSeparateColorMaps = []
extractEdges1Display.BlockMapScalars = []
extractEdges1Display.BlockInterpolateScalarsBeforeMappings = []
extractEdges1Display.BlockOpacities = []
extractEdges1Display.BlockMapScalarsGUI = 1
extractEdges1Display.BlockInterpolateScalarsBeforeMappingsGUI = 1
extractEdges1Display.BlockOpacitiesGUI = 1.0
extractEdges1Display.Orient = 0
extractEdges1Display.OrientationMode = 'Direction'
extractEdges1Display.SelectOrientationVectors = 'None'
extractEdges1Display.Scaling = 0
extractEdges1Display.ScaleMode = 'No Data Scaling Off'
extractEdges1Display.ScaleFactor = 34.06010131835938
extractEdges1Display.SelectScaleArray = 'None'
extractEdges1Display.GlyphType = 'Arrow'
extractEdges1Display.UseGlyphTable = 0
extractEdges1Display.GlyphTableIndexArray = 'None'
extractEdges1Display.UseCompositeGlyphTable = 0
extractEdges1Display.UseGlyphCullingAndLOD = 0
extractEdges1Display.LODValues = []
extractEdges1Display.ColorByLODIndex = 0
extractEdges1Display.GaussianRadius = 1.7030050659179687
extractEdges1Display.ShaderPreset = 'Sphere'
extractEdges1Display.CustomTriangleScale = 3
extractEdges1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
extractEdges1Display.Emissive = 0
extractEdges1Display.ScaleByArray = 0
extractEdges1Display.SetScaleArray = [None, '']
extractEdges1Display.ScaleArrayComponent = 0
extractEdges1Display.UseScaleFunction = 1
extractEdges1Display.ScaleTransferFunction = 'Piecewise Function'
extractEdges1Display.OpacityByArray = 0
extractEdges1Display.OpacityArray = [None, '']
extractEdges1Display.OpacityArrayComponent = 0
extractEdges1Display.OpacityTransferFunction = 'Piecewise Function'
extractEdges1Display.DataAxesGrid = 'Grid Axes Representation'
extractEdges1Display.SelectionCellLabelBold = 0
extractEdges1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
extractEdges1Display.SelectionCellLabelFontFamily = 'Arial'
extractEdges1Display.SelectionCellLabelFontFile = ''
extractEdges1Display.SelectionCellLabelFontSize = 18
extractEdges1Display.SelectionCellLabelItalic = 0
extractEdges1Display.SelectionCellLabelJustification = 'Left'
extractEdges1Display.SelectionCellLabelOpacity = 1.0
extractEdges1Display.SelectionCellLabelShadow = 0
extractEdges1Display.SelectionPointLabelBold = 0
extractEdges1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
extractEdges1Display.SelectionPointLabelFontFamily = 'Arial'
extractEdges1Display.SelectionPointLabelFontFile = ''
extractEdges1Display.SelectionPointLabelFontSize = 18
extractEdges1Display.SelectionPointLabelItalic = 0
extractEdges1Display.SelectionPointLabelJustification = 'Left'
extractEdges1Display.SelectionPointLabelOpacity = 1.0
extractEdges1Display.SelectionPointLabelShadow = 0
extractEdges1Display.PolarAxes = 'Polar Axes Representation'
extractEdges1Display.SelectInputVectors = [None, '']
extractEdges1Display.NumberOfSteps = 40
extractEdges1Display.StepSize = 0.25
extractEdges1Display.NormalizeVectors = 1
extractEdges1Display.EnhancedLIC = 1
extractEdges1Display.ColorMode = 'Blend'
extractEdges1Display.LICIntensity = 0.8
extractEdges1Display.MapModeBias = 0.0
extractEdges1Display.EnhanceContrast = 'Off'
extractEdges1Display.LowLICContrastEnhancementFactor = 0.0
extractEdges1Display.HighLICContrastEnhancementFactor = 0.0
extractEdges1Display.LowColorContrastEnhancementFactor = 0.0
extractEdges1Display.HighColorContrastEnhancementFactor = 0.0
extractEdges1Display.AntiAlias = 0
extractEdges1Display.MaskOnSurface = 1
extractEdges1Display.MaskThreshold = 0.0
extractEdges1Display.MaskIntensity = 0.0
extractEdges1Display.MaskColor = [0.5, 0.5, 0.5]
extractEdges1Display.GenerateNoiseTexture = 0
extractEdges1Display.NoiseType = 'Gaussian'
extractEdges1Display.NoiseTextureSize = 128
extractEdges1Display.NoiseGrainSize = 2
extractEdges1Display.MinNoiseValue = 0.0
extractEdges1Display.MaxNoiseValue = 0.8
extractEdges1Display.NumberOfNoiseLevels = 1024
extractEdges1Display.ImpulseNoiseProbability = 1.0
extractEdges1Display.ImpulseNoiseBackgroundValue = 0.0
extractEdges1Display.NoiseGeneratorSeed = 1
extractEdges1Display.CompositeStrategy = 'AUTO'
extractEdges1Display.UseLICForLOD = 0
extractEdges1Display.WriteLog = ''

# init the 'Transform2' selected for 'TextureTransform'
extractEdges1Display.TextureTransform.Translate = [0.0, 0.0, 0.0]
extractEdges1Display.TextureTransform.Rotate = [0.0, 0.0, 0.0]
extractEdges1Display.TextureTransform.Scale = [1.0, 1.0, 1.0]

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
extractEdges1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
extractEdges1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
extractEdges1Display.GlyphType.TipResolution = 6
extractEdges1Display.GlyphType.TipRadius = 0.1
extractEdges1Display.GlyphType.TipLength = 0.35
extractEdges1Display.GlyphType.ShaftResolution = 6
extractEdges1Display.GlyphType.ShaftRadius = 0.03
extractEdges1Display.GlyphType.Invert = 0

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
extractEdges1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
extractEdges1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
extractEdges1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
extractEdges1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
extractEdges1Display.DataAxesGrid.XTitle = 'X Axis'
extractEdges1Display.DataAxesGrid.YTitle = 'Y Axis'
extractEdges1Display.DataAxesGrid.ZTitle = 'Z Axis'
extractEdges1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
extractEdges1Display.DataAxesGrid.XTitleFontFile = ''
extractEdges1Display.DataAxesGrid.XTitleBold = 0
extractEdges1Display.DataAxesGrid.XTitleItalic = 0
extractEdges1Display.DataAxesGrid.XTitleFontSize = 12
extractEdges1Display.DataAxesGrid.XTitleShadow = 0
extractEdges1Display.DataAxesGrid.XTitleOpacity = 1.0
extractEdges1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
extractEdges1Display.DataAxesGrid.YTitleFontFile = ''
extractEdges1Display.DataAxesGrid.YTitleBold = 0
extractEdges1Display.DataAxesGrid.YTitleItalic = 0
extractEdges1Display.DataAxesGrid.YTitleFontSize = 12
extractEdges1Display.DataAxesGrid.YTitleShadow = 0
extractEdges1Display.DataAxesGrid.YTitleOpacity = 1.0
extractEdges1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
extractEdges1Display.DataAxesGrid.ZTitleFontFile = ''
extractEdges1Display.DataAxesGrid.ZTitleBold = 0
extractEdges1Display.DataAxesGrid.ZTitleItalic = 0
extractEdges1Display.DataAxesGrid.ZTitleFontSize = 12
extractEdges1Display.DataAxesGrid.ZTitleShadow = 0
extractEdges1Display.DataAxesGrid.ZTitleOpacity = 1.0
extractEdges1Display.DataAxesGrid.FacesToRender = 63
extractEdges1Display.DataAxesGrid.CullBackface = 0
extractEdges1Display.DataAxesGrid.CullFrontface = 1
extractEdges1Display.DataAxesGrid.ShowGrid = 0
extractEdges1Display.DataAxesGrid.ShowEdges = 1
extractEdges1Display.DataAxesGrid.ShowTicks = 1
extractEdges1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
extractEdges1Display.DataAxesGrid.AxesToLabel = 63
extractEdges1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
extractEdges1Display.DataAxesGrid.XLabelFontFile = ''
extractEdges1Display.DataAxesGrid.XLabelBold = 0
extractEdges1Display.DataAxesGrid.XLabelItalic = 0
extractEdges1Display.DataAxesGrid.XLabelFontSize = 12
extractEdges1Display.DataAxesGrid.XLabelShadow = 0
extractEdges1Display.DataAxesGrid.XLabelOpacity = 1.0
extractEdges1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
extractEdges1Display.DataAxesGrid.YLabelFontFile = ''
extractEdges1Display.DataAxesGrid.YLabelBold = 0
extractEdges1Display.DataAxesGrid.YLabelItalic = 0
extractEdges1Display.DataAxesGrid.YLabelFontSize = 12
extractEdges1Display.DataAxesGrid.YLabelShadow = 0
extractEdges1Display.DataAxesGrid.YLabelOpacity = 1.0
extractEdges1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
extractEdges1Display.DataAxesGrid.ZLabelFontFile = ''
extractEdges1Display.DataAxesGrid.ZLabelBold = 0
extractEdges1Display.DataAxesGrid.ZLabelItalic = 0
extractEdges1Display.DataAxesGrid.ZLabelFontSize = 12
extractEdges1Display.DataAxesGrid.ZLabelShadow = 0
extractEdges1Display.DataAxesGrid.ZLabelOpacity = 1.0
extractEdges1Display.DataAxesGrid.XAxisNotation = 'Mixed'
extractEdges1Display.DataAxesGrid.XAxisPrecision = 2
extractEdges1Display.DataAxesGrid.XAxisUseCustomLabels = 0
extractEdges1Display.DataAxesGrid.XAxisLabels = []
extractEdges1Display.DataAxesGrid.YAxisNotation = 'Mixed'
extractEdges1Display.DataAxesGrid.YAxisPrecision = 2
extractEdges1Display.DataAxesGrid.YAxisUseCustomLabels = 0
extractEdges1Display.DataAxesGrid.YAxisLabels = []
extractEdges1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
extractEdges1Display.DataAxesGrid.ZAxisPrecision = 2
extractEdges1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
extractEdges1Display.DataAxesGrid.ZAxisLabels = []
extractEdges1Display.DataAxesGrid.UseCustomBounds = 0
extractEdges1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
extractEdges1Display.PolarAxes.Visibility = 0
extractEdges1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
extractEdges1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
extractEdges1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
extractEdges1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
extractEdges1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
extractEdges1Display.PolarAxes.EnableCustomRange = 0
extractEdges1Display.PolarAxes.CustomRange = [0.0, 1.0]
extractEdges1Display.PolarAxes.AutoPole = 1
extractEdges1Display.PolarAxes.PolarAxisVisibility = 1
extractEdges1Display.PolarAxes.RadialAxesVisibility = 1
extractEdges1Display.PolarAxes.DrawRadialGridlines = 1
extractEdges1Display.PolarAxes.PolarArcsVisibility = 1
extractEdges1Display.PolarAxes.DrawPolarArcsGridlines = 1
extractEdges1Display.PolarAxes.NumberOfRadialAxes = 0
extractEdges1Display.PolarAxes.DeltaAngleRadialAxes = 45.0
extractEdges1Display.PolarAxes.NumberOfPolarAxes = 5
extractEdges1Display.PolarAxes.DeltaRangePolarAxes = 0.0
extractEdges1Display.PolarAxes.CustomMinRadius = 1
extractEdges1Display.PolarAxes.MinimumRadius = 0.0
extractEdges1Display.PolarAxes.CustomMaxRadius = 0
extractEdges1Display.PolarAxes.MaximumRadius = 1.0
extractEdges1Display.PolarAxes.CustomAngles = 1
extractEdges1Display.PolarAxes.MinimumAngle = 0.0
extractEdges1Display.PolarAxes.MaximumAngle = 90.0
extractEdges1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
extractEdges1Display.PolarAxes.PolarArcResolutionPerDegree = 0.2
extractEdges1Display.PolarAxes.Ratio = 1.0
extractEdges1Display.PolarAxes.EnableOverallColor = 1
extractEdges1Display.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
extractEdges1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
extractEdges1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
extractEdges1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
extractEdges1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
extractEdges1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
extractEdges1Display.PolarAxes.PolarAxisTitleVisibility = 1
extractEdges1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
extractEdges1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
extractEdges1Display.PolarAxes.PolarTitleOffset = [20.0, 20.0]
extractEdges1Display.PolarAxes.PolarLabelVisibility = 1
extractEdges1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
extractEdges1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
extractEdges1Display.PolarAxes.PolarLabelOffset = 10.0
extractEdges1Display.PolarAxes.PolarExponentOffset = 5.0
extractEdges1Display.PolarAxes.RadialTitleVisibility = 1
extractEdges1Display.PolarAxes.RadialTitleFormat = '%-#3.1f'
extractEdges1Display.PolarAxes.RadialTitleLocation = 'Bottom'
extractEdges1Display.PolarAxes.RadialTitleOffset = [20.0, 0.0]
extractEdges1Display.PolarAxes.RadialUnitsVisibility = 1
extractEdges1Display.PolarAxes.ScreenSize = 10.0
extractEdges1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
extractEdges1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
extractEdges1Display.PolarAxes.PolarAxisTitleFontFile = ''
extractEdges1Display.PolarAxes.PolarAxisTitleBold = 0
extractEdges1Display.PolarAxes.PolarAxisTitleItalic = 0
extractEdges1Display.PolarAxes.PolarAxisTitleShadow = 0
extractEdges1Display.PolarAxes.PolarAxisTitleFontSize = 12
extractEdges1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
extractEdges1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
extractEdges1Display.PolarAxes.PolarAxisLabelFontFile = ''
extractEdges1Display.PolarAxes.PolarAxisLabelBold = 0
extractEdges1Display.PolarAxes.PolarAxisLabelItalic = 0
extractEdges1Display.PolarAxes.PolarAxisLabelShadow = 0
extractEdges1Display.PolarAxes.PolarAxisLabelFontSize = 12
extractEdges1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
extractEdges1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
extractEdges1Display.PolarAxes.LastRadialAxisTextFontFile = ''
extractEdges1Display.PolarAxes.LastRadialAxisTextBold = 0
extractEdges1Display.PolarAxes.LastRadialAxisTextItalic = 0
extractEdges1Display.PolarAxes.LastRadialAxisTextShadow = 0
extractEdges1Display.PolarAxes.LastRadialAxisTextFontSize = 12
extractEdges1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
extractEdges1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
extractEdges1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
extractEdges1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
extractEdges1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
extractEdges1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
extractEdges1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
extractEdges1Display.PolarAxes.EnableDistanceLOD = 1
extractEdges1Display.PolarAxes.DistanceLODThreshold = 0.7
extractEdges1Display.PolarAxes.EnableViewAngleLOD = 1
extractEdges1Display.PolarAxes.ViewAngleLODThreshold = 0.7
extractEdges1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
extractEdges1Display.PolarAxes.PolarTicksVisibility = 1
extractEdges1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
extractEdges1Display.PolarAxes.TickLocation = 'Both'
extractEdges1Display.PolarAxes.AxisTickVisibility = 1
extractEdges1Display.PolarAxes.AxisMinorTickVisibility = 0
extractEdges1Display.PolarAxes.AxisTickMatchesPolarAxes = 1
extractEdges1Display.PolarAxes.DeltaRangeMajor = 1.0
extractEdges1Display.PolarAxes.DeltaRangeMinor = 0.5
extractEdges1Display.PolarAxes.ArcTickVisibility = 1
extractEdges1Display.PolarAxes.ArcMinorTickVisibility = 0
extractEdges1Display.PolarAxes.ArcTickMatchesRadialAxes = 1
extractEdges1Display.PolarAxes.DeltaAngleMajor = 10.0
extractEdges1Display.PolarAxes.DeltaAngleMinor = 5.0
extractEdges1Display.PolarAxes.TickRatioRadiusSize = 0.02
extractEdges1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
extractEdges1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
extractEdges1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
extractEdges1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
extractEdges1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
extractEdges1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
extractEdges1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
extractEdges1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
extractEdges1Display.PolarAxes.ArcMajorTickSize = 0.0
extractEdges1Display.PolarAxes.ArcTickRatioSize = 0.3
extractEdges1Display.PolarAxes.ArcMajorTickThickness = 1.0
extractEdges1Display.PolarAxes.ArcTickRatioThickness = 0.5
extractEdges1Display.PolarAxes.Use2DMode = 0
extractEdges1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(site_boundary_shiftedobj, renderView1)
renderView1.OrientationAxesVisibility = 0
# update the view to ensure updated data information
renderView1.Update()

# create a new 'Transform'
transform1 = Transform(registrationName='Transform1', Input=extractEdges1)
transform1.Transform = 'Transform'
transform1.TransformAllInputVectors = 1

# init the 'Transform' selected for 'Transform'
transform1.Transform.Translate = [0.0, 0.0, 0.0]
transform1.Transform.Rotate = [0.0, 0.0, 0.0]
transform1.Transform.Scale = [1.0, 1.0, 1.0]

# Properties modified on transform1.Transform
transform1.Transform.Translate = [0.0, 0.0, slice_z+0.1*slice_z]

# show data in view
transform1Display = Show(transform1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
transform1Display.Selection = None
transform1Display.Representation = 'Surface'
transform1Display.ColorArrayName = [None, '']
transform1Display.LookupTable = None
transform1Display.MapScalars = 1
transform1Display.MultiComponentsMapping = 0
transform1Display.InterpolateScalarsBeforeMapping = 1
transform1Display.UseNanColorForMissingArrays = 0
transform1Display.Opacity = 1.0
transform1Display.PointSize = 2.0
transform1Display.LineWidth = 1.0
transform1Display.RenderLinesAsTubes = 0
transform1Display.RenderPointsAsSpheres = 0
transform1Display.DisableLighting = 0
transform1Display.Diffuse = 1.0
transform1Display.Interpolation = 'Gouraud'
transform1Display.Specular = 0.0
transform1Display.SpecularColor = [1.0, 1.0, 1.0]
transform1Display.SpecularPower = 100.0
transform1Display.Luminosity = 0.0
transform1Display.Ambient = 0.0
transform1Display.Roughness = 0.3
transform1Display.Metallic = 0.0
transform1Display.EdgeTint = [1.0, 1.0, 1.0]
transform1Display.Anisotropy = 0.0
transform1Display.AnisotropyRotation = 0.0
transform1Display.BaseIOR = 1.5
transform1Display.CoatStrength = 0.0
transform1Display.CoatIOR = 2.0
transform1Display.CoatRoughness = 0.0
transform1Display.CoatColor = [1.0, 1.0, 1.0]
transform1Display.SelectNormalArray = 'None'
transform1Display.SelectTangentArray = 'None'
transform1Display.ComputePointNormals = 0
transform1Display.Splitting = 1
transform1Display.FeatureAngle = 30.0
transform1Display.SelectTCoordArray = 'None'
transform1Display.Texture = None
transform1Display.RepeatTextures = 1
transform1Display.InterpolateTextures = 0
transform1Display.SeamlessU = 0
transform1Display.SeamlessV = 0
transform1Display.UseMipmapTextures = 0
transform1Display.ShowTexturesOnBackface = 1
transform1Display.BaseColorTexture = None
transform1Display.NormalTexture = None
transform1Display.NormalScale = 1.0
transform1Display.CoatNormalTexture = None
transform1Display.CoatNormalScale = 1.0
transform1Display.MaterialTexture = None
transform1Display.OcclusionStrength = 1.0
transform1Display.AnisotropyTexture = None
transform1Display.EmissiveTexture = None
transform1Display.EmissiveFactor = [1.0, 1.0, 1.0]
transform1Display.TextureTransform = 'Transform2'
transform1Display.EdgeOpacity = 1.0
transform1Display.BackfaceRepresentation = 'Follow Frontface'
transform1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
transform1Display.BackfaceOpacity = 1.0
transform1Display.Translation = [0.0, 0.0, 0.0]
transform1Display.Scale = [1.0, 1.0, 1.0]
transform1Display.Orientation = [0.0, 0.0, 0.0]
transform1Display.Origin = [0.0, 0.0, 0.0]
transform1Display.CoordinateShiftScaleMethod = 'Always Auto Shift Scale'
transform1Display.Pickable = 1
transform1Display.Triangulate = 0
transform1Display.UseShaderReplacements = 0
transform1Display.ShaderReplacements = ''
transform1Display.NonlinearSubdivisionLevel = 1
transform1Display.MatchBoundariesIgnoringCellOrder = 0
transform1Display.UseDataPartitions = 0
transform1Display.OSPRayUseScaleArray = 'All Approximate'
transform1Display.OSPRayScaleArray = ''
transform1Display.OSPRayScaleFunction = 'Piecewise Function'
transform1Display.OSPRayMaterial = 'None'
transform1Display.Assembly = ''
transform1Display.SelectedBlockSelectors = ['']
transform1Display.BlockSelectors = ['/']
transform1Display.BlockColors = []
transform1Display.BlockColorArrayNames = []
transform1Display.BlockLookupTables = []
transform1Display.BlockUseSeparateColorMaps = []
transform1Display.BlockMapScalars = []
transform1Display.BlockInterpolateScalarsBeforeMappings = []
transform1Display.BlockOpacities = []
transform1Display.BlockMapScalarsGUI = 1
transform1Display.BlockInterpolateScalarsBeforeMappingsGUI = 1
transform1Display.BlockOpacitiesGUI = 1.0
transform1Display.Orient = 0
transform1Display.OrientationMode = 'Direction'
transform1Display.SelectOrientationVectors = 'None'
transform1Display.Scaling = 0
transform1Display.ScaleMode = 'No Data Scaling Off'
transform1Display.ScaleFactor = 34.06010131835938
transform1Display.SelectScaleArray = 'None'
transform1Display.GlyphType = 'Arrow'
transform1Display.UseGlyphTable = 0
transform1Display.GlyphTableIndexArray = 'None'
transform1Display.UseCompositeGlyphTable = 0
transform1Display.UseGlyphCullingAndLOD = 0
transform1Display.LODValues = []
transform1Display.ColorByLODIndex = 0
transform1Display.GaussianRadius = 1.7030050659179687
transform1Display.ShaderPreset = 'Sphere'
transform1Display.CustomTriangleScale = 3
transform1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
transform1Display.Emissive = 0
transform1Display.ScaleByArray = 0
transform1Display.SetScaleArray = [None, '']
transform1Display.ScaleArrayComponent = 0
transform1Display.UseScaleFunction = 1
transform1Display.ScaleTransferFunction = 'Piecewise Function'
transform1Display.OpacityByArray = 0
transform1Display.OpacityArray = [None, '']
transform1Display.OpacityArrayComponent = 0
transform1Display.OpacityTransferFunction = 'Piecewise Function'
transform1Display.DataAxesGrid = 'Grid Axes Representation'
transform1Display.SelectionCellLabelBold = 0
transform1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
transform1Display.SelectionCellLabelFontFamily = 'Arial'
transform1Display.SelectionCellLabelFontFile = ''
transform1Display.SelectionCellLabelFontSize = 18
transform1Display.SelectionCellLabelItalic = 0
transform1Display.SelectionCellLabelJustification = 'Left'
transform1Display.SelectionCellLabelOpacity = 1.0
transform1Display.SelectionCellLabelShadow = 0
transform1Display.SelectionPointLabelBold = 0
transform1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
transform1Display.SelectionPointLabelFontFamily = 'Arial'
transform1Display.SelectionPointLabelFontFile = ''
transform1Display.SelectionPointLabelFontSize = 18
transform1Display.SelectionPointLabelItalic = 0
transform1Display.SelectionPointLabelJustification = 'Left'
transform1Display.SelectionPointLabelOpacity = 1.0
transform1Display.SelectionPointLabelShadow = 0
transform1Display.PolarAxes = 'Polar Axes Representation'
transform1Display.SelectInputVectors = [None, '']
transform1Display.NumberOfSteps = 40
transform1Display.StepSize = 0.25
transform1Display.NormalizeVectors = 1
transform1Display.EnhancedLIC = 1
transform1Display.ColorMode = 'Blend'
transform1Display.LICIntensity = 0.8
transform1Display.MapModeBias = 0.0
transform1Display.EnhanceContrast = 'Off'
transform1Display.LowLICContrastEnhancementFactor = 0.0
transform1Display.HighLICContrastEnhancementFactor = 0.0
transform1Display.LowColorContrastEnhancementFactor = 0.0
transform1Display.HighColorContrastEnhancementFactor = 0.0
transform1Display.AntiAlias = 0
transform1Display.MaskOnSurface = 1
transform1Display.MaskThreshold = 0.0
transform1Display.MaskIntensity = 0.0
transform1Display.MaskColor = [0.5, 0.5, 0.5]
transform1Display.GenerateNoiseTexture = 0
transform1Display.NoiseType = 'Gaussian'
transform1Display.NoiseTextureSize = 128
transform1Display.NoiseGrainSize = 2
transform1Display.MinNoiseValue = 0.0
transform1Display.MaxNoiseValue = 0.8
transform1Display.NumberOfNoiseLevels = 1024
transform1Display.ImpulseNoiseProbability = 1.0
transform1Display.ImpulseNoiseBackgroundValue = 0.0
transform1Display.NoiseGeneratorSeed = 1
transform1Display.CompositeStrategy = 'AUTO'
transform1Display.UseLICForLOD = 0
transform1Display.WriteLog = ''

# init the 'Transform2' selected for 'TextureTransform'
transform1Display.TextureTransform.Translate = [0.0, 0.0, 0.0]
transform1Display.TextureTransform.Rotate = [0.0, 0.0, 0.0]
transform1Display.TextureTransform.Scale = [1.0, 1.0, 1.0]

# init the 'Piecewise Function' selected for 'OSPRayScaleFunction'
transform1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
transform1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
transform1Display.GlyphType.TipResolution = 6
transform1Display.GlyphType.TipRadius = 0.1
transform1Display.GlyphType.TipLength = 0.35
transform1Display.GlyphType.ShaftResolution = 6
transform1Display.GlyphType.ShaftRadius = 0.03
transform1Display.GlyphType.Invert = 0

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
transform1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
transform1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
transform1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
transform1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'Grid Axes Representation' selected for 'DataAxesGrid'
transform1Display.DataAxesGrid.XTitle = 'X Axis'
transform1Display.DataAxesGrid.YTitle = 'Y Axis'
transform1Display.DataAxesGrid.ZTitle = 'Z Axis'
transform1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
transform1Display.DataAxesGrid.XTitleFontFile = ''
transform1Display.DataAxesGrid.XTitleBold = 0
transform1Display.DataAxesGrid.XTitleItalic = 0
transform1Display.DataAxesGrid.XTitleFontSize = 12
transform1Display.DataAxesGrid.XTitleShadow = 0
transform1Display.DataAxesGrid.XTitleOpacity = 1.0
transform1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
transform1Display.DataAxesGrid.YTitleFontFile = ''
transform1Display.DataAxesGrid.YTitleBold = 0
transform1Display.DataAxesGrid.YTitleItalic = 0
transform1Display.DataAxesGrid.YTitleFontSize = 12
transform1Display.DataAxesGrid.YTitleShadow = 0
transform1Display.DataAxesGrid.YTitleOpacity = 1.0
transform1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
transform1Display.DataAxesGrid.ZTitleFontFile = ''
transform1Display.DataAxesGrid.ZTitleBold = 0
transform1Display.DataAxesGrid.ZTitleItalic = 0
transform1Display.DataAxesGrid.ZTitleFontSize = 12
transform1Display.DataAxesGrid.ZTitleShadow = 0
transform1Display.DataAxesGrid.ZTitleOpacity = 1.0
transform1Display.DataAxesGrid.FacesToRender = 63
transform1Display.DataAxesGrid.CullBackface = 0
transform1Display.DataAxesGrid.CullFrontface = 1
transform1Display.DataAxesGrid.ShowGrid = 0
transform1Display.DataAxesGrid.ShowEdges = 1
transform1Display.DataAxesGrid.ShowTicks = 1
transform1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
transform1Display.DataAxesGrid.AxesToLabel = 63
transform1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
transform1Display.DataAxesGrid.XLabelFontFile = ''
transform1Display.DataAxesGrid.XLabelBold = 0
transform1Display.DataAxesGrid.XLabelItalic = 0
transform1Display.DataAxesGrid.XLabelFontSize = 12
transform1Display.DataAxesGrid.XLabelShadow = 0
transform1Display.DataAxesGrid.XLabelOpacity = 1.0
transform1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
transform1Display.DataAxesGrid.YLabelFontFile = ''
transform1Display.DataAxesGrid.YLabelBold = 0
transform1Display.DataAxesGrid.YLabelItalic = 0
transform1Display.DataAxesGrid.YLabelFontSize = 12
transform1Display.DataAxesGrid.YLabelShadow = 0
transform1Display.DataAxesGrid.YLabelOpacity = 1.0
transform1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
transform1Display.DataAxesGrid.ZLabelFontFile = ''
transform1Display.DataAxesGrid.ZLabelBold = 0
transform1Display.DataAxesGrid.ZLabelItalic = 0
transform1Display.DataAxesGrid.ZLabelFontSize = 12
transform1Display.DataAxesGrid.ZLabelShadow = 0
transform1Display.DataAxesGrid.ZLabelOpacity = 1.0
transform1Display.DataAxesGrid.XAxisNotation = 'Mixed'
transform1Display.DataAxesGrid.XAxisPrecision = 2
transform1Display.DataAxesGrid.XAxisUseCustomLabels = 0
transform1Display.DataAxesGrid.XAxisLabels = []
transform1Display.DataAxesGrid.YAxisNotation = 'Mixed'
transform1Display.DataAxesGrid.YAxisPrecision = 2
transform1Display.DataAxesGrid.YAxisUseCustomLabels = 0
transform1Display.DataAxesGrid.YAxisLabels = []
transform1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
transform1Display.DataAxesGrid.ZAxisPrecision = 2
transform1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
transform1Display.DataAxesGrid.ZAxisLabels = []
transform1Display.DataAxesGrid.UseCustomBounds = 0
transform1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
transform1Display.PolarAxes.Visibility = 0
transform1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
transform1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
transform1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
transform1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
transform1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
transform1Display.PolarAxes.EnableCustomRange = 0
transform1Display.PolarAxes.CustomRange = [0.0, 1.0]
transform1Display.PolarAxes.AutoPole = 1
transform1Display.PolarAxes.PolarAxisVisibility = 1
transform1Display.PolarAxes.RadialAxesVisibility = 1
transform1Display.PolarAxes.DrawRadialGridlines = 1
transform1Display.PolarAxes.PolarArcsVisibility = 1
transform1Display.PolarAxes.DrawPolarArcsGridlines = 1
transform1Display.PolarAxes.NumberOfRadialAxes = 0
transform1Display.PolarAxes.DeltaAngleRadialAxes = 45.0
transform1Display.PolarAxes.NumberOfPolarAxes = 5
transform1Display.PolarAxes.DeltaRangePolarAxes = 0.0
transform1Display.PolarAxes.CustomMinRadius = 1
transform1Display.PolarAxes.MinimumRadius = 0.0
transform1Display.PolarAxes.CustomMaxRadius = 0
transform1Display.PolarAxes.MaximumRadius = 1.0
transform1Display.PolarAxes.CustomAngles = 1
transform1Display.PolarAxes.MinimumAngle = 0.0
transform1Display.PolarAxes.MaximumAngle = 90.0
transform1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
transform1Display.PolarAxes.PolarArcResolutionPerDegree = 0.2
transform1Display.PolarAxes.Ratio = 1.0
transform1Display.PolarAxes.EnableOverallColor = 1
transform1Display.PolarAxes.OverallColor = [1.0, 1.0, 1.0]
transform1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
transform1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
transform1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
transform1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
transform1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
transform1Display.PolarAxes.PolarAxisTitleVisibility = 1
transform1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
transform1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
transform1Display.PolarAxes.PolarTitleOffset = [20.0, 20.0]
transform1Display.PolarAxes.PolarLabelVisibility = 1
transform1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
transform1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
transform1Display.PolarAxes.PolarLabelOffset = 10.0
transform1Display.PolarAxes.PolarExponentOffset = 5.0
transform1Display.PolarAxes.RadialTitleVisibility = 1
transform1Display.PolarAxes.RadialTitleFormat = '%-#3.1f'
transform1Display.PolarAxes.RadialTitleLocation = 'Bottom'
transform1Display.PolarAxes.RadialTitleOffset = [20.0, 0.0]
transform1Display.PolarAxes.RadialUnitsVisibility = 1
transform1Display.PolarAxes.ScreenSize = 10.0
transform1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
transform1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
transform1Display.PolarAxes.PolarAxisTitleFontFile = ''
transform1Display.PolarAxes.PolarAxisTitleBold = 0
transform1Display.PolarAxes.PolarAxisTitleItalic = 0
transform1Display.PolarAxes.PolarAxisTitleShadow = 0
transform1Display.PolarAxes.PolarAxisTitleFontSize = 12
transform1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
transform1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
transform1Display.PolarAxes.PolarAxisLabelFontFile = ''
transform1Display.PolarAxes.PolarAxisLabelBold = 0
transform1Display.PolarAxes.PolarAxisLabelItalic = 0
transform1Display.PolarAxes.PolarAxisLabelShadow = 0
transform1Display.PolarAxes.PolarAxisLabelFontSize = 12
transform1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
transform1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
transform1Display.PolarAxes.LastRadialAxisTextFontFile = ''
transform1Display.PolarAxes.LastRadialAxisTextBold = 0
transform1Display.PolarAxes.LastRadialAxisTextItalic = 0
transform1Display.PolarAxes.LastRadialAxisTextShadow = 0
transform1Display.PolarAxes.LastRadialAxisTextFontSize = 12
transform1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
transform1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
transform1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
transform1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
transform1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
transform1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
transform1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
transform1Display.PolarAxes.EnableDistanceLOD = 1
transform1Display.PolarAxes.DistanceLODThreshold = 0.7
transform1Display.PolarAxes.EnableViewAngleLOD = 1
transform1Display.PolarAxes.ViewAngleLODThreshold = 0.7
transform1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
transform1Display.PolarAxes.PolarTicksVisibility = 1
transform1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
transform1Display.PolarAxes.TickLocation = 'Both'
transform1Display.PolarAxes.AxisTickVisibility = 1
transform1Display.PolarAxes.AxisMinorTickVisibility = 0
transform1Display.PolarAxes.AxisTickMatchesPolarAxes = 1
transform1Display.PolarAxes.DeltaRangeMajor = 1.0
transform1Display.PolarAxes.DeltaRangeMinor = 0.5
transform1Display.PolarAxes.ArcTickVisibility = 1
transform1Display.PolarAxes.ArcMinorTickVisibility = 0
transform1Display.PolarAxes.ArcTickMatchesRadialAxes = 1
transform1Display.PolarAxes.DeltaAngleMajor = 10.0
transform1Display.PolarAxes.DeltaAngleMinor = 5.0
transform1Display.PolarAxes.TickRatioRadiusSize = 0.02
transform1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
transform1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
transform1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
transform1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
transform1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
transform1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
transform1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
transform1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
transform1Display.PolarAxes.ArcMajorTickSize = 0.0
transform1Display.PolarAxes.ArcTickRatioSize = 0.3
transform1Display.PolarAxes.ArcMajorTickThickness = 1.0
transform1Display.PolarAxes.ArcTickRatioThickness = 0.5
transform1Display.PolarAxes.Use2DMode = 0
transform1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(extractEdges1, renderView1)


# Subrat: Set the color bar to left
UpdatePipeline(proxy=foamfoam)
bounds_info = foamfoam.GetDataInformation().GetBounds()
# xmin_d, xmax_d = bounds_info[0], bounds_info[1]
# ymin_d, ymax_d = bounds_info[2], bounds_info[3]

xmin_d, xmax_d = x_min_site_boundary, x_max_site_boundary
ymin_d, ymax_d = y_min_site_boundary, y_max_site_boundary


zmin_d, zmax_d = bounds_info[4], bounds_info[5]
xcen_d = 0.5 * (xmin_d + xmax_d)
ycen_d = 0.5 * (ymin_d + ymax_d)
zcen_d = 0.5 * (zmin_d + zmax_d)
span_x = xmax_d - xmin_d
span_y = ymax_d - ymin_d


uLUTColorBar = GetScalarBar(uLUT, renderView1)
uLUTColorBar.Title = 'Wind Speed (m/s)'
uLUTColorBar.ComponentTitle = ''

# norm_x = (span_x - xmin_d) / span_x if span_x != 0 else 0.9
# norm_y = (span_y - ymin_d) / span_y if span_y != 0 else 0.1
# # clamp to [0,1]
# norm_x = max(0.0, min(1.0, norm_x))
# norm_y = max(0.0, min(1.0, norm_y))
# uLUTColorBar.WindowLocation = 'Any Location'
# uLUTColorBar.Position = [0.01, 0.05]
# # uLUTColorBar.Position = [0.1, 0.01]

# print(norm_y)
# uLUTColorBar.ScalarBarLength = norm_y * 0.9
# Colorbar spans exactly the data Y extent (= the contour frame).
# Uses data bounds directly — no camera assumptions needed.
bar_bottom = 0.0
bar_length = 1.0
uLUTColorBar.WindowLocation = 'Any Location'
uLUTColorBar.Position = [0.85, bar_bottom+bar_length*0.1]
uLUTColorBar.ScalarBarLength = bar_length*0.8

uLUTColorBar.LabelFormat = '%.2f'
uLUTColorBar.RangeLabelFormat = '%.2f'

uLUTColorBar.DrawBackground = 1
uLUTColorBar.BackgroundColor = [1.0, 1.0, 1.0, 0.9]

print(bar_length)



# update the view to ensure updated data information
renderView1.Update()

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# # get layout
# layout1 = GetLayout()

#END: SUBRAT


# get layout
layout1 = GetLayout()

# Derive pixel dimensions from domain aspect ratio so image matches the frame.
# right_pad adds space on the right for the colorbar (same value used below).
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
layout1.SetSize(image_width_px, image_height_px)

# current camera placement for renderView1 — data-driven, no hardcoded values
renderView1.CameraFocalPoint = [xcen_d, ycen_d, zcen_d]
renderView1.CameraParallelScale = span_y / 2.0



# save animation
# --- Universal camera logic for top-down view and full x-y domain coverage ---
import math




# ─── Camera: top-down, full domain ────────────────────────────────────────────

image_width  = image_width_px
image_height = image_height_px

# domain_aspect = span_x / span_y if span_y != 0 else 1.0
# image_aspect  = image_width / image_height

# renderView1.CameraParallelProjection = 0
# fov_rad = math.radians(30.0)

# if domain_aspect > image_aspect:
#     fit_span = span_x
#     fit_dim  = image_width
# else:
#     fit_span = span_y
#     fit_dim  = image_height

# z_required = (fit_span / (2.0 * math.tan(fov_rad / 2.0))) * (image_height / fit_dim)
# z_required *= 1.05

# renderView1.CameraPosition  = [xcen_d, ycen_d, zcen_d + z_required]
# renderView1.CameraFocalPoint = [xcen_d, ycen_d, zcen_d]
# renderView1.CameraViewUp     = [0, 1, 0]

right_pad = _right_pad  # defined above alongside image dimensions

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



# Ensure the pipeline is updated before getting bounds
renderView1.Update()

# # Get bounds of the full domain (foamfoam) after update
# bounds = foamfoam.GetDataInformation().GetBounds()
# # bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
# xmin, xmax, ymin, ymax, zmin, zmax = bounds
# xcenter = 0.5 * (xmin + xmax)
# ycenter = 0.5 * (ymin + ymax)
# zcenter = 0.5 * (zmin + zmax)

# # Camera position: much higher above center, looking down

# # Camera position: above center, looking down
# z_offset = max(2.0 * (zmax - zmin), 1.0)
# renderView1.CameraPosition = [xcenter, ycenter, zmax + z_offset]
# renderView1.CameraFocalPoint = [xcenter, ycenter, zcenter]
# renderView1.CameraViewUp = [0, 1, 0]

# # --- Fit largest domain span to smallest image size ---
# # Get image resolution from SaveAnimation call below
# image_width = 800
# image_height = 800
# try:
#   image_width = ImageResolution[0]
#   image_height = ImageResolution[1]
# except Exception:
#   pass

# span_x = xmax - xmin
# span_y = ymax - ymin
# domain_aspect = span_x / span_y if span_y != 0 else 1.0
# image_aspect = image_width / image_height if image_height != 0 else 1.0

# # Switch to perspective projection
# renderView1.CameraParallelProjection = 0

# # Use a typical vertical field of view (in degrees)
# fov_deg = 30.0
# fov_rad = math.radians(fov_deg)

# # Fit the largest domain span to the smallest image size
# if domain_aspect > image_aspect:
#   # Domain is wider than image, fit x to width
#   fit_span = span_x
#   fit_dim = image_width
# else:
#   # Domain is taller than image, fit y to height
#   fit_span = span_y
#   fit_dim = image_height

# # Calculate required camera distance (z) so fit_span fits in FOV
# # For vertical FOV: fit_span = 2 * z * tan(fov/2) * (fit_dim / image_height)
# # Solve for z:
# z_required = (fit_span / (2.0 * math.tan(fov_rad / 2.0))) * (image_height / fit_dim)
# # Add a small margin
# z_required *= 1.05

# # Set camera position
# renderView1.CameraPosition = [xcenter, ycenter, zcenter + z_required]
# renderView1.CameraFocalPoint = [xcenter, ycenter, zcenter]
# renderView1.CameraViewUp = [0, 1, 0]

# Save animation as before
SaveAnimation(filename='temporal.jpeg', viewOrLayout=renderView1, location=16, ImageResolution=[image_width_px, image_height_px],
  FontScaling='Scale fonts proportionally',
  OverrideColorPalette='WhiteBackground',
  StereoMode='No change',
  TransparentBackground=0,
  FrameRate=1,
  FrameStride=1,
  FrameWindow=[0, 149], 
  # JPEG options
  Quality=100,
  Progressive=1,
  SuffixFormat='.%04d')
# Adjust camera

# current camera placement for renderView1
renderView1.CameraPosition = [1.8850021362304688, 1.7292518615722656, 359.0917608932871]
renderView1.CameraFocalPoint = [1.8850021362304688, 1.7292518615722656, 24.780000686645508]
renderView1.CameraParallelScale = 153.2865309386078

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(image_width_px, image_height_px)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [1.8850021362304688, 1.7292518615722656, 359.0917608932871]
renderView1.CameraFocalPoint = [1.8850021362304688, 1.7292518615722656, 24.780000686645508]
renderView1.CameraParallelScale = 153.2865309386078


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------
