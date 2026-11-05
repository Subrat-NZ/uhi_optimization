#!/bin/bash

file="translateParameters"

# Read values from file
xMin=$(grep xMin $file | awk '{print $2}' | tr -d ';')
xMax=$(grep xMax $file | awk '{print $2}' | tr -d ';')

yMin=$(grep yMin $file | awk '{print $2}' | tr -d ';')
yMax=$(grep yMax $file | awk '{print $2}' | tr -d ';')

zMin=$(grep zMin $file | awk '{print $2}' | tr -d ';')
zMax=$(grep zMax $file | awk '{print $2}' | tr -d ';')

# Compute center
xCenter=$(awk "BEGIN {print ($xMin + $xMax)/2}")
yCenter=$(awk "BEGIN {print ($yMin + $yMax)/2}")
zCenter=$(awk "BEGIN {print ($zMin + $zMax)/2}")

# Translation required to move to origin
tx=$(awk "BEGIN {print -1*$xCenter}")
ty=$(awk "BEGIN {print -1*$yCenter}")
tz=$(awk "BEGIN {print 0.0*$zMin}")


surfaceTransformPoints -translate "($tx $ty $tz)" buildings.obj building_shifted.obj
surfaceTransformPoints -translate "($tx $ty $tz)" site_boundary.obj site_boundary_shifted.obj
