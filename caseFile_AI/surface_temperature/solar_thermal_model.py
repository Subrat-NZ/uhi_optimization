"""
Analytical solar-exposure + convective/radiative surface-temperature model
for buildings exported from the scene JSON (SNU-SPH / urban-microclimate
style export: buildings[].faces[] with vertices, normal, surfaceType,
materialProperties.thermalProperties; optionally trees[] for canopy shading).

No CFD, and no full (multi-bounce) ray tracing -- everything below is
geometry plus visibility rays toward the sun at each weather time:

  1. SOLAR GEOMETRY  (Cooper's declination + standard hour-angle equations)
     -> sun altitude & azimuth for any date / time / lat-lon.

  2. CLEAR-SKY IRRADIANCE  (Hottel, 1976 atmospheric-transmittance model)
     -> direct-normal (I_DN) and diffuse-horizontal (I_dH) irradiance from
        the solar altitude alone -- used only when you DON'T have a real
        weather-data export (see `run_analysis`). If you DO have one (a
        `weatherData/MMDD/` folder with Tambient, windSpeed, IDN, Idif,
        sunPosVector, ...), use `run_analysis_from_folder` instead: it
        reads the real sun-position vector and real DNI/diffuse straight
        from those files and skips this model entirely -- more accurate,
        and removes any ambiguity about axis conventions since the sun
        vector is already given in the same (East, North, Up) axes as the
        building JSON's face normals.

  3. PER-FACE SUNLIT TEST  -- cos(theta) = dot(face_normal, sun_unit_vector).
     A face receives direct beam radiation only when cos(theta) > 0 AND the
     sun is above the horizon (self-orientation), AND (new) a visibility ray
     from the face toward the sun is not blocked by another building or tree
     in the scene (mutual shading -- see Section 5, "Shadow occlusion").

  4. CONVECTION -- classic linear wind correlation h_c = a + b*V (McAdams-
     type, W/m^2K), applied per face using your wind-speed time series.

  5. SURFACE TEMPERATURE: each exposed building or requested ground triangle
     is a 0.23 m slab with five nodes through its thickness, heat capacity
     rho*cp*L per unit area, and
     conductance k/dx between adjacent nodes. A matrix exponential advances
     conduction/storage exactly for the spatially discretized system, holding
     weather and ambient-linearized longwave coefficients constant between
     records. A building inner face exchanges heat with fixed 24 C indoor air;
     a ground back face exchanges heat with fixed deep soil. These settings,
     materials, and initial temperatures are configurable.
     T_solair_C remains an equivalent driving temperature for the bulk model.

  6. WHOLE-BUILDING TEMPERATURE -- single-node RC (lumped capacitance,
     C = rho*V*cp) network per building, driven by that building's own
     faces' sol-air temperatures. Because C and the effective conductance
     h_eff are taken as constant within each hour, the ODE
     C dT/dt = h_eff*(T_eq - T) has the exact analytical solution

         T(t+dt) = T_eq + (T(t) - T_eq) * exp(-dt / tau),   tau = C/h_eff

     so the whole march-forward is a closed-form exponential step, not a
     numerical ODE integration. Each building in the scene gets its own
     node (its own C, its own faces) -- they are only coupled through the
     shadows they cast on each other, not thermally.

Assumptions you should sanity-check against your own scene / site
(flagged again inline where they are used):
  - Scene axes are treated as ENU: +x = East, +y = North, +z = Up. If your
    tool defines "north" differently, set `north_offset_deg` below.
  - Longwave emissivity is not in the JSON -> defaulted to 0.9 (typical
    concrete/masonry). Change `DEFAULT_EMISSIVITY` if you know better.
  - Ground/surroundings albedo for the reflected-radiation term is not in
     the JSON -> defaulted to 0.2. Change `DEFAULT_GROUND_ALBEDO`.
  - The optional OBJ ground calculation uses the case soil defaults k=1.5
    W/m/K, rho=1150 kg/m3 and cp=650 J/kg/K. Its active depth is 0.23 m and
    its default lower conductance k/L connects it to fixed 24 C deep soil.
  - The floor face (surfaceType == "floor") is excluded from solar +
    outdoor convection (it's in contact with the ground, not outdoor air).
    Toggle `INCLUDE_FLOOR` if you want to add a simple ground-conduction
    term instead. A "ceiling" face (surfaceType == "ceiling") is *always*
    excluded the same way, with no toggle -- in multi-floor exports (one
    JSON "building" per story, stacked in z) this tags the boundary between
    two stacked stories, which never touches outdoor air regardless of
    settings. Only "wall" and "roof" faces are treated as always outdoor-
    exposed.
  - Two of the face normals in the source JSON point the "wrong" way for
    what they geometrically are (roof normal given as (0,0,-1), i.e.
    inward, floor as (0,0,1), also inward). This script auto-corrects
    normal sign using `surfaceType` so the physics is right regardless of
    what the exporter wrote. Also, wall face "area" was exported as 0 in
    this file -- areas are recomputed from the vertices, never trusted
    from the JSON.
  - The OBJ path retains individual mesh triangles, with three visibility
    samples per triangle by default, accelerated by a triangle BVH. The JSON
    path still tests one centroid per polygon. Only the direct
    beam is blockable; the diffuse-sky and ground-reflected components
    still use the unobstructed sky-view-factor formulas (a standard
    simplification -- properly reducing diffuse view factors for nearby
    obstructions would need a more expensive hemispherical integration).
  - In the JSON path, tree canopies are approximated as a bounding sphere for shadow
    purposes (a common simplification in urban solar-access studies), not
    actual leaf/branch geometry. The JSON field names used to build that
    sphere are guessed from common conventions (see `build_tree_occluder`)
    since no example tree entry was available when this was written --
    check `_TREE_*_KEYS` against your real tree schema and adjust if needed.
    The OBJ path traces the actual tree triangles, treating them as opaque.

MULTI-SURFACE OBJ MODE (buildings, ground, pathway, lawn, pond, trees)
  Every component is optional; whatever has valid triangles is solved and the
  rest is skipped with a log line. Geometry comes from the OBJ files, material
  properties from the scene JSON when present (else the documented defaults;
  the log lists the source of every property). Ground-like classes use the
  same transient slab solver as the walls. Ponds, lawns and tree canopies
  additionally lose latent heat by evaporation/transpiration (Lewis analogy,
  linearised about ambient so the exact matrix-exponential update still
  applies), and tree canopies exchange heat with ambient air on their back
  side. Ground cells under building roofs or under pond/pathway/lawn cells
  are removed so nothing is solved twice. Neighbouring ground/pathway/lawn/pond
  cells also exchange heat sideways (LateralCoupler; --no_lateral turns it off).
  Several weather days can be given:
  each becomes its own set of outputs, run in parallel with --workers.
"""

import concurrent.futures
import glob
import json
import math
import multiprocessing
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Tunable defaults (not present in the JSON -- see docstring above)
# --------------------------------------------------------------------------
DEFAULT_EMISSIVITY = 0.90
DEFAULT_GROUND_ALBEDO = 0.20
STEFAN_BOLTZMANN = 5.670374e-8          # W/m^2/K^4
SOLAR_CONSTANT = 1367.0                 # W/m^2
INCLUDE_FLOOR = False                   # floor treated as adiabatic (ground contact)


# ==========================================================================
# 1. Geometry: load scene, fix normals, compute real areas
# ==========================================================================

@dataclass
class Face:
    building_id: str
    face_id: str
    surface_type: str
    normal: np.ndarray          # outward unit normal, corrected
    area: float                 # m^2, computed from vertices (JSON area ignored)
    tilt_deg: float             # 0 = horizontal facing up, 90 = vertical, 180 = facing down
    center: np.ndarray
    vertices: np.ndarray        # (N,3) planar polygon loop, as given (for geometry export)
    thermal_conductivity: float
    density: float
    specific_heat: float
    albedo: float
    emissivity: float = DEFAULT_EMISSIVITY
    parent_face_id: str = ""    # logical wall/roof; OBJ physics stays per triangle


def _polygon_area_3d(verts: np.ndarray) -> float:
    """Area of a planar polygon in 3D via Newell's method (cross-product sum)."""
    n = len(verts)
    normal_sum = np.zeros(3)
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        normal_sum += np.cross(v1, v2)
    return 0.5 * np.linalg.norm(normal_sum)


def load_scene(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_faces(scene: dict) -> list:
    """Flatten every building's faces into a list of corrected Face objects."""
    faces = []
    for b in scene.get("buildings", []):
        bid = b.get("id", "building")
        for f in b.get("faces", []):
            verts = np.array([[v["x"], v["y"], v["z"]] for v in f["vertices"]])
            area = _polygon_area_3d(verts)          # never trust JSON "area"
            n = np.array([f["normal"]["x"], f["normal"]["y"], f["normal"]["z"]], dtype=float)
            n = n / (np.linalg.norm(n) + 1e-12)

            stype = f.get("surfaceType", "wall")
            # --- fix inward-pointing roof/floor normals (see module docstring) ---
            if stype == "roof" and n[2] < 0:
                n = -n
            if stype == "floor" and n[2] > 0:
                n = -n

            tilt_deg = math.degrees(math.acos(np.clip(n[2], -1.0, 1.0)))
            c = f.get("center", {"x": 0, "y": 0, "z": 0})
            center = np.array([c["x"], c["y"], c["z"]])

            def _prop(name, default):
                value = _lookup_property(f, name)
                return default if value is None else value
            faces.append(Face(
                building_id=bid,
                face_id=f["id"],
                surface_type=stype,
                normal=n,
                area=area,
                tilt_deg=tilt_deg,
                center=center,
                vertices=verts,
                thermal_conductivity=_prop("thermal_conductivity", DEFAULT_MATERIAL["thermal_conductivity"]),
                density=_prop("density", DEFAULT_MATERIAL["density"]),
                specific_heat=_prop("specific_heat", DEFAULT_MATERIAL["specific_heat"]),
                albedo=_prop("albedo", DEFAULT_ALBEDO_ROOF if stype == "roof" else DEFAULT_ALBEDO_WALL),
                emissivity=_prop("emissivity", DEFAULT_EMISSIVITY),
            ))
    return faces


def building_volume(scene: dict, building_id: Optional[str] = None) -> float:
    for b in scene.get("buildings", []):
        if building_id is None or b.get("id") == building_id:
            geo = b.get("geometry", {})
            if "volume" in geo:
                return geo["volume"]
            return geo.get("footprintArea", 0) * b.get("height", 0)
    return 0.0


def site_location(scene: dict) -> tuple:
    """Returns (lat_deg, lon_deg) -- prefers shadowSettings, falls back to site centroid."""
    ss = scene.get("settings", {}).get("shadowSettings")
    if ss and "latitude" in ss:
        return ss["latitude"], ss["longitude"]
    coords = scene["site"]["properties"]["originalCoordinates"]
    lat = float(np.mean([c["lat"] for c in coords]))
    lon = float(np.mean([c["lng"] for c in coords]))
    return lat, lon


# ==========================================================================
# 2. Solar geometry
# ==========================================================================

def solar_position(lat_deg: float, lon_deg: float, day_of_year: int,
                    local_std_hour: float, utc_offset_hr: float = 0.0) -> tuple:
    """
    Standard solar-position equations (Cooper's declination + hour angle).
    Returns (altitude_deg, azimuth_deg); azimuth measured clockwise from
    true North (0=N, 90=E, 180=S, 270=W). altitude < 0 => sun below horizon.
    """
    lat = math.radians(lat_deg)
    n = day_of_year

    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0)))

    # Equation of time (minutes), Duffie & Beckman form
    B = math.radians(360.0 * (n - 81) / 364.0)
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    std_meridian = 15.0 * utc_offset_hr
    time_corr_min = 4.0 * (lon_deg - std_meridian) + eot
    solar_time = local_std_hour + time_corr_min / 60.0

    hour_angle = math.radians(15.0 * (solar_time - 12.0))

    sin_alt = (math.sin(lat) * math.sin(decl) +
               math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
    altitude = math.asin(np.clip(sin_alt, -1.0, 1.0))

    # Azimuth, measured clockwise from true North (standard NOAA-style form):
    #   cos(A) = (sin(decl) - sin(lat)*sin(alt)) / (cos(lat)*cos(alt))
    # acos() only returns 0-180 deg, so mirror into 180-360 for the
    # afternoon (hour_angle > 0) to get the correct NE-at-sunrise /
    # NW-at-sunset quadrant.
    cos_alt = max(math.cos(altitude), 1e-9)
    cos_A = (math.sin(decl) - math.sin(lat) * math.sin(altitude)) / (math.cos(lat) * cos_alt)
    az_raw = math.degrees(math.acos(np.clip(cos_A, -1.0, 1.0)))
    azimuth = az_raw if hour_angle <= 0 else 360.0 - az_raw

    return math.degrees(altitude), azimuth


def sun_unit_vector(altitude_deg: float, azimuth_deg: float,
                     north_offset_deg: float = 0.0) -> np.ndarray:
    """
    Sun direction as a unit vector in scene coordinates, assuming
    +x = East, +y = North, +z = Up (rotate with north_offset_deg if your
    scene's "north" is not aligned with +y).
    """
    alt = math.radians(altitude_deg)
    az = math.radians(azimuth_deg - north_offset_deg)
    east = math.cos(alt) * math.sin(az)
    north = math.cos(alt) * math.cos(az)
    up = math.sin(alt)
    return np.array([east, north, up])


# ==========================================================================
# 3. Clear-sky irradiance (Hottel, 1976)
# ==========================================================================

def hottel_clear_sky(altitude_deg: float, site_altitude_km: float = 0.0,
                      r0: float = 1.0, r1: float = 1.0, rk: float = 1.0) -> tuple:
    """
    Hottel's (1976) atmospheric-transmittance clear-sky model.
    r0, r1, rk are the climate-type correction factors from Hottel's paper
    (defaults =1.0, i.e. the reference "mid-latitude summer" atmosphere --
    nudge by a few % for your own climate/visibility if you know it).
    Returns (I_DN, I_dH) in W/m^2 -- direct-normal and diffuse-horizontal.
    Returns (0, 0) when the sun is below the horizon.
    """
    if altitude_deg <= 0:
        return 0.0, 0.0
    A = site_altitude_km
    sin_b = math.sin(math.radians(altitude_deg))

    a0 = r0 * (0.4237 - 0.00821 * (6.0 - A) ** 2)
    a1 = r1 * (0.5055 + 0.00595 * (6.5 - A) ** 2)
    k = rk * (0.2711 + 0.01858 * (2.5 - A) ** 2)

    tau_b = a0 + a1 * math.exp(-k / sin_b)
    tau_d = 0.2710 - 0.2939 * tau_b   # Liu-Jordan diffuse-transmittance approx

    i_dn = SOLAR_CONSTANT * tau_b
    i_dh = SOLAR_CONSTANT * tau_d * sin_b
    return max(i_dn, 0.0), max(i_dh, 0.0)


def sky_temperature(t_amb_k: float) -> float:
    """Swinbank (1963) clear-sky effective sky temperature, in Kelvin."""
    return 0.0552 * t_amb_k ** 1.5


# ==========================================================================
# 4. Per-face irradiance & sol-air temperature
# ==========================================================================

def face_irradiance(face: Face, sun_vec: np.ndarray, altitude_deg: float,
                     i_dn: float, i_dh: float,
                     ground_albedo: float = DEFAULT_GROUND_ALBEDO,
                     external_shadow: bool = False) -> dict:
    """
    Direct + diffuse (isotropic sky) + ground-reflected irradiance on one face.
    `external_shadow=True` means a ray/occlusion test (Section 5) found this
    face blocked by another building or tree this hour -- the direct beam is
    then zeroed regardless of the face's own orientation toward the sun.
    """
    cos_theta = float(np.dot(face.normal, sun_vec))
    self_lit = (altitude_deg > 0) and (cos_theta > 0)
    sunlit = self_lit and not external_shadow

    i_direct = i_dn * cos_theta if sunlit else 0.0

    tilt = math.radians(face.tilt_deg)
    view_sky = (1 + math.cos(tilt)) / 2.0     # 1 for roof, 0.5 for wall, 0 for floor
    view_ground = 1.0 - view_sky
    i_diffuse = i_dh * view_sky

    i_gh = i_dn * math.sin(math.radians(max(altitude_deg, 0.0))) + i_dh
    i_reflected = ground_albedo * i_gh * view_ground

    return dict(sunlit=sunlit, self_lit=self_lit, shadowed=external_shadow,
                cos_theta=cos_theta, view_sky=view_sky,
                i_direct=i_direct, i_diffuse=i_diffuse, i_reflected=i_reflected,
                i_total=i_direct + i_diffuse + i_reflected)


def convective_h(wind_speed_ms: float, a: float = 5.7, b: float = 3.8) -> float:
    """McAdams-type linear wind correlation (W/m^2K)."""
    return a + b * max(wind_speed_ms, 0.0)


def sol_air_temperature(t_amb_c: float, i_total: float, h_c: float,
                         albedo: float, view_sky: float,
                         emissivity: float = DEFAULT_EMISSIVITY) -> float:
    """
    ASHRAE sol-air temperature: the fictitious outdoor air temperature that
    would give the same solar + convective + long-wave heat gain as the
    real conditions, so a face just "sees" T_sol_air via plain convection.
    """
    alpha = 1.0 - albedo
    t_amb_k = t_amb_c + 273.15
    t_sky_k = sky_temperature(t_amb_k)
    # net long-wave LOSS to the sky, scaled by how much sky this face can see
    q_lw_loss = view_sky * emissivity * STEFAN_BOLTZMANN * (t_amb_k ** 4 - t_sky_k ** 4)
    return t_amb_c + (alpha * i_total - q_lw_loss) / h_c


def surface_temperature(t_amb_c, i_total, h_c, albedo, view_sky,
                        emissivity=DEFAULT_EMISSIVITY):
    """Local steady exterior temperature (C), with no wall conduction/storage.

    Solve alpha*I = h_c*(Ts-Ta) + eps*sigma*(Ts^4 - Fsky*Tsky^4
                                                - (1-Fsky)*Ta^4).
    Ground/surroundings are at ambient temperature; sky uses Swinbank.
    Supports scalar or broadcastable array inputs. Unlike sol-air temperature,
    outgoing longwave radiation is evaluated at the actual surface temperature.
    """
    ta = np.asarray(t_amb_c) + 273.15
    absorbed = (1.0 - np.asarray(albedo)) * np.asarray(i_total)
    eps_sigma = np.asarray(emissivity) * STEFAN_BOLTZMANN
    background = np.asarray(view_sky) * sky_temperature(ta)**4 + (1-np.asarray(view_sky))*ta**4
    ts = np.asarray(ta + absorbed / h_c)
    for _ in range(30):
        residual = h_c*(ts-ta) + eps_sigma*(ts**4-background) - absorbed
        step = residual / (h_c + 4*eps_sigma*ts**3)
        ts = ts - step
        if np.max(np.abs(step), initial=0.0) < 1e-8:
            break
    return ts - 273.15


@dataclass(frozen=True)
class WallSettings:
    """Uniform slab settings for exposed walls and roofs; SI units except C."""
    thickness_m: float = 0.23
    nodes: int = 5
    indoor_temperature_c: float = 24.0
    indoor_h_wm2k: float = 8.0

    def __post_init__(self):
        if not np.isfinite(self.thickness_m) or self.thickness_m <= 0:
            raise ValueError("wall thickness must be finite and positive")
        if not isinstance(self.nodes, int) or self.nodes < 2:
            raise ValueError("wall nodes must be an integer >= 2")
        if not np.isfinite(self.indoor_temperature_c) or self.indoor_temperature_c <= -273.15:
            raise ValueError("indoor temperature must be finite and above absolute zero")
        if not np.isfinite(self.indoor_h_wm2k) or self.indoor_h_wm2k < 0:
            raise ValueError("indoor heat-transfer coefficient must be finite and nonnegative")


@dataclass(frozen=True)
class GroundSettings:
    """Thermal properties and lower boundary for exposed ground triangles.

    The active soil layer uses the same nodal slab solver as a wall. Its lower
    face exchanges heat with fixed deep-soil temperature through
    ``deep_h_wm2k``. When that value is omitted, k/depth is used, representing
    conduction through one additional layer of the same thickness.
    """
    depth_m: float = 0.23
    nodes: int = 5
    deep_temperature_c: float = 24.0
    deep_h_wm2k: Optional[float] = None
    thermal_conductivity: float = 1.5
    density: float = 1150.0
    specific_heat: float = 650.0
    albedo: float = DEFAULT_GROUND_ALBEDO
    emissivity: float = 0.95
    wetness: float = 0.0                  # fraction of potential evaporation (0 dry .. 1 water)
    back_follows_ambient: bool = False    # back face exchanges with ambient air (tree canopy)

    def __post_init__(self):
        if not np.isfinite(self.depth_m) or self.depth_m <= 0:
            raise ValueError("ground depth must be finite and positive")
        if not np.isfinite(self.wetness) or not 0 <= self.wetness <= 1:
            raise ValueError("wetness must be finite and between 0 and 1")
        WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
                     self.effective_deep_h_wm2k)
        material = np.array([self.thermal_conductivity, self.density,
                             self.specific_heat], dtype=float)
        if np.any(~np.isfinite(material)) or np.any(material <= 0):
            raise ValueError("ground conductivity, density and heat capacity must be positive and finite")
        if not np.isfinite(self.albedo) or not 0 <= self.albedo <= 1:
            raise ValueError("ground albedo must be finite and between 0 and 1")
        if not np.isfinite(self.emissivity) or not 0 <= self.emissivity <= 1:
            raise ValueError("ground emissivity must be finite and between 0 and 1")

    @property
    def effective_deep_h_wm2k(self) -> float:
        return (self.thermal_conductivity / self.depth_m
                if self.deep_h_wm2k is None else self.deep_h_wm2k)

    def slab_settings(self) -> WallSettings:
        return WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
                            self.effective_deep_h_wm2k)


class SlabConduction:
    """Independent 1-D slabs, with an exact exponential update in time.

    Boundary nodes lie on the actual surfaces and have half control volumes.
    Interior nodes have full control volumes. Per-area nodal capacities sum
    to rho*cp*L; adjacent conductance is k/dx. Sideways coupling, when enabled,
    is applied between steps by LateralCoupler (this class itself is 1-D).
    Grouping equal materials lets all their triangles share a small matrix.
    """

    def __init__(self, conductivity, density, specific_heat, emissivity,
                 initial_c, settings=None):
        self.settings = settings or WallSettings()
        properties = np.column_stack(np.broadcast_arrays(
            conductivity, density, specific_heat, emissivity))
        if np.any(~np.isfinite(properties)) or np.any(properties[:, :3] <= 0):
            raise ValueError("wall conductivity, density and heat capacity must be positive and finite")
        self.dx = self.settings.thickness_m / (self.settings.nodes-1)
        widths = np.full(self.settings.nodes, self.dx)
        widths[[0, -1]] *= 0.5
        self.capacity = properties[:, 1, None]*properties[:, 2, None]*widths
        self.conductance = properties[:, 0] / self.dx
        self.temperatures = np.full((len(properties), self.settings.nodes), initial_c, dtype=float)
        if not np.isfinite(initial_c) or initial_c <= -273.15:
            raise ValueError("initial wall temperature must be finite and above absolute zero")
        self.initial_c = initial_c
        self.back_h = self.settings.indoor_h_wm2k
        self.back_temperature = self.settings.indoor_temperature_c
        _, inverse = np.unique(properties, axis=0, return_inverse=True)
        self.groups = [np.flatnonzero(inverse == i) for i in range(inverse.max()+1)]

    def set_back(self, h, temperature):
        """Replace the back-face coefficient / temperature (time-varying back
        boundary, e.g. ambient air behind a tree canopy). Used by advance()
        and diagnostics() until changed again."""
        self.back_h = float(h)
        self.back_temperature = float(temperature)

    def advance(self, dt_seconds, external_h, external_source):
        """C*Tdot = -K*T + b; exterior input is source - h*T[0].

        Input coefficients are held constant for dt. Transform with sqrt(C)
        to a symmetric matrix, then use its eigendecomposition to evaluate
        exp(A*dt) and the integrated forcing. No SciPy dependency or explicit
        time stepping; zero eigenvalues are handled for insulated slabs.
        """
        if not np.isfinite(dt_seconds) or dt_seconds < 0:
            raise ValueError("wall time step must be finite and nonnegative")
        if dt_seconds == 0:
            return
        n = self.settings.nodes
        external_h = np.broadcast_to(external_h, len(self.temperatures))
        external_source = np.broadcast_to(external_source, len(self.temperatures))
        for ids in self.groups:
            # Radiation coefficient depends on ambient and emissivity only,
            # so it is common to a material group within a weather interval.
            if not np.allclose(external_h[ids], external_h[ids[0]], rtol=0, atol=1e-12):
                raise ValueError("external h must be equal within each material group")
            c = self.capacity[ids[0]]
            root_c = np.sqrt(c)
            g = self.conductance[ids[0]]
            K = np.diag(np.full(n, 2*g)) + np.diag(np.full(n-1, -g), 1) + np.diag(np.full(n-1, -g), -1)
            K[0,0] = g + external_h[ids[0]]
            K[-1,-1] = g + self.back_h
            A = -K / root_c[:, None] / root_c[None, :]
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            exponent = np.exp(eigenvalues*dt_seconds)
            phi = np.full(n, dt_seconds, dtype=float)
            np.divide(np.expm1(eigenvalues*dt_seconds), eigenvalues,
                      out=phi, where=np.abs(eigenvalues) > 1e-15)
            b = np.zeros((len(ids), n))
            b[:,0] = external_source[ids]
            b[:,-1] = self.back_h*self.back_temperature
            modes = (self.temperatures[ids]*root_c) @ eigenvectors
            forcing = (b/root_c) @ eigenvectors
            self.temperatures[ids] = ((modes*exponent + forcing*phi) @ eigenvectors.T) / root_c

    def diagnostics(self, external_h, external_source):
        """Instantaneous conductive/storage fluxes at the current output time."""
        t = self.temperatures
        q_outer = external_source-external_h*t[:,0]
        q_inner = self.back_h*(t[:,-1]-self.back_temperature)
        q_conduction = self.conductance*(t[:,0]-t[:,1])
        t_mean = np.sum(self.capacity*t,axis=1)/self.capacity.sum(axis=1)
        capacity = self.capacity.sum(axis=1)
        return dict(T_inner_C=t[:,-1].copy(), T_back_surface_C=t[:,-1].copy(),
                    T_wall_mean_C=t_mean, T_slab_mean_C=t_mean,
                    q_conduction_in_Wm2=q_conduction, q_indoor_out_Wm2=q_inner,
                    q_back_out_Wm2=q_inner,
                    q_storage_Wm2=q_outer-q_inner,
                    q_outer_node_storage_Wm2=q_outer-q_conduction,
                    stored_heat_Jm2=np.sum(self.capacity*(t-self.initial_c),axis=1),
                    wall_capacity_Jm2K=capacity, slab_capacity_Jm2K=capacity)


# ==========================================================================
# 5. Shadow occlusion -- cheap two-phase ray tracing
#
#    Phase 1 (broad, O(1) per occluder): does the ray toward the sun even
#             pass through the occluder's axis-aligned bounding box?
#    Phase 2 (narrow, exact): only for occluders that survive phase 1 --
#             Moller-Trumbore ray-triangle tests against the occluder's
#             actual wall/roof triangles (buildings), or a closed-form
#             ray-sphere test (trees, canopy approximated as a sphere).
#
#    This is the standard bounding-volume-culling idea from ray tracing,
#    kept deliberately simple (no BVH/kd-tree): for scenes with up to a
#    few hundred buildings this keeps the great majority of rays from ever
#    touching a triangle, which is normally the expensive part.
# ==========================================================================

@dataclass
class Occluder:
    kind: str                                   # "building" or "tree"
    source_id: str
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    triangles: list = field(default_factory=list)     # [(p0,p1,p2), ...] for buildings
    sphere_center: Optional[np.ndarray] = None          # for trees
    sphere_radius: Optional[float] = None


def _triangulate_quad(v0, v1, v2, v3):
    return [(v0, v1, v2), (v0, v2, v3)]


def _fan_triangulate(verts):
    """Fan triangulation from the first vertex. Exact for convex polygons
    (true for simple rectangular footprints); an approximation for
    concave ones -- flag this if you have complex (e.g. OSM-derived,
    L-shaped) footprints and it matters for your results."""
    return [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]


def build_building_occluder(building_json: dict) -> Optional[Occluder]:
    """
    Triangulates one building's wall + roof faces for use as an occluder of
    OTHER buildings/faces. The floor is skipped -- sitting at ground level,
    it cannot shadow anything above it, so leaving it out is a free
    reduction in triangle count with no effect on the result.
    """
    tris, all_pts = [], []
    for f in building_json.get("faces", []):
        if f.get("surfaceType") == "floor":
            continue
        verts = [np.array([v["x"], v["y"], v["z"]]) for v in f["vertices"]]
        all_pts.extend(verts)
        if len(verts) == 4:
            tris.extend(_triangulate_quad(*verts))
        elif len(verts) == 3:
            tris.append(tuple(verts))
        elif len(verts) > 4:
            tris.extend(_fan_triangulate(verts))
    if not tris:
        return None
    pts = np.array(all_pts)
    return Occluder(kind="building", source_id=building_json.get("id", "building"),
                     aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tris)


# Field-name guesses for tree entries -- NOT verified against a real export
# (none was available when this was written). Check these against your
# actual tree JSON and adjust if the names differ.
_TREE_RADIUS_KEYS = ["crownRadius", "canopyRadius", "radius"]
_TREE_DIAMETER_KEYS = ["crownDiameter", "canopyDiameter", "diameter"]
_TREE_HEIGHT_KEYS = ["height", "totalHeight", "crownHeight"]
_TREE_TRUNK_KEYS = ["trunkHeight", "trunk_height"]


def resolve_tree_geometry(tree_json: dict) -> Optional[tuple]:
    """
    Shared field-lookup used both for the shadow-occluder sphere and for the
    fuller trunk+crown mesh in the ParaView export. Checks a nested
    "dimensions" sub-object first (more precise, seen in real exports) and
    falls back to top-level fields. Returns
    (pos [3], trunk_height, crown_height, crown_radius) or None if no
    recognisable position/size fields are found.
    """
    pos = None
    if "position" in tree_json:
        p = tree_json["position"]
        pos = np.array([p.get("x", 0), p.get("y", 0), p.get("z", 0)])
    elif all(k in tree_json for k in ("x", "y", "z")):
        pos = np.array([tree_json["x"], tree_json["y"], tree_json["z"]])
    if pos is None:
        return None

    dims = tree_json.get("dimensions", {})

    def find(keys, diameter_keys=()):
        for k in keys:
            if k in dims:
                return dims[k]
        for k in keys:
            if k in tree_json:
                return tree_json[k]
        for k in diameter_keys:
            if k in dims:
                return dims[k] / 2.0
        for k in diameter_keys:
            if k in tree_json:
                return tree_json[k] / 2.0
        return None

    radius = find(_TREE_RADIUS_KEYS, _TREE_DIAMETER_KEYS)
    height = find(_TREE_HEIGHT_KEYS)
    trunk = find(_TREE_TRUNK_KEYS)
    if trunk is None:
        trunk = 0.0
    if radius is None and height is None:
        return None
    if radius is None:
        radius = 0.3 * height             # rough canopy-width/height fallback ratio
    if height is None:
        height = trunk + 2.5 * radius
    crown_height = max(height - trunk, 0.1)
    return pos, trunk, crown_height, radius


def build_tree_occluder(tree_json: dict) -> Optional[Occluder]:
    """
    Approximates a tree canopy as a single bounding sphere for shadow
    purposes -- a standard simplification in urban solar-access studies
    (real foliage isn't meshed anyway). Returns None if no recognisable
    position/size fields are found, rather than guessing at a made-up size.
    """
    geo = resolve_tree_geometry(tree_json)
    if geo is None:
        return None
    pos, trunk, crown_height, radius = geo
    canopy_center = pos + np.array([0.0, 0.0, trunk + crown_height / 2.0])
    return Occluder(kind="tree", source_id=tree_json.get("id", "tree"),
                     aabb_min=canopy_center - radius, aabb_max=canopy_center + radius,
                     sphere_center=canopy_center, sphere_radius=radius)


def build_occluders_by_building(scene: dict) -> dict:
    """
    Returns {building_id: [Occluder, ...]} -- for each building, the list of
    every OTHER building's triangulated mesh plus every tree's bounding
    sphere (a building is never tested against its own geometry).
    """
    buildings = scene.get("buildings", [])
    building_occs = {}
    for b in buildings:
        occ = build_building_occluder(b)
        if occ is not None:
            building_occs[b.get("id", "building")] = occ

    tree_occs = []
    skipped = 0
    for t in scene.get("trees", []):
        occ = build_tree_occluder(t)
        if occ is not None:
            tree_occs.append(occ)
        else:
            skipped += 1
    if skipped:
        print(f"[shadows] Skipped {skipped} tree(s) in this scene -- no recognised "
              f"position/size fields. Check _TREE_*_KEYS in solar_thermal_model.py "
              f"against your tree JSON schema.")

    by_building = {}
    for bid in building_occs:
        others = [o for oid, o in building_occs.items() if oid != bid]
        by_building[bid] = others + tree_occs
    return by_building


def _ray_aabb_hit(origin: np.ndarray, direction: np.ndarray,
                   box_min: np.ndarray, box_max: np.ndarray) -> bool:
    """Standard slab test for ray-AABB intersection, t > 0 only."""
    t_min, t_max = 0.0, np.inf
    for i in range(3):
        d = direction[i]
        if abs(d) < 1e-12:
            if origin[i] < box_min[i] or origin[i] > box_max[i]:
                return False
            continue
        t1 = (box_min[i] - origin[i]) / d
        t2 = (box_max[i] - origin[i]) / d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def _ray_triangle_hit(origin: np.ndarray, direction: np.ndarray,
                       v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                       eps: float = 1e-9) -> bool:
    """Moller-Trumbore ray-triangle intersection; only t > eps counts (so the
    face's own near-coplanar geometry can never register as a self-hit)."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    if -eps < a < eps:
        return False
    f = 1.0 / a
    s = origin - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * np.dot(edge2, q)
    return t > eps


def _ray_sphere_hit(origin: np.ndarray, direction: np.ndarray,
                     center: np.ndarray, radius: float, eps: float = 1e-9) -> bool:
    oc = origin - center
    b = np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1, t2 = -b - sq, -b + sq
    return t1 > eps or t2 > eps


def is_shadowed(point: np.ndarray, sun_vec: np.ndarray, occluders: list) -> bool:
    """
    True if the ray from `point` toward the sun is blocked by any occluder
    before reaching it. AABB broad-phase first (cheap); triangle/sphere
    narrow-phase only for occluders the ray could plausibly hit.
    """
    origin = point  # small offset added by the caller along the face normal
    for occ in occluders:
        if not _ray_aabb_hit(origin, sun_vec, occ.aabb_min, occ.aabb_max):
            continue
        if occ.kind == "tree":
            if _ray_sphere_hit(origin, sun_vec, occ.sphere_center, occ.sphere_radius):
                return True
        else:
            for v0, v1, v2 in occ.triangles:
                if _ray_triangle_hit(origin, sun_vec, v0, v1, v2):
                    return True
    return False


class TriangleRayTracer:
    """NumPy packet ray tracing over a median-split triangle BVH.

    Rays in a packet share the sun direction. Bounds prune whole triangle
    groups; leaves test at most 16 triangles at once. Both triangle sides
    occlude, and positive hit distance plus an outward offset avoid self hits.
    No optional ray-tracing libraries are required.
    """

    def __init__(self, triangles, leaf_size=16):
        self.triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
        self.nodes = []
        if not len(self.triangles):
            return
        lo = self.triangles.min(axis=1)
        hi = self.triangles.max(axis=1)
        centers = (lo + hi) / 2

        def build(ids):
            idx = len(self.nodes)
            bounds = (lo[ids].min(axis=0), hi[ids].max(axis=0))
            self.nodes.append(None)
            if len(ids) <= leaf_size:
                self.nodes[idx] = (*bounds, ids, -1, -1)
            else:
                axis = np.argmax(np.ptp(centers[ids], axis=0))
                mid = len(ids) // 2
                ids = ids[np.argpartition(centers[ids, axis], mid)]
                left, right = build(ids[:mid]), build(ids[mid:])
                self.nodes[idx] = (*bounds, None, left, right)
            return idx

        build(np.arange(len(self.triangles)))

    def blocked(self, origins, direction, chunk_size=8192):
        origins = np.asarray(origins, dtype=float).reshape(-1, 3)
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        blocked = np.zeros(len(origins), dtype=bool)
        if not self.nodes:
            return blocked
        v0 = self.triangles[:, 0]
        e1 = self.triangles[:, 1] - v0
        e2 = self.triangles[:, 2] - v0
        cross = np.cross(direction, e2)
        det = np.einsum('ij,ij->i', e1, cross)
        valid = np.abs(det) > 1e-10
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
        for start in range(0, len(origins), chunk_size):
            stack = [(0, np.arange(start, min(start+chunk_size, len(origins))))]
            while stack:
                node, rays = stack.pop()
                rays = rays[~blocked[rays]]
                if not len(rays):
                    continue
                lo, hi, ids, left, right = self.nodes[node]
                points = origins[rays]
                near = np.zeros(len(rays))
                far = np.full(len(rays), np.inf)
                for axis in range(3):
                    if abs(direction[axis]) < 1e-12:
                        outside = (points[:, axis] < lo[axis]-1e-9) | (points[:, axis] > hi[axis]+1e-9)
                        far[outside] = -np.inf
                    else:
                        t1 = (lo[axis]-1e-9-points[:, axis]) / direction[axis]
                        t2 = (hi[axis]+1e-9-points[:, axis]) / direction[axis]
                        near = np.maximum(near, np.minimum(t1, t2))
                        far = np.minimum(far, np.maximum(t1, t2))
                rays = rays[far >= near]
                if not len(rays):
                    continue
                if ids is None:
                    stack.extend([(left, rays), (right, rays)])
                    continue
                ids = ids[valid[ids]]
                if not len(ids):
                    continue
                delta = origins[rays, None, :] - v0[ids]
                u = np.einsum('rtj,tj->rt', delta, cross[ids]) * inv[ids]
                q = np.cross(delta, e1[ids])
                v = np.einsum('rtj,j->rt', q, direction) * inv[ids]
                distance = np.einsum('rtj,tj->rt', q, e2[ids]) * inv[ids]
                hit = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (distance > 1e-7)
                blocked[rays[np.any(hit, axis=1)]] = True
        return blocked


def triangle_sunlit_fraction(faces, sun_vec, ray_tracer, samples=3):
    """Visible area estimate using one centroid or three equal-weight samples.

    Resolution is the input mesh. Three interior barycentric samples provide
    fractional direct-beam exposure for triangles crossed by a shadow edge.
    """
    if samples not in (1, 3):
        raise ValueError("shadow_samples must be 1 or 3")
    normals = np.array([f.normal for f in faces])
    eligible = np.flatnonzero(normals @ sun_vec > 0) if sun_vec[2] > 0 else np.array([], dtype=int)
    fractions = np.zeros(len(faces))
    if not len(eligible):
        return fractions
    vertices = np.array([faces[i].vertices for i in eligible])
    weights = np.array([[1/3, 1/3, 1/3]]) if samples == 1 else np.array([
        [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    origins = np.einsum('sj,tjk->tsk', weights, vertices)
    origins += normals[eligible, None, :] * _SHADOW_RAY_OFFSET
    hits = ray_tracer.blocked(origins.reshape(-1, 3), sun_vec).reshape(-1, samples)
    fractions[eligible] = 1 - hits.mean(axis=1)
    return fractions


# ==========================================================================
# 6. Weather input (plug your real data in here)
# ==========================================================================

def load_weather(csv_path: Optional[str] = None, n_hours: int = 24,
                  t_mean_c: float = 24.0, t_amp_c: float = 6.0,
                  wind_mean_ms: float = 3.0) -> pd.DataFrame:
    """
    If csv_path is given, expects columns: hour, T_amb_C, wind_speed_ms
    (optionally GHI_Wm2, DNI_Wm2, DHI_Wm2 to override the clear-sky model
    with your own measured/forecast radiation).

    If csv_path is None, a synthetic diurnal profile is generated as a
    PLACEHOLDER only (sinusoidal T_amb, constant wind) so the pipeline is
    runnable end-to-end -- swap this out for your real weather data.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        return df

    hours = np.arange(n_hours)
    # min T_amb near 05:00, max near 15:00 -- simple diurnal sine, placeholder only
    t_amb = t_mean_c - t_amp_c * np.cos(2 * np.pi * (hours - 15) / 24.0)
    wind = np.full(n_hours, wind_mean_ms)
    return pd.DataFrame({"hour": hours, "T_amb_C": t_amb, "wind_speed_ms": wind})


# --------------------------------------------------------------------------
# Real weather-data-folder loader: OpenFOAM-style time-table export
# (weatherData/MMDD/{Tambient, windSpeed, IDN, Idif, sunPosVector, ...})
#
# Each file looks like:
#   // comment
#   (
#   ( <time_s> <scalar> )                    <- scalar field
#   ( <time_s> ( <vx> <vy> <vz> ) )          <- vector field
#   ...
#   )
# --------------------------------------------------------------------------

_VEC_RE = re.compile(
    r'\(\s*(-?[0-9.eE+-]+)\s+\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)\s*\)'
)
_SCALAR_RE = re.compile(r'\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)')


def parse_foam_field(path: str) -> tuple:
    """Parses one OpenFOAM-style time-table file. Returns (times_s, values),
    values is a 1D array for scalar fields or an (N,3) array for vector fields."""
    with open(path) as fh:
        content = re.sub(r'//.*', '', fh.read())   # strip comments

    vec = _VEC_RE.findall(content)
    if vec:
        times = np.array([float(m[0]) for m in vec])
        values = np.array([[float(m[1]), float(m[2]), float(m[3])] for m in vec])
        return times, values

    sca = _SCALAR_RE.findall(content)
    times = np.array([float(m[0]) for m in sca])
    values = np.array([float(m[1]) for m in sca])
    return times, values


def resolve_weather_folder(root: str, month: int, day: int) -> str:
    """weatherData/<root> + MMDD (e.g. month=12, day=11 -> '1211')."""
    mmdd = f"{month:02d}{day:02d}"
    candidate = os.path.join(root, mmdd)
    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"No such weather folder: {candidate}")
    return candidate


def load_weather_folder(weather_dir: str) -> dict:
    """
    Reads every field file in a `weatherData/MMDD/` folder and returns a
    dict of hourly arrays ready for `run_analysis_from_folder`:
        hour, T_amb_C, wind_speed_ms, IDN_Wm2, Idif_Wm2,
        sun_vec (N,3) [East, North, Up unit vector, straight from the file],
        altitude_deg  [from solarElevation if present, else asin(sun_vec_z)]
    Any other fields present (cloudCover, relativeHumidity, atmosphericPressure,
    wetBulbTemperature, wambient, wPonds, windVelocity, ...) are passed through
    unchanged under their own file name, in case you want them for other
    calculations (e.g. evaporative cooling from the ponds).
    """
    files = {f: os.path.join(weather_dir, f) for f in os.listdir(weather_dir)
              if os.path.isfile(os.path.join(weather_dir, f))}

    parsed = {name: parse_foam_field(path) for name, path in files.items()}
    # all fields share the same time base -- use whichever field is present first
    times_s = next(iter(parsed.values()))[0]
    hours = times_s / 3600.0

    def get(name, default=None):
        return parsed[name][1] if name in parsed else default

    t_amb_k = get("Tambient")
    wx = dict(
        hour=hours,
        T_amb_C=(t_amb_k - 273.15) if t_amb_k is not None else None,
        wind_speed_ms=get("windSpeed"),
        IDN_Wm2=get("IDN", np.zeros_like(hours)),
        Idif_Wm2=get("Idif", np.zeros_like(hours)),
    )

    if "sunPosVector" in parsed:
        wx["sun_vec"] = parsed["sunPosVector"][1]                 # (N,3) already East/North/Up
        lengths = np.linalg.norm(wx["sun_vec"], axis=1)
        if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("sunPosVector must contain finite, nonzero vectors")
        wx["sun_vec"] = wx["sun_vec"] / lengths[:, None]
        wx["altitude_deg"] = np.degrees(np.arcsin(np.clip(wx["sun_vec"][:, 2], -1, 1)))
    elif "solarElevation" in parsed and "solarAzimuth" in parsed:
        alt = parsed["solarElevation"][1]
        az = parsed["solarAzimuth"][1]
        wx["altitude_deg"] = alt
        wx["sun_vec"] = np.array([sun_unit_vector(a, z) for a, z in zip(alt, az)])
    else:
        raise ValueError("Weather folder has neither sunPosVector nor "
                          "solarElevation+solarAzimuth -- cannot get sun direction.")

    if wx["T_amb_C"] is None:
        raise ValueError("Weather folder is missing a 'Tambient' file.")
    if wx["wind_speed_ms"] is None:
        raise ValueError("Weather folder is missing a 'windSpeed' file.")

    # keep any extra fields around too, unmodified
    for name, (t, v) in parsed.items():
        if name not in ("Tambient", "windSpeed", "IDN", "Idif", "sunPosVector",
                         "solarElevation", "solarAzimuth"):
            wx[name] = v

    return wx


# ==========================================================================
# 7. Shared per-hour physics loop + per-building RC response
#    (used by BOTH the clear-sky/solar-formula path and the real
#    weather-data-folder path below)
# ==========================================================================

_SHADOW_RAY_OFFSET = 1e-3   # m, nudge the ray origin off the face surface

def _simulate_steps(faces: list, capacitance_by_building: dict, records: list,
              ground_albedo: float = DEFAULT_GROUND_ALBEDO,
              occluders_by_building: Optional[dict] = None,
              t_initial_c: Optional[float] = None,
              ray_tracer=None, shadow_samples: int = 3,
              wall_settings: Optional[WallSettings] = None,
              surface_group: str = "building",
              wetness: float = 0.0, back_follows_ambient: bool = False,
              workers: int = 1, lockstep: bool = False) -> tuple:
    """Per-cell solar/thermal balance plus the separate legacy bulk RC estimate.

    OBJ cells use packet rays; JSON faces retain their centroid visibility test.
    Output temperatures are states at the labelled timestamps. The first
    output is the initial condition; every later record supplies the constant
    boundary condition over the interval ending at that timestamp. This keeps
    each output's shadow/solar fields aligned with its thermal state.
    """
    excluded = {"ceiling"} | (set() if INCLUDE_FLOOR else {"floor"})
    exposed = [f for f in faces if f.surface_type not in excluded]
    if not exposed or not records:
        raise ValueError("Simulation requires exposed faces and weather records")
    bids = list(capacitance_by_building)
    bindex = {bid: i for i, bid in enumerate(bids)}
    indices = np.array([bindex[f.building_id] for f in exposed])
    normals = np.array([f.normal for f in exposed])
    areas = np.array([f.area for f in exposed])
    albedo = np.array([f.albedo for f in exposed])
    emissivity = np.array([f.emissivity for f in exposed])
    view_sky = (1 + normals[:, 2]) / 2
    static = dict(building_id=[f.building_id for f in exposed],
                  face_id=[f.face_id for f in exposed],
                  parent_face_id=[f.parent_face_id or f.face_id for f in exposed],
                  surface_type=[f.surface_type for f in exposed],
                  surface_group=surface_group, area_m2=areas,
                  tilt_deg=[f.tilt_deg for f in exposed])
    frames, bulk_rows = [], []
    bulk_T = np.full(len(bids), t_initial_c if t_initial_c is not None else records[0]["T_amb_C"], dtype=float)
    caps = np.array([capacitance_by_building[b] for b in bids])
    wall_settings = wall_settings or WallSettings()
    wall_initial_c = (t_initial_c if t_initial_c is not None
                      else wall_settings.indoor_temperature_c)
    slab = SlabConduction([f.thermal_conductivity for f in exposed],
                          [f.density for f in exposed], [f.specific_heat for f in exposed],
                          emissivity, wall_initial_c, wall_settings)
    # Sunlit fractions depend only on geometry and sun position, not on the
    # thermal state, so all hours are ray-traced up front (optionally threaded).
    shadow_fractions = {}
    if ray_tracer is not None:
        sun_vecs = [np.asarray(r["sun_vec"], float) / np.linalg.norm(r["sun_vec"]) for r in records]
        todo = [i for i, r in enumerate(records) if r["altitude_deg"] > 0]
        verts_arr = np.array([f.vertices for f in exposed], dtype=float)
        print(f"[shadows] {surface_group}: tracing {len(exposed)} cells x {len(todo)} hours "
              f"(threads={max(1, int(workers))})", flush=True)

        def _shadow_job(i):
            return i, sunlit_fractions_from_arrays(normals, verts_arr, sun_vecs[i],
                                                   ray_tracer, shadow_samples)
        if workers > 1 and len(todo) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as pool:
                shadow_fractions = dict(pool.map(_shadow_job, todo))
        else:
            shadow_fractions = dict(_shadow_job(i) for i in todo)
    for ri, rec in enumerate(records):
        t_amb = rec["T_amb_C"]
        sun = np.asarray(rec["sun_vec"], dtype=float)
        sun = sun / np.linalg.norm(sun)
        alt = rec["altitude_deg"]
        h_c = convective_h(rec["wind_speed_ms"])
        cos_theta = normals @ sun
        self_lit = (alt > 0) & (cos_theta > 0)
        fraction = self_lit.astype(float)
        if ray_tracer is not None and alt > 0:
            fraction = shadow_fractions[ri]
        elif occluders_by_building:
            for i in np.flatnonzero(self_lit):
                f = exposed[i]
                fraction[i] = not is_shadowed(f.center + f.normal*_SHADOW_RAY_OFFSET,
                                              sun, occluders_by_building.get(f.building_id, []))
        shadow_fraction = np.where(self_lit, 1-fraction, 0)
        direct = max(rec["i_dn"], 0) * np.maximum(cos_theta, 0) * fraction
        diffuse = max(rec["i_dh"], 0) * view_sky
        ghi = max(rec["i_dn"], 0) * max(sun[2], 0) + max(rec["i_dh"], 0)
        reflected = ground_albedo * ghi * (1-view_sky)
        total = direct + diffuse + reflected
        solair = sol_air_temperature(t_amb, total, h_c, albedo, view_sky, emissivity)
        q_solar = (1-albedo)*total
        ta_k = t_amb + 273.15
        # Tangent linearization at ambient, frozen for this weather interval:
        # q_lw(Ts) ~= q_lw(Ta) + h_rad*(Ts-Ta). This permits the exact
        # matrix-exponential wall update; full Ts^4 radiation is nonlinear.
        q_lw_ambient = emissivity*STEFAN_BOLTZMANN*view_sky*(ta_k**4-sky_temperature(ta_k)**4)
        h_rad = 4*emissivity*STEFAN_BOLTZMANN*ta_k**3
        # Evaporation / transpiration, same tangent linearisation at ambient:
        # q_evap(Ts) ~= e0 + e_slope*(Ts-Ta); coef = wetness*L*h_c/cp_air (Lewis).
        e0 = e_slope = 0.0
        if wetness > 0:
            p_atm = rec.get("p_atm", 101325.0)
            w_amb = rec.get("w_amb")
            if w_amb is None:
                w_amb = _humidity_ratio(t_amb, rec.get("rh", 0.5), p_atm)
            ws, dws = _saturation_humidity_ratio(t_amb, p_atm)
            coef = wetness*LATENT_HEAT_VAPORIZATION*h_c/CP_AIR
            e0, e_slope = coef*(ws-w_amb), coef*dws
        external_h = h_c+h_rad+e_slope
        external_source = q_solar-q_lw_ambient-e0 + external_h*t_amb
        if back_follows_ambient:
            slab.set_back(h_c, t_amb)

        if ri > 0:
            dt = (rec["hour"] - records[ri-1]["hour"])*3600
            if dt <= 0:
                raise ValueError("Weather times must be strictly increasing")
            if lockstep:
                yield dt, slab          # lets LateralCoupler act on the state
            slab.advance(dt, external_h, external_source)

        ts = slab.temperatures[:,0].copy()
        q_conv = h_c*(ts-t_amb)
        q_evap = e0 + e_slope*(ts-t_amb)
        q_lw = q_lw_ambient + h_rad*(ts-t_amb)
        wall_fields = slab.diagnostics(external_h, external_source)
        wall_fields["stored_heat_J"] = wall_fields["stored_heat_Jm2"]*areas
        wall_fields["wall_capacity_JK"] = wall_fields["wall_capacity_Jm2K"]*areas
        wall_fields["slab_capacity_JK"] = wall_fields["slab_capacity_Jm2K"]*areas
        frames.append(pd.DataFrame(dict(static, hour=rec["hour"], solar_altitude_deg=alt,
            self_lit=self_lit, shadowed=shadow_fraction > 0, sunlit=fraction > 0,
            sunlit_fraction=fraction, shadow_fraction=shadow_fraction, cos_theta=cos_theta,
            I_direct_Wm2=direct, I_diffuse_Wm2=diffuse, I_reflected_Wm2=reflected,
            I_total_Wm2=total, h_conv_Wm2K=h_c, T_amb_C=t_amb,
            T_solair_C=solair, T_surface_C=ts, q_solar_absorbed_Wm2=q_solar,
            q_conv_out_Wm2=q_conv, q_longwave_out_Wm2=q_lw, q_evap_out_Wm2=q_evap,
            q_lateral_in_Wm2=getattr(slab, "lateral_flux_Wm2", 0.0),
            wall_thickness_m=wall_settings.thickness_m,
            slab_thickness_m=wall_settings.thickness_m,
            T_indoor_air_C=slab.back_temperature,
            T_back_boundary_C=slab.back_temperature,
            h_indoor_Wm2K=slab.back_h,
            h_back_Wm2K=slab.back_h, **wall_fields)))
        ah = np.bincount(indices, weights=areas*h_c, minlength=len(bids))
        aht = np.bincount(indices, weights=areas*h_c*solair, minlength=len(bids))
        teq = np.divide(aht, ah, out=np.full(len(bids), t_amb, dtype=float), where=ah > 0)
        tau = np.divide(caps, ah, out=np.full(len(bids), np.inf), where=ah > 0)
        for bi, bid in enumerate(bids):
            bulk_rows.append(dict(building_id=bid, hour=rec["hour"], T_amb_C=t_amb,
                T_eq_C=teq[bi], h_eff_WK=ah[bi], tau_hr=tau[bi]/3600,
                T_building_C=bulk_T[bi]))
        if ri+1 < len(records):
            dt_bulk = (records[ri+1]["hour"] - rec["hour"])*3600
            if dt_bulk <= 0:
                raise ValueError("Weather times must be strictly increasing")
            decay = np.exp(-np.divide(
                dt_bulk, tau, out=np.full_like(tau, np.inf), where=tau > 0))
            bulk_T = teq + (bulk_T-teq)*decay
        if ray_tracer is not None:
            print(f"[simulation] hour {rec['hour']:g}: {len(exposed)} surface cells", flush=True)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(bulk_rows)


def _simulate(*args, **kwargs) -> tuple:
    """Run _simulate_steps to completion and return its (face_df, bulk_df)."""
    gen = _simulate_steps(*args, **kwargs)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def compute_capacitances(scene: dict, faces: list) -> dict:
    """{building_id: C [J/K]} for every building in the scene, C = rho*V*cp."""
    caps = {}
    faces_by_bid = {}
    for f in faces:
        faces_by_bid.setdefault(f.building_id, f)
    for bid, f0 in faces_by_bid.items():
        vol = building_volume(scene, bid)
        caps[bid] = f0.density * vol * f0.specific_heat
    return caps


def run_analysis(json_path: str, day: date, weather: pd.DataFrame,
                  utc_offset_hr: float = 0.0, north_offset_deg: float = 0.0,
                  site_altitude_km: float = 0.0,
                  consider_shadows: bool = True,
                  t_initial_c: Optional[float] = None,
                  wall_settings: Optional[WallSettings] = None) -> tuple:
    """
    CLEAR-SKY / SOLAR-FORMULA path: use this when you do NOT have a real
    weather-data export, only a date + location + basic T_amb/wind. Solar
    position and irradiance are both modelled analytically (see functions
    `solar_position` and `hottel_clear_sky` above).
    Returns (face_hourly_df, building_hourly_df, faces).
    """
    scene = _scene_from(json_path)
    faces = extract_faces(scene)
    lat, lon = site_location(scene)
    capacitances = compute_capacitances(scene, faces)
    occluders = build_occluders_by_building(scene) if consider_shadows else None
    day_of_year = day.timetuple().tm_yday

    records = []
    for _, row in weather.iterrows():
        hr = row["hour"]
        alt, az = solar_position(lat, lon, day_of_year, hr, utc_offset_hr)
        sun_vec = sun_unit_vector(alt, az, north_offset_deg)
        if "GHI_Wm2" in row and not pd.isna(row.get("DNI_Wm2", np.nan)):
            i_dn, i_dh = row["DNI_Wm2"], row["DHI_Wm2"]
        else:
            i_dn, i_dh = hottel_clear_sky(alt, site_altitude_km)
        records.append(dict(hour=hr, T_amb_C=row["T_amb_C"], wind_speed_ms=row["wind_speed_ms"],
                             sun_vec=sun_vec, altitude_deg=alt, i_dn=i_dn, i_dh=i_dh))

    face_df, building_df = _simulate(faces, capacitances, records,
                                      occluders_by_building=occluders, t_initial_c=t_initial_c,
                                      wall_settings=wall_settings)
    return face_df, building_df, faces


def run_analysis_from_folder(json_path: str, weather_dir: str,
                              ground_albedo: float = DEFAULT_GROUND_ALBEDO,
                              consider_shadows: bool = True,
                              t_initial_c: Optional[float] = None,
                  wall_settings: Optional[WallSettings] = None) -> tuple:
    """
    REAL WEATHER-DATA path (preferred whenever you have it): reads Tambient,
    windSpeed, IDN, Idif and sunPosVector straight from an exported
    `weatherData/MMDD/` folder (see `load_weather_folder`). No solar-position
    formula and no clear-sky model are used here at all -- the sun vector and
    irradiance are exactly what your own pipeline computed, in the same
    (x=East, y=North, z=Up) axes as the building JSON's face normals, so
    there is no coordinate-convention guessing.
    Returns (face_hourly_df, building_hourly_df, faces).
    """
    scene = _scene_from(json_path)
    faces = extract_faces(scene)
    capacitances = compute_capacitances(scene, faces)
    occluders = build_occluders_by_building(scene) if consider_shadows else None
    wx = load_weather_folder(weather_dir)

    records = []
    n = len(wx["hour"])
    for i in range(n):
        records.append(dict(
            hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
            sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
            i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i],
        ))

    face_df, building_df = _simulate(faces, capacitances, records, ground_albedo=ground_albedo,
                                      occluders_by_building=occluders, t_initial_c=t_initial_c,
                                      wall_settings=wall_settings)
    return face_df, building_df, faces


# ==========================================================================
# 8. ParaView export -- writes the same results as a time series of legacy
#    VTK PolyData files (plus a .pvd collection), so you can open one file
#    in ParaView, scrub an "hour of day" timeline, and see the building
#    (colored by sol-air temperature, shadowed/sunlit status, or irradiance)
#    together with the trees, with no plugins and no `pip install vtk` --
#    this writes plain-text legacy VTK, which ParaView reads natively.
# ==========================================================================

_SURFACE_TYPE_ID = {"wall": 0, "roof": 1, "floor": 2, "ceiling": 3, "tree": 5, "ground": 6,
                    "pathway": 7, "lawn": 8, "pond": 9}
TREE_TRUNK_TYPE_ID = 4
TREE_CROWN_TYPE_ID = 5


def _ring_polygon(cx: float, cy: float, z: float, radius: float, n: int) -> list:
    return [(cx + radius * math.cos(2 * math.pi * i / n),
              cy + radius * math.sin(2 * math.pi * i / n), z) for i in range(n)]


def _cylinder_polys(cx: float, cy: float, z0: float, z1: float, radius: float, n: int = 8) -> list:
    """Side quads + a bottom cap (top is hidden inside the tree crown)."""
    ring0 = _ring_polygon(cx, cy, z0, radius, n)
    ring1 = _ring_polygon(cx, cy, z1, radius, n)
    polys = []
    for i in range(n):
        j = (i + 1) % n
        polys.append([ring0[i], ring0[j], ring1[j], ring1[i]])
    polys.append(list(reversed(ring0)))   # bottom cap, wound outward-facing-down
    return polys


def _cone_polys(cx: float, cy: float, base_z: float, apex_z: float, radius: float, n: int = 8) -> list:
    """Side triangles (apex at top) + a base cap."""
    ring = _ring_polygon(cx, cy, base_z, radius, n)
    apex = (cx, cy, apex_z)
    polys = [[ring[i], ring[(i + 1) % n], apex] for i in range(n)]
    polys.append(list(ring))              # base cap
    return polys


def _tree_mesh(tree_json: dict, trunk_radius_ratio: float = 0.15) -> tuple:
    """Returns (polygons, surface_type_ids) for one tree's trunk + crown, or
    (None, None) if the tree's geometry couldn't be resolved (see
    resolve_tree_geometry)."""
    geo = resolve_tree_geometry(tree_json)
    if geo is None:
        return None, None
    pos, trunk_h, crown_h, crown_r = geo
    trunk_r = max(crown_r * trunk_radius_ratio, 0.1)
    trunk_polys = _cylinder_polys(pos[0], pos[1], pos[2], pos[2] + trunk_h, trunk_r)
    crown_polys = _cone_polys(pos[0], pos[1], pos[2] + trunk_h, pos[2] + trunk_h + crown_h, crown_r)
    polys = trunk_polys + crown_polys
    type_ids = [TREE_TRUNK_TYPE_ID] * len(trunk_polys) + [TREE_CROWN_TYPE_ID] * len(crown_polys)
    return polys, type_ids


def _write_vtk_polydata_slow(path: str, polygons: list, cell_scalars: dict) -> None:
    """
    polygons: list of polygons, each a list of (x,y,z) tuples (a planar loop).
        Points are NOT shared between polygons -- every polygon gets its own
        copy of its corner points. That is deliberate: colouring is by CELL
        data (one value per face), and un-shared points guarantee ParaView
        renders each face as a single flat colour instead of interpolating
        across a shared edge with a differently-valued neighbour.
    cell_scalars: {name: (values, "float"|"int")}, one value per polygon,
        same length and order as `polygons`.
    """
    points, polygon_indices = [], []
    idx = 0
    for poly in polygons:
        n = len(poly)
        polygon_indices.append(list(range(idx, idx + n)))
        points.extend(poly)
        idx += n

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("solar_thermal_model export\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} float\n")
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        total_size = sum(len(idxs) + 1 for idxs in polygon_indices)
        f.write(f"POLYGONS {len(polygon_indices)} {total_size}\n")
        for idxs in polygon_indices:
            f.write(f"{len(idxs)} " + " ".join(str(i) for i in idxs) + "\n")
        f.write(f"CELL_DATA {len(polygons)}\n")
        for name, (values, dtype) in cell_scalars.items():
            f.write(f"SCALARS {name} {dtype} 1\n")
            f.write("LOOKUP_TABLE default\n")
            fmt = "{}\n" if dtype == "int" else "{:.4f}\n"
            for v in values:
                f.write(fmt.format(v))


def export_vtk_series(json_path: str, weather_dir: str, out_dir: str,
                       consider_shadows: bool = True,
                       wall_settings: Optional[WallSettings] = None,
                       t_initial_c: Optional[float] = None) -> str:
    """
    Runs the real-weather-data analysis (see `run_analysis_from_folder`) and
    writes one legacy-VTK PolyData file per hour plus a `scene.pvd` ParaView
    collection tying them together as a time series. Open `scene.pvd` in
    ParaView, colour by any of the per-face arrays below, and scrub the time
    slider to watch temperature and shadowing move through the day together.

    Per-face (cell) data written on every building face:
        T_C            transient outer surface temperature, deg C
        I_total_Wm2    total incident solar irradiance, W/m^2
        shadowed       1 if ray-blocked by another building/tree this hour
        sunlit         1 if actually receiving direct beam this hour
                        (self_lit AND NOT shadowed)
        surface_type_id  0=wall 1=roof 2=floor 3=ceiling 4=tree trunk 5=tree crown
        building_index  which building/floor this face belongs to (-1 for trees)
        is_building     1 for building faces, 0 for tree geometry

    Floor/ceiling faces are excluded from the thermal simulation itself (see
    the module docstring) but ARE still drawn here, with T_C set to that
    hour's ambient air temperature, so the building reads as a solid, complete
    shape rather than having visible holes -- filter on surface_type_id in
    ParaView if you want only the faces that were actually simulated.
    Trees are static (their geometry doesn't change hour to hour) but are
    re-written into every frame file so each one opens as a complete scene on
    its own; T_C for tree geometry is likewise just that hour's ambient
    temperature (canopy temperature isn't modelled).
    """
    scene = _scene_from(json_path)
    faces = extract_faces(scene)
    face_df, building_df, _ = run_analysis_from_folder(
        json_path, weather_dir, consider_shadows=consider_shadows,
        wall_settings=wall_settings, t_initial_c=t_initial_c)
    os.makedirs(out_dir, exist_ok=True)

    tree_polys, tree_type_ids = [], []
    n_tree_geo_ok, n_tree_geo_skipped = 0, 0
    for t in scene.get("trees", []):
        polys, type_ids = _tree_mesh(t)
        if polys is None:
            n_tree_geo_skipped += 1
            continue
        n_tree_geo_ok += 1
        tree_polys.extend(polys)
        tree_type_ids.extend(type_ids)
    if n_tree_geo_skipped:
        print(f"[vtk export] Skipped {n_tree_geo_skipped} tree(s) with unrecognised geometry fields.")

    building_ids_sorted = sorted({f.building_id for f in faces})
    building_index_map = {bid: i for i, bid in enumerate(building_ids_sorted)}

    face_lookup = {}
    for _, r in face_df.iterrows():
        face_lookup[(r["building_id"], r["face_id"], r["hour"])] = r

    hours = sorted(face_df["hour"].unique())
    pvd_entries = []
    for h in hours:
        amb_rows = building_df[building_df["hour"] == h]
        amb_c = float(amb_rows["T_amb_C"].iloc[0]) if len(amb_rows) else 0.0

        polys, T_C, shadowed, sunlit, I_tot, surf_id, bidx, is_bldg = [], [], [], [], [], [], [], []
        for f in faces:
            row = face_lookup.get((f.building_id, f.face_id, h))
            polys.append([tuple(v) for v in f.vertices])
            if row is None:                          # floor/ceiling: not simulated
                T_C.append(amb_c); shadowed.append(0); sunlit.append(0); I_tot.append(0.0)
            else:
                T_C.append(float(row["T_surface_C"])); shadowed.append(int(bool(row["shadowed"])))
                sunlit.append(int(bool(row["sunlit"]))); I_tot.append(float(row["I_total_Wm2"]))
            surf_id.append(_SURFACE_TYPE_ID.get(f.surface_type, -1))
            bidx.append(building_index_map[f.building_id])
            is_bldg.append(1)

        polys.extend(tree_polys)
        T_C.extend([amb_c] * len(tree_polys))
        shadowed.extend([0] * len(tree_polys))
        sunlit.extend([0] * len(tree_polys))
        I_tot.extend([0.0] * len(tree_polys))
        surf_id.extend(tree_type_ids)
        bidx.extend([-1] * len(tree_polys))
        is_bldg.extend([0] * len(tree_polys))

        fname = f"frame_{int(round(h * 100)):04d}.vtk"   # e.g. hour 13.5 -> frame_1350.vtk
        _write_vtk_polydata(os.path.join(out_dir, fname), polys, {
            "T_C": (T_C, "float"),
            "I_total_Wm2": (I_tot, "float"),
            "shadowed": (shadowed, "int"),
            "sunlit": (sunlit, "int"),
            "surface_type_id": (surf_id, "int"),
            "building_index": (bidx, "int"),
            "is_building": (is_bldg, "int"),
        })
        pvd_entries.append((float(h), fname))

    pvd_path = os.path.join(out_dir, "scene.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for t, fname in pvd_entries:
            f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')

    print(f"[vtk export] Wrote {len(pvd_entries)} frame(s) + scene.pvd to {out_dir} "
          f"({n_tree_geo_ok} tree(s) meshed).")
    return pvd_path


# ==========================================================================
# 9. OBJ pipeline: retain input triangles for partial-wall shadows.
#    Coplanar grouping is used only for parent IDs and bulk-volume estimates.
#    OBJ does not carry thermal materials: defaults below assume concrete.
#    Spatial resolution is limited by the input mesh; use a finer OBJ for
#    sharper shadow edges. Trees are opaque meshes, without leaf transmission.
# ==========================================================================

DEFAULT_MATERIAL = dict(thermal_conductivity=2.3, density=2400, specific_heat=880)
DEFAULT_ALBEDO_WALL = 0.3
DEFAULT_ALBEDO_ROOF = 0.4
ROOF_NORMAL_Z_THRESHOLD = 0.7   # |normal.z| above this -> "roof", else "wall"


def _obj_vertex_index(token: str, n_verts_so_far: int) -> int:
    """
    Parses one OBJ face-vertex token in any of the four standard forms
    (`v`, `v/vt`, `v/vt/vn`, `v//vn`) and returns a 0-based vertex index.
    Handles negative (relative-to-current-end) indices too, per the OBJ
    spec.
    """
    idx = int(token.split("/")[0])
    return (n_verts_so_far + idx) if idx < 0 else (idx - 1)


def load_obj_mesh(path: str) -> tuple:
    """
    OBJ parser for the buildings_refined.obj / trees.obj this pipeline
    consumes. json_to_obj.py's own write_obj() only ever emits plain
    `v x y z` / `f i j k` lines, but real pipelines often pass the file
    through something else afterwards (normal generation, mesh repair,
    etc.) that adds `vn`/`vt` references -- seen in practice as
    `f 1//1 2//1 3//1` style faces. This is intentionally more permissive
    than that one exact format: it accepts all four standard face-vertex
    forms (`v`, `v/vt`, `v/vt/vn`, `v//vn`), skips `vn`/`vt` lines outright
    (this pipeline recomputes every normal from triangle geometry anyway,
    never reads one from the file), and fan-triangulates any face with more
    than 3 vertices rather than assuming triangles-only.
    Returns (vertices [N,3] float array, triangles [list of (i,j,k) 0-based
    int tuples]).
    """
    verts, tris = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                idxs = [_obj_vertex_index(tok, len(verts)) for tok in tokens]
                for t in range(1, len(idxs) - 1):     # fan triangulation (no-op for triangles)
                    tris.append((idxs[0], idxs[t], idxs[t + 1]))
    return np.array(verts, dtype=float), tris


def _triangle_normal_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple:
    n = np.cross(v1 - v0, v2 - v0)
    area = float(np.linalg.norm(n)) / 2.0
    if area < 1e-12:
        return None, 0.0
    return n / (2.0 * area), area


def group_coplanar(vertices: np.ndarray, triangles: list,
                    normal_decimals: int = 3, offset_decimals: int = 2) -> dict:
    """
    Merges triangles sharing the same (rounded outward normal, rounded plane
    offset) into "logical faces". Exact for flat, regularly-subdivided quads
    (this OBJ format's building geometry): every wall segment along one
    straight facade -- including across every stacked storey of a
    multi-floor tower, since they're all perfectly coplanar -- collapses
    back into ONE logical wall face.
    Returns {plane_key: {"triangles": [(i,j,k), ...], "normal", "area",
                          "center", "z_min", "z_max"}}.
    """
    groups = {}
    for (i, j, k) in triangles:
        v0, v1, v2 = vertices[i], vertices[j], vertices[k]
        n, area = _triangle_normal_area(v0, v1, v2)
        if n is None:
            continue
        d = float(np.dot(n, v0))
        key = (round(float(n[0]), normal_decimals), round(float(n[1]), normal_decimals),
               round(float(n[2]), normal_decimals), round(d, offset_decimals))
        g = groups.setdefault(key, dict(triangles=[], normal_sum=np.zeros(3), area=0.0,
                                          centroid_sum=np.zeros(3), z_min=np.inf, z_max=-np.inf))
        centroid = (v0 + v1 + v2) / 3.0
        g["triangles"].append((i, j, k))
        g["normal_sum"] += n * area
        g["area"] += area
        g["centroid_sum"] += centroid * area
        g["z_min"] = min(g["z_min"], float(v0[2]), float(v1[2]), float(v2[2]))
        g["z_max"] = max(g["z_max"], float(v0[2]), float(v1[2]), float(v2[2]))

    out = {}
    for key, g in groups.items():
        if g["area"] <= 0:
            continue
        normal = g["normal_sum"] / (np.linalg.norm(g["normal_sum"]) + 1e-12)
        center = g["centroid_sum"] / g["area"]
        out[key] = dict(triangles=g["triangles"], normal=normal, area=g["area"],
                          center=center, z_min=g["z_min"], z_max=g["z_max"])
    return out


def _mesh_to_occluder(vertices: np.ndarray, triangles: list, source_id: str) -> Optional[Occluder]:
    if not triangles:
        return None
    tri_pts = [(vertices[i], vertices[j], vertices[k]) for (i, j, k) in triangles]
    idx = sorted({v for tri in triangles for v in tri})
    pts = vertices[idx]
    return Occluder(kind="building", source_id=source_id,
                     aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tri_pts)


def _position_key(p, decimals: int = 4) -> tuple:
    return (round(float(p[0]), decimals), round(float(p[1]), decimals), round(float(p[2]), decimals))


def _obj_components_by_shared_position(vertices: np.ndarray, triangles: list,
                                         decimals: int = 4) -> list:
    """
    Union-find over vertex POSITIONS, not indices -- this is the piece that
    was missing before. json_to_obj.py does not share vertex indices across
    different original JSON faces, even when they are geometrically
    touching (e.g. two wall quads meeting at a shared vertical edge), so an
    index-based union-find (see _obj_connected_components, used for trees)
    cannot tell "two touching walls of the same building" apart from "two
    unrelated triangles that happen to have unrelated index numbers" -- it
    under-merges everything into one-triangle-ish fragments. Position-based
    matching (round each vertex to `decimals` and treat equal rounded
    positions as the same point) is what actually reconstructs "which
    triangles are physically part of the same connected building".
    Returns a list of triangle-index lists, one per physically distinct
    building.
    """
    pos_to_id: dict = {}
    vertex_logical_id = np.empty(len(vertices), dtype=np.int64)
    for v_idx in range(len(vertices)):
        key = _position_key(vertices[v_idx], decimals)
        lid = pos_to_id.get(key)
        if lid is None:
            lid = len(pos_to_id)
            pos_to_id[key] = lid
        vertex_logical_id[v_idx] = lid

    parent = list(range(len(pos_to_id)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j, k) in triangles:
        li, lj, lk = int(vertex_logical_id[i]), int(vertex_logical_id[j]), int(vertex_logical_id[k])
        union(li, lj)
        union(lj, lk)

    comp_triangles: dict = {}
    for t_idx, (i, j, k) in enumerate(triangles):
        root = find(int(vertex_logical_id[i]))
        comp_triangles.setdefault(root, []).append(t_idx)
    return list(comp_triangles.values())


def _obj_connected_components(vertices: np.ndarray, triangles: list) -> list:
    """
    Union-find over shared vertex INDICES. This groups triangles generated
    together as one contiguous primitive (e.g. one tree's trunk cylinder) --
    json_to_obj.py DOES share vertex indices within a single primitive's own
    subdivision, so this works there, but NOT across different original
    JSON faces (see _obj_components_by_shared_position for that -- used for
    buildings, where genuine cross-face adjacency detection matters).
    Verified on a real trees.obj: 49 trees x (1 trunk + 1 crown, each its
    own component) = 98 components.
    Returns a list of triangle-index lists, one per component.
    """
    parent = list(range(len(vertices)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j, k) in triangles:
        union(i, j)
        union(j, k)

    comp_triangles = {}
    for t_idx, (i, j, k) in enumerate(triangles):
        comp_triangles.setdefault(find(i), []).append(t_idx)
    return list(comp_triangles.values())


def build_tree_occluders_from_obj(trees_obj_path: str) -> list:
    """Loads trees.obj and returns one Occluder per connected primitive
    (trunk/crown separately) -- more, smaller occluders only help the
    broad-phase AABB culling, so there's no reason to merge them further."""
    if not trees_obj_path or not os.path.isfile(trees_obj_path):
        return []
    V, F = load_obj_mesh(trees_obj_path)
    if not F:
        return []
    occluders = []
    for c_idx, tri_indices in enumerate(_obj_connected_components(V, F)):
        occ = _mesh_to_occluder(V, [F[t] for t in tri_indices], f"tree_part_{c_idx}")
        if occ is not None:
            occluders.append(occ)
    return occluders


# --------------------------------------------------------------------------
# 9b. Multi-surface OBJ pipeline: buildings, ground, pathway, lawn, pond, trees
#
#     Every component is optional. A component is solved when its OBJ holds
#     valid triangles (or, for trees only, when the JSON describes them);
#     anything missing or unusable is reported in the log and skipped.
#     Geometry comes from the OBJ files; material properties come from the
#     scene JSON when it has them, otherwise from the documented defaults in
#     LAYER_DEFAULTS / DEFAULT_MATERIAL (the log says which, per property).
# --------------------------------------------------------------------------

LATENT_HEAT_VAPORIZATION = 2.45e6      # J/kg
CP_AIR = 1006.0                        # J/kg/K
LAYER_CLASSES = ("ground", "pathway", "lawn", "pond", "tree")
LAYER_PRIORITY = ("pond", "pathway", "lawn", "ground")   # top-most first


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[warning] {message}", flush=True)


def _scene_from(scene_or_path):
    """Accept either a parsed scene dict or a path to a scene JSON."""
    return scene_or_path if isinstance(scene_or_path, dict) else load_scene(scene_or_path)


# ---- material properties from the scene JSON -----------------------------

def _norm_key(key) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


_PROPERTY_ALIASES = {
    "thermal_conductivity": ("thermalconductivity", "conductivity"),
    "density": ("density", "rho"),
    "specific_heat": ("specificheat", "cp"),
    "albedo": ("albedo", "reflectivity"),
    "emissivity": ("emissivity", "emittance"),
    "depth_m": ("depthm", "depth", "thicknessm", "thickness"),
    "wetness": ("wetness", "evaporationfraction", "moistureavailability"),
    "mixing_factor": ("mixingfactor", "mixingmultiplier"),
}
_CONTAINER_KEYS = ("thermalproperties", "thermalprops")
_MATERIAL_KEYS = ("materialproperties", "material", "properties")
_CLASS_KEYS = {
    "ground": ("ground", "terrain", "soil", "groundproperties"),
    "pathway": ("pathways", "pathway", "paths", "path", "roads", "road",
                "pavements", "pavement", "walkways"),
    "lawn": ("lawns", "lawn", "grass", "grasslands", "greenareas"),
    "pond": ("ponds", "pond", "water", "waterbodies"),
    "tree": ("trees", "tree"),
}


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _thermal_containers(item) -> list:
    """Nested 'thermalProperties' dicts anywhere (shallowly) inside an item."""
    found = []

    def walk(obj, depth):
        if depth > 4:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _norm_key(k) in _CONTAINER_KEYS and isinstance(v, dict):
                    found.append(v)
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj[:3]:
                walk(v, depth + 1)

    walk(item, 0)
    return found


def _lookup_property(item, prop: str) -> Optional[float]:
    """Look up a numeric property by alias. Search order: thermalProperties
    containers, materialProperties-like dicts, then the item's own top level.
    A bare top-level 'density' is ignored (too ambiguous, e.g. tree density)."""
    if not isinstance(item, dict):
        return None
    dicts = _thermal_containers(item)
    for k, v in item.items():
        if _norm_key(k) in _MATERIAL_KEYS and isinstance(v, dict):
            dicts.append(v)
    if prop != "density":
        dicts.append(item)
    aliases = _PROPERTY_ALIASES[prop]
    for d in dicts:
        norm = {_norm_key(k): v for k, v in d.items()}
        for alias in aliases:
            x = _as_float(norm.get(alias))
            if x is not None:
                return x
    return None


def _scene_items(scene: dict, cls: str) -> list:
    wanted = _CLASS_KEYS[cls]
    containers = [scene]
    if isinstance(scene.get("settings"), dict):
        containers.append(scene["settings"])
    items = []
    for c in containers:
        for k, v in c.items():
            if _norm_key(k) in wanted:
                if isinstance(v, list):
                    items.extend(x for x in v if isinstance(x, dict))
                elif isinstance(v, dict):
                    items.append(v)
    return items


def json_layer_properties(scene: Optional[dict], cls: str) -> dict:
    """Mean of each property over all JSON items of this class."""
    if not isinstance(scene, dict):
        return {}
    items = _scene_items(scene, cls)
    out = {}
    for prop in _PROPERTY_ALIASES:
        vals = [v for v in (_lookup_property(it, prop) for it in items) if v is not None]
        if vals:
            out[prop] = float(np.mean(vals))
    return out


# Documented defaults. Change these (or give the property in the JSON / CLI).
_LAYER_BASE = dict(depth_m=0.23, nodes=5, deep_temperature_c=24.0, deep_h_wm2k=None,
                   thermal_conductivity=1.5, density=1150.0, specific_heat=650.0,
                   albedo=DEFAULT_GROUND_ALBEDO, emissivity=0.95, wetness=0.0,
                   back_follows_ambient=False,
                   mixing_factor=1.0)   # multiplies conductivity (turbulent mixing, water only)
LAYER_DEFAULTS = {
    "ground": {},
    # paving (concrete/asphalt-like)
    "pathway": dict(thermal_conductivity=1.4, density=2100.0, specific_heat=880.0,
                    albedo=0.25, emissivity=0.92, depth_m=0.15),
    # turf over the same soil as the ground; wetness = fraction of potential
    # evaporation (0 = dry surface, 1 = free water surface)
    "lawn": dict(albedo=0.25, emissivity=0.97, wetness=0.25),
    # shallow water with free-surface evaporation. Still water conducts heat
    # poorly (k = 0.6), so a real pond is mixed by wind and convection; this is
    # represented by an effective conductivity k_eff = mixing_factor * k.
    # mixing_factor = 1 gives still water (surface skin heats strongly).
    "pond": dict(depth_m=0.5, thermal_conductivity=0.6, density=997.0,
                 specific_heat=4182.0, albedo=0.06, emissivity=0.96, wetness=1.0,
                 mixing_factor=30.0),
    # thin canopy skin, back side exchanging with ambient air, partial transpiration
    "tree": dict(depth_m=0.002, nodes=2, thermal_conductivity=0.15, density=700.0,
                 specific_heat=3500.0, albedo=0.20, emissivity=0.96, wetness=0.30,
                 back_follows_ambient=True),
}


def resolve_layer_settings(cls: str, scene: Optional[dict] = None,
                           overrides: Optional[dict] = None) -> tuple:
    """Precedence: built-in default < class default < scene JSON < CLI override.
    Returns (GroundSettings, provenance {property: source})."""
    base = dict(_LAYER_BASE)
    prov = {k: "default" for k in base}
    for k, v in LAYER_DEFAULTS.get(cls, {}).items():
        base[k] = v
        prov[k] = f"default:{cls}"
    json_vals = json_layer_properties(scene, cls)
    cli_vals = {k: v for k, v in (overrides or {}).items() if v is not None}

    def build(use_json):
        d, p = dict(base), dict(prov)
        if use_json:
            for k, v in json_vals.items():
                d[k], p[k] = v, "json"
        for k, v in cli_vals.items():
            d[k], p[k] = v, "cli"
        d["wetness"] = min(max(float(d["wetness"]), 0.0), 1.0)
        if cls == "ground":
            d["wetness"] = 0.0          # ground is dry soil: no evaporation, whatever the JSON says
            p["wetness"] = "dry soil"
        d["nodes"] = int(d["nodes"])
        mixing = float(d.pop("mixing_factor"))
        p.pop("mixing_factor", None)
        if not math.isfinite(mixing) or mixing <= 0:
            raise ValueError("mixing_factor must be finite and positive")
        if mixing != 1.0:
            d["thermal_conductivity"] = float(d["thermal_conductivity"]) * mixing
            p["thermal_conductivity"] = f"{p['thermal_conductivity']} x{mixing:g} mixing"
        return GroundSettings(**d), p

    try:
        return build(True)
    except ValueError as exc:
        warn(f"{cls}: invalid property in JSON ({exc}); falling back to defaults")
        return build(False)


def describe_layer_settings(cls: str, gs: GroundSettings, prov: dict) -> str:
    parts = []
    for key, label, fmt in (("thermal_conductivity", "k", "{:.3g}"), ("density", "rho", "{:.0f}"),
                            ("specific_heat", "cp", "{:.0f}"), ("albedo", "albedo", "{:.2f}"),
                            ("emissivity", "eps", "{:.2f}"), ("depth_m", "depth_m", "{:.3g}"),
                            ("wetness", "wetness", "{:.2f}")):
        parts.append(f"{label}={fmt.format(getattr(gs, key))}[{prov.get(key, 'given')}]")
    return f"[properties] {cls}: " + " ".join(parts)


class BuildingMaterials:
    """Wall/roof properties for OBJ buildings taken from the scene JSON.

    JSON faces are averaged (area-weighted) per surface type. An OBJ building
    is matched to JSON building(s) whose XY bounding box (+1 m) contains its
    centre; unmatched OBJ buildings get the scene-wide wall/roof means.
    Missing properties fall back to DEFAULT_MATERIAL / DEFAULT_ALBEDO_*.
    """
    _KEYS = ("thermal_conductivity", "density", "specific_heat", "albedo", "emissivity")

    def __init__(self, scene: Optional[dict] = None):
        self.defaults = {
            "wall": dict(DEFAULT_MATERIAL, albedo=DEFAULT_ALBEDO_WALL, emissivity=DEFAULT_EMISSIVITY),
            "roof": dict(DEFAULT_MATERIAL, albedo=DEFAULT_ALBEDO_ROOF, emissivity=DEFAULT_EMISSIVITY)}
        self.overall = {k: dict(v) for k, v in self.defaults.items()}
        self.provenance = {"wall": "default", "roof": "default"}
        self.buildings = []
        self.n_matched = 0
        self.n_unmatched = 0
        if isinstance(scene, dict):
            try:
                self._read_scene(scene)
            except Exception as exc:                      # never let JSON quirks stop the run
                warn(f"buildings: could not read materials from JSON ({exc}); using defaults")

    def _merge(self, base: dict, samples: list) -> dict:
        out = dict(base)
        for k in self._KEYS:
            vals = [(a, f[k]) for a, f in samples if k in f]
            if vals:
                out[k] = float(np.average([v for _, v in vals], weights=[a for a, _ in vals]))
        if (any(not math.isfinite(out[k]) or out[k] <= 0 for k in self._KEYS[:3])
                or not 0 <= out["albedo"] <= 1 or not 0 <= out["emissivity"] <= 1):
            warn("buildings: out-of-range material values in JSON ignored")
            return dict(base)
        return out

    def _read_scene(self, scene: dict):
        per_building, overall = [], {"wall": [], "roof": []}
        for bi, b in enumerate(scene.get("buildings", []) or []):
            if not isinstance(b, dict):
                continue
            per, pts = {"wall": [], "roof": []}, []
            for f in b.get("faces", []) or []:
                try:
                    verts = np.array([[v["x"], v["y"], v["z"]] for v in f["vertices"]], float)
                except Exception:
                    continue
                if len(verts) < 3 or not np.isfinite(verts).all():
                    continue
                pts.append(verts)
                kind = {"roof": "roof", "wall": "wall"}.get(str(f.get("surfaceType", "wall")).lower())
                if kind is None:
                    continue
                found = {k: _lookup_property(f, k) for k in self._KEYS}
                found = {k: v for k, v in found.items() if v is not None}
                if found:
                    sample = (max(_polygon_area_3d(verts), 1e-9), found)
                    per[kind].append(sample)
                    overall[kind].append(sample)
            if pts:
                allp = np.vstack(pts)
                per_building.append((str(b.get("id", f"building_{bi}")),
                                     allp.min(axis=0)[:2], allp.max(axis=0)[:2], per))
        for kind in ("wall", "roof"):
            if overall[kind]:
                self.overall[kind] = self._merge(self.defaults[kind], overall[kind])
                self.provenance[kind] = "json"
        for bid, lo, hi, per in per_building:
            entry = {kind: self._merge(self.overall[kind], per[kind]) for kind in ("wall", "roof")}
            self.buildings.append((bid, lo, hi, entry))

    def for_component(self, center_xy) -> dict:
        c = np.asarray(center_xy, float)
        hits = [e for _, lo, hi, e in self.buildings
                if np.all(c >= lo - 1.0) and np.all(c <= hi + 1.0)]
        if not hits:
            self.n_unmatched += 1
            return self.overall
        self.n_matched += 1
        if len(hits) == 1:
            return hits[0]
        return {kind: {k: float(np.mean([h[kind][k] for h in hits])) for k in self._KEYS}
                for kind in ("wall", "roof")}

    def describe(self) -> str:
        parts = []
        for kind in ("wall", "roof"):
            m = self.overall[kind]
            parts.append(f"{kind}: k={m['thermal_conductivity']:.3g} rho={m['density']:.0f} "
                         f"cp={m['specific_heat']:.0f} albedo={m['albedo']:.2f} "
                         f"eps={m['emissivity']:.2f} [{self.provenance[kind]}]")
        return ("[properties] buildings: " + "; ".join(parts) +
                f"; OBJ buildings matched to JSON: {self.n_matched}, using scene mean: {self.n_unmatched}")


# ---- OBJ loading with validation -------------------------------------------

def _clean_mesh(V, F, label: str):
    """Drop invalid / degenerate triangles. Returns (V, F ndarray) or None."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
    n0 = len(F)
    if V.ndim != 2 or len(V) == 0 or n0 == 0:
        warn(f"{label}: no vertices/faces")
        return None
    F = F[((F >= 0) & (F < len(V))).all(axis=1)]
    F = F[np.isfinite(V)[F].all(axis=(1, 2))] if len(F) else F
    if len(F):
        tri = V[F]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        F = F[area > 1e-12]
    if n0 - len(F):
        warn(f"{label}: dropped {n0 - len(F)} invalid/degenerate triangle(s) of {n0}")
    if not len(F):
        warn(f"{label}: no valid triangles left")
        return None
    return V, F


def load_obj_checked(path, label: str):
    """Load one OBJ (or a list, merged). Returns (V, F) or None; never raises."""
    if not path:
        return None
    paths = [path] if isinstance(path, str) else list(path)
    meshes = []
    for p in paths:
        if not os.path.isfile(p):
            warn(f"{label}: file not found, skipping: {p}")
            continue
        try:
            V, F = load_obj_mesh(p)
        except Exception as exc:
            warn(f"{label}: could not parse {p} ({exc})")
            continue
        mesh = _clean_mesh(V, F, f"{label} {os.path.basename(p)}")
        if mesh is not None:
            meshes.append(mesh)
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    offset, Vs, Fs = 0, [], []
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + offset)
        offset += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def _json_trees_mesh(scene: Optional[dict]):
    """Fallback tree geometry (trunk + cone crown) built from JSON tree entries."""
    if not isinstance(scene, dict):
        return None
    polys = []
    for t in scene.get("trees", []) or []:
        if not isinstance(t, dict):
            continue
        try:
            p, _ = _tree_mesh(t)
        except Exception:
            continue
        if p:
            polys.extend(p)
    if not polys:
        return None
    V, F = [], []
    for poly in polys:
        base = len(V)
        V.extend(poly)
        F.extend((base, base + i, base + i + 1) for i in range(1, len(poly) - 1))
    return _clean_mesh(V, F, "trees (from JSON)")


# ---- XY coverage test (remove ground cells hidden under overlay surfaces) --

class XYTriangleIndex:
    """Uniform-grid index of triangles projected on the XY plane."""

    def __init__(self, tri_xyz):
        t = np.asarray(tri_xyz, float)[:, :, :2]
        self.t = t
        lo, hi = t.min(axis=1), t.max(axis=1)
        span = (hi - lo).max(axis=1)
        extent = float((hi.max(axis=0) - lo.min(axis=0)).max())
        self.cell = max(float(np.percentile(span, 90)), extent / 200.0, 1e-6)
        self.origin = lo.min(axis=0)
        i0 = np.floor((lo - self.origin) / self.cell).astype(int)
        i1 = np.floor((hi - self.origin) / self.cell).astype(int)
        self.grid = {}
        for k in range(len(t)):
            for ix in range(i0[k, 0], i1[k, 0] + 1):
                for iy in range(i0[k, 1], i1[k, 1] + 1):
                    self.grid.setdefault((ix, iy), []).append(k)

    def locate(self, pts_xy) -> np.ndarray:
        """Index of a triangle containing each XY point, or -1."""
        pts = np.asarray(pts_xy, float).reshape(-1, 2)
        out = np.full(len(pts), -1, dtype=np.int64)
        cells = np.floor((pts - self.origin) / self.cell).astype(int)
        keys, inv = np.unique(cells, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
        for ki, (cx, cy) in enumerate(keys):
            tris = self.grid.get((int(cx), int(cy)))
            if not tris:
                continue
            idx = order[bounds[ki]:bounds[ki + 1]]
            tt = self.t[tris]
            a = tt[:, 0]
            v0, v1 = tt[:, 1] - a, tt[:, 2] - a
            den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            ok = np.abs(den) > 1e-14
            den = np.where(ok, den, 1.0)
            d = pts[idx][:, None, :] - a[None]
            u = (d[..., 0] * v1[:, 1] - v1[:, 0] * d[..., 1]) / den
            v = (v0[:, 0] * d[..., 1] - d[..., 0] * v0[:, 1]) / den
            inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
            hit = inside.any(axis=1)
            out[idx[hit]] = np.asarray(tris)[inside[hit].argmax(axis=1)]
        return out

    def covers(self, pts_xy) -> np.ndarray:
        return self.locate(pts_xy) >= 0


# ---- lateral conduction between neighbouring surface cells -------------------

HORIZONTAL_CLASSES = ("ground", "pathway", "lawn", "pond")   # classes that conduct sideways


def _neighbour_pairs(V, area, kcond, cen, tol: float = 1e-4):
    """Neighbouring triangles and the lateral contact factor F = Lc/(da/ka + db/kb)
    [W/(m K)] between them (multiply by a node height to get a conductance in W/K).

    Triangles sharing a whole edge (same vertex positions) use that edge exactly.
    Edges used by one triangle only (mesh boundary, next to a removed cell, or
    next to another class that does not share vertices) are probed with three
    points just across the edge, and the triangle found there is the neighbour.
    The contact length is split over the probes; da, db are the distances from
    each centroid to the edge line. Both directions are evaluated and the larger
    value is kept, so non-matching meshes still give a symmetric coupling."""
    N = len(V)
    _, vid = np.unique(np.round(V.reshape(-1, 3) / tol).astype(np.int64), axis=0, return_inverse=True)
    vid = np.asarray(vid).reshape(N, 3)
    nv = int(vid.max()) + 1
    ia, ib = np.array([0, 1, 2]), np.array([1, 2, 0])
    e0, e1 = vid[:, ia].ravel(), vid[:, ib].ravel()
    owner = np.repeat(np.arange(N), 3)
    p0, p1 = V[:, ia, :].reshape(-1, 3), V[:, ib, :].reshape(-1, 3)
    elen = np.linalg.norm(p1 - p0, axis=1)
    key = np.minimum(e0, e1).astype(np.int64) * (nv + 1) + np.maximum(e0, e1)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).reshape(-1)
    per_edge = cnt[inv]

    # -- edges shared by exactly two triangles --------------------------------
    idx = np.flatnonzero(per_edge == 2)
    order = idx[np.argsort(inv[idx], kind="stable")]
    f_i, s_i = order[0::2], order[1::2]
    a, b, L = owner[f_i], owner[s_i], elen[f_i]
    keep = (a != b) & (L > 1e-12)
    a, b, L = a[keep], b[keep], L[keep]
    da, db = 2 * area[a] / (3 * L), 2 * area[b] / (3 * L)
    F_int = L / (da / kcond[a] + db / kcond[b])
    lo_i, hi_i = np.minimum(a, b), np.maximum(a, b)

    # -- boundary edges: probe the cell across the edge -----------------------
    idx = np.flatnonzero(per_edge == 1)
    pa, pb = p0[idx], p1[idx]
    own, L = owner[idx], elen[idx]
    e_xy = (pb - pa)[:, :2]
    lxy = np.linalg.norm(e_xy, axis=1)
    ok = (lxy > 1e-9) & (L > 1e-12)
    idx, pa, pb, own, L, e_xy, lxy = idx[ok], pa[ok], pb[ok], own[ok], L[ok], e_xy[ok], lxy[ok]
    nrm = np.column_stack([e_xy[:, 1], -e_xy[:, 0]]) / lxy[:, None]
    mid = 0.5 * (pa + pb)[:, :2]
    flip = np.einsum("ij,ij->i", mid - cen[own][:, :2], nrm) < 0
    nrm[flip] *= -1
    lo_b = hi_b = np.empty(0, dtype=np.int64)
    F_b = np.empty(0)
    if len(idx):
        index = XYTriangleIndex(V)
        fr = np.array([1 / 6, 1 / 2, 5 / 6])
        base = pa[:, None, :2] + (pb - pa)[:, None, :2] * fr[None, :, None]       # (M, 3, 2)
        found = np.full((len(own), 3), -1, dtype=np.int64)
        # first probe just across the edge; unresolved probes look progressively
        # further (gaps left by removing overlapped cells are up to about one cell wide)
        for dist in (np.clip(0.02 * L, 1e-3, 0.25), 0.25 * L, 0.6 * L, 1.0 * L):
            rows, cols = np.nonzero(found < 0)
            if not len(rows):
                break
            pts = base[rows, cols] + nrm[rows] * dist[rows, None]
            hit = index.locate(pts)
            hit[hit == own[rows]] = -1
            found[rows, cols] = hit
        a_s = np.repeat(own, 3)
        b_s = found.ravel()
        sel = (b_s >= 0) & (b_s != a_s)
        a_s, b_s = a_s[sel], b_s[sel]
        rep = np.repeat(np.arange(len(own)), 3)[sel]
        da_s = 2 * area[a_s] / (3 * L[rep])
        db_s = np.abs(np.einsum("ij,ij->i", cen[b_s][:, :2] - pa[rep][:, :2], nrm[rep]))
        db_s = np.maximum(db_s, 1e-3)
        F_s = (L[rep] / 3) / (da_s / kcond[a_s] + db_s / kcond[b_s])
        lo_s, hi_s = np.minimum(a_s, b_s), np.maximum(a_s, b_s)
        k2 = lo_s.astype(np.int64) * N + hi_s
        uk, inv2 = np.unique(k2, return_inverse=True)
        inv2 = np.asarray(inv2).reshape(-1)
        fwd = a_s < b_s
        s_fwd = np.bincount(inv2, weights=F_s * fwd, minlength=len(uk))
        s_bwd = np.bincount(inv2, weights=F_s * ~fwd, minlength=len(uk))
        F_b = np.maximum(s_fwd, s_bwd)
        lo_b, hi_b = uk // N, uk % N

    lo = np.concatenate([lo_i, lo_b]).astype(np.int64)
    hi = np.concatenate([hi_i, hi_b]).astype(np.int64)
    F = np.concatenate([F_int, F_b])
    if not len(lo):
        return lo, hi, F
    uk, inv3 = np.unique(lo * N + hi, return_inverse=True)
    F = np.bincount(np.asarray(inv3).reshape(-1), weights=F, minlength=len(uk))
    return uk // N, uk % N, F


class LateralCoupler:
    """Sideways heat conduction between neighbouring ground / pathway / lawn / pond
    cells, node by node (same depth), across shared or touching cell edges.

    The vertical slabs are still solved exactly per cell. Between two vertical
    steps the sideways exchange is applied as one implicit (backward Euler) step
    of  C dT/dt = sum_b G_ab (T_b - T_a)  over all coupled nodes, solved with
    Jacobi-preconditioned conjugate gradients (no SciPy needed). It conserves
    energy and is unconditionally stable. Cells with different layer thickness
    exchange heat at equal depth (the finer-resolved cell sets the node height).
    Trees and buildings are not coupled."""

    def __init__(self, layer_faces: dict, layer_settings: dict):
        self.classes = [c for c in HORIZONTAL_CLASSES if layer_faces.get(c)]
        self.offset, self.count, self.nodes = {}, {}, {}
        V, area, kc, cen, rhocp, dxs, nn, thick, cid = [], [], [], [], [], [], [], [], []
        n = 0
        for ci, cls in enumerate(self.classes):
            faces, gs = layer_faces[cls], layer_settings[cls]
            self.offset[cls], self.count[cls], self.nodes[cls] = n, len(faces), gs.nodes
            n += len(faces)
            V.append(np.array([f.vertices for f in faces], float))
            area.append(np.array([f.area for f in faces], float))
            kc.append(np.array([f.thermal_conductivity for f in faces], float))
            cen.append(np.array([f.center for f in faces], float))
            rhocp.append(np.array([f.density * f.specific_heat for f in faces], float))
            dxs.append(np.full(len(faces), gs.depth_m / (gs.nodes - 1)))
            nn.append(np.full(len(faces), gs.nodes))
            thick.append(np.full(len(faces), gs.depth_m))
            cid.append(np.full(len(faces), ci))
        self.n_cells = n
        V, area, kc, cen = map(np.concatenate, (V, area, kc, cen))
        rhocp, dxs, nn, thick, cid = map(np.concatenate, (rhocp, dxs, nn, thick, cid))
        self.area = area
        nmax = int(nn.max())
        self.nmax = nmax
        lo, hi, F = _neighbour_pairs(V, area, kc, cen)
        self.n_pairs = len(lo)
        self.pair_counts = {}
        for ca in range(len(self.classes)):
            for cb in range(ca, len(self.classes)):
                m = ((cid[lo] == ca) & (cid[hi] == cb)) | ((cid[lo] == cb) & (cid[hi] == ca))
                if m.any():
                    self.pair_counts[(self.classes[ca], self.classes[cb])] = int(m.sum())

        # node capacities (J/K); padding nodes of coarser slabs get a dummy 1
        width = np.ones((n, nmax))
        for j in range(nmax):
            w = dxs * np.where((j == 0) | (j == nn - 1), 0.5, 1.0)
            width[:, j] = np.where(j < nn, w, 0.0)
        cap = rhocp[:, None] * width * area[:, None]
        self.cap = np.where(np.arange(nmax)[None, :] < nn[:, None], cap, 1.0).ravel()

        # node-level conductances; the finer-resolved cell of a pair sets the node height
        ea, eb, gg = [], [], []
        if len(lo):
            fine_is_lo = dxs[lo] <= dxs[hi]
            f = np.where(fine_is_lo, lo, hi)
            c = np.where(fine_is_lo, hi, lo)
            for j in range(nmax):
                z = j * dxs[f]
                ok = (j < nn[f]) & (z <= thick[c] + 1e-9)
                if not ok.any():
                    continue
                jc = np.clip(np.rint(z[ok] / dxs[c][ok]).astype(int), 0, nn[c][ok] - 1)
                h = dxs[f][ok] * (0.5 if j == 0 else 1.0)
                h = np.where(j == nn[f][ok] - 1, 0.5 * dxs[f][ok], h)
                ea.append(f[ok] * nmax + j)
                eb.append(c[ok] * nmax + jc)
                gg.append(F[ok] * h)
        if ea:
            self.ea, self.eb, self.g = np.concatenate(ea), np.concatenate(eb), np.concatenate(gg)
        else:
            self.ea = self.eb = np.empty(0, dtype=np.int64)
            self.g = np.empty(0)
        size = n * nmax
        self.deg = (np.bincount(self.ea, weights=self.g, minlength=size)
                    + np.bincount(self.eb, weights=self.g, minlength=size))
        self.last_max_flux = 0.0

    def describe(self) -> str:
        kinds = ", ".join(f"{a}-{b}: {n}" if a != b else f"{a}: {n}"
                          for (a, b), n in sorted(self.pair_counts.items()))
        return (f"[lateral] {self.n_cells} cells coupled, {self.n_pairs} neighbour pairs "
                f"({kinds if kinds else 'none'})")

    def _laplacian(self, x):
        return self.deg * x - (np.bincount(self.ea, weights=self.g * x[self.eb], minlength=len(x))
                               + np.bincount(self.eb, weights=self.g * x[self.ea], minlength=len(x)))

    def apply(self, slabs: dict, dt: float) -> None:
        """One implicit sideways step of length dt (s) on the slabs' temperatures."""
        if dt <= 0 or not len(self.g):
            return
        T = np.zeros((self.n_cells, self.nmax))
        for cls in self.classes:
            o, n, m = self.offset[cls], self.count[cls], self.nodes[cls]
            T[o:o + n, :m] = slabs[cls].temperatures
        x0 = T.ravel()
        rhs = -dt * self._laplacian(x0)
        d = np.zeros_like(x0)
        rnorm0 = float(np.linalg.norm(rhs))
        if rnorm0 > 0:
            diag = self.cap + dt * self.deg
            r, z = rhs.copy(), rhs / diag
            p, rz = z.copy(), float(r @ z)
            for _ in range(500):
                Ap = self.cap * p + dt * self._laplacian(p)
                alpha = rz / float(p @ Ap)
                d += alpha * p
                r -= alpha * Ap
                if np.linalg.norm(r) <= 1e-11 * rnorm0:
                    break
                z = r / diag
                rz_new = float(r @ z)
                p = z + (rz_new / rz) * p
                rz = rz_new
        change = d.reshape(self.n_cells, self.nmax)
        heat = (self.cap.reshape(self.n_cells, self.nmax) * change).sum(axis=1) / dt   # W per cell
        flux = heat / self.area
        self.last_max_flux = max(self.last_max_flux, float(np.abs(flux).max()))
        newT = T + change
        for cls in self.classes:
            o, n, m = self.offset[cls], self.count[cls], self.nodes[cls]
            slabs[cls].temperatures[:] = newT[o:o + n, :m]
            slabs[cls].lateral_flux_Wm2 = flux[o:o + n].copy()


def _run_lockstep(coupler: LateralCoupler, jobs: dict) -> dict:
    """Step several surface classes hour by hour so the sideways coupling can act
    between steps. jobs = {class: (args, kwargs)} for _simulate_steps."""
    gens = {cls: _simulate_steps(*a, lockstep=True, **kw) for cls, (a, kw) in jobs.items()}
    pending, done = {}, {}

    def advance(cls):
        try:
            pending[cls] = next(gens[cls])
        except StopIteration as stop:
            done[cls] = stop.value
            pending.pop(cls, None)

    for cls in gens:
        advance(cls)
    while pending:
        if set(pending) != set(gens):
            raise RuntimeError("surface classes fell out of step")
        dts = {round(p[0], 6) for p in pending.values()}
        if len(dts) != 1:
            raise RuntimeError("surface classes have different time steps")
        coupler.apply({cls: p[1] for cls, p in pending.items()}, dts.pop())
        for cls in list(pending):
            advance(cls)
    return done


# ---- building faces from OBJ ----------------------------------------------

def build_faces_from_obj(buildings_obj_path=None, materials: Optional[BuildingMaterials] = None,
                         mesh=None) -> tuple:
    """Recover buildings, retaining every nondegenerate triangle as a cell.

    Coplanar groups provide parent wall IDs and approximate bulk volumes only;
    visibility, irradiance, temperature and VTK values belong to each triangle.
    Wall/roof materials come from `materials` (JSON-derived when available).
    Returns (faces, capacitances, face_triangles, occluders); the last is kept
    for API compatibility and is empty (rays use the BVH built from the mesh).
    """
    if mesh is None:
        mesh = load_obj_checked(buildings_obj_path, "buildings")
    if mesh is None:
        return [], {}, {}, {}
    V, Farr = mesh
    materials = materials or BuildingMaterials()
    F = [tuple(t) for t in Farr.tolist()]
    faces, face_triangles, capacitances = [], {}, {}
    for b_idx, tri_indices in enumerate(_obj_components_by_shared_position(V, F)):
        bid = f"building_{b_idx:02d}"
        comp_tris = [F[t] for t in tri_indices]
        groups = group_coplanar(V, comp_tris)
        if not groups:
            continue
        z_min = min(g["z_min"] for g in groups.values())
        z_max = max(g["z_max"] for g in groups.values())
        pts = V[np.unique(np.asarray(comp_tris).ravel())]
        mats = materials.for_component(0.5 * (pts.min(axis=0) + pts.max(axis=0))[:2])
        wm = mats["wall"]
        roof_area = sum(g["area"] * abs(g["normal"][2]) for g in groups.values()
                        if abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
                        and g["center"][2] > z_min + 1e-6)
        capacitances[bid] = wm["density"] * roof_area * (z_max - z_min) * wm["specific_heat"]
        for gi, g in enumerate(groups.values()):
            horizontal = abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
            bottom = horizontal and g["z_max"] <= z_min + 1e-6
            stype = "floor" if bottom else ("roof" if horizontal else "wall")
            normal = g["normal"].copy()
            if (stype == "roof" and normal[2] < 0) or (stype == "floor" and normal[2] > 0):
                normal = -normal
            tilt = math.degrees(math.acos(np.clip(normal[2], -1, 1)))
            parent = f"{bid}_face_{gi:03d}_{stype}"
            m = mats["roof" if stype == "roof" else "wall"]
            for ti, tri in enumerate(g["triangles"]):
                verts = V[list(tri)]
                _, area = _triangle_normal_area(*verts)
                fid = f"{parent}_tri_{ti:06d}"
                faces.append(Face(
                    building_id=bid, face_id=fid, parent_face_id=parent, surface_type=stype,
                    normal=normal, area=area, tilt_deg=tilt, center=verts.mean(axis=0),
                    vertices=verts, thermal_conductivity=m["thermal_conductivity"],
                    density=m["density"], specific_heat=m["specific_heat"],
                    albedo=m["albedo"], emissivity=m["emissivity"]))
                face_triangles[fid] = [tri]
    return faces, capacitances, face_triangles, {}


# ---- ground / pathway / lawn / pond / tree cells ---------------------------

def _layer_geometry(V, F, orient: str):
    """Per-triangle vertices, unit normals and areas. orient='up' flips normals
    to point upward; orient='outward' flips them away from the centre of each
    connected piece (closed tree proxies)."""
    tri = V[F].copy()
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    area = 0.5 * norm
    nrm = n / norm[:, None]
    if orient == "up":
        flip = nrm[:, 2] < 0
    else:
        flip = np.zeros(len(F), dtype=bool)
        centroids = tri.mean(axis=1)
        for comp in _obj_components_by_shared_position(V, F):
            idx = np.asarray(comp)
            centre = tri[idx].reshape(-1, 3).mean(axis=0)
            flip[idx] = np.einsum("ij,ij->i", nrm[idx], centroids[idx] - centre) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    nrm[flip] *= -1
    return tri, nrm, area


def build_layer_faces(cls: str, V, F, settings: GroundSettings, building_tracer=None,
                      higher_index: Optional[XYTriangleIndex] = None,
                      orient: str = "up") -> list:
    """One Face per mesh triangle. Cells under a building roof (vertical ray)
    and cells whose centroid lies under a higher-priority surface (e.g. ground
    under a pond) are removed so no point is solved twice."""
    tri, nrm, area = _layer_geometry(V, F, orient)
    cen = tri.mean(axis=1)
    keep = np.ones(len(tri), dtype=bool)
    n_under = n_cov = 0
    if building_tracer is not None and cls != "tree":
        blocked = building_tracer.blocked(cen + np.array([0.0, 0.0, _SHADOW_RAY_OFFSET]),
                                          np.array([0.0, 0.0, 1.0]))
        n_under = int(blocked.sum())
        keep &= ~blocked
    if higher_index is not None and cls != "tree":
        covered = higher_index.covers(cen[:, :2])
        n_cov = int((covered & keep).sum())
        keep &= ~covered
    faces = []
    for ti in np.flatnonzero(keep):
        faces.append(Face(
            building_id=cls, face_id=f"{cls}_tri_{ti:06d}", parent_face_id=f"{cls}_surface",
            surface_type=cls, normal=nrm[ti], area=float(area[ti]),
            tilt_deg=math.degrees(math.acos(np.clip(nrm[ti][2], -1, 1))),
            center=cen[ti], vertices=tri[ti],
            thermal_conductivity=settings.thermal_conductivity, density=settings.density,
            specific_heat=settings.specific_heat, albedo=settings.albedo,
            emissivity=settings.emissivity))
    log(f"[input] {cls}: {len(faces)} cells solved "
        f"({n_under} under buildings and {n_cov} under other surfaces removed), "
        f"area {sum(f.area for f in faces):.1f} m2")
    return faces


@dataclass
class ObjBundle:
    building_faces: list
    capacitances: dict
    layer_faces: dict
    layer_settings: dict
    tracer: object
    wall_settings: WallSettings
    lateral: object = None

    @property
    def n_cells(self) -> int:
        return len(self.building_faces) + sum(len(v) for v in self.layer_faces.values())

    def all_faces(self) -> list:
        out = list(self.building_faces)
        for cls in LAYER_CLASSES:
            out += self.layer_faces.get(cls, [])
        return out


def prepare_obj_bundle(buildings_obj=None, ground_obj=None, trees_obj=None, pathway_obj=None,
                       lawn_obj=None, pond_obj=None, scene: Optional[dict] = None,
                       wall_settings: Optional[WallSettings] = None,
                       layer_overrides: Optional[dict] = None,
                       fixed_layer_settings: Optional[dict] = None,
                       consider_shadows: bool = True, remove_overlaps: bool = True,
                       lateral_conduction: bool = True) -> ObjBundle:
    """Load every available component once (geometry, materials, BVH). The
    result is reused for every weather day."""
    wall_settings = wall_settings or WallSettings()
    layer_overrides = layer_overrides or {}
    fixed_layer_settings = fixed_layer_settings or {}
    paths = dict(ground=ground_obj, pathway=pathway_obj, lawn=lawn_obj, pond=pond_obj, tree=trees_obj)
    meshes = {}
    for cls, p in paths.items():
        mesh = load_obj_checked(p, f"{cls} OBJ") if p else None
        if mesh is None and cls == "tree":
            mesh = _json_trees_mesh(scene)
            if mesh is not None:
                log("[input] tree: no usable trees OBJ, using tree geometry from the JSON")
        if mesh is not None:
            meshes[cls] = mesh
            log(f"[input] {cls}: {len(mesh[1])} valid triangles")
        elif p:
            warn(f"{cls}: nothing solved ({p})")
    bmesh = load_obj_checked(buildings_obj, "buildings OBJ") if buildings_obj else None

    materials = BuildingMaterials(scene)
    building_faces, capacitances = [], {}
    if bmesh is not None:
        building_faces, capacitances, _, _ = build_faces_from_obj(mesh=bmesh, materials=materials)
        n_b = len(capacitances)
        log(f"[input] buildings: {n_b} building(s), {len(building_faces)} cells")
        log(materials.describe())
    elif buildings_obj:
        warn("buildings: nothing solved")

    b_tris = bmesh[0][bmesh[1]] if bmesh is not None else np.empty((0, 3, 3))
    building_tracer = TriangleRayTracer(b_tris) if len(b_tris) else None
    tracer = None
    if consider_shadows:
        tree_tris = meshes["tree"][0][meshes["tree"][1]] if "tree" in meshes else np.empty((0, 3, 3))
        if len(tree_tris):
            tracer = TriangleRayTracer(np.concatenate([b_tris, tree_tris]) if len(b_tris) else tree_tris)
        else:
            tracer = building_tracer
        if tracer is not None:
            log(f"[shadows] occluders: {len(b_tris)} building + {len(tree_tris)} tree triangles")
        else:
            log("[shadows] no occluders present: only self-orientation (cos) shading applies")

    tri_cache = {cls: m[0][m[1]] for cls, m in meshes.items()}
    layer_faces, layer_settings = {}, {}
    for cls in LAYER_CLASSES:
        if cls not in meshes:
            continue
        if cls in fixed_layer_settings:
            gs, prov = fixed_layer_settings[cls], {}
        else:
            gs, prov = resolve_layer_settings(cls, scene, layer_overrides.get(cls))
        log(describe_layer_settings(cls, gs, prov))
        higher = None
        if remove_overlaps and cls in LAYER_PRIORITY:
            above = [tri_cache[c] for c in LAYER_PRIORITY[:LAYER_PRIORITY.index(cls)] if c in tri_cache]
            if above:
                higher = XYTriangleIndex(np.concatenate(above))
        V, F = meshes[cls]
        faces = build_layer_faces(cls, V, F, gs, building_tracer, higher,
                                  "outward" if cls == "tree" else "up")
        if faces:
            layer_faces[cls], layer_settings[cls] = faces, gs
        else:
            warn(f"{cls}: no cells left after filtering; skipped")
    coupler = None
    if lateral_conduction and any(layer_faces.get(c) for c in HORIZONTAL_CLASSES):
        try:
            coupler = LateralCoupler(layer_faces, layer_settings)
            log(coupler.describe())
            if not coupler.n_pairs:
                coupler = None
        except Exception as exc:                      # never let coupling stop the run
            warn(f"lateral conduction disabled ({exc!r})")
            coupler = None
    return ObjBundle(building_faces, capacitances, layer_faces, layer_settings, tracer,
                     wall_settings, coupler)


# ---- weather -> per-record dicts (adds humidity / pressure when available) --

def _series(values, n):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    return arr if arr.ndim == 1 and len(arr) == n else None


def records_from_weather(wx: dict) -> list:
    n = len(wx["hour"])
    p = _series(wx.get("atmosphericPressure"), n)
    if p is not None:                       # accept Pa, hPa or kPa
        med = float(np.nanmedian(p))
        p = p * (1000.0 if med < 200 else 100.0 if med < 2000 else 1.0)
    w_amb = _series(wx.get("wambient"), n)
    rh = _series(wx.get("relativeHumidity"), n)
    if rh is not None and float(np.nanmax(rh)) > 1.5:
        rh = rh / 100.0
    records = []
    for i in range(n):
        rec = dict(hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
                   sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
                   i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i])
        if p is not None and np.isfinite(p[i]):
            rec["p_atm"] = float(p[i])
        if w_amb is not None and np.isfinite(w_amb[i]) and 0 <= w_amb[i] < 0.1:
            rec["w_amb"] = float(w_amb[i])
        if rh is not None and np.isfinite(rh[i]):
            rec["rh"] = float(np.clip(rh[i], 0.0, 1.0))
        records.append(rec)
    return records


def _saturation_pressure_pa(t_c: float) -> float:
    return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


def _humidity_ratio(t_c: float, rh: float, p_pa: float) -> float:
    pv = min(rh * _saturation_pressure_pa(t_c), 0.99 * p_pa)
    return 0.622 * pv / (p_pa - pv)


def _saturation_humidity_ratio(t_c: float, p_pa: float) -> tuple:
    """(w_sat, d w_sat / dT) at temperature t_c."""
    es = min(_saturation_pressure_pa(t_c), 0.99 * p_pa)
    des = es * 17.625 * 243.04 / (t_c + 243.04) ** 2
    return 0.622 * es / (p_pa - es), 0.622 * p_pa * des / (p_pa - es) ** 2


def sunlit_fractions_from_arrays(normals, verts, sun_vec, ray_tracer, samples=3):
    """Array version of triangle_sunlit_fraction (no per-call Python loops)."""
    fractions = np.zeros(len(normals))
    if sun_vec[2] <= 0:
        return fractions
    eligible = np.flatnonzero(normals @ sun_vec > 0)
    if not len(eligible):
        return fractions
    weights = np.array([[1/3, 1/3, 1/3]]) if samples == 1 else np.array([
        [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    origins = np.einsum('sj,tjk->tsk', weights, verts[eligible])
    origins += normals[eligible, None, :] * _SHADOW_RAY_OFFSET
    hits = ray_tracer.blocked(origins.reshape(-1, 3), sun_vec).reshape(-1, samples)
    fractions[eligible] = 1 - hits.mean(axis=1)
    return fractions


def _tag_layer_columns(df: pd.DataFrame) -> None:
    """Non-building classes: expose the slab fields under the ground_* names
    (they hold the sub-surface layer of whatever class `surface_group` says)
    and blank the building-only columns."""
    df["ground_depth_m"] = df["slab_thickness_m"]
    df["T_ground_bottom_C"] = df["T_back_surface_C"]
    df["T_deep_ground_C"] = df["T_back_boundary_C"]
    df["h_deep_ground_Wm2K"] = df["h_back_Wm2K"]
    df["q_deep_ground_out_Wm2"] = df["q_back_out_Wm2"]
    df["T_ground_mean_C"] = df["T_slab_mean_C"]
    df["ground_capacity_Jm2K"] = df["slab_capacity_Jm2K"]
    df["ground_capacity_JK"] = df["slab_capacity_JK"]
    building_only = ["wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K", "T_inner_C",
                     "T_wall_mean_C", "q_indoor_out_Wm2", "wall_capacity_Jm2K", "wall_capacity_JK"]
    df[building_only] = np.nan


def solve_obj_day(bundle: ObjBundle, weather_dir: str, t_initial_c: Optional[float] = None,
                  ground_initial_c: Optional[float] = None,
                  ground_albedo: float = DEFAULT_GROUND_ALBEDO,
                  shadow_samples: int = 3, workers: int = 1) -> tuple:
    """Solve every component in the bundle for one weather folder.
    Returns (face_df, building_df, faces)."""
    wx = load_weather_folder(weather_dir)
    records = records_from_weather(wx)
    frames, bframes, faces_out = [], [], []
    if bundle.building_faces:
        fdf, bdf = _simulate(bundle.building_faces, bundle.capacitances, records,
                             ground_albedo=ground_albedo, t_initial_c=t_initial_c,
                             ray_tracer=bundle.tracer, shadow_samples=shadow_samples,
                             wall_settings=bundle.wall_settings, workers=workers)
        frames.append(fdf)
        bframes.append(bdf)
        faces_out += bundle.building_faces
    coupler = bundle.lateral
    coupled = set(coupler.classes) if coupler is not None else set()
    jobs, results = {}, {}
    for cls in LAYER_CLASSES:
        faces = bundle.layer_faces.get(cls)
        if not faces:
            continue
        gs = bundle.layer_settings[cls]
        cap = gs.density * gs.specific_heat * gs.depth_m * sum(f.area for f in faces)
        init = records[0]["T_amb_C"] if cls == "tree" else ground_initial_c
        args = (faces, {cls: cap}, records)
        kwargs = dict(ground_albedo=ground_albedo, t_initial_c=init, ray_tracer=bundle.tracer,
                      shadow_samples=shadow_samples, wall_settings=gs.slab_settings(),
                      surface_group=cls, wetness=gs.wetness,
                      back_follows_ambient=gs.back_follows_ambient, workers=workers)
        if cls in coupled:
            jobs[cls] = (args, kwargs)
        else:
            results[cls] = _simulate(*args, **kwargs)
    if jobs:
        coupler.last_max_flux = 0.0
        results.update(_run_lockstep(coupler, jobs))
        log(f"[lateral] largest sideways heat flux in this day: {coupler.last_max_flux:.3g} W/m2")
    for cls in LAYER_CLASSES:
        if cls not in results:
            continue
        ldf, _ = results[cls]
        _tag_layer_columns(ldf)
        frames.append(ldf)
        faces_out += bundle.layer_faces[cls]
    if not frames:
        raise ValueError("no solvable components in this scene")
    face_df = pd.concat(frames, ignore_index=True, sort=False)
    building_df = (pd.concat(bframes, ignore_index=True) if bframes else
                   pd.DataFrame(columns=["building_id", "hour", "T_amb_C", "T_eq_C",
                                         "h_eff_WK", "tau_hr", "T_building_C"]))
    return face_df, building_df, faces_out


def run_analysis_from_obj(buildings_obj_path=None, weather_dir: Optional[str] = None,
                          trees_obj_path: Optional[str] = None, consider_shadows: bool = True,
                          t_initial_c: Optional[float] = None, shadow_samples: int = 3,
                          wall_settings: Optional[WallSettings] = None,
                          ground_obj_path: Optional[str] = None,
                          ground_initial_c: Optional[float] = None,
                          ground_settings: Optional[GroundSettings] = None,
                          pathway_obj_path: Optional[str] = None,
                          lawn_obj_path: Optional[str] = None,
                          pond_obj_path: Optional[str] = None,
                          scene=None, workers: int = 1, lateral_conduction: bool = True) -> tuple:
    """Convenience wrapper: prepare + solve one weather folder.
    Returns (face_df, building_df, faces)."""
    if shadow_samples not in (1, 3):
        raise ValueError("shadow_samples must be 1 or 3")
    bundle = prepare_obj_bundle(
        buildings_obj_path, ground_obj_path, trees_obj_path, pathway_obj_path, lawn_obj_path,
        pond_obj_path, scene=_scene_from(scene) if scene is not None else None,
        wall_settings=wall_settings,
        fixed_layer_settings={"ground": ground_settings} if ground_settings else None,
        consider_shadows=consider_shadows, lateral_conduction=lateral_conduction)
    return solve_obj_day(bundle, weather_dir, t_initial_c=t_initial_c,
                         ground_initial_c=ground_initial_c, shadow_samples=shadow_samples,
                         workers=workers)


def export_vtk_series_from_analysis(analysis: tuple, out_dir: str) -> str:
    """Write one legacy-VTK file per hour (+ scene.pvd) with per-cell CELL_DATA
    for every solved component. T_C aliases T_surface_C. surface_type_id:
    0 wall, 1 roof, 2 floor, 3 ceiling, 5 tree, 6 ground, 7 pathway, 8 lawn,
    9 pond. Tree/pond/lawn cells also set is_tree/is_pond/is_lawn flags."""
    face_df, building_df, faces = analysis
    os.makedirs(out_dir, exist_ok=True)
    polygons = [np.asarray(f.vertices, float) for f in faces]
    ids = [f.face_id for f in faces]
    building_ids = dict.fromkeys(f.building_id for f in faces if f.surface_type not in LAYER_CLASSES)
    bmap = {bid: i for i, bid in enumerate(building_ids)}
    pmap = {pid: i for i, pid in enumerate(dict.fromkeys(f.parent_face_id for f in faces))}
    static = {
        "surface_type_id": ([_SURFACE_TYPE_ID.get(f.surface_type, -1) for f in faces], "int"),
        "is_building": ([int(f.surface_type not in LAYER_CLASSES) for f in faces], "int"),
        "building_index": ([bmap.get(f.building_id, -1) for f in faces], "int"),
        "parent_face_index": ([pmap[f.parent_face_id] for f in faces], "int"),
    }
    for cls in LAYER_CLASSES:
        static[f"is_{cls}"] = ([int(f.surface_type == cls) for f in faces], "int")
    fields = ["T_surface_C", "T_solair_C", "I_direct_Wm2", "I_total_Wm2",
              "sunlit_fraction", "shadow_fraction", "shadowed", "sunlit",
              "h_conv_Wm2K", "q_solar_absorbed_Wm2", "q_conv_out_Wm2", "q_longwave_out_Wm2",
              "q_evap_out_Wm2", "q_lateral_in_Wm2", "T_inner_C", "T_wall_mean_C", "q_conduction_in_Wm2",
              "q_indoor_out_Wm2", "q_storage_Wm2", "q_outer_node_storage_Wm2",
              "stored_heat_Jm2", "stored_heat_J", "wall_capacity_Jm2K", "wall_capacity_JK",
              "wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K", "T_back_surface_C",
              "T_slab_mean_C", "q_back_out_Wm2", "slab_capacity_Jm2K", "slab_capacity_JK",
              "slab_thickness_m", "T_back_boundary_C", "h_back_Wm2K", "ground_depth_m",
              "T_ground_bottom_C", "T_deep_ground_C", "h_deep_ground_Wm2K",
              "q_deep_ground_out_Wm2", "T_ground_mean_C", "ground_capacity_Jm2K",
              "ground_capacity_JK"]
    fields = [name for name in fields if name in face_df.columns]
    entries = []
    for h, frame in face_df.groupby("hour", sort=True):
        amb_c = float(frame["T_amb_C"].iloc[0])
        frame = frame.set_index("face_id").reindex(ids)
        scalars = dict(static)
        for name in fields:
            fill = amb_c if name.startswith("T_") else 0
            dtype = "int" if name in ("shadowed", "sunlit") else "float"
            values = frame[name].fillna(fill).to_numpy(dtype=int if dtype == "int" else float)
            scalars[name] = (values, dtype)
        scalars["T_C"] = scalars["T_surface_C"]
        fname = f"frame_{int(round(h * 100)):04d}.vtk"
        _write_vtk_polydata(os.path.join(out_dir, fname), polygons, scalars)
        entries.append((float(h), fname))
    pvd_path = os.path.join(out_dir, "scene.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for t, fname in entries:
            f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')
    log(f"[vtk export] Wrote {len(entries)} frame(s) + scene.pvd to {out_dir}")
    return pvd_path


def export_vtk_series_from_obj(buildings_obj_path, weather_dir: str, out_dir: str,
                               trees_obj_path: Optional[str] = None, consider_shadows: bool = True,
                               shadow_samples: int = 3, analysis=None,
                               wall_settings: Optional[WallSettings] = None,
                               t_initial_c: Optional[float] = None,
                               ground_obj_path: Optional[str] = None,
                               ground_settings: Optional[GroundSettings] = None,
                               ground_initial_c: Optional[float] = None,
                               pathway_obj_path: Optional[str] = None,
                               lawn_obj_path: Optional[str] = None,
                               pond_obj_path: Optional[str] = None, scene=None) -> str:
    """Backward-compatible wrapper: run (unless `analysis` is given) and export."""
    if analysis is None:
        analysis = run_analysis_from_obj(
            buildings_obj_path, weather_dir, trees_obj_path=trees_obj_path,
            consider_shadows=consider_shadows, t_initial_c=t_initial_c,
            shadow_samples=shadow_samples, wall_settings=wall_settings,
            ground_obj_path=ground_obj_path, ground_initial_c=ground_initial_c,
            ground_settings=ground_settings, pathway_obj_path=pathway_obj_path,
            lawn_obj_path=lawn_obj_path, pond_obj_path=pond_obj_path, scene=scene)
    return export_vtk_series_from_analysis(analysis, out_dir)


# ---- fast legacy-VTK writer (falls back to the slow one for mixed polygons) --

VTK_BINARY = False      # set by --vtk_binary: big-endian legacy BINARY files (much faster/smaller)


def _write_vtk_polydata_binary(path, arr, cell_scalars) -> None:
    n_poly, n_v, _ = arr.shape
    conn = np.concatenate([np.full((n_poly, 1), n_v), np.arange(n_poly * n_v).reshape(n_poly, n_v)],
                          axis=1).astype(">i4")
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nsolar_thermal_model export\nBINARY\nDATASET POLYDATA\n")
        f.write(f"POINTS {n_poly * n_v} float\n".encode())
        f.write(arr.reshape(-1, 3).astype(">f4").tobytes())
        f.write(f"\nPOLYGONS {n_poly} {n_poly * (n_v + 1)}\n".encode())
        f.write(conn.tobytes())
        f.write(f"\nCELL_DATA {n_poly}\n".encode())
        for name, (values, dtype) in cell_scalars.items():
            f.write(f"SCALARS {name} {dtype} 1\nLOOKUP_TABLE default\n".encode())
            if dtype == "int":
                f.write(np.asarray(values).astype(">i4").tobytes())
            else:
                f.write(np.nan_to_num(np.asarray(values, dtype=float)).astype(">f4").tobytes())
            f.write(b"\n")


def _write_vtk_polydata(path: str, polygons: list, cell_scalars: dict) -> None:
    try:
        arr = np.asarray(polygons, dtype=float)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("not a uniform polygon array")
    except (ValueError, TypeError):
        return _write_vtk_polydata_slow(path, polygons, cell_scalars)
    if VTK_BINARY:
        return _write_vtk_polydata_binary(path, arr, cell_scalars)
    n_poly, n_v, _ = arr.shape
    conn = np.concatenate([np.full((n_poly, 1), n_v), np.arange(n_poly * n_v).reshape(n_poly, n_v)], axis=1)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nsolar_thermal_model export\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {n_poly * n_v} float\n")
        np.savetxt(f, arr.reshape(-1, 3), fmt="%.4f")
        f.write(f"POLYGONS {n_poly} {n_poly * (n_v + 1)}\n")
        np.savetxt(f, conn, fmt="%d")
        f.write(f"CELL_DATA {n_poly}\n")
        for name, (values, dtype) in cell_scalars.items():
            f.write(f"SCALARS {name} {dtype} 1\nLOOKUP_TABLE default\n")
            if dtype == "int":
                np.savetxt(f, np.asarray(values).astype(np.int64), fmt="%d")
            else:
                np.savetxt(f, np.nan_to_num(np.asarray(values, dtype=float)), fmt="%.4f")


def class_summary(face_df: pd.DataFrame) -> pd.DataFrame:
    """Area-weighted hourly means per surface_group / surface_type."""
    cols = ["T_surface_C", "I_total_Wm2", "sunlit_fraction", "q_evap_out_Wm2"]
    d = face_df[["surface_group", "surface_type", "hour", "area_m2"] + cols].copy()
    for c in cols:
        d[c] = d[c] * d["area_m2"]
    keys = ["surface_group", "surface_type", "hour"]
    g = d.groupby(keys, as_index=False).sum(numeric_only=True)
    for c in cols:
        g[c] = g[c] / g["area_m2"]
    mm = face_df.groupby(keys)["T_surface_C"].agg(["min", "max"]).reset_index()
    g = g.merge(mm, on=keys)
    return g.rename(columns={"T_surface_C": "T_surface_mean_C", "I_total_Wm2": "I_total_mean_Wm2",
                             "sunlit_fraction": "sunlit_fraction_mean",
                             "q_evap_out_Wm2": "q_evap_out_mean_Wm2",
                             "min": "T_surface_min_C", "max": "T_surface_max_C"})


# ==========================================================================
# 10. Plots
# ==========================================================================

def plot_results(face_df: pd.DataFrame, building_df: pd.DataFrame, out_prefix: str,
                  building_id: Optional[str] = None):
    import matplotlib.pyplot as plt

    bldg_ids = building_df["building_id"].unique()
    single = building_id if building_id is not None else (bldg_ids[0] if len(bldg_ids) == 1 else None)

    face_sub = face_df if single is None else face_df[face_df["building_id"] == single]
    bldg_sub = building_df if single is None else building_df[building_df["building_id"] == single]
    # Plot area-weighted parent walls, avoiding one curve per mesh triangle.
    if "parent_face_id" in face_sub and face_sub.parent_face_id.nunique() < face_sub.face_id.nunique():
        keys = ["building_id", "parent_face_id", "hour"]
        values = ["I_total_Wm2", "T_solair_C", "T_surface_C"]
        weighted = face_sub[keys + ["area_m2"]].copy()
        for name in values:
            weighted[name] = face_sub[name] * face_sub.area_m2
        grouped = weighted.groupby(keys, as_index=False).sum()
        for name in values:
            grouped[name] /= grouped.area_m2
        face_sub = grouped.rename(columns={"parent_face_id": "face_id"})

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    # --- Panel 1: incident solar flux per face vs hour (shows exposure windows) ---
    ax = axes[0]
    for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
        sub = sub.sort_values("hour")
        lab = fid if single is not None else f"{bid}:{fid}"
        ax.plot(sub["hour"], sub["I_total_Wm2"], marker="o", ms=3, label=lab)
    ax.set_ylabel("Incident solar flux, W/m$^2$")
    title = "Per-face solar exposure through the day (sunlit windows)"
    ax.set_title(title if single is None else f"{title} -- {single}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # --- Panel 2: surface temperature + ambient + legacy bulk estimate ---
    ax = axes[1]
    for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
        sub = sub.sort_values("hour")
        lab = f"{fid} (surface)" if single is not None else f"{bid}:{fid} (surface)"
        ax.plot(sub["hour"], sub["T_surface_C"], marker="o", ms=3, label=lab)
    ax.plot(bldg_sub.groupby("hour")["T_amb_C"].first().index,
            bldg_sub.groupby("hour")["T_amb_C"].first().values,
            "k--", lw=2, label="T_amb (air)")
    for bid, sub in bldg_sub.groupby("building_id"):
        sub = sub.sort_values("hour")
        lab = "T_building (lumped, RC)" if single is not None else f"T_building {bid} (RC)"
        ax.plot(sub["hour"], sub["T_building_C"], lw=2.5, label=lab)
    ax.set_xlabel("Hour of day (local standard time)")
    ax.set_ylabel("Temperature, \u00b0C")
    ax.set_title("Surface temperature per face vs. ambient vs. building bulk temperature")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_plot.png", dpi=150)
    print(f"Wrote {out_prefix}_plot.png")


# ==========================================================================
# 11. Script entry point
# ==========================================================================

_SCENE_KEY_NORMS = {"buildings", "site", "settings"} | {k for ks in _CLASS_KEYS.values() for k in ks}
_NAME_PATTERNS = {"ground": r"ground|terrain", "tree": r"tree", "pathway": r"pathway|walkway|road",
                  "lawn": r"lawn|grass", "pond": r"pond|water"}


def _expand_paths(values) -> list:
    """Flatten CLI path lists; expand quoted glob patterns that the shell did not."""
    out = []
    for v in values or []:
        if any(c in v for c in "*?[") and not os.path.exists(v):
            out.extend(sorted(glob.glob(v)))
        else:
            out.append(v)
    return out


def load_scenes(paths) -> tuple:
    """Load every scene-like JSON among `paths`; merge component lists by id.
    Returns (scene_dict_or_None, [files used])."""
    scenes, used = [], []
    for p in _expand_paths(paths):
        if not os.path.isfile(p):
            warn(f"json: not a file, skipping: {p}")
            continue
        try:
            d = load_scene(p)
        except Exception as exc:
            warn(f"json: cannot parse {p} ({exc}); skipping")
            continue
        if not isinstance(d, dict) or not any(_norm_key(k) in _SCENE_KEY_NORMS for k in d):
            log(f"[input] json: {p} does not look like a scene file; ignored")
            continue
        scenes.append(d)
        used.append(p)
    if not scenes:
        return None, []
    merged = dict(scenes[0])
    for other in scenes[1:]:
        for key, val in other.items():
            if isinstance(val, list):
                mine = list(merged.get(key) or [])
                seen = {x.get("id") for x in mine if isinstance(x, dict) and x.get("id") is not None}
                for x in val:
                    if not (isinstance(x, dict) and x.get("id") in seen):
                        mine.append(x)
                merged[key] = mine
            elif key not in merged:
                merged[key] = val
    return merged, used


def classify_obj_inputs(args) -> tuple:
    """Sort the OBJ arguments. `--buildings_obj *.obj` typically expands to every
    OBJ in the folder, so files that are given explicitly for another class (or
    whose name says ground/tree/pathway/lawn/pond) are removed from the
    buildings list; with several building meshes left, 'refined' ones win.
    Returns (building_obj_list, {class: path or None})."""
    explicit = {"ground": args.ground_obj, "tree": args.trees_obj, "pathway": args.pathway_obj,
                "lawn": args.lawn_obj, "pond": args.pond_obj}
    explicit_real = {os.path.realpath(p) for p in explicit.values() if p}
    remaining = []
    for p in _expand_paths(args.buildings_obj):
        if os.path.realpath(p) in explicit_real:
            continue
        stem = os.path.basename(p).lower()
        owner = next((c for c, pat in _NAME_PATTERNS.items() if re.search(pat, stem)), None)
        if owner is None:
            remaining.append(p)
        elif not explicit[owner]:
            explicit[owner] = p
            log(f"[input] {p}: name says '{owner}'; using it as the {owner} surface")
        else:
            log(f"[input] {p}: ignored in --buildings_obj (a {owner} file was given explicitly)")
    named = [p for p in remaining if "building" in os.path.basename(p).lower()]
    pool = named or remaining
    refined = [p for p in pool if "refined" in os.path.basename(p).lower()]
    chosen = refined or pool
    if len(chosen) > 1:
        log(f"[input] several building meshes found, merging: {', '.join(chosen)}")
    return chosen, explicit


def _day_tags(dirs: list) -> list:
    seen, out = {}, []
    for d in dirs:
        t = os.path.basename(os.path.normpath(d)) or "day"
        n = seen.get(t, 0)
        seen[t] = n + 1
        out.append(t if n == 0 else f"{t}_{n}")
    return out


_FORK_STATE = {}


def _day_worker(job):
    return _FORK_STATE["fn"](*job)


def run_days(day_fn, jobs: list, workers: int) -> list:
    """jobs = [(weather_dir, tag)]. Several days + workers>1: one forked process
    per day (geometry/BVH are inherited, not rebuilt). Otherwise days run in
    order and `workers` threads trace shadow rays inside each day."""
    if len(jobs) > 1 and workers > 1 and "fork" in multiprocessing.get_all_start_methods():
        _FORK_STATE["fn"] = lambda wd, tag: day_fn(wd, tag, 1)
        results = []
        try:
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=min(workers, len(jobs)),
                    mp_context=multiprocessing.get_context("fork")) as ex:
                futures = [ex.submit(_day_worker, j) for j in jobs]
                for j, fut in zip(jobs, futures):
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        results.append((j[1] or j[0], False, f"worker failed: {exc!r}"))
        finally:
            _FORK_STATE.clear()
        return results
    return [day_fn(wd, tag, workers) for wd, tag in jobs]


def _out_prefix(args, tag) -> str:
    return os.path.join(args.output_dir, f"{args.out_prefix}_{tag}" if tag else args.out_prefix)


def _log_summary(label: str, face_df: pd.DataFrame) -> None:
    s = class_summary(face_df)
    for (grp, typ), sub in s.groupby(["surface_group", "surface_type"]):
        hot, cold = sub.loc[sub["T_surface_mean_C"].idxmax()], sub.loc[sub["T_surface_mean_C"].idxmin()]
        log(f"[day {label}] {grp}/{typ}: area {sub['area_m2'].iloc[0]:.1f} m2, mean T_surface "
            f"max {hot['T_surface_mean_C']:.1f} C at {hot['hour']:g} h, "
            f"min {cold['T_surface_mean_C']:.1f} C at {cold['hour']:g} h")


COMPACT_FACE_COLUMNS = [
    "building_id", "face_id", "surface_type", "surface_group", "area_m2", "hour", "T_amb_C",
    "sunlit_fraction", "I_total_Wm2", "T_surface_C", "T_solair_C", "h_conv_Wm2K",
    "q_solar_absorbed_Wm2", "q_conv_out_Wm2", "q_longwave_out_Wm2", "q_evap_out_Wm2",
    "q_lateral_in_Wm2", "q_conduction_in_Wm2", "q_storage_Wm2"]
_CSV_STATE = {}


def _csv_part(job):
    part, lo, hi, header = job
    _CSV_STATE["df"].iloc[lo:hi].to_csv(part, index=False, header=header)
    return part


def write_csv_parallel(df: pd.DataFrame, path: str, workers: int = 1) -> None:
    """DataFrame.to_csv is single-threaded and dominates the run time for millions
    of rows. With workers > 1 the rows are split into slices that forked processes
    write at the same time; the slices are then joined byte-for-byte. The result
    is identical to a plain to_csv."""
    n = len(df)
    if workers <= 1 or n < 200_000 or "fork" not in multiprocessing.get_all_start_methods():
        df.to_csv(path, index=False)
        return
    bounds = np.linspace(0, n, workers * 4 + 1).astype(int)
    jobs = [(f"{path}.part{i}", int(bounds[i]), int(bounds[i + 1]), i == 0)
            for i in range(len(bounds) - 1)]
    _CSV_STATE["df"] = df
    try:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("fork")) as ex:
            list(ex.map(_csv_part, jobs))
        with open(path, "wb") as out:
            for job in jobs:
                with open(job[0], "rb") as part:
                    shutil.copyfileobj(part, out, 1 << 24)
    finally:
        _CSV_STATE.clear()
        for job in jobs:
            try:
                os.remove(job[0])
            except OSError:
                pass


def _write_day_csvs(prefix: str, face_df, building_df, workers: int = 1, faces_csv: str = "full") -> None:
    t0 = time.time()
    written = []
    if faces_csv != "none":
        out = face_df if faces_csv == "full" else face_df[
            [c for c in COMPACT_FACE_COLUMNS if c in face_df.columns]]
        write_csv_parallel(out, f"{prefix}_faces_hourly.csv", workers)
        written.append(f"{prefix}_faces_hourly.csv")
    building_df.to_csv(f"{prefix}_building_hourly.csv", index=False)
    class_summary(face_df).to_csv(f"{prefix}_class_summary.csv", index=False)
    written += [f"{prefix}_building_hourly.csv", f"{prefix}_class_summary.csv"]
    log(f"Wrote {', '.join(written)} ({time.time() - t0:.0f} s)")


def make_obj_day_fn(bundle: ObjBundle, args):
    def day_fn(weather_dir, tag, threads):
        label = tag or os.path.basename(os.path.normpath(weather_dir))
        t0 = time.time()
        try:
            log(f"[day {label}] start: {weather_dir} (threads={threads})")
            analysis = solve_obj_day(
                bundle, weather_dir, t_initial_c=args.initial_temp_c,
                ground_initial_c=args.ground_initial_temp_c, ground_albedo=args.ground_albedo
                if args.ground_albedo is not None else DEFAULT_GROUND_ALBEDO,
                shadow_samples=args.shadow_samples, workers=threads)
            face_df, building_df, faces = analysis
            log(f"[day {label}] solved {len(face_df)} cell-hours in {time.time() - t0:.0f} s")
            prefix = _out_prefix(args, tag)
            _write_day_csvs(prefix, face_df, building_df, threads, args.faces_csv)
            _log_summary(label, face_df)
            log(f"[day {label}] CSV written at {time.time() - t0:.0f} s")
            if args.vtk_out is not None:
                export_vtk_series_from_analysis(
                    analysis, os.path.join(args.vtk_out, tag) if tag else args.vtk_out)
            if args.plot and len(building_df):
                try:
                    plot_results(face_df, building_df, prefix)
                except Exception as exc:
                    warn(f"plot skipped: {exc}")
            return label, True, f"done in {time.time() - t0:.0f} s"
        except Exception as exc:
            log(f"[day {label}] FAILED: {exc!r}\n{traceback.format_exc()}")
            return label, False, repr(exc)
    return day_fn


def make_json_day_fn(scene: dict, args, wall_settings):
    def day_fn(weather_dir, tag, threads):
        label = tag or (os.path.basename(os.path.normpath(weather_dir)) if weather_dir else "clear-sky")
        t0 = time.time()
        try:
            log(f"[day {label}] start (JSON pipeline)")
            if weather_dir is not None:
                if args.vtk_out is not None:
                    export_vtk_series(scene, weather_dir,
                                      os.path.join(args.vtk_out, tag) if tag else args.vtk_out,
                                      consider_shadows=not args.no_shadows,
                                      wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
                face_df, building_df, faces = run_analysis_from_folder(
                    scene, weather_dir, consider_shadows=not args.no_shadows,
                    wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
            else:
                if args.vtk_out is not None:
                    warn("--vtk_out needs --weather_dir/--weather_root; VTK skipped")
                face_df, building_df, faces = run_analysis(
                    scene, date(args.year, args.month, args.day), load_weather(args.weather_csv),
                    utc_offset_hr=args.utc_offset, consider_shadows=not args.no_shadows,
                    wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
            prefix = _out_prefix(args, tag)
            _write_day_csvs(prefix, face_df, building_df, threads, args.faces_csv)
            _log_summary(label, face_df)
            if args.plot:
                plot_results(face_df, building_df, prefix)
            return label, True, f"done in {time.time() - t0:.0f} s"
        except Exception as exc:
            log(f"[day {label}] FAILED: {exc!r}\n{traceback.format_exc()}")
            return label, False, repr(exc)
    return day_fn


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Solar, conduction and storage model for buildings, ground, pathways, "
                    "lawns, ponds and trees. Every input is optional; whatever is present "
                    "and valid is solved.")
    p.add_argument("--json", nargs="*", default=None,
                   help="Scene JSON file(s) (globs fine). Used for material properties, and "
                        "as the geometry source when no OBJ is given. Default: building_scene.json")
    p.add_argument("--buildings_obj", nargs="*", default=None,
                   help="Building OBJ(s). A shell glob like ../triSurface/*.obj is fine: files "
                        "given for other classes (or named ground/tree/pathway/lawn/pond) are "
                        "sorted out automatically.")
    p.add_argument("--ground_obj", default=None)
    p.add_argument("--trees_obj", default=None,
                   help="Tree OBJ: opaque shadow casters AND solved canopy surfaces.")
    p.add_argument("--pathway_obj", default=None)
    p.add_argument("--lawn_obj", default=None)
    p.add_argument("--pond_obj", default=None)
    p.add_argument("--shadow_samples", type=int, choices=(1, 3), default=3,
                   help="Rays per OBJ triangle: 3 (default) estimates partial shade; 1 uses the centroid.")
    p.add_argument("--weather_csv", default=None,
                   help="CSV hour,T_amb_C,wind_speed_ms[,...] (clear-sky path, JSON geometry only)")
    p.add_argument("--weather_dir", nargs="*", default=None,
                   help="One or more weatherData/MMDD folders (a glob like weatherData/*/. is fine); "
                        "each is solved as a separate day.")
    p.add_argument("--weather_root", default=None,
                   help="weatherData/ root; combined with --month/--day to find MMDD")
    p.add_argument("--month", type=int, default=6)
    p.add_argument("--day", type=int, default=21)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--utc_offset", type=float, default=0.0)
    p.add_argument("--out_prefix", default="results")
    p.add_argument("--output_dir", default=".", help="Directory for CSV/PNG outputs.")
    p.add_argument("--vtk_out", default=None,
                   help="Directory for the ParaView time series (a sub-folder per day when "
                        "several weather days are given).")
    p.add_argument("--vtk_ascii", action="store_true",
                   help="Write ASCII legacy VTK instead of the default binary (about 100x slower "
                        "and 3x larger; ParaView reads both).")
    p.add_argument("--vtk_binary", action="store_true", help=argparse.SUPPRESS)  # now the default
    p.add_argument("--faces_csv", choices=("full", "compact", "none"), default="full",
                   help="Per-cell hourly CSV: all 56 columns (default), 18 key columns, or skip it "
                        "(the VTK files carry the same fields). With --workers > 1 it is written in parallel.")
    p.add_argument("--workers", type=int, default=1,
                   help="Several weather days: that many days run in parallel (processes). "
                        "A single day: that many threads trace shadow rays.")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--no_shadows", action="store_true", help="Skip shadow casting (faster).")
    p.add_argument("--no_lateral", action="store_true",
                   help="Do not conduct heat sideways between neighbouring ground/pathway/lawn/pond cells.")
    p.add_argument("--no_overlap_removal", action="store_true",
                   help="Keep ground cells that lie under lawn/pond/pathway cells.")
    p.add_argument("--wall_thickness_m", type=float, default=0.23)
    p.add_argument("--wall_nodes", type=int, default=5)
    p.add_argument("--indoor_temp_c", type=float, default=24.0)
    p.add_argument("--indoor_h", type=float, default=8.0)
    p.add_argument("--initial_temp_c", type=float, default=None,
                   help="Initial uniform building slab temperature; default --indoor_temp_c.")
    # Overrides. Left unset (None) so JSON / built-in defaults apply.
    p.add_argument("--ground_depth_m", type=float, default=None)
    p.add_argument("--ground_nodes", type=int, default=None)
    p.add_argument("--deep_ground_temp_c", type=float, default=None,
                   help="Fixed deep temperature below ground/pathway/lawn/pond layers (default 24).")
    p.add_argument("--ground_deep_h", type=float, default=None)
    p.add_argument("--ground_initial_temp_c", type=float, default=None,
                   help="Initial ground/pathway/lawn/pond temperature; default is the deep temperature.")
    p.add_argument("--ground_conductivity", type=float, default=None)
    p.add_argument("--ground_density", type=float, default=None)
    p.add_argument("--ground_specific_heat", type=float, default=None)
    p.add_argument("--ground_albedo", type=float, default=None)
    p.add_argument("--ground_emissivity", type=float, default=None)
    p.add_argument("--lawn_wetness", type=float, default=None,
                   help="Lawn evaporation as a fraction of free-water evaporation (default 0.25).")
    p.add_argument("--tree_wetness", type=float, default=None,
                   help="Canopy transpiration fraction (default 0.30).")
    p.add_argument("--pond_depth_m", type=float, default=None, help="Water depth (default 0.5).")
    p.add_argument("--pond_mixing_factor", type=float, default=None,
                   help="Pond effective conductivity = factor x 0.6 W/m/K, representing wind and "
                        "convective mixing (default 30; 1 = still water, which heats only a thin skin).")
    return p


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.workers = max(1, args.workers)
    global VTK_BINARY
    VTK_BINARY = not args.vtk_ascii
    if args.pond_mixing_factor is not None and not (math.isfinite(args.pond_mixing_factor)
                                                   and args.pond_mixing_factor > 0):
        parser.error("--pond_mixing_factor must be a positive number")
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        wall_settings = WallSettings(args.wall_thickness_m, args.wall_nodes,
                                     args.indoor_temp_c, args.indoor_h)
    except ValueError as error:
        parser.error(str(error))

    # ---- inputs -----------------------------------------------------------
    json_args = args.json if args.json else (["building_scene.json"]
                                             if os.path.isfile("building_scene.json") else [])
    scene, scene_files = load_scenes(json_args)
    log(f"[input] scene JSON: {', '.join(scene_files) if scene_files else 'none'}")
    building_objs, class_objs = classify_obj_inputs(args)

    weather_dirs, seen = [], set()
    for d in _expand_paths(args.weather_dir):
        rp = os.path.realpath(d)
        if os.path.isdir(d) and rp not in seen:
            seen.add(rp)
            weather_dirs.append(d)
        elif not os.path.isdir(d):
            warn(f"weather_dir: not a directory, skipping: {d}")
    if not weather_dirs and args.weather_root:
        try:
            weather_dirs = [resolve_weather_folder(args.weather_root, args.month, args.day)]
        except FileNotFoundError as exc:
            warn(str(exc))
    log(f"[input] weather days: {len(weather_dirs)}  workers: {args.workers}")

    overrides = {
        "ground": dict(depth_m=args.ground_depth_m, nodes=args.ground_nodes,
                       deep_h_wm2k=args.ground_deep_h, thermal_conductivity=args.ground_conductivity,
                       density=args.ground_density, specific_heat=args.ground_specific_heat,
                       albedo=args.ground_albedo, emissivity=args.ground_emissivity,
                       deep_temperature_c=args.deep_ground_temp_c),
        "pathway": dict(deep_temperature_c=args.deep_ground_temp_c),
        "lawn": dict(deep_temperature_c=args.deep_ground_temp_c, wetness=args.lawn_wetness),
        "pond": dict(deep_temperature_c=args.deep_ground_temp_c, depth_m=args.pond_depth_m,
                     mixing_factor=args.pond_mixing_factor),
        "tree": dict(wetness=args.tree_wetness),
    }

    # ---- choose pipeline --------------------------------------------------
    day_fn = None
    jobs = []
    any_obj = bool(building_objs) or any(class_objs.values())
    if any_obj:
        if not weather_dirs:
            log("ERROR: the OBJ pipeline needs real weather data: pass --weather_dir or --weather_root")
            return 2
        bundle = prepare_obj_bundle(
            building_objs or None, class_objs["ground"], class_objs["tree"], class_objs["pathway"],
            class_objs["lawn"], class_objs["pond"], scene=scene, wall_settings=wall_settings,
            layer_overrides=overrides, consider_shadows=not args.no_shadows,
            remove_overlaps=not args.no_overlap_removal, lateral_conduction=not args.no_lateral)
        if bundle.n_cells:
            log(f"[input] OBJ pipeline: {bundle.n_cells} cells in total")
            day_fn = make_obj_day_fn(bundle, args)
        else:
            warn("no valid OBJ geometry found")
    if day_fn is None:
        if scene is None or not scene.get("buildings"):
            log("ERROR: nothing to solve: no valid OBJ files and no buildings in a scene JSON")
            return 2
        log("[input] JSON pipeline (buildings from the scene JSON; shadows from buildings and "
            "tree spheres; lawns/ponds/ground need OBJ files)")
        day_fn = make_json_day_fn(scene, args, wall_settings)
    if weather_dirs:
        tags = _day_tags(weather_dirs) if len(weather_dirs) > 1 else [None]
        jobs = list(zip(weather_dirs, tags))
    else:
        jobs = [(None, None)]

    results = run_days(day_fn, jobs, args.workers)
    failed = [r for r in results if not r[1]]
    for label, ok, msg in results:
        log(f"[result] {label}: {'OK' if ok else 'FAILED'} - {msg}")
    return 1 if failed and len(failed) == len(results) else 0


if __name__ == "__main__":
    sys.exit(main())



#Subrat: The below code doesn't take into account the lateral heat transfer between the adjacent walls. It only considers the heat transfer in the direction normal to the wall surface. This is a simplification that may not be accurate for all scenarios, especially in cases where there are significant temperature gradients or heat sources/sinks on adjacent walls.
# """
# Analytical solar-exposure + convective/radiative surface-temperature model
# for buildings exported from the scene JSON (SNU-SPH / urban-microclimate
# style export: buildings[].faces[] with vertices, normal, surfaceType,
# materialProperties.thermalProperties; optionally trees[] for canopy shading).

# No CFD, and no full (multi-bounce) ray tracing -- everything below is
# geometry plus visibility rays toward the sun at each weather time:

#   1. SOLAR GEOMETRY  (Cooper's declination + standard hour-angle equations)
#      -> sun altitude & azimuth for any date / time / lat-lon.

#   2. CLEAR-SKY IRRADIANCE  (Hottel, 1976 atmospheric-transmittance model)
#      -> direct-normal (I_DN) and diffuse-horizontal (I_dH) irradiance from
#         the solar altitude alone -- used only when you DON'T have a real
#         weather-data export (see `run_analysis`). If you DO have one (a
#         `weatherData/MMDD/` folder with Tambient, windSpeed, IDN, Idif,
#         sunPosVector, ...), use `run_analysis_from_folder` instead: it
#         reads the real sun-position vector and real DNI/diffuse straight
#         from those files and skips this model entirely -- more accurate,
#         and removes any ambiguity about axis conventions since the sun
#         vector is already given in the same (East, North, Up) axes as the
#         building JSON's face normals.

#   3. PER-FACE SUNLIT TEST  -- cos(theta) = dot(face_normal, sun_unit_vector).
#      A face receives direct beam radiation only when cos(theta) > 0 AND the
#      sun is above the horizon (self-orientation), AND (new) a visibility ray
#      from the face toward the sun is not blocked by another building or tree
#      in the scene (mutual shading -- see Section 5, "Shadow occlusion").

#   4. CONVECTION -- classic linear wind correlation h_c = a + b*V (McAdams-
#      type, W/m^2K), applied per face using your wind-speed time series.

#   5. SURFACE TEMPERATURE: each exposed building or requested ground triangle
#      is a 0.23 m slab with five nodes through its thickness, heat capacity
#      rho*cp*L per unit area, and
#      conductance k/dx between adjacent nodes. A matrix exponential advances
#      conduction/storage exactly for the spatially discretized system, holding
#      weather and ambient-linearized longwave coefficients constant between
#      records. A building inner face exchanges heat with fixed 24 C indoor air;
#      a ground back face exchanges heat with fixed deep soil. These settings,
#      materials, and initial temperatures are configurable.
#      T_solair_C remains an equivalent driving temperature for the bulk model.

#   6. WHOLE-BUILDING TEMPERATURE -- single-node RC (lumped capacitance,
#      C = rho*V*cp) network per building, driven by that building's own
#      faces' sol-air temperatures. Because C and the effective conductance
#      h_eff are taken as constant within each hour, the ODE
#      C dT/dt = h_eff*(T_eq - T) has the exact analytical solution

#          T(t+dt) = T_eq + (T(t) - T_eq) * exp(-dt / tau),   tau = C/h_eff

#      so the whole march-forward is a closed-form exponential step, not a
#      numerical ODE integration. Each building in the scene gets its own
#      node (its own C, its own faces) -- they are only coupled through the
#      shadows they cast on each other, not thermally.

# Assumptions you should sanity-check against your own scene / site
# (flagged again inline where they are used):
#   - Scene axes are treated as ENU: +x = East, +y = North, +z = Up. If your
#     tool defines "north" differently, set `north_offset_deg` below.
#   - Longwave emissivity is not in the JSON -> defaulted to 0.9 (typical
#     concrete/masonry). Change `DEFAULT_EMISSIVITY` if you know better.
#   - Ground/surroundings albedo for the reflected-radiation term is not in
#      the JSON -> defaulted to 0.2. Change `DEFAULT_GROUND_ALBEDO`.
#   - The optional OBJ ground calculation uses the case soil defaults k=1.5
#     W/m/K, rho=1150 kg/m3 and cp=650 J/kg/K. Its active depth is 0.23 m and
#     its default lower conductance k/L connects it to fixed 24 C deep soil.
#   - The floor face (surfaceType == "floor") is excluded from solar +
#     outdoor convection (it's in contact with the ground, not outdoor air).
#     Toggle `INCLUDE_FLOOR` if you want to add a simple ground-conduction
#     term instead. A "ceiling" face (surfaceType == "ceiling") is *always*
#     excluded the same way, with no toggle -- in multi-floor exports (one
#     JSON "building" per story, stacked in z) this tags the boundary between
#     two stacked stories, which never touches outdoor air regardless of
#     settings. Only "wall" and "roof" faces are treated as always outdoor-
#     exposed.
#   - Two of the face normals in the source JSON point the "wrong" way for
#     what they geometrically are (roof normal given as (0,0,-1), i.e.
#     inward, floor as (0,0,1), also inward). This script auto-corrects
#     normal sign using `surfaceType` so the physics is right regardless of
#     what the exporter wrote. Also, wall face "area" was exported as 0 in
#     this file -- areas are recomputed from the vertices, never trusted
#     from the JSON.
#   - The OBJ path retains individual mesh triangles, with three visibility
#     samples per triangle by default, accelerated by a triangle BVH. The JSON
#     path still tests one centroid per polygon. Only the direct
#     beam is blockable; the diffuse-sky and ground-reflected components
#     still use the unobstructed sky-view-factor formulas (a standard
#     simplification -- properly reducing diffuse view factors for nearby
#     obstructions would need a more expensive hemispherical integration).
#   - In the JSON path, tree canopies are approximated as a bounding sphere for shadow
#     purposes (a common simplification in urban solar-access studies), not
#     actual leaf/branch geometry. The JSON field names used to build that
#     sphere are guessed from common conventions (see `build_tree_occluder`)
#     since no example tree entry was available when this was written --
#     check `_TREE_*_KEYS` against your real tree schema and adjust if needed.
#     The OBJ path traces the actual tree triangles, treating them as opaque.

# MULTI-SURFACE OBJ MODE (buildings, ground, pathway, lawn, pond, trees)
#   Every component is optional; whatever has valid triangles is solved and the
#   rest is skipped with a log line. Geometry comes from the OBJ files, material
#   properties from the scene JSON when present (else the documented defaults;
#   the log lists the source of every property). Ground-like classes use the
#   same transient slab solver as the walls. Ponds, lawns and tree canopies
#   additionally lose latent heat by evaporation/transpiration (Lewis analogy,
#   linearised about ambient so the exact matrix-exponential update still
#   applies), and tree canopies exchange heat with ambient air on their back
#   side. Ground cells under building roofs or under pond/pathway/lawn cells
#   are removed so nothing is solved twice. Several weather days can be given:
#   each becomes its own set of outputs, run in parallel with --workers.
# """

# import concurrent.futures
# import glob
# import json
# import math
# import multiprocessing
# import os
# import re
# import shutil
# import sys
# import time
# import traceback
# from dataclasses import dataclass, field
# from datetime import date
# from typing import Optional

# import numpy as np
# import pandas as pd

# # --------------------------------------------------------------------------
# # Tunable defaults (not present in the JSON -- see docstring above)
# # --------------------------------------------------------------------------
# DEFAULT_EMISSIVITY = 0.90
# DEFAULT_GROUND_ALBEDO = 0.20
# STEFAN_BOLTZMANN = 5.670374e-8          # W/m^2/K^4
# SOLAR_CONSTANT = 1367.0                 # W/m^2
# INCLUDE_FLOOR = False                   # floor treated as adiabatic (ground contact)


# # ==========================================================================
# # 1. Geometry: load scene, fix normals, compute real areas
# # ==========================================================================

# @dataclass
# class Face:
#     building_id: str
#     face_id: str
#     surface_type: str
#     normal: np.ndarray          # outward unit normal, corrected
#     area: float                 # m^2, computed from vertices (JSON area ignored)
#     tilt_deg: float             # 0 = horizontal facing up, 90 = vertical, 180 = facing down
#     center: np.ndarray
#     vertices: np.ndarray        # (N,3) planar polygon loop, as given (for geometry export)
#     thermal_conductivity: float
#     density: float
#     specific_heat: float
#     albedo: float
#     emissivity: float = DEFAULT_EMISSIVITY
#     parent_face_id: str = ""    # logical wall/roof; OBJ physics stays per triangle


# def _polygon_area_3d(verts: np.ndarray) -> float:
#     """Area of a planar polygon in 3D via Newell's method (cross-product sum)."""
#     n = len(verts)
#     normal_sum = np.zeros(3)
#     for i in range(n):
#         v1 = verts[i]
#         v2 = verts[(i + 1) % n]
#         normal_sum += np.cross(v1, v2)
#     return 0.5 * np.linalg.norm(normal_sum)


# def load_scene(json_path: str) -> dict:
#     with open(json_path, "r") as f:
#         return json.load(f)


# def extract_faces(scene: dict) -> list:
#     """Flatten every building's faces into a list of corrected Face objects."""
#     faces = []
#     for b in scene.get("buildings", []):
#         bid = b.get("id", "building")
#         for f in b.get("faces", []):
#             verts = np.array([[v["x"], v["y"], v["z"]] for v in f["vertices"]])
#             area = _polygon_area_3d(verts)          # never trust JSON "area"
#             n = np.array([f["normal"]["x"], f["normal"]["y"], f["normal"]["z"]], dtype=float)
#             n = n / (np.linalg.norm(n) + 1e-12)

#             stype = f.get("surfaceType", "wall")
#             # --- fix inward-pointing roof/floor normals (see module docstring) ---
#             if stype == "roof" and n[2] < 0:
#                 n = -n
#             if stype == "floor" and n[2] > 0:
#                 n = -n

#             tilt_deg = math.degrees(math.acos(np.clip(n[2], -1.0, 1.0)))
#             c = f.get("center", {"x": 0, "y": 0, "z": 0})
#             center = np.array([c["x"], c["y"], c["z"]])

#             def _prop(name, default):
#                 value = _lookup_property(f, name)
#                 return default if value is None else value
#             faces.append(Face(
#                 building_id=bid,
#                 face_id=f["id"],
#                 surface_type=stype,
#                 normal=n,
#                 area=area,
#                 tilt_deg=tilt_deg,
#                 center=center,
#                 vertices=verts,
#                 thermal_conductivity=_prop("thermal_conductivity", DEFAULT_MATERIAL["thermal_conductivity"]),
#                 density=_prop("density", DEFAULT_MATERIAL["density"]),
#                 specific_heat=_prop("specific_heat", DEFAULT_MATERIAL["specific_heat"]),
#                 albedo=_prop("albedo", DEFAULT_ALBEDO_ROOF if stype == "roof" else DEFAULT_ALBEDO_WALL),
#                 emissivity=_prop("emissivity", DEFAULT_EMISSIVITY),
#             ))
#     return faces


# def building_volume(scene: dict, building_id: Optional[str] = None) -> float:
#     for b in scene.get("buildings", []):
#         if building_id is None or b.get("id") == building_id:
#             geo = b.get("geometry", {})
#             if "volume" in geo:
#                 return geo["volume"]
#             return geo.get("footprintArea", 0) * b.get("height", 0)
#     return 0.0


# def site_location(scene: dict) -> tuple:
#     """Returns (lat_deg, lon_deg) -- prefers shadowSettings, falls back to site centroid."""
#     ss = scene.get("settings", {}).get("shadowSettings")
#     if ss and "latitude" in ss:
#         return ss["latitude"], ss["longitude"]
#     coords = scene["site"]["properties"]["originalCoordinates"]
#     lat = float(np.mean([c["lat"] for c in coords]))
#     lon = float(np.mean([c["lng"] for c in coords]))
#     return lat, lon


# # ==========================================================================
# # 2. Solar geometry
# # ==========================================================================

# def solar_position(lat_deg: float, lon_deg: float, day_of_year: int,
#                     local_std_hour: float, utc_offset_hr: float = 0.0) -> tuple:
#     """
#     Standard solar-position equations (Cooper's declination + hour angle).
#     Returns (altitude_deg, azimuth_deg); azimuth measured clockwise from
#     true North (0=N, 90=E, 180=S, 270=W). altitude < 0 => sun below horizon.
#     """
#     lat = math.radians(lat_deg)
#     n = day_of_year

#     decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0)))

#     # Equation of time (minutes), Duffie & Beckman form
#     B = math.radians(360.0 * (n - 81) / 364.0)
#     eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

#     std_meridian = 15.0 * utc_offset_hr
#     time_corr_min = 4.0 * (lon_deg - std_meridian) + eot
#     solar_time = local_std_hour + time_corr_min / 60.0

#     hour_angle = math.radians(15.0 * (solar_time - 12.0))

#     sin_alt = (math.sin(lat) * math.sin(decl) +
#                math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
#     altitude = math.asin(np.clip(sin_alt, -1.0, 1.0))

#     # Azimuth, measured clockwise from true North (standard NOAA-style form):
#     #   cos(A) = (sin(decl) - sin(lat)*sin(alt)) / (cos(lat)*cos(alt))
#     # acos() only returns 0-180 deg, so mirror into 180-360 for the
#     # afternoon (hour_angle > 0) to get the correct NE-at-sunrise /
#     # NW-at-sunset quadrant.
#     cos_alt = max(math.cos(altitude), 1e-9)
#     cos_A = (math.sin(decl) - math.sin(lat) * math.sin(altitude)) / (math.cos(lat) * cos_alt)
#     az_raw = math.degrees(math.acos(np.clip(cos_A, -1.0, 1.0)))
#     azimuth = az_raw if hour_angle <= 0 else 360.0 - az_raw

#     return math.degrees(altitude), azimuth


# def sun_unit_vector(altitude_deg: float, azimuth_deg: float,
#                      north_offset_deg: float = 0.0) -> np.ndarray:
#     """
#     Sun direction as a unit vector in scene coordinates, assuming
#     +x = East, +y = North, +z = Up (rotate with north_offset_deg if your
#     scene's "north" is not aligned with +y).
#     """
#     alt = math.radians(altitude_deg)
#     az = math.radians(azimuth_deg - north_offset_deg)
#     east = math.cos(alt) * math.sin(az)
#     north = math.cos(alt) * math.cos(az)
#     up = math.sin(alt)
#     return np.array([east, north, up])


# # ==========================================================================
# # 3. Clear-sky irradiance (Hottel, 1976)
# # ==========================================================================

# def hottel_clear_sky(altitude_deg: float, site_altitude_km: float = 0.0,
#                       r0: float = 1.0, r1: float = 1.0, rk: float = 1.0) -> tuple:
#     """
#     Hottel's (1976) atmospheric-transmittance clear-sky model.
#     r0, r1, rk are the climate-type correction factors from Hottel's paper
#     (defaults =1.0, i.e. the reference "mid-latitude summer" atmosphere --
#     nudge by a few % for your own climate/visibility if you know it).
#     Returns (I_DN, I_dH) in W/m^2 -- direct-normal and diffuse-horizontal.
#     Returns (0, 0) when the sun is below the horizon.
#     """
#     if altitude_deg <= 0:
#         return 0.0, 0.0
#     A = site_altitude_km
#     sin_b = math.sin(math.radians(altitude_deg))

#     a0 = r0 * (0.4237 - 0.00821 * (6.0 - A) ** 2)
#     a1 = r1 * (0.5055 + 0.00595 * (6.5 - A) ** 2)
#     k = rk * (0.2711 + 0.01858 * (2.5 - A) ** 2)

#     tau_b = a0 + a1 * math.exp(-k / sin_b)
#     tau_d = 0.2710 - 0.2939 * tau_b   # Liu-Jordan diffuse-transmittance approx

#     i_dn = SOLAR_CONSTANT * tau_b
#     i_dh = SOLAR_CONSTANT * tau_d * sin_b
#     return max(i_dn, 0.0), max(i_dh, 0.0)


# def sky_temperature(t_amb_k: float) -> float:
#     """Swinbank (1963) clear-sky effective sky temperature, in Kelvin."""
#     return 0.0552 * t_amb_k ** 1.5


# # ==========================================================================
# # 4. Per-face irradiance & sol-air temperature
# # ==========================================================================

# def face_irradiance(face: Face, sun_vec: np.ndarray, altitude_deg: float,
#                      i_dn: float, i_dh: float,
#                      ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                      external_shadow: bool = False) -> dict:
#     """
#     Direct + diffuse (isotropic sky) + ground-reflected irradiance on one face.
#     `external_shadow=True` means a ray/occlusion test (Section 5) found this
#     face blocked by another building or tree this hour -- the direct beam is
#     then zeroed regardless of the face's own orientation toward the sun.
#     """
#     cos_theta = float(np.dot(face.normal, sun_vec))
#     self_lit = (altitude_deg > 0) and (cos_theta > 0)
#     sunlit = self_lit and not external_shadow

#     i_direct = i_dn * cos_theta if sunlit else 0.0

#     tilt = math.radians(face.tilt_deg)
#     view_sky = (1 + math.cos(tilt)) / 2.0     # 1 for roof, 0.5 for wall, 0 for floor
#     view_ground = 1.0 - view_sky
#     i_diffuse = i_dh * view_sky

#     i_gh = i_dn * math.sin(math.radians(max(altitude_deg, 0.0))) + i_dh
#     i_reflected = ground_albedo * i_gh * view_ground

#     return dict(sunlit=sunlit, self_lit=self_lit, shadowed=external_shadow,
#                 cos_theta=cos_theta, view_sky=view_sky,
#                 i_direct=i_direct, i_diffuse=i_diffuse, i_reflected=i_reflected,
#                 i_total=i_direct + i_diffuse + i_reflected)


# def convective_h(wind_speed_ms: float, a: float = 5.7, b: float = 3.8) -> float:
#     """McAdams-type linear wind correlation (W/m^2K)."""
#     return a + b * max(wind_speed_ms, 0.0)


# def sol_air_temperature(t_amb_c: float, i_total: float, h_c: float,
#                          albedo: float, view_sky: float,
#                          emissivity: float = DEFAULT_EMISSIVITY) -> float:
#     """
#     ASHRAE sol-air temperature: the fictitious outdoor air temperature that
#     would give the same solar + convective + long-wave heat gain as the
#     real conditions, so a face just "sees" T_sol_air via plain convection.
#     """
#     alpha = 1.0 - albedo
#     t_amb_k = t_amb_c + 273.15
#     t_sky_k = sky_temperature(t_amb_k)
#     # net long-wave LOSS to the sky, scaled by how much sky this face can see
#     q_lw_loss = view_sky * emissivity * STEFAN_BOLTZMANN * (t_amb_k ** 4 - t_sky_k ** 4)
#     return t_amb_c + (alpha * i_total - q_lw_loss) / h_c


# def surface_temperature(t_amb_c, i_total, h_c, albedo, view_sky,
#                         emissivity=DEFAULT_EMISSIVITY):
#     """Local steady exterior temperature (C), with no wall conduction/storage.

#     Solve alpha*I = h_c*(Ts-Ta) + eps*sigma*(Ts^4 - Fsky*Tsky^4
#                                                 - (1-Fsky)*Ta^4).
#     Ground/surroundings are at ambient temperature; sky uses Swinbank.
#     Supports scalar or broadcastable array inputs. Unlike sol-air temperature,
#     outgoing longwave radiation is evaluated at the actual surface temperature.
#     """
#     ta = np.asarray(t_amb_c) + 273.15
#     absorbed = (1.0 - np.asarray(albedo)) * np.asarray(i_total)
#     eps_sigma = np.asarray(emissivity) * STEFAN_BOLTZMANN
#     background = np.asarray(view_sky) * sky_temperature(ta)**4 + (1-np.asarray(view_sky))*ta**4
#     ts = np.asarray(ta + absorbed / h_c)
#     for _ in range(30):
#         residual = h_c*(ts-ta) + eps_sigma*(ts**4-background) - absorbed
#         step = residual / (h_c + 4*eps_sigma*ts**3)
#         ts = ts - step
#         if np.max(np.abs(step), initial=0.0) < 1e-8:
#             break
#     return ts - 273.15


# @dataclass(frozen=True)
# class WallSettings:
#     """Uniform slab settings for exposed walls and roofs; SI units except C."""
#     thickness_m: float = 0.23
#     nodes: int = 5
#     indoor_temperature_c: float = 24.0
#     indoor_h_wm2k: float = 8.0

#     def __post_init__(self):
#         if not np.isfinite(self.thickness_m) or self.thickness_m <= 0:
#             raise ValueError("wall thickness must be finite and positive")
#         if not isinstance(self.nodes, int) or self.nodes < 2:
#             raise ValueError("wall nodes must be an integer >= 2")
#         if not np.isfinite(self.indoor_temperature_c) or self.indoor_temperature_c <= -273.15:
#             raise ValueError("indoor temperature must be finite and above absolute zero")
#         if not np.isfinite(self.indoor_h_wm2k) or self.indoor_h_wm2k < 0:
#             raise ValueError("indoor heat-transfer coefficient must be finite and nonnegative")


# @dataclass(frozen=True)
# class GroundSettings:
#     """Thermal properties and lower boundary for exposed ground triangles.

#     The active soil layer uses the same nodal slab solver as a wall. Its lower
#     face exchanges heat with fixed deep-soil temperature through
#     ``deep_h_wm2k``. When that value is omitted, k/depth is used, representing
#     conduction through one additional layer of the same thickness.
#     """
#     depth_m: float = 0.23
#     nodes: int = 5
#     deep_temperature_c: float = 24.0
#     deep_h_wm2k: Optional[float] = None
#     thermal_conductivity: float = 1.5
#     density: float = 1150.0
#     specific_heat: float = 650.0
#     albedo: float = DEFAULT_GROUND_ALBEDO
#     emissivity: float = 0.95
#     wetness: float = 0.05                  # fraction of potential evaporation (0 dry .. 1 water)
#     back_follows_ambient: bool = False    # back face exchanges with ambient air (tree canopy)

#     def __post_init__(self):
#         if not np.isfinite(self.depth_m) or self.depth_m <= 0:
#             raise ValueError("ground depth must be finite and positive")
#         if not np.isfinite(self.wetness) or not 0 <= self.wetness <= 1:
#             raise ValueError("wetness must be finite and between 0 and 1")
#         WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
#                      self.effective_deep_h_wm2k)
#         material = np.array([self.thermal_conductivity, self.density,
#                              self.specific_heat], dtype=float)
#         if np.any(~np.isfinite(material)) or np.any(material <= 0):
#             raise ValueError("ground conductivity, density and heat capacity must be positive and finite")
#         if not np.isfinite(self.albedo) or not 0 <= self.albedo <= 1:
#             raise ValueError("ground albedo must be finite and between 0 and 1")
#         if not np.isfinite(self.emissivity) or not 0 <= self.emissivity <= 1:
#             raise ValueError("ground emissivity must be finite and between 0 and 1")

#     @property
#     def effective_deep_h_wm2k(self) -> float:
#         return (self.thermal_conductivity / self.depth_m
#                 if self.deep_h_wm2k is None else self.deep_h_wm2k)

#     def slab_settings(self) -> WallSettings:
#         return WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
#                             self.effective_deep_h_wm2k)


# class SlabConduction:
#     """Independent 1-D slabs, with an exact exponential update in time.

#     Boundary nodes lie on the actual surfaces and have half control volumes.
#     Interior nodes have full control volumes. Per-area nodal capacities sum
#     to rho*cp*L; adjacent conductance is k/dx. No lateral cell coupling.
#     Grouping equal materials lets all their triangles share a small matrix.
#     """

#     def __init__(self, conductivity, density, specific_heat, emissivity,
#                  initial_c, settings=None):
#         self.settings = settings or WallSettings()
#         properties = np.column_stack(np.broadcast_arrays(
#             conductivity, density, specific_heat, emissivity))
#         if np.any(~np.isfinite(properties)) or np.any(properties[:, :3] <= 0):
#             raise ValueError("wall conductivity, density and heat capacity must be positive and finite")
#         self.dx = self.settings.thickness_m / (self.settings.nodes-1)
#         widths = np.full(self.settings.nodes, self.dx)
#         widths[[0, -1]] *= 0.5
#         self.capacity = properties[:, 1, None]*properties[:, 2, None]*widths
#         self.conductance = properties[:, 0] / self.dx
#         self.temperatures = np.full((len(properties), self.settings.nodes), initial_c, dtype=float)
#         if not np.isfinite(initial_c) or initial_c <= -273.15:
#             raise ValueError("initial wall temperature must be finite and above absolute zero")
#         self.initial_c = initial_c
#         self.back_h = self.settings.indoor_h_wm2k
#         self.back_temperature = self.settings.indoor_temperature_c
#         _, inverse = np.unique(properties, axis=0, return_inverse=True)
#         self.groups = [np.flatnonzero(inverse == i) for i in range(inverse.max()+1)]

#     def set_back(self, h, temperature):
#         """Replace the back-face coefficient / temperature (time-varying back
#         boundary, e.g. ambient air behind a tree canopy). Used by advance()
#         and diagnostics() until changed again."""
#         self.back_h = float(h)
#         self.back_temperature = float(temperature)

#     def advance(self, dt_seconds, external_h, external_source):
#         """C*Tdot = -K*T + b; exterior input is source - h*T[0].

#         Input coefficients are held constant for dt. Transform with sqrt(C)
#         to a symmetric matrix, then use its eigendecomposition to evaluate
#         exp(A*dt) and the integrated forcing. No SciPy dependency or explicit
#         time stepping; zero eigenvalues are handled for insulated slabs.
#         """
#         if not np.isfinite(dt_seconds) or dt_seconds < 0:
#             raise ValueError("wall time step must be finite and nonnegative")
#         if dt_seconds == 0:
#             return
#         n = self.settings.nodes
#         external_h = np.broadcast_to(external_h, len(self.temperatures))
#         external_source = np.broadcast_to(external_source, len(self.temperatures))
#         for ids in self.groups:
#             # Radiation coefficient depends on ambient and emissivity only,
#             # so it is common to a material group within a weather interval.
#             if not np.allclose(external_h[ids], external_h[ids[0]], rtol=0, atol=1e-12):
#                 raise ValueError("external h must be equal within each material group")
#             c = self.capacity[ids[0]]
#             root_c = np.sqrt(c)
#             g = self.conductance[ids[0]]
#             K = np.diag(np.full(n, 2*g)) + np.diag(np.full(n-1, -g), 1) + np.diag(np.full(n-1, -g), -1)
#             K[0,0] = g + external_h[ids[0]]
#             K[-1,-1] = g + self.back_h
#             A = -K / root_c[:, None] / root_c[None, :]
#             eigenvalues, eigenvectors = np.linalg.eigh(A)
#             exponent = np.exp(eigenvalues*dt_seconds)
#             phi = np.full(n, dt_seconds, dtype=float)
#             np.divide(np.expm1(eigenvalues*dt_seconds), eigenvalues,
#                       out=phi, where=np.abs(eigenvalues) > 1e-15)
#             b = np.zeros((len(ids), n))
#             b[:,0] = external_source[ids]
#             b[:,-1] = self.back_h*self.back_temperature
#             modes = (self.temperatures[ids]*root_c) @ eigenvectors
#             forcing = (b/root_c) @ eigenvectors
#             self.temperatures[ids] = ((modes*exponent + forcing*phi) @ eigenvectors.T) / root_c

#     def diagnostics(self, external_h, external_source):
#         """Instantaneous conductive/storage fluxes at the current output time."""
#         t = self.temperatures
#         q_outer = external_source-external_h*t[:,0]
#         q_inner = self.back_h*(t[:,-1]-self.back_temperature)
#         q_conduction = self.conductance*(t[:,0]-t[:,1])
#         t_mean = np.sum(self.capacity*t,axis=1)/self.capacity.sum(axis=1)
#         capacity = self.capacity.sum(axis=1)
#         return dict(T_inner_C=t[:,-1].copy(), T_back_surface_C=t[:,-1].copy(),
#                     T_wall_mean_C=t_mean, T_slab_mean_C=t_mean,
#                     q_conduction_in_Wm2=q_conduction, q_indoor_out_Wm2=q_inner,
#                     q_back_out_Wm2=q_inner,
#                     q_storage_Wm2=q_outer-q_inner,
#                     q_outer_node_storage_Wm2=q_outer-q_conduction,
#                     stored_heat_Jm2=np.sum(self.capacity*(t-self.initial_c),axis=1),
#                     wall_capacity_Jm2K=capacity, slab_capacity_Jm2K=capacity)


# # ==========================================================================
# # 5. Shadow occlusion -- cheap two-phase ray tracing
# #
# #    Phase 1 (broad, O(1) per occluder): does the ray toward the sun even
# #             pass through the occluder's axis-aligned bounding box?
# #    Phase 2 (narrow, exact): only for occluders that survive phase 1 --
# #             Moller-Trumbore ray-triangle tests against the occluder's
# #             actual wall/roof triangles (buildings), or a closed-form
# #             ray-sphere test (trees, canopy approximated as a sphere).
# #
# #    This is the standard bounding-volume-culling idea from ray tracing,
# #    kept deliberately simple (no BVH/kd-tree): for scenes with up to a
# #    few hundred buildings this keeps the great majority of rays from ever
# #    touching a triangle, which is normally the expensive part.
# # ==========================================================================

# @dataclass
# class Occluder:
#     kind: str                                   # "building" or "tree"
#     source_id: str
#     aabb_min: np.ndarray
#     aabb_max: np.ndarray
#     triangles: list = field(default_factory=list)     # [(p0,p1,p2), ...] for buildings
#     sphere_center: Optional[np.ndarray] = None          # for trees
#     sphere_radius: Optional[float] = None


# def _triangulate_quad(v0, v1, v2, v3):
#     return [(v0, v1, v2), (v0, v2, v3)]


# def _fan_triangulate(verts):
#     """Fan triangulation from the first vertex. Exact for convex polygons
#     (true for simple rectangular footprints); an approximation for
#     concave ones -- flag this if you have complex (e.g. OSM-derived,
#     L-shaped) footprints and it matters for your results."""
#     return [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]


# def build_building_occluder(building_json: dict) -> Optional[Occluder]:
#     """
#     Triangulates one building's wall + roof faces for use as an occluder of
#     OTHER buildings/faces. The floor is skipped -- sitting at ground level,
#     it cannot shadow anything above it, so leaving it out is a free
#     reduction in triangle count with no effect on the result.
#     """
#     tris, all_pts = [], []
#     for f in building_json.get("faces", []):
#         if f.get("surfaceType") == "floor":
#             continue
#         verts = [np.array([v["x"], v["y"], v["z"]]) for v in f["vertices"]]
#         all_pts.extend(verts)
#         if len(verts) == 4:
#             tris.extend(_triangulate_quad(*verts))
#         elif len(verts) == 3:
#             tris.append(tuple(verts))
#         elif len(verts) > 4:
#             tris.extend(_fan_triangulate(verts))
#     if not tris:
#         return None
#     pts = np.array(all_pts)
#     return Occluder(kind="building", source_id=building_json.get("id", "building"),
#                      aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tris)


# # Field-name guesses for tree entries -- NOT verified against a real export
# # (none was available when this was written). Check these against your
# # actual tree JSON and adjust if the names differ.
# _TREE_RADIUS_KEYS = ["crownRadius", "canopyRadius", "radius"]
# _TREE_DIAMETER_KEYS = ["crownDiameter", "canopyDiameter", "diameter"]
# _TREE_HEIGHT_KEYS = ["height", "totalHeight", "crownHeight"]
# _TREE_TRUNK_KEYS = ["trunkHeight", "trunk_height"]


# def resolve_tree_geometry(tree_json: dict) -> Optional[tuple]:
#     """
#     Shared field-lookup used both for the shadow-occluder sphere and for the
#     fuller trunk+crown mesh in the ParaView export. Checks a nested
#     "dimensions" sub-object first (more precise, seen in real exports) and
#     falls back to top-level fields. Returns
#     (pos [3], trunk_height, crown_height, crown_radius) or None if no
#     recognisable position/size fields are found.
#     """
#     pos = None
#     if "position" in tree_json:
#         p = tree_json["position"]
#         pos = np.array([p.get("x", 0), p.get("y", 0), p.get("z", 0)])
#     elif all(k in tree_json for k in ("x", "y", "z")):
#         pos = np.array([tree_json["x"], tree_json["y"], tree_json["z"]])
#     if pos is None:
#         return None

#     dims = tree_json.get("dimensions", {})

#     def find(keys, diameter_keys=()):
#         for k in keys:
#             if k in dims:
#                 return dims[k]
#         for k in keys:
#             if k in tree_json:
#                 return tree_json[k]
#         for k in diameter_keys:
#             if k in dims:
#                 return dims[k] / 2.0
#         for k in diameter_keys:
#             if k in tree_json:
#                 return tree_json[k] / 2.0
#         return None

#     radius = find(_TREE_RADIUS_KEYS, _TREE_DIAMETER_KEYS)
#     height = find(_TREE_HEIGHT_KEYS)
#     trunk = find(_TREE_TRUNK_KEYS)
#     if trunk is None:
#         trunk = 0.0
#     if radius is None and height is None:
#         return None
#     if radius is None:
#         radius = 0.3 * height             # rough canopy-width/height fallback ratio
#     if height is None:
#         height = trunk + 2.5 * radius
#     crown_height = max(height - trunk, 0.1)
#     return pos, trunk, crown_height, radius


# def build_tree_occluder(tree_json: dict) -> Optional[Occluder]:
#     """
#     Approximates a tree canopy as a single bounding sphere for shadow
#     purposes -- a standard simplification in urban solar-access studies
#     (real foliage isn't meshed anyway). Returns None if no recognisable
#     position/size fields are found, rather than guessing at a made-up size.
#     """
#     geo = resolve_tree_geometry(tree_json)
#     if geo is None:
#         return None
#     pos, trunk, crown_height, radius = geo
#     canopy_center = pos + np.array([0.0, 0.0, trunk + crown_height / 2.0])
#     return Occluder(kind="tree", source_id=tree_json.get("id", "tree"),
#                      aabb_min=canopy_center - radius, aabb_max=canopy_center + radius,
#                      sphere_center=canopy_center, sphere_radius=radius)


# def build_occluders_by_building(scene: dict) -> dict:
#     """
#     Returns {building_id: [Occluder, ...]} -- for each building, the list of
#     every OTHER building's triangulated mesh plus every tree's bounding
#     sphere (a building is never tested against its own geometry).
#     """
#     buildings = scene.get("buildings", [])
#     building_occs = {}
#     for b in buildings:
#         occ = build_building_occluder(b)
#         if occ is not None:
#             building_occs[b.get("id", "building")] = occ

#     tree_occs = []
#     skipped = 0
#     for t in scene.get("trees", []):
#         occ = build_tree_occluder(t)
#         if occ is not None:
#             tree_occs.append(occ)
#         else:
#             skipped += 1
#     if skipped:
#         print(f"[shadows] Skipped {skipped} tree(s) in this scene -- no recognised "
#               f"position/size fields. Check _TREE_*_KEYS in solar_thermal_model.py "
#               f"against your tree JSON schema.")

#     by_building = {}
#     for bid in building_occs:
#         others = [o for oid, o in building_occs.items() if oid != bid]
#         by_building[bid] = others + tree_occs
#     return by_building


# def _ray_aabb_hit(origin: np.ndarray, direction: np.ndarray,
#                    box_min: np.ndarray, box_max: np.ndarray) -> bool:
#     """Standard slab test for ray-AABB intersection, t > 0 only."""
#     t_min, t_max = 0.0, np.inf
#     for i in range(3):
#         d = direction[i]
#         if abs(d) < 1e-12:
#             if origin[i] < box_min[i] or origin[i] > box_max[i]:
#                 return False
#             continue
#         t1 = (box_min[i] - origin[i]) / d
#         t2 = (box_max[i] - origin[i]) / d
#         if t1 > t2:
#             t1, t2 = t2, t1
#         t_min = max(t_min, t1)
#         t_max = min(t_max, t2)
#         if t_min > t_max:
#             return False
#     return True


# def _ray_triangle_hit(origin: np.ndarray, direction: np.ndarray,
#                        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
#                        eps: float = 1e-9) -> bool:
#     """Moller-Trumbore ray-triangle intersection; only t > eps counts (so the
#     face's own near-coplanar geometry can never register as a self-hit)."""
#     edge1 = v1 - v0
#     edge2 = v2 - v0
#     h = np.cross(direction, edge2)
#     a = np.dot(edge1, h)
#     if -eps < a < eps:
#         return False
#     f = 1.0 / a
#     s = origin - v0
#     u = f * np.dot(s, h)
#     if u < 0.0 or u > 1.0:
#         return False
#     q = np.cross(s, edge1)
#     v = f * np.dot(direction, q)
#     if v < 0.0 or u + v > 1.0:
#         return False
#     t = f * np.dot(edge2, q)
#     return t > eps


# def _ray_sphere_hit(origin: np.ndarray, direction: np.ndarray,
#                      center: np.ndarray, radius: float, eps: float = 1e-9) -> bool:
#     oc = origin - center
#     b = np.dot(oc, direction)
#     c = np.dot(oc, oc) - radius * radius
#     disc = b * b - c
#     if disc < 0:
#         return False
#     sq = math.sqrt(disc)
#     t1, t2 = -b - sq, -b + sq
#     return t1 > eps or t2 > eps


# def is_shadowed(point: np.ndarray, sun_vec: np.ndarray, occluders: list) -> bool:
#     """
#     True if the ray from `point` toward the sun is blocked by any occluder
#     before reaching it. AABB broad-phase first (cheap); triangle/sphere
#     narrow-phase only for occluders the ray could plausibly hit.
#     """
#     origin = point  # small offset added by the caller along the face normal
#     for occ in occluders:
#         if not _ray_aabb_hit(origin, sun_vec, occ.aabb_min, occ.aabb_max):
#             continue
#         if occ.kind == "tree":
#             if _ray_sphere_hit(origin, sun_vec, occ.sphere_center, occ.sphere_radius):
#                 return True
#         else:
#             for v0, v1, v2 in occ.triangles:
#                 if _ray_triangle_hit(origin, sun_vec, v0, v1, v2):
#                     return True
#     return False


# class TriangleRayTracer:
#     """NumPy packet ray tracing over a median-split triangle BVH.

#     Rays in a packet share the sun direction. Bounds prune whole triangle
#     groups; leaves test at most 16 triangles at once. Both triangle sides
#     occlude, and positive hit distance plus an outward offset avoid self hits.
#     No optional ray-tracing libraries are required.
#     """

#     def __init__(self, triangles, leaf_size=16):
#         self.triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
#         self.nodes = []
#         if not len(self.triangles):
#             return
#         lo = self.triangles.min(axis=1)
#         hi = self.triangles.max(axis=1)
#         centers = (lo + hi) / 2

#         def build(ids):
#             idx = len(self.nodes)
#             bounds = (lo[ids].min(axis=0), hi[ids].max(axis=0))
#             self.nodes.append(None)
#             if len(ids) <= leaf_size:
#                 self.nodes[idx] = (*bounds, ids, -1, -1)
#             else:
#                 axis = np.argmax(np.ptp(centers[ids], axis=0))
#                 mid = len(ids) // 2
#                 ids = ids[np.argpartition(centers[ids, axis], mid)]
#                 left, right = build(ids[:mid]), build(ids[mid:])
#                 self.nodes[idx] = (*bounds, None, left, right)
#             return idx

#         build(np.arange(len(self.triangles)))

#     def blocked(self, origins, direction, chunk_size=8192):
#         origins = np.asarray(origins, dtype=float).reshape(-1, 3)
#         direction = np.asarray(direction, dtype=float)
#         direction = direction / np.linalg.norm(direction)
#         blocked = np.zeros(len(origins), dtype=bool)
#         if not self.nodes:
#             return blocked
#         v0 = self.triangles[:, 0]
#         e1 = self.triangles[:, 1] - v0
#         e2 = self.triangles[:, 2] - v0
#         cross = np.cross(direction, e2)
#         det = np.einsum('ij,ij->i', e1, cross)
#         valid = np.abs(det) > 1e-10
#         inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
#         for start in range(0, len(origins), chunk_size):
#             stack = [(0, np.arange(start, min(start+chunk_size, len(origins))))]
#             while stack:
#                 node, rays = stack.pop()
#                 rays = rays[~blocked[rays]]
#                 if not len(rays):
#                     continue
#                 lo, hi, ids, left, right = self.nodes[node]
#                 points = origins[rays]
#                 near = np.zeros(len(rays))
#                 far = np.full(len(rays), np.inf)
#                 for axis in range(3):
#                     if abs(direction[axis]) < 1e-12:
#                         outside = (points[:, axis] < lo[axis]-1e-9) | (points[:, axis] > hi[axis]+1e-9)
#                         far[outside] = -np.inf
#                     else:
#                         t1 = (lo[axis]-1e-9-points[:, axis]) / direction[axis]
#                         t2 = (hi[axis]+1e-9-points[:, axis]) / direction[axis]
#                         near = np.maximum(near, np.minimum(t1, t2))
#                         far = np.minimum(far, np.maximum(t1, t2))
#                 rays = rays[far >= near]
#                 if not len(rays):
#                     continue
#                 if ids is None:
#                     stack.extend([(left, rays), (right, rays)])
#                     continue
#                 ids = ids[valid[ids]]
#                 if not len(ids):
#                     continue
#                 delta = origins[rays, None, :] - v0[ids]
#                 u = np.einsum('rtj,tj->rt', delta, cross[ids]) * inv[ids]
#                 q = np.cross(delta, e1[ids])
#                 v = np.einsum('rtj,j->rt', q, direction) * inv[ids]
#                 distance = np.einsum('rtj,tj->rt', q, e2[ids]) * inv[ids]
#                 hit = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (distance > 1e-7)
#                 blocked[rays[np.any(hit, axis=1)]] = True
#         return blocked


# def triangle_sunlit_fraction(faces, sun_vec, ray_tracer, samples=3):
#     """Visible area estimate using one centroid or three equal-weight samples.

#     Resolution is the input mesh. Three interior barycentric samples provide
#     fractional direct-beam exposure for triangles crossed by a shadow edge.
#     """
#     if samples not in (1, 3):
#         raise ValueError("shadow_samples must be 1 or 3")
#     normals = np.array([f.normal for f in faces])
#     eligible = np.flatnonzero(normals @ sun_vec > 0) if sun_vec[2] > 0 else np.array([], dtype=int)
#     fractions = np.zeros(len(faces))
#     if not len(eligible):
#         return fractions
#     vertices = np.array([faces[i].vertices for i in eligible])
#     weights = np.array([[1/3, 1/3, 1/3]]) if samples == 1 else np.array([
#         [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
#     origins = np.einsum('sj,tjk->tsk', weights, vertices)
#     origins += normals[eligible, None, :] * _SHADOW_RAY_OFFSET
#     hits = ray_tracer.blocked(origins.reshape(-1, 3), sun_vec).reshape(-1, samples)
#     fractions[eligible] = 1 - hits.mean(axis=1)
#     return fractions


# # ==========================================================================
# # 6. Weather input (plug your real data in here)
# # ==========================================================================

# def load_weather(csv_path: Optional[str] = None, n_hours: int = 24,
#                   t_mean_c: float = 24.0, t_amp_c: float = 6.0,
#                   wind_mean_ms: float = 3.0) -> pd.DataFrame:
#     """
#     If csv_path is given, expects columns: hour, T_amb_C, wind_speed_ms
#     (optionally GHI_Wm2, DNI_Wm2, DHI_Wm2 to override the clear-sky model
#     with your own measured/forecast radiation).

#     If csv_path is None, a synthetic diurnal profile is generated as a
#     PLACEHOLDER only (sinusoidal T_amb, constant wind) so the pipeline is
#     runnable end-to-end -- swap this out for your real weather data.
#     """
#     if csv_path:
#         df = pd.read_csv(csv_path)
#         return df

#     hours = np.arange(n_hours)
#     # min T_amb near 05:00, max near 15:00 -- simple diurnal sine, placeholder only
#     t_amb = t_mean_c - t_amp_c * np.cos(2 * np.pi * (hours - 15) / 24.0)
#     wind = np.full(n_hours, wind_mean_ms)
#     return pd.DataFrame({"hour": hours, "T_amb_C": t_amb, "wind_speed_ms": wind})


# # --------------------------------------------------------------------------
# # Real weather-data-folder loader: OpenFOAM-style time-table export
# # (weatherData/MMDD/{Tambient, windSpeed, IDN, Idif, sunPosVector, ...})
# #
# # Each file looks like:
# #   // comment
# #   (
# #   ( <time_s> <scalar> )                    <- scalar field
# #   ( <time_s> ( <vx> <vy> <vz> ) )          <- vector field
# #   ...
# #   )
# # --------------------------------------------------------------------------

# _VEC_RE = re.compile(
#     r'\(\s*(-?[0-9.eE+-]+)\s+\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)\s*\)'
# )
# _SCALAR_RE = re.compile(r'\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)')


# def parse_foam_field(path: str) -> tuple:
#     """Parses one OpenFOAM-style time-table file. Returns (times_s, values),
#     values is a 1D array for scalar fields or an (N,3) array for vector fields."""
#     with open(path) as fh:
#         content = re.sub(r'//.*', '', fh.read())   # strip comments

#     vec = _VEC_RE.findall(content)
#     if vec:
#         times = np.array([float(m[0]) for m in vec])
#         values = np.array([[float(m[1]), float(m[2]), float(m[3])] for m in vec])
#         return times, values

#     sca = _SCALAR_RE.findall(content)
#     times = np.array([float(m[0]) for m in sca])
#     values = np.array([float(m[1]) for m in sca])
#     return times, values


# def resolve_weather_folder(root: str, month: int, day: int) -> str:
#     """weatherData/<root> + MMDD (e.g. month=12, day=11 -> '1211')."""
#     mmdd = f"{month:02d}{day:02d}"
#     candidate = os.path.join(root, mmdd)
#     if not os.path.isdir(candidate):
#         raise FileNotFoundError(f"No such weather folder: {candidate}")
#     return candidate


# def load_weather_folder(weather_dir: str) -> dict:
#     """
#     Reads every field file in a `weatherData/MMDD/` folder and returns a
#     dict of hourly arrays ready for `run_analysis_from_folder`:
#         hour, T_amb_C, wind_speed_ms, IDN_Wm2, Idif_Wm2,
#         sun_vec (N,3) [East, North, Up unit vector, straight from the file],
#         altitude_deg  [from solarElevation if present, else asin(sun_vec_z)]
#     Any other fields present (cloudCover, relativeHumidity, atmosphericPressure,
#     wetBulbTemperature, wambient, wPonds, windVelocity, ...) are passed through
#     unchanged under their own file name, in case you want them for other
#     calculations (e.g. evaporative cooling from the ponds).
#     """
#     files = {f: os.path.join(weather_dir, f) for f in os.listdir(weather_dir)
#               if os.path.isfile(os.path.join(weather_dir, f))}

#     parsed = {name: parse_foam_field(path) for name, path in files.items()}
#     # all fields share the same time base -- use whichever field is present first
#     times_s = next(iter(parsed.values()))[0]
#     hours = times_s / 3600.0

#     def get(name, default=None):
#         return parsed[name][1] if name in parsed else default

#     t_amb_k = get("Tambient")
#     wx = dict(
#         hour=hours,
#         T_amb_C=(t_amb_k - 273.15) if t_amb_k is not None else None,
#         wind_speed_ms=get("windSpeed"),
#         IDN_Wm2=get("IDN", np.zeros_like(hours)),
#         Idif_Wm2=get("Idif", np.zeros_like(hours)),
#     )

#     if "sunPosVector" in parsed:
#         wx["sun_vec"] = parsed["sunPosVector"][1]                 # (N,3) already East/North/Up
#         lengths = np.linalg.norm(wx["sun_vec"], axis=1)
#         if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
#             raise ValueError("sunPosVector must contain finite, nonzero vectors")
#         wx["sun_vec"] = wx["sun_vec"] / lengths[:, None]
#         wx["altitude_deg"] = np.degrees(np.arcsin(np.clip(wx["sun_vec"][:, 2], -1, 1)))
#     elif "solarElevation" in parsed and "solarAzimuth" in parsed:
#         alt = parsed["solarElevation"][1]
#         az = parsed["solarAzimuth"][1]
#         wx["altitude_deg"] = alt
#         wx["sun_vec"] = np.array([sun_unit_vector(a, z) for a, z in zip(alt, az)])
#     else:
#         raise ValueError("Weather folder has neither sunPosVector nor "
#                           "solarElevation+solarAzimuth -- cannot get sun direction.")

#     if wx["T_amb_C"] is None:
#         raise ValueError("Weather folder is missing a 'Tambient' file.")
#     if wx["wind_speed_ms"] is None:
#         raise ValueError("Weather folder is missing a 'windSpeed' file.")

#     # keep any extra fields around too, unmodified
#     for name, (t, v) in parsed.items():
#         if name not in ("Tambient", "windSpeed", "IDN", "Idif", "sunPosVector",
#                          "solarElevation", "solarAzimuth"):
#             wx[name] = v

#     return wx


# # ==========================================================================
# # 7. Shared per-hour physics loop + per-building RC response
# #    (used by BOTH the clear-sky/solar-formula path and the real
# #    weather-data-folder path below)
# # ==========================================================================

# _SHADOW_RAY_OFFSET = 1e-3   # m, nudge the ray origin off the face surface

# def _simulate(faces: list, capacitance_by_building: dict, records: list,
#               ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#               occluders_by_building: Optional[dict] = None,
#               t_initial_c: Optional[float] = None,
#               ray_tracer=None, shadow_samples: int = 3,
#               wall_settings: Optional[WallSettings] = None,
#               surface_group: str = "building",
#               wetness: float = 0.0, back_follows_ambient: bool = False,
#               workers: int = 1) -> tuple:
#     """Per-cell solar/thermal balance plus the separate legacy bulk RC estimate.

#     OBJ cells use packet rays; JSON faces retain their centroid visibility test.
#     Output temperatures are states at the labelled timestamps. The first
#     output is the initial condition; every later record supplies the constant
#     boundary condition over the interval ending at that timestamp. This keeps
#     each output's shadow/solar fields aligned with its thermal state.
#     """
#     excluded = {"ceiling"} | (set() if INCLUDE_FLOOR else {"floor"})
#     exposed = [f for f in faces if f.surface_type not in excluded]
#     if not exposed or not records:
#         raise ValueError("Simulation requires exposed faces and weather records")
#     bids = list(capacitance_by_building)
#     bindex = {bid: i for i, bid in enumerate(bids)}
#     indices = np.array([bindex[f.building_id] for f in exposed])
#     normals = np.array([f.normal for f in exposed])
#     areas = np.array([f.area for f in exposed])
#     albedo = np.array([f.albedo for f in exposed])
#     emissivity = np.array([f.emissivity for f in exposed])
#     view_sky = (1 + normals[:, 2]) / 2
#     static = dict(building_id=[f.building_id for f in exposed],
#                   face_id=[f.face_id for f in exposed],
#                   parent_face_id=[f.parent_face_id or f.face_id for f in exposed],
#                   surface_type=[f.surface_type for f in exposed],
#                   surface_group=surface_group, area_m2=areas,
#                   tilt_deg=[f.tilt_deg for f in exposed])
#     frames, bulk_rows = [], []
#     bulk_T = np.full(len(bids), t_initial_c if t_initial_c is not None else records[0]["T_amb_C"], dtype=float)
#     caps = np.array([capacitance_by_building[b] for b in bids])
#     wall_settings = wall_settings or WallSettings()
#     wall_initial_c = (t_initial_c if t_initial_c is not None
#                       else wall_settings.indoor_temperature_c)
#     slab = SlabConduction([f.thermal_conductivity for f in exposed],
#                           [f.density for f in exposed], [f.specific_heat for f in exposed],
#                           emissivity, wall_initial_c, wall_settings)
#     # Sunlit fractions depend only on geometry and sun position, not on the
#     # thermal state, so all hours are ray-traced up front (optionally threaded).
#     shadow_fractions = {}
#     if ray_tracer is not None:
#         sun_vecs = [np.asarray(r["sun_vec"], float) / np.linalg.norm(r["sun_vec"]) for r in records]
#         todo = [i for i, r in enumerate(records) if r["altitude_deg"] > 0]
#         verts_arr = np.array([f.vertices for f in exposed], dtype=float)
#         print(f"[shadows] {surface_group}: tracing {len(exposed)} cells x {len(todo)} hours "
#               f"(threads={max(1, int(workers))})", flush=True)

#         def _shadow_job(i):
#             return i, sunlit_fractions_from_arrays(normals, verts_arr, sun_vecs[i],
#                                                    ray_tracer, shadow_samples)
#         if workers > 1 and len(todo) > 1:
#             with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as pool:
#                 shadow_fractions = dict(pool.map(_shadow_job, todo))
#         else:
#             shadow_fractions = dict(_shadow_job(i) for i in todo)
#     for ri, rec in enumerate(records):
#         t_amb = rec["T_amb_C"]
#         sun = np.asarray(rec["sun_vec"], dtype=float)
#         sun = sun / np.linalg.norm(sun)
#         alt = rec["altitude_deg"]
#         h_c = convective_h(rec["wind_speed_ms"])
#         cos_theta = normals @ sun
#         self_lit = (alt > 0) & (cos_theta > 0)
#         fraction = self_lit.astype(float)
#         if ray_tracer is not None and alt > 0:
#             fraction = shadow_fractions[ri]
#         elif occluders_by_building:
#             for i in np.flatnonzero(self_lit):
#                 f = exposed[i]
#                 fraction[i] = not is_shadowed(f.center + f.normal*_SHADOW_RAY_OFFSET,
#                                               sun, occluders_by_building.get(f.building_id, []))
#         shadow_fraction = np.where(self_lit, 1-fraction, 0)
#         direct = max(rec["i_dn"], 0) * np.maximum(cos_theta, 0) * fraction
#         diffuse = max(rec["i_dh"], 0) * view_sky
#         ghi = max(rec["i_dn"], 0) * max(sun[2], 0) + max(rec["i_dh"], 0)
#         reflected = ground_albedo * ghi * (1-view_sky)
#         total = direct + diffuse + reflected
#         solair = sol_air_temperature(t_amb, total, h_c, albedo, view_sky, emissivity)
#         q_solar = (1-albedo)*total
#         ta_k = t_amb + 273.15
#         # Tangent linearization at ambient, frozen for this weather interval:
#         # q_lw(Ts) ~= q_lw(Ta) + h_rad*(Ts-Ta). This permits the exact
#         # matrix-exponential wall update; full Ts^4 radiation is nonlinear.
#         q_lw_ambient = emissivity*STEFAN_BOLTZMANN*view_sky*(ta_k**4-sky_temperature(ta_k)**4)
#         h_rad = 4*emissivity*STEFAN_BOLTZMANN*ta_k**3
#         # Evaporation / transpiration, same tangent linearisation at ambient:
#         # q_evap(Ts) ~= e0 + e_slope*(Ts-Ta); coef = wetness*L*h_c/cp_air (Lewis).
#         e0 = e_slope = 0.0
#         if wetness > 0:
#             p_atm = rec.get("p_atm", 101325.0)
#             w_amb = rec.get("w_amb")
#             if w_amb is None:
#                 w_amb = _humidity_ratio(t_amb, rec.get("rh", 0.5), p_atm)
#             ws, dws = _saturation_humidity_ratio(t_amb, p_atm)
#             coef = wetness*LATENT_HEAT_VAPORIZATION*h_c/CP_AIR
#             e0, e_slope = coef*(ws-w_amb), coef*dws
#         external_h = h_c+h_rad+e_slope
#         external_source = q_solar-q_lw_ambient-e0 + external_h*t_amb
#         if back_follows_ambient:
#             slab.set_back(h_c, t_amb)

#         if ri > 0:
#             dt = (rec["hour"] - records[ri-1]["hour"])*3600
#             if dt <= 0:
#                 raise ValueError("Weather times must be strictly increasing")
#             slab.advance(dt, external_h, external_source)

#         ts = slab.temperatures[:,0].copy()
#         q_conv = h_c*(ts-t_amb)
#         q_evap = e0 + e_slope*(ts-t_amb)
#         q_lw = q_lw_ambient + h_rad*(ts-t_amb)
#         wall_fields = slab.diagnostics(external_h, external_source)
#         wall_fields["stored_heat_J"] = wall_fields["stored_heat_Jm2"]*areas
#         wall_fields["wall_capacity_JK"] = wall_fields["wall_capacity_Jm2K"]*areas
#         wall_fields["slab_capacity_JK"] = wall_fields["slab_capacity_Jm2K"]*areas
#         frames.append(pd.DataFrame(dict(static, hour=rec["hour"], solar_altitude_deg=alt,
#             self_lit=self_lit, shadowed=shadow_fraction > 0, sunlit=fraction > 0,
#             sunlit_fraction=fraction, shadow_fraction=shadow_fraction, cos_theta=cos_theta,
#             I_direct_Wm2=direct, I_diffuse_Wm2=diffuse, I_reflected_Wm2=reflected,
#             I_total_Wm2=total, h_conv_Wm2K=h_c, T_amb_C=t_amb,
#             T_solair_C=solair, T_surface_C=ts, q_solar_absorbed_Wm2=q_solar,
#             q_conv_out_Wm2=q_conv, q_longwave_out_Wm2=q_lw, q_evap_out_Wm2=q_evap,
#             wall_thickness_m=wall_settings.thickness_m,
#             slab_thickness_m=wall_settings.thickness_m,
#             T_indoor_air_C=slab.back_temperature,
#             T_back_boundary_C=slab.back_temperature,
#             h_indoor_Wm2K=slab.back_h,
#             h_back_Wm2K=slab.back_h, **wall_fields)))
#         ah = np.bincount(indices, weights=areas*h_c, minlength=len(bids))
#         aht = np.bincount(indices, weights=areas*h_c*solair, minlength=len(bids))
#         teq = np.divide(aht, ah, out=np.full(len(bids), t_amb, dtype=float), where=ah > 0)
#         tau = np.divide(caps, ah, out=np.full(len(bids), np.inf), where=ah > 0)
#         for bi, bid in enumerate(bids):
#             bulk_rows.append(dict(building_id=bid, hour=rec["hour"], T_amb_C=t_amb,
#                 T_eq_C=teq[bi], h_eff_WK=ah[bi], tau_hr=tau[bi]/3600,
#                 T_building_C=bulk_T[bi]))
#         if ri+1 < len(records):
#             dt_bulk = (records[ri+1]["hour"] - rec["hour"])*3600
#             if dt_bulk <= 0:
#                 raise ValueError("Weather times must be strictly increasing")
#             decay = np.exp(-np.divide(
#                 dt_bulk, tau, out=np.full_like(tau, np.inf), where=tau > 0))
#             bulk_T = teq + (bulk_T-teq)*decay
#         if ray_tracer is not None:
#             print(f"[simulation] hour {rec['hour']:g}: {len(exposed)} surface cells", flush=True)
#     return pd.concat(frames, ignore_index=True), pd.DataFrame(bulk_rows)


# def compute_capacitances(scene: dict, faces: list) -> dict:
#     """{building_id: C [J/K]} for every building in the scene, C = rho*V*cp."""
#     caps = {}
#     faces_by_bid = {}
#     for f in faces:
#         faces_by_bid.setdefault(f.building_id, f)
#     for bid, f0 in faces_by_bid.items():
#         vol = building_volume(scene, bid)
#         caps[bid] = f0.density * vol * f0.specific_heat
#     return caps


# def run_analysis(json_path: str, day: date, weather: pd.DataFrame,
#                   utc_offset_hr: float = 0.0, north_offset_deg: float = 0.0,
#                   site_altitude_km: float = 0.0,
#                   consider_shadows: bool = True,
#                   t_initial_c: Optional[float] = None,
#                   wall_settings: Optional[WallSettings] = None) -> tuple:
#     """
#     CLEAR-SKY / SOLAR-FORMULA path: use this when you do NOT have a real
#     weather-data export, only a date + location + basic T_amb/wind. Solar
#     position and irradiance are both modelled analytically (see functions
#     `solar_position` and `hottel_clear_sky` above).
#     Returns (face_hourly_df, building_hourly_df, faces).
#     """
#     scene = _scene_from(json_path)
#     faces = extract_faces(scene)
#     lat, lon = site_location(scene)
#     capacitances = compute_capacitances(scene, faces)
#     occluders = build_occluders_by_building(scene) if consider_shadows else None
#     day_of_year = day.timetuple().tm_yday

#     records = []
#     for _, row in weather.iterrows():
#         hr = row["hour"]
#         alt, az = solar_position(lat, lon, day_of_year, hr, utc_offset_hr)
#         sun_vec = sun_unit_vector(alt, az, north_offset_deg)
#         if "GHI_Wm2" in row and not pd.isna(row.get("DNI_Wm2", np.nan)):
#             i_dn, i_dh = row["DNI_Wm2"], row["DHI_Wm2"]
#         else:
#             i_dn, i_dh = hottel_clear_sky(alt, site_altitude_km)
#         records.append(dict(hour=hr, T_amb_C=row["T_amb_C"], wind_speed_ms=row["wind_speed_ms"],
#                              sun_vec=sun_vec, altitude_deg=alt, i_dn=i_dn, i_dh=i_dh))

#     face_df, building_df = _simulate(faces, capacitances, records,
#                                       occluders_by_building=occluders, t_initial_c=t_initial_c,
#                                       wall_settings=wall_settings)
#     return face_df, building_df, faces


# def run_analysis_from_folder(json_path: str, weather_dir: str,
#                               ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                               consider_shadows: bool = True,
#                               t_initial_c: Optional[float] = None,
#                   wall_settings: Optional[WallSettings] = None) -> tuple:
#     """
#     REAL WEATHER-DATA path (preferred whenever you have it): reads Tambient,
#     windSpeed, IDN, Idif and sunPosVector straight from an exported
#     `weatherData/MMDD/` folder (see `load_weather_folder`). No solar-position
#     formula and no clear-sky model are used here at all -- the sun vector and
#     irradiance are exactly what your own pipeline computed, in the same
#     (x=East, y=North, z=Up) axes as the building JSON's face normals, so
#     there is no coordinate-convention guessing.
#     Returns (face_hourly_df, building_hourly_df, faces).
#     """
#     scene = _scene_from(json_path)
#     faces = extract_faces(scene)
#     capacitances = compute_capacitances(scene, faces)
#     occluders = build_occluders_by_building(scene) if consider_shadows else None
#     wx = load_weather_folder(weather_dir)

#     records = []
#     n = len(wx["hour"])
#     for i in range(n):
#         records.append(dict(
#             hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
#             sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
#             i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i],
#         ))

#     face_df, building_df = _simulate(faces, capacitances, records, ground_albedo=ground_albedo,
#                                       occluders_by_building=occluders, t_initial_c=t_initial_c,
#                                       wall_settings=wall_settings)
#     return face_df, building_df, faces


# # ==========================================================================
# # 8. ParaView export -- writes the same results as a time series of legacy
# #    VTK PolyData files (plus a .pvd collection), so you can open one file
# #    in ParaView, scrub an "hour of day" timeline, and see the building
# #    (colored by sol-air temperature, shadowed/sunlit status, or irradiance)
# #    together with the trees, with no plugins and no `pip install vtk` --
# #    this writes plain-text legacy VTK, which ParaView reads natively.
# # ==========================================================================

# _SURFACE_TYPE_ID = {"wall": 0, "roof": 1, "floor": 2, "ceiling": 3, "tree": 5, "ground": 6,
#                     "pathway": 7, "lawn": 8, "pond": 9}
# TREE_TRUNK_TYPE_ID = 4
# TREE_CROWN_TYPE_ID = 5


# def _ring_polygon(cx: float, cy: float, z: float, radius: float, n: int) -> list:
#     return [(cx + radius * math.cos(2 * math.pi * i / n),
#               cy + radius * math.sin(2 * math.pi * i / n), z) for i in range(n)]


# def _cylinder_polys(cx: float, cy: float, z0: float, z1: float, radius: float, n: int = 8) -> list:
#     """Side quads + a bottom cap (top is hidden inside the tree crown)."""
#     ring0 = _ring_polygon(cx, cy, z0, radius, n)
#     ring1 = _ring_polygon(cx, cy, z1, radius, n)
#     polys = []
#     for i in range(n):
#         j = (i + 1) % n
#         polys.append([ring0[i], ring0[j], ring1[j], ring1[i]])
#     polys.append(list(reversed(ring0)))   # bottom cap, wound outward-facing-down
#     return polys


# def _cone_polys(cx: float, cy: float, base_z: float, apex_z: float, radius: float, n: int = 8) -> list:
#     """Side triangles (apex at top) + a base cap."""
#     ring = _ring_polygon(cx, cy, base_z, radius, n)
#     apex = (cx, cy, apex_z)
#     polys = [[ring[i], ring[(i + 1) % n], apex] for i in range(n)]
#     polys.append(list(ring))              # base cap
#     return polys


# def _tree_mesh(tree_json: dict, trunk_radius_ratio: float = 0.15) -> tuple:
#     """Returns (polygons, surface_type_ids) for one tree's trunk + crown, or
#     (None, None) if the tree's geometry couldn't be resolved (see
#     resolve_tree_geometry)."""
#     geo = resolve_tree_geometry(tree_json)
#     if geo is None:
#         return None, None
#     pos, trunk_h, crown_h, crown_r = geo
#     trunk_r = max(crown_r * trunk_radius_ratio, 0.1)
#     trunk_polys = _cylinder_polys(pos[0], pos[1], pos[2], pos[2] + trunk_h, trunk_r)
#     crown_polys = _cone_polys(pos[0], pos[1], pos[2] + trunk_h, pos[2] + trunk_h + crown_h, crown_r)
#     polys = trunk_polys + crown_polys
#     type_ids = [TREE_TRUNK_TYPE_ID] * len(trunk_polys) + [TREE_CROWN_TYPE_ID] * len(crown_polys)
#     return polys, type_ids


# def _write_vtk_polydata_slow(path: str, polygons: list, cell_scalars: dict) -> None:
#     """
#     polygons: list of polygons, each a list of (x,y,z) tuples (a planar loop).
#         Points are NOT shared between polygons -- every polygon gets its own
#         copy of its corner points. That is deliberate: colouring is by CELL
#         data (one value per face), and un-shared points guarantee ParaView
#         renders each face as a single flat colour instead of interpolating
#         across a shared edge with a differently-valued neighbour.
#     cell_scalars: {name: (values, "float"|"int")}, one value per polygon,
#         same length and order as `polygons`.
#     """
#     points, polygon_indices = [], []
#     idx = 0
#     for poly in polygons:
#         n = len(poly)
#         polygon_indices.append(list(range(idx, idx + n)))
#         points.extend(poly)
#         idx += n

#     with open(path, "w") as f:
#         f.write("# vtk DataFile Version 3.0\n")
#         f.write("solar_thermal_model export\n")
#         f.write("ASCII\n")
#         f.write("DATASET POLYDATA\n")
#         f.write(f"POINTS {len(points)} float\n")
#         for p in points:
#             f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
#         total_size = sum(len(idxs) + 1 for idxs in polygon_indices)
#         f.write(f"POLYGONS {len(polygon_indices)} {total_size}\n")
#         for idxs in polygon_indices:
#             f.write(f"{len(idxs)} " + " ".join(str(i) for i in idxs) + "\n")
#         f.write(f"CELL_DATA {len(polygons)}\n")
#         for name, (values, dtype) in cell_scalars.items():
#             f.write(f"SCALARS {name} {dtype} 1\n")
#             f.write("LOOKUP_TABLE default\n")
#             fmt = "{}\n" if dtype == "int" else "{:.4f}\n"
#             for v in values:
#                 f.write(fmt.format(v))


# def export_vtk_series(json_path: str, weather_dir: str, out_dir: str,
#                        consider_shadows: bool = True,
#                        wall_settings: Optional[WallSettings] = None,
#                        t_initial_c: Optional[float] = None) -> str:
#     """
#     Runs the real-weather-data analysis (see `run_analysis_from_folder`) and
#     writes one legacy-VTK PolyData file per hour plus a `scene.pvd` ParaView
#     collection tying them together as a time series. Open `scene.pvd` in
#     ParaView, colour by any of the per-face arrays below, and scrub the time
#     slider to watch temperature and shadowing move through the day together.

#     Per-face (cell) data written on every building face:
#         T_C            transient outer surface temperature, deg C
#         I_total_Wm2    total incident solar irradiance, W/m^2
#         shadowed       1 if ray-blocked by another building/tree this hour
#         sunlit         1 if actually receiving direct beam this hour
#                         (self_lit AND NOT shadowed)
#         surface_type_id  0=wall 1=roof 2=floor 3=ceiling 4=tree trunk 5=tree crown
#         building_index  which building/floor this face belongs to (-1 for trees)
#         is_building     1 for building faces, 0 for tree geometry

#     Floor/ceiling faces are excluded from the thermal simulation itself (see
#     the module docstring) but ARE still drawn here, with T_C set to that
#     hour's ambient air temperature, so the building reads as a solid, complete
#     shape rather than having visible holes -- filter on surface_type_id in
#     ParaView if you want only the faces that were actually simulated.
#     Trees are static (their geometry doesn't change hour to hour) but are
#     re-written into every frame file so each one opens as a complete scene on
#     its own; T_C for tree geometry is likewise just that hour's ambient
#     temperature (canopy temperature isn't modelled).
#     """
#     scene = _scene_from(json_path)
#     faces = extract_faces(scene)
#     face_df, building_df, _ = run_analysis_from_folder(
#         json_path, weather_dir, consider_shadows=consider_shadows,
#         wall_settings=wall_settings, t_initial_c=t_initial_c)
#     os.makedirs(out_dir, exist_ok=True)

#     tree_polys, tree_type_ids = [], []
#     n_tree_geo_ok, n_tree_geo_skipped = 0, 0
#     for t in scene.get("trees", []):
#         polys, type_ids = _tree_mesh(t)
#         if polys is None:
#             n_tree_geo_skipped += 1
#             continue
#         n_tree_geo_ok += 1
#         tree_polys.extend(polys)
#         tree_type_ids.extend(type_ids)
#     if n_tree_geo_skipped:
#         print(f"[vtk export] Skipped {n_tree_geo_skipped} tree(s) with unrecognised geometry fields.")

#     building_ids_sorted = sorted({f.building_id for f in faces})
#     building_index_map = {bid: i for i, bid in enumerate(building_ids_sorted)}

#     face_lookup = {}
#     for _, r in face_df.iterrows():
#         face_lookup[(r["building_id"], r["face_id"], r["hour"])] = r

#     hours = sorted(face_df["hour"].unique())
#     pvd_entries = []
#     for h in hours:
#         amb_rows = building_df[building_df["hour"] == h]
#         amb_c = float(amb_rows["T_amb_C"].iloc[0]) if len(amb_rows) else 0.0

#         polys, T_C, shadowed, sunlit, I_tot, surf_id, bidx, is_bldg = [], [], [], [], [], [], [], []
#         for f in faces:
#             row = face_lookup.get((f.building_id, f.face_id, h))
#             polys.append([tuple(v) for v in f.vertices])
#             if row is None:                          # floor/ceiling: not simulated
#                 T_C.append(amb_c); shadowed.append(0); sunlit.append(0); I_tot.append(0.0)
#             else:
#                 T_C.append(float(row["T_surface_C"])); shadowed.append(int(bool(row["shadowed"])))
#                 sunlit.append(int(bool(row["sunlit"]))); I_tot.append(float(row["I_total_Wm2"]))
#             surf_id.append(_SURFACE_TYPE_ID.get(f.surface_type, -1))
#             bidx.append(building_index_map[f.building_id])
#             is_bldg.append(1)

#         polys.extend(tree_polys)
#         T_C.extend([amb_c] * len(tree_polys))
#         shadowed.extend([0] * len(tree_polys))
#         sunlit.extend([0] * len(tree_polys))
#         I_tot.extend([0.0] * len(tree_polys))
#         surf_id.extend(tree_type_ids)
#         bidx.extend([-1] * len(tree_polys))
#         is_bldg.extend([0] * len(tree_polys))

#         fname = f"frame_{int(round(h * 100)):04d}.vtk"   # e.g. hour 13.5 -> frame_1350.vtk
#         _write_vtk_polydata(os.path.join(out_dir, fname), polys, {
#             "T_C": (T_C, "float"),
#             "I_total_Wm2": (I_tot, "float"),
#             "shadowed": (shadowed, "int"),
#             "sunlit": (sunlit, "int"),
#             "surface_type_id": (surf_id, "int"),
#             "building_index": (bidx, "int"),
#             "is_building": (is_bldg, "int"),
#         })
#         pvd_entries.append((float(h), fname))

#     pvd_path = os.path.join(out_dir, "scene.pvd")
#     with open(pvd_path, "w") as f:
#         f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
#         for t, fname in pvd_entries:
#             f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
#         f.write('</Collection>\n</VTKFile>\n')

#     print(f"[vtk export] Wrote {len(pvd_entries)} frame(s) + scene.pvd to {out_dir} "
#           f"({n_tree_geo_ok} tree(s) meshed).")
#     return pvd_path


# # ==========================================================================
# # 9. OBJ pipeline: retain input triangles for partial-wall shadows.
# #    Coplanar grouping is used only for parent IDs and bulk-volume estimates.
# #    OBJ does not carry thermal materials: defaults below assume concrete.
# #    Spatial resolution is limited by the input mesh; use a finer OBJ for
# #    sharper shadow edges. Trees are opaque meshes, without leaf transmission.
# # ==========================================================================

# DEFAULT_MATERIAL = dict(thermal_conductivity=2.3, density=2400, specific_heat=880)
# DEFAULT_ALBEDO_WALL = 0.3
# DEFAULT_ALBEDO_ROOF = 0.4
# ROOF_NORMAL_Z_THRESHOLD = 0.7   # |normal.z| above this -> "roof", else "wall"


# def _obj_vertex_index(token: str, n_verts_so_far: int) -> int:
#     """
#     Parses one OBJ face-vertex token in any of the four standard forms
#     (`v`, `v/vt`, `v/vt/vn`, `v//vn`) and returns a 0-based vertex index.
#     Handles negative (relative-to-current-end) indices too, per the OBJ
#     spec.
#     """
#     idx = int(token.split("/")[0])
#     return (n_verts_so_far + idx) if idx < 0 else (idx - 1)


# def load_obj_mesh(path: str) -> tuple:
#     """
#     OBJ parser for the buildings_refined.obj / trees.obj this pipeline
#     consumes. json_to_obj.py's own write_obj() only ever emits plain
#     `v x y z` / `f i j k` lines, but real pipelines often pass the file
#     through something else afterwards (normal generation, mesh repair,
#     etc.) that adds `vn`/`vt` references -- seen in practice as
#     `f 1//1 2//1 3//1` style faces. This is intentionally more permissive
#     than that one exact format: it accepts all four standard face-vertex
#     forms (`v`, `v/vt`, `v/vt/vn`, `v//vn`), skips `vn`/`vt` lines outright
#     (this pipeline recomputes every normal from triangle geometry anyway,
#     never reads one from the file), and fan-triangulates any face with more
#     than 3 vertices rather than assuming triangles-only.
#     Returns (vertices [N,3] float array, triangles [list of (i,j,k) 0-based
#     int tuples]).
#     """
#     verts, tris = [], []
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("v "):
#                 p = line.split()
#                 verts.append((float(p[1]), float(p[2]), float(p[3])))
#             elif line.startswith("f "):
#                 tokens = line.split()[1:]
#                 idxs = [_obj_vertex_index(tok, len(verts)) for tok in tokens]
#                 for t in range(1, len(idxs) - 1):     # fan triangulation (no-op for triangles)
#                     tris.append((idxs[0], idxs[t], idxs[t + 1]))
#     return np.array(verts, dtype=float), tris


# def _triangle_normal_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple:
#     n = np.cross(v1 - v0, v2 - v0)
#     area = float(np.linalg.norm(n)) / 2.0
#     if area < 1e-12:
#         return None, 0.0
#     return n / (2.0 * area), area


# def group_coplanar(vertices: np.ndarray, triangles: list,
#                     normal_decimals: int = 3, offset_decimals: int = 2) -> dict:
#     """
#     Merges triangles sharing the same (rounded outward normal, rounded plane
#     offset) into "logical faces". Exact for flat, regularly-subdivided quads
#     (this OBJ format's building geometry): every wall segment along one
#     straight facade -- including across every stacked storey of a
#     multi-floor tower, since they're all perfectly coplanar -- collapses
#     back into ONE logical wall face.
#     Returns {plane_key: {"triangles": [(i,j,k), ...], "normal", "area",
#                           "center", "z_min", "z_max"}}.
#     """
#     groups = {}
#     for (i, j, k) in triangles:
#         v0, v1, v2 = vertices[i], vertices[j], vertices[k]
#         n, area = _triangle_normal_area(v0, v1, v2)
#         if n is None:
#             continue
#         d = float(np.dot(n, v0))
#         key = (round(float(n[0]), normal_decimals), round(float(n[1]), normal_decimals),
#                round(float(n[2]), normal_decimals), round(d, offset_decimals))
#         g = groups.setdefault(key, dict(triangles=[], normal_sum=np.zeros(3), area=0.0,
#                                           centroid_sum=np.zeros(3), z_min=np.inf, z_max=-np.inf))
#         centroid = (v0 + v1 + v2) / 3.0
#         g["triangles"].append((i, j, k))
#         g["normal_sum"] += n * area
#         g["area"] += area
#         g["centroid_sum"] += centroid * area
#         g["z_min"] = min(g["z_min"], float(v0[2]), float(v1[2]), float(v2[2]))
#         g["z_max"] = max(g["z_max"], float(v0[2]), float(v1[2]), float(v2[2]))

#     out = {}
#     for key, g in groups.items():
#         if g["area"] <= 0:
#             continue
#         normal = g["normal_sum"] / (np.linalg.norm(g["normal_sum"]) + 1e-12)
#         center = g["centroid_sum"] / g["area"]
#         out[key] = dict(triangles=g["triangles"], normal=normal, area=g["area"],
#                           center=center, z_min=g["z_min"], z_max=g["z_max"])
#     return out


# def _mesh_to_occluder(vertices: np.ndarray, triangles: list, source_id: str) -> Optional[Occluder]:
#     if not triangles:
#         return None
#     tri_pts = [(vertices[i], vertices[j], vertices[k]) for (i, j, k) in triangles]
#     idx = sorted({v for tri in triangles for v in tri})
#     pts = vertices[idx]
#     return Occluder(kind="building", source_id=source_id,
#                      aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tri_pts)


# def _position_key(p, decimals: int = 4) -> tuple:
#     return (round(float(p[0]), decimals), round(float(p[1]), decimals), round(float(p[2]), decimals))


# def _obj_components_by_shared_position(vertices: np.ndarray, triangles: list,
#                                          decimals: int = 4) -> list:
#     """
#     Union-find over vertex POSITIONS, not indices -- this is the piece that
#     was missing before. json_to_obj.py does not share vertex indices across
#     different original JSON faces, even when they are geometrically
#     touching (e.g. two wall quads meeting at a shared vertical edge), so an
#     index-based union-find (see _obj_connected_components, used for trees)
#     cannot tell "two touching walls of the same building" apart from "two
#     unrelated triangles that happen to have unrelated index numbers" -- it
#     under-merges everything into one-triangle-ish fragments. Position-based
#     matching (round each vertex to `decimals` and treat equal rounded
#     positions as the same point) is what actually reconstructs "which
#     triangles are physically part of the same connected building".
#     Returns a list of triangle-index lists, one per physically distinct
#     building.
#     """
#     pos_to_id: dict = {}
#     vertex_logical_id = np.empty(len(vertices), dtype=np.int64)
#     for v_idx in range(len(vertices)):
#         key = _position_key(vertices[v_idx], decimals)
#         lid = pos_to_id.get(key)
#         if lid is None:
#             lid = len(pos_to_id)
#             pos_to_id[key] = lid
#         vertex_logical_id[v_idx] = lid

#     parent = list(range(len(pos_to_id)))

#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra != rb:
#             parent[ra] = rb

#     for (i, j, k) in triangles:
#         li, lj, lk = int(vertex_logical_id[i]), int(vertex_logical_id[j]), int(vertex_logical_id[k])
#         union(li, lj)
#         union(lj, lk)

#     comp_triangles: dict = {}
#     for t_idx, (i, j, k) in enumerate(triangles):
#         root = find(int(vertex_logical_id[i]))
#         comp_triangles.setdefault(root, []).append(t_idx)
#     return list(comp_triangles.values())


# def _obj_connected_components(vertices: np.ndarray, triangles: list) -> list:
#     """
#     Union-find over shared vertex INDICES. This groups triangles generated
#     together as one contiguous primitive (e.g. one tree's trunk cylinder) --
#     json_to_obj.py DOES share vertex indices within a single primitive's own
#     subdivision, so this works there, but NOT across different original
#     JSON faces (see _obj_components_by_shared_position for that -- used for
#     buildings, where genuine cross-face adjacency detection matters).
#     Verified on a real trees.obj: 49 trees x (1 trunk + 1 crown, each its
#     own component) = 98 components.
#     Returns a list of triangle-index lists, one per component.
#     """
#     parent = list(range(len(vertices)))

#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra != rb:
#             parent[ra] = rb

#     for (i, j, k) in triangles:
#         union(i, j)
#         union(j, k)

#     comp_triangles = {}
#     for t_idx, (i, j, k) in enumerate(triangles):
#         comp_triangles.setdefault(find(i), []).append(t_idx)
#     return list(comp_triangles.values())


# def build_tree_occluders_from_obj(trees_obj_path: str) -> list:
#     """Loads trees.obj and returns one Occluder per connected primitive
#     (trunk/crown separately) -- more, smaller occluders only help the
#     broad-phase AABB culling, so there's no reason to merge them further."""
#     if not trees_obj_path or not os.path.isfile(trees_obj_path):
#         return []
#     V, F = load_obj_mesh(trees_obj_path)
#     if not F:
#         return []
#     occluders = []
#     for c_idx, tri_indices in enumerate(_obj_connected_components(V, F)):
#         occ = _mesh_to_occluder(V, [F[t] for t in tri_indices], f"tree_part_{c_idx}")
#         if occ is not None:
#             occluders.append(occ)
#     return occluders


# # --------------------------------------------------------------------------
# # 9b. Multi-surface OBJ pipeline: buildings, ground, pathway, lawn, pond, trees
# #
# #     Every component is optional. A component is solved when its OBJ holds
# #     valid triangles (or, for trees only, when the JSON describes them);
# #     anything missing or unusable is reported in the log and skipped.
# #     Geometry comes from the OBJ files; material properties come from the
# #     scene JSON when it has them, otherwise from the documented defaults in
# #     LAYER_DEFAULTS / DEFAULT_MATERIAL (the log says which, per property).
# # --------------------------------------------------------------------------

# LATENT_HEAT_VAPORIZATION = 2.45e6      # J/kg
# CP_AIR = 1006.0                        # J/kg/K
# LAYER_CLASSES = ("ground", "pathway", "lawn", "pond", "tree")
# LAYER_PRIORITY = ("pond", "pathway", "lawn", "ground")   # top-most first


# def log(message: str) -> None:
#     print(message, flush=True)


# def warn(message: str) -> None:
#     print(f"[warning] {message}", flush=True)


# def _scene_from(scene_or_path):
#     """Accept either a parsed scene dict or a path to a scene JSON."""
#     return scene_or_path if isinstance(scene_or_path, dict) else load_scene(scene_or_path)


# # ---- material properties from the scene JSON -----------------------------

# def _norm_key(key) -> str:
#     return re.sub(r"[^a-z0-9]", "", str(key).lower())


# _PROPERTY_ALIASES = {
#     "thermal_conductivity": ("thermalconductivity", "conductivity"),
#     "density": ("density", "rho"),
#     "specific_heat": ("specificheat", "cp"),
#     "albedo": ("albedo", "reflectivity"),
#     "emissivity": ("emissivity", "emittance"),
#     "depth_m": ("depthm", "depth", "thicknessm", "thickness"),
#     "wetness": ("wetness", "evaporationfraction", "moistureavailability"),
#     "mixing_factor": ("mixingfactor", "mixingmultiplier"),
# }
# _CONTAINER_KEYS = ("thermalproperties", "thermalprops")
# _MATERIAL_KEYS = ("materialproperties", "material", "properties")
# _CLASS_KEYS = {
#     "ground": ("ground", "terrain", "soil", "groundproperties"),
#     "pathway": ("pathways", "pathway", "paths", "path", "roads", "road",
#                 "pavements", "pavement", "walkways"),
#     "lawn": ("lawns", "lawn", "grass", "grasslands", "greenareas"),
#     "pond": ("ponds", "pond", "water", "waterbodies"),
#     "tree": ("trees", "tree"),
# }


# def _as_float(value) -> Optional[float]:
#     if isinstance(value, bool):
#         return None
#     try:
#         x = float(value)
#     except (TypeError, ValueError):
#         return None
#     return x if math.isfinite(x) else None


# def _thermal_containers(item) -> list:
#     """Nested 'thermalProperties' dicts anywhere (shallowly) inside an item."""
#     found = []

#     def walk(obj, depth):
#         if depth > 4:
#             return
#         if isinstance(obj, dict):
#             for k, v in obj.items():
#                 if _norm_key(k) in _CONTAINER_KEYS and isinstance(v, dict):
#                     found.append(v)
#             for v in obj.values():
#                 walk(v, depth + 1)
#         elif isinstance(obj, list):
#             for v in obj[:3]:
#                 walk(v, depth + 1)

#     walk(item, 0)
#     return found


# def _lookup_property(item, prop: str) -> Optional[float]:
#     """Look up a numeric property by alias. Search order: thermalProperties
#     containers, materialProperties-like dicts, then the item's own top level.
#     A bare top-level 'density' is ignored (too ambiguous, e.g. tree density)."""
#     if not isinstance(item, dict):
#         return None
#     dicts = _thermal_containers(item)
#     for k, v in item.items():
#         if _norm_key(k) in _MATERIAL_KEYS and isinstance(v, dict):
#             dicts.append(v)
#     if prop != "density":
#         dicts.append(item)
#     aliases = _PROPERTY_ALIASES[prop]
#     for d in dicts:
#         norm = {_norm_key(k): v for k, v in d.items()}
#         for alias in aliases:
#             x = _as_float(norm.get(alias))
#             if x is not None:
#                 return x
#     return None


# def _scene_items(scene: dict, cls: str) -> list:
#     wanted = _CLASS_KEYS[cls]
#     containers = [scene]
#     if isinstance(scene.get("settings"), dict):
#         containers.append(scene["settings"])
#     items = []
#     for c in containers:
#         for k, v in c.items():
#             if _norm_key(k) in wanted:
#                 if isinstance(v, list):
#                     items.extend(x for x in v if isinstance(x, dict))
#                 elif isinstance(v, dict):
#                     items.append(v)
#     return items


# def json_layer_properties(scene: Optional[dict], cls: str) -> dict:
#     """Mean of each property over all JSON items of this class."""
#     if not isinstance(scene, dict):
#         return {}
#     items = _scene_items(scene, cls)
#     out = {}
#     for prop in _PROPERTY_ALIASES:
#         vals = [v for v in (_lookup_property(it, prop) for it in items) if v is not None]
#         if vals:
#             out[prop] = float(np.mean(vals))
#     return out


# # Documented defaults. Change these (or give the property in the JSON / CLI).
# _LAYER_BASE = dict(depth_m=0.23, nodes=5, deep_temperature_c=24.0, deep_h_wm2k=None,
#                    thermal_conductivity=1.5, density=1150.0, specific_heat=650.0,
#                    albedo=DEFAULT_GROUND_ALBEDO, emissivity=0.95, wetness=0.0,
#                    back_follows_ambient=False,
#                    mixing_factor=1.0)   # multiplies conductivity (turbulent mixing, water only)
# LAYER_DEFAULTS = {
#     "ground": {},
#     # paving (concrete/asphalt-like)
#     "pathway": dict(thermal_conductivity=1.4, density=2100.0, specific_heat=880.0,
#                     albedo=0.25, emissivity=0.92, depth_m=0.15),
#     # turf over the same soil as the ground; wetness = fraction of potential
#     # evaporation (0 = dry surface, 1 = free water surface)
#     "lawn": dict(albedo=0.25, emissivity=0.97, wetness=0.25),
#     # shallow water with free-surface evaporation. Still water conducts heat
#     # poorly (k = 0.6), so a real pond is mixed by wind and convection; this is
#     # represented by an effective conductivity k_eff = mixing_factor * k.
#     # mixing_factor = 1 gives still water (surface skin heats strongly).
#     "pond": dict(depth_m=0.5, thermal_conductivity=0.6, density=997.0,
#                  specific_heat=4182.0, albedo=0.06, emissivity=0.96, wetness=1.0,
#                  mixing_factor=30.0),
#     # thin canopy skin, back side exchanging with ambient air, partial transpiration
#     "tree": dict(depth_m=0.002, nodes=2, thermal_conductivity=0.15, density=700.0,
#                  specific_heat=3500.0, albedo=0.20, emissivity=0.96, wetness=0.30,
#                  back_follows_ambient=True),
# }


# def resolve_layer_settings(cls: str, scene: Optional[dict] = None,
#                            overrides: Optional[dict] = None) -> tuple:
#     """Precedence: built-in default < class default < scene JSON < CLI override.
#     Returns (GroundSettings, provenance {property: source})."""
#     base = dict(_LAYER_BASE)
#     prov = {k: "default" for k in base}
#     for k, v in LAYER_DEFAULTS.get(cls, {}).items():
#         base[k] = v
#         prov[k] = f"default:{cls}"
#     json_vals = json_layer_properties(scene, cls)
#     cli_vals = {k: v for k, v in (overrides or {}).items() if v is not None}

#     def build(use_json):
#         d, p = dict(base), dict(prov)
#         if use_json:
#             for k, v in json_vals.items():
#                 d[k], p[k] = v, "json"
#         for k, v in cli_vals.items():
#             d[k], p[k] = v, "cli"
#         d["wetness"] = min(max(float(d["wetness"]), 0.0), 1.0)
#         d["nodes"] = int(d["nodes"])
#         mixing = float(d.pop("mixing_factor"))
#         p.pop("mixing_factor", None)
#         if not math.isfinite(mixing) or mixing <= 0:
#             raise ValueError("mixing_factor must be finite and positive")
#         if mixing != 1.0:
#             d["thermal_conductivity"] = float(d["thermal_conductivity"]) * mixing
#             p["thermal_conductivity"] = f"{p['thermal_conductivity']} x{mixing:g} mixing"
#         return GroundSettings(**d), p

#     try:
#         return build(True)
#     except ValueError as exc:
#         warn(f"{cls}: invalid property in JSON ({exc}); falling back to defaults")
#         return build(False)


# def describe_layer_settings(cls: str, gs: GroundSettings, prov: dict) -> str:
#     parts = []
#     for key, label, fmt in (("thermal_conductivity", "k", "{:.3g}"), ("density", "rho", "{:.0f}"),
#                             ("specific_heat", "cp", "{:.0f}"), ("albedo", "albedo", "{:.2f}"),
#                             ("emissivity", "eps", "{:.2f}"), ("depth_m", "depth_m", "{:.3g}"),
#                             ("wetness", "wetness", "{:.2f}")):
#         parts.append(f"{label}={fmt.format(getattr(gs, key))}[{prov.get(key, 'given')}]")
#     return f"[properties] {cls}: " + " ".join(parts)


# class BuildingMaterials:
#     """Wall/roof properties for OBJ buildings taken from the scene JSON.

#     JSON faces are averaged (area-weighted) per surface type. An OBJ building
#     is matched to JSON building(s) whose XY bounding box (+1 m) contains its
#     centre; unmatched OBJ buildings get the scene-wide wall/roof means.
#     Missing properties fall back to DEFAULT_MATERIAL / DEFAULT_ALBEDO_*.
#     """
#     _KEYS = ("thermal_conductivity", "density", "specific_heat", "albedo", "emissivity")

#     def __init__(self, scene: Optional[dict] = None):
#         self.defaults = {
#             "wall": dict(DEFAULT_MATERIAL, albedo=DEFAULT_ALBEDO_WALL, emissivity=DEFAULT_EMISSIVITY),
#             "roof": dict(DEFAULT_MATERIAL, albedo=DEFAULT_ALBEDO_ROOF, emissivity=DEFAULT_EMISSIVITY)}
#         self.overall = {k: dict(v) for k, v in self.defaults.items()}
#         self.provenance = {"wall": "default", "roof": "default"}
#         self.buildings = []
#         self.n_matched = 0
#         self.n_unmatched = 0
#         if isinstance(scene, dict):
#             try:
#                 self._read_scene(scene)
#             except Exception as exc:                      # never let JSON quirks stop the run
#                 warn(f"buildings: could not read materials from JSON ({exc}); using defaults")

#     def _merge(self, base: dict, samples: list) -> dict:
#         out = dict(base)
#         for k in self._KEYS:
#             vals = [(a, f[k]) for a, f in samples if k in f]
#             if vals:
#                 out[k] = float(np.average([v for _, v in vals], weights=[a for a, _ in vals]))
#         if (any(not math.isfinite(out[k]) or out[k] <= 0 for k in self._KEYS[:3])
#                 or not 0 <= out["albedo"] <= 1 or not 0 <= out["emissivity"] <= 1):
#             warn("buildings: out-of-range material values in JSON ignored")
#             return dict(base)
#         return out

#     def _read_scene(self, scene: dict):
#         per_building, overall = [], {"wall": [], "roof": []}
#         for bi, b in enumerate(scene.get("buildings", []) or []):
#             if not isinstance(b, dict):
#                 continue
#             per, pts = {"wall": [], "roof": []}, []
#             for f in b.get("faces", []) or []:
#                 try:
#                     verts = np.array([[v["x"], v["y"], v["z"]] for v in f["vertices"]], float)
#                 except Exception:
#                     continue
#                 if len(verts) < 3 or not np.isfinite(verts).all():
#                     continue
#                 pts.append(verts)
#                 kind = {"roof": "roof", "wall": "wall"}.get(str(f.get("surfaceType", "wall")).lower())
#                 if kind is None:
#                     continue
#                 found = {k: _lookup_property(f, k) for k in self._KEYS}
#                 found = {k: v for k, v in found.items() if v is not None}
#                 if found:
#                     sample = (max(_polygon_area_3d(verts), 1e-9), found)
#                     per[kind].append(sample)
#                     overall[kind].append(sample)
#             if pts:
#                 allp = np.vstack(pts)
#                 per_building.append((str(b.get("id", f"building_{bi}")),
#                                      allp.min(axis=0)[:2], allp.max(axis=0)[:2], per))
#         for kind in ("wall", "roof"):
#             if overall[kind]:
#                 self.overall[kind] = self._merge(self.defaults[kind], overall[kind])
#                 self.provenance[kind] = "json"
#         for bid, lo, hi, per in per_building:
#             entry = {kind: self._merge(self.overall[kind], per[kind]) for kind in ("wall", "roof")}
#             self.buildings.append((bid, lo, hi, entry))

#     def for_component(self, center_xy) -> dict:
#         c = np.asarray(center_xy, float)
#         hits = [e for _, lo, hi, e in self.buildings
#                 if np.all(c >= lo - 1.0) and np.all(c <= hi + 1.0)]
#         if not hits:
#             self.n_unmatched += 1
#             return self.overall
#         self.n_matched += 1
#         if len(hits) == 1:
#             return hits[0]
#         return {kind: {k: float(np.mean([h[kind][k] for h in hits])) for k in self._KEYS}
#                 for kind in ("wall", "roof")}

#     def describe(self) -> str:
#         parts = []
#         for kind in ("wall", "roof"):
#             m = self.overall[kind]
#             parts.append(f"{kind}: k={m['thermal_conductivity']:.3g} rho={m['density']:.0f} "
#                          f"cp={m['specific_heat']:.0f} albedo={m['albedo']:.2f} "
#                          f"eps={m['emissivity']:.2f} [{self.provenance[kind]}]")
#         return ("[properties] buildings: " + "; ".join(parts) +
#                 f"; OBJ buildings matched to JSON: {self.n_matched}, using scene mean: {self.n_unmatched}")


# # ---- OBJ loading with validation -------------------------------------------

# def _clean_mesh(V, F, label: str):
#     """Drop invalid / degenerate triangles. Returns (V, F ndarray) or None."""
#     V = np.asarray(V, dtype=float)
#     F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
#     n0 = len(F)
#     if V.ndim != 2 or len(V) == 0 or n0 == 0:
#         warn(f"{label}: no vertices/faces")
#         return None
#     F = F[((F >= 0) & (F < len(V))).all(axis=1)]
#     F = F[np.isfinite(V)[F].all(axis=(1, 2))] if len(F) else F
#     if len(F):
#         tri = V[F]
#         area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
#         F = F[area > 1e-12]
#     if n0 - len(F):
#         warn(f"{label}: dropped {n0 - len(F)} invalid/degenerate triangle(s) of {n0}")
#     if not len(F):
#         warn(f"{label}: no valid triangles left")
#         return None
#     return V, F


# def load_obj_checked(path, label: str):
#     """Load one OBJ (or a list, merged). Returns (V, F) or None; never raises."""
#     if not path:
#         return None
#     paths = [path] if isinstance(path, str) else list(path)
#     meshes = []
#     for p in paths:
#         if not os.path.isfile(p):
#             warn(f"{label}: file not found, skipping: {p}")
#             continue
#         try:
#             V, F = load_obj_mesh(p)
#         except Exception as exc:
#             warn(f"{label}: could not parse {p} ({exc})")
#             continue
#         mesh = _clean_mesh(V, F, f"{label} {os.path.basename(p)}")
#         if mesh is not None:
#             meshes.append(mesh)
#     if not meshes:
#         return None
#     if len(meshes) == 1:
#         return meshes[0]
#     offset, Vs, Fs = 0, [], []
#     for V, F in meshes:
#         Vs.append(V)
#         Fs.append(F + offset)
#         offset += len(V)
#     return np.vstack(Vs), np.vstack(Fs)


# def _json_trees_mesh(scene: Optional[dict]):
#     """Fallback tree geometry (trunk + cone crown) built from JSON tree entries."""
#     if not isinstance(scene, dict):
#         return None
#     polys = []
#     for t in scene.get("trees", []) or []:
#         if not isinstance(t, dict):
#             continue
#         try:
#             p, _ = _tree_mesh(t)
#         except Exception:
#             continue
#         if p:
#             polys.extend(p)
#     if not polys:
#         return None
#     V, F = [], []
#     for poly in polys:
#         base = len(V)
#         V.extend(poly)
#         F.extend((base, base + i, base + i + 1) for i in range(1, len(poly) - 1))
#     return _clean_mesh(V, F, "trees (from JSON)")


# # ---- XY coverage test (remove ground cells hidden under overlay surfaces) --

# class XYTriangleIndex:
#     """Uniform-grid index of triangles projected on the XY plane."""

#     def __init__(self, tri_xyz):
#         t = np.asarray(tri_xyz, float)[:, :, :2]
#         self.t = t
#         lo, hi = t.min(axis=1), t.max(axis=1)
#         span = (hi - lo).max(axis=1)
#         extent = float((hi.max(axis=0) - lo.min(axis=0)).max())
#         self.cell = max(float(np.percentile(span, 90)), extent / 200.0, 1e-6)
#         self.origin = lo.min(axis=0)
#         i0 = np.floor((lo - self.origin) / self.cell).astype(int)
#         i1 = np.floor((hi - self.origin) / self.cell).astype(int)
#         self.grid = {}
#         for k in range(len(t)):
#             for ix in range(i0[k, 0], i1[k, 0] + 1):
#                 for iy in range(i0[k, 1], i1[k, 1] + 1):
#                     self.grid.setdefault((ix, iy), []).append(k)

#     def covers(self, pts_xy) -> np.ndarray:
#         pts = np.asarray(pts_xy, float).reshape(-1, 2)
#         out = np.zeros(len(pts), dtype=bool)
#         cells = np.floor((pts - self.origin) / self.cell).astype(int)
#         keys, inv = np.unique(cells, axis=0, return_inverse=True)
#         inv = inv.ravel()
#         order = np.argsort(inv, kind="stable")
#         bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
#         for ki, (cx, cy) in enumerate(keys):
#             tris = self.grid.get((int(cx), int(cy)))
#             if not tris:
#                 continue
#             idx = order[bounds[ki]:bounds[ki + 1]]
#             tt = self.t[tris]
#             a = tt[:, 0]
#             v0, v1 = tt[:, 1] - a, tt[:, 2] - a
#             den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
#             ok = np.abs(den) > 1e-14
#             den = np.where(ok, den, 1.0)
#             d = pts[idx][:, None, :] - a[None]
#             u = (d[..., 0] * v1[:, 1] - v1[:, 0] * d[..., 1]) / den
#             v = (v0[:, 0] * d[..., 1] - d[..., 0] * v0[:, 1]) / den
#             inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
#             out[idx] = inside.any(axis=1)
#         return out


# # ---- building faces from OBJ ----------------------------------------------

# def build_faces_from_obj(buildings_obj_path=None, materials: Optional[BuildingMaterials] = None,
#                          mesh=None) -> tuple:
#     """Recover buildings, retaining every nondegenerate triangle as a cell.

#     Coplanar groups provide parent wall IDs and approximate bulk volumes only;
#     visibility, irradiance, temperature and VTK values belong to each triangle.
#     Wall/roof materials come from `materials` (JSON-derived when available).
#     Returns (faces, capacitances, face_triangles, occluders); the last is kept
#     for API compatibility and is empty (rays use the BVH built from the mesh).
#     """
#     if mesh is None:
#         mesh = load_obj_checked(buildings_obj_path, "buildings")
#     if mesh is None:
#         return [], {}, {}, {}
#     V, Farr = mesh
#     materials = materials or BuildingMaterials()
#     F = [tuple(t) for t in Farr.tolist()]
#     faces, face_triangles, capacitances = [], {}, {}
#     for b_idx, tri_indices in enumerate(_obj_components_by_shared_position(V, F)):
#         bid = f"building_{b_idx:02d}"
#         comp_tris = [F[t] for t in tri_indices]
#         groups = group_coplanar(V, comp_tris)
#         if not groups:
#             continue
#         z_min = min(g["z_min"] for g in groups.values())
#         z_max = max(g["z_max"] for g in groups.values())
#         pts = V[np.unique(np.asarray(comp_tris).ravel())]
#         mats = materials.for_component(0.5 * (pts.min(axis=0) + pts.max(axis=0))[:2])
#         wm = mats["wall"]
#         roof_area = sum(g["area"] * abs(g["normal"][2]) for g in groups.values()
#                         if abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
#                         and g["center"][2] > z_min + 1e-6)
#         capacitances[bid] = wm["density"] * roof_area * (z_max - z_min) * wm["specific_heat"]
#         for gi, g in enumerate(groups.values()):
#             horizontal = abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
#             bottom = horizontal and g["z_max"] <= z_min + 1e-6
#             stype = "floor" if bottom else ("roof" if horizontal else "wall")
#             normal = g["normal"].copy()
#             if (stype == "roof" and normal[2] < 0) or (stype == "floor" and normal[2] > 0):
#                 normal = -normal
#             tilt = math.degrees(math.acos(np.clip(normal[2], -1, 1)))
#             parent = f"{bid}_face_{gi:03d}_{stype}"
#             m = mats["roof" if stype == "roof" else "wall"]
#             for ti, tri in enumerate(g["triangles"]):
#                 verts = V[list(tri)]
#                 _, area = _triangle_normal_area(*verts)
#                 fid = f"{parent}_tri_{ti:06d}"
#                 faces.append(Face(
#                     building_id=bid, face_id=fid, parent_face_id=parent, surface_type=stype,
#                     normal=normal, area=area, tilt_deg=tilt, center=verts.mean(axis=0),
#                     vertices=verts, thermal_conductivity=m["thermal_conductivity"],
#                     density=m["density"], specific_heat=m["specific_heat"],
#                     albedo=m["albedo"], emissivity=m["emissivity"]))
#                 face_triangles[fid] = [tri]
#     return faces, capacitances, face_triangles, {}


# # ---- ground / pathway / lawn / pond / tree cells ---------------------------

# def _layer_geometry(V, F, orient: str):
#     """Per-triangle vertices, unit normals and areas. orient='up' flips normals
#     to point upward; orient='outward' flips them away from the centre of each
#     connected piece (closed tree proxies)."""
#     tri = V[F].copy()
#     n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
#     norm = np.linalg.norm(n, axis=1)
#     area = 0.5 * norm
#     nrm = n / norm[:, None]
#     if orient == "up":
#         flip = nrm[:, 2] < 0
#     else:
#         flip = np.zeros(len(F), dtype=bool)
#         centroids = tri.mean(axis=1)
#         for comp in _obj_components_by_shared_position(V, F):
#             idx = np.asarray(comp)
#             centre = tri[idx].reshape(-1, 3).mean(axis=0)
#             flip[idx] = np.einsum("ij,ij->i", nrm[idx], centroids[idx] - centre) < 0
#     tri[flip] = tri[flip][:, [0, 2, 1]]
#     nrm[flip] *= -1
#     return tri, nrm, area


# def build_layer_faces(cls: str, V, F, settings: GroundSettings, building_tracer=None,
#                       higher_index: Optional[XYTriangleIndex] = None,
#                       orient: str = "up") -> list:
#     """One Face per mesh triangle. Cells under a building roof (vertical ray)
#     and cells whose centroid lies under a higher-priority surface (e.g. ground
#     under a pond) are removed so no point is solved twice."""
#     tri, nrm, area = _layer_geometry(V, F, orient)
#     cen = tri.mean(axis=1)
#     keep = np.ones(len(tri), dtype=bool)
#     n_under = n_cov = 0
#     if building_tracer is not None and cls != "tree":
#         blocked = building_tracer.blocked(cen + np.array([0.0, 0.0, _SHADOW_RAY_OFFSET]),
#                                           np.array([0.0, 0.0, 1.0]))
#         n_under = int(blocked.sum())
#         keep &= ~blocked
#     if higher_index is not None and cls != "tree":
#         covered = higher_index.covers(cen[:, :2])
#         n_cov = int((covered & keep).sum())
#         keep &= ~covered
#     faces = []
#     for ti in np.flatnonzero(keep):
#         faces.append(Face(
#             building_id=cls, face_id=f"{cls}_tri_{ti:06d}", parent_face_id=f"{cls}_surface",
#             surface_type=cls, normal=nrm[ti], area=float(area[ti]),
#             tilt_deg=math.degrees(math.acos(np.clip(nrm[ti][2], -1, 1))),
#             center=cen[ti], vertices=tri[ti],
#             thermal_conductivity=settings.thermal_conductivity, density=settings.density,
#             specific_heat=settings.specific_heat, albedo=settings.albedo,
#             emissivity=settings.emissivity))
#     log(f"[input] {cls}: {len(faces)} cells solved "
#         f"({n_under} under buildings and {n_cov} under other surfaces removed), "
#         f"area {sum(f.area for f in faces):.1f} m2")
#     return faces


# @dataclass
# class ObjBundle:
#     building_faces: list
#     capacitances: dict
#     layer_faces: dict
#     layer_settings: dict
#     tracer: object
#     wall_settings: WallSettings

#     @property
#     def n_cells(self) -> int:
#         return len(self.building_faces) + sum(len(v) for v in self.layer_faces.values())

#     def all_faces(self) -> list:
#         out = list(self.building_faces)
#         for cls in LAYER_CLASSES:
#             out += self.layer_faces.get(cls, [])
#         return out


# def prepare_obj_bundle(buildings_obj=None, ground_obj=None, trees_obj=None, pathway_obj=None,
#                        lawn_obj=None, pond_obj=None, scene: Optional[dict] = None,
#                        wall_settings: Optional[WallSettings] = None,
#                        layer_overrides: Optional[dict] = None,
#                        fixed_layer_settings: Optional[dict] = None,
#                        consider_shadows: bool = True, remove_overlaps: bool = True) -> ObjBundle:
#     """Load every available component once (geometry, materials, BVH). The
#     result is reused for every weather day."""
#     wall_settings = wall_settings or WallSettings()
#     layer_overrides = layer_overrides or {}
#     fixed_layer_settings = fixed_layer_settings or {}
#     paths = dict(ground=ground_obj, pathway=pathway_obj, lawn=lawn_obj, pond=pond_obj, tree=trees_obj)
#     meshes = {}
#     for cls, p in paths.items():
#         mesh = load_obj_checked(p, f"{cls} OBJ") if p else None
#         if mesh is None and cls == "tree":
#             mesh = _json_trees_mesh(scene)
#             if mesh is not None:
#                 log("[input] tree: no usable trees OBJ, using tree geometry from the JSON")
#         if mesh is not None:
#             meshes[cls] = mesh
#             log(f"[input] {cls}: {len(mesh[1])} valid triangles")
#         elif p:
#             warn(f"{cls}: nothing solved ({p})")
#     bmesh = load_obj_checked(buildings_obj, "buildings OBJ") if buildings_obj else None

#     materials = BuildingMaterials(scene)
#     building_faces, capacitances = [], {}
#     if bmesh is not None:
#         building_faces, capacitances, _, _ = build_faces_from_obj(mesh=bmesh, materials=materials)
#         n_b = len(capacitances)
#         log(f"[input] buildings: {n_b} building(s), {len(building_faces)} cells")
#         log(materials.describe())
#     elif buildings_obj:
#         warn("buildings: nothing solved")

#     b_tris = bmesh[0][bmesh[1]] if bmesh is not None else np.empty((0, 3, 3))
#     building_tracer = TriangleRayTracer(b_tris) if len(b_tris) else None
#     tracer = None
#     if consider_shadows:
#         tree_tris = meshes["tree"][0][meshes["tree"][1]] if "tree" in meshes else np.empty((0, 3, 3))
#         if len(tree_tris):
#             tracer = TriangleRayTracer(np.concatenate([b_tris, tree_tris]) if len(b_tris) else tree_tris)
#         else:
#             tracer = building_tracer
#         if tracer is not None:
#             log(f"[shadows] occluders: {len(b_tris)} building + {len(tree_tris)} tree triangles")
#         else:
#             log("[shadows] no occluders present: only self-orientation (cos) shading applies")

#     tri_cache = {cls: m[0][m[1]] for cls, m in meshes.items()}
#     layer_faces, layer_settings = {}, {}
#     for cls in LAYER_CLASSES:
#         if cls not in meshes:
#             continue
#         if cls in fixed_layer_settings:
#             gs, prov = fixed_layer_settings[cls], {}
#         else:
#             gs, prov = resolve_layer_settings(cls, scene, layer_overrides.get(cls))
#         log(describe_layer_settings(cls, gs, prov))
#         higher = None
#         if remove_overlaps and cls in LAYER_PRIORITY:
#             above = [tri_cache[c] for c in LAYER_PRIORITY[:LAYER_PRIORITY.index(cls)] if c in tri_cache]
#             if above:
#                 higher = XYTriangleIndex(np.concatenate(above))
#         V, F = meshes[cls]
#         faces = build_layer_faces(cls, V, F, gs, building_tracer, higher,
#                                   "outward" if cls == "tree" else "up")
#         if faces:
#             layer_faces[cls], layer_settings[cls] = faces, gs
#         else:
#             warn(f"{cls}: no cells left after filtering; skipped")
#     return ObjBundle(building_faces, capacitances, layer_faces, layer_settings, tracer, wall_settings)


# # ---- weather -> per-record dicts (adds humidity / pressure when available) --

# def _series(values, n):
#     if values is None:
#         return None
#     arr = np.asarray(values, dtype=float)
#     return arr if arr.ndim == 1 and len(arr) == n else None


# def records_from_weather(wx: dict) -> list:
#     n = len(wx["hour"])
#     p = _series(wx.get("atmosphericPressure"), n)
#     if p is not None:                       # accept Pa, hPa or kPa
#         med = float(np.nanmedian(p))
#         p = p * (1000.0 if med < 200 else 100.0 if med < 2000 else 1.0)
#     w_amb = _series(wx.get("wambient"), n)
#     rh = _series(wx.get("relativeHumidity"), n)
#     if rh is not None and float(np.nanmax(rh)) > 1.5:
#         rh = rh / 100.0
#     records = []
#     for i in range(n):
#         rec = dict(hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
#                    sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
#                    i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i])
#         if p is not None and np.isfinite(p[i]):
#             rec["p_atm"] = float(p[i])
#         if w_amb is not None and np.isfinite(w_amb[i]) and 0 <= w_amb[i] < 0.1:
#             rec["w_amb"] = float(w_amb[i])
#         if rh is not None and np.isfinite(rh[i]):
#             rec["rh"] = float(np.clip(rh[i], 0.0, 1.0))
#         records.append(rec)
#     return records


# def _saturation_pressure_pa(t_c: float) -> float:
#     return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


# def _humidity_ratio(t_c: float, rh: float, p_pa: float) -> float:
#     pv = min(rh * _saturation_pressure_pa(t_c), 0.99 * p_pa)
#     return 0.622 * pv / (p_pa - pv)


# def _saturation_humidity_ratio(t_c: float, p_pa: float) -> tuple:
#     """(w_sat, d w_sat / dT) at temperature t_c."""
#     es = min(_saturation_pressure_pa(t_c), 0.99 * p_pa)
#     des = es * 17.625 * 243.04 / (t_c + 243.04) ** 2
#     return 0.622 * es / (p_pa - es), 0.622 * p_pa * des / (p_pa - es) ** 2


# def sunlit_fractions_from_arrays(normals, verts, sun_vec, ray_tracer, samples=3):
#     """Array version of triangle_sunlit_fraction (no per-call Python loops)."""
#     fractions = np.zeros(len(normals))
#     if sun_vec[2] <= 0:
#         return fractions
#     eligible = np.flatnonzero(normals @ sun_vec > 0)
#     if not len(eligible):
#         return fractions
#     weights = np.array([[1/3, 1/3, 1/3]]) if samples == 1 else np.array([
#         [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
#     origins = np.einsum('sj,tjk->tsk', weights, verts[eligible])
#     origins += normals[eligible, None, :] * _SHADOW_RAY_OFFSET
#     hits = ray_tracer.blocked(origins.reshape(-1, 3), sun_vec).reshape(-1, samples)
#     fractions[eligible] = 1 - hits.mean(axis=1)
#     return fractions


# def _tag_layer_columns(df: pd.DataFrame) -> None:
#     """Non-building classes: expose the slab fields under the ground_* names
#     (they hold the sub-surface layer of whatever class `surface_group` says)
#     and blank the building-only columns."""
#     df["ground_depth_m"] = df["slab_thickness_m"]
#     df["T_ground_bottom_C"] = df["T_back_surface_C"]
#     df["T_deep_ground_C"] = df["T_back_boundary_C"]
#     df["h_deep_ground_Wm2K"] = df["h_back_Wm2K"]
#     df["q_deep_ground_out_Wm2"] = df["q_back_out_Wm2"]
#     df["T_ground_mean_C"] = df["T_slab_mean_C"]
#     df["ground_capacity_Jm2K"] = df["slab_capacity_Jm2K"]
#     df["ground_capacity_JK"] = df["slab_capacity_JK"]
#     building_only = ["wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K", "T_inner_C",
#                      "T_wall_mean_C", "q_indoor_out_Wm2", "wall_capacity_Jm2K", "wall_capacity_JK"]
#     df[building_only] = np.nan


# def solve_obj_day(bundle: ObjBundle, weather_dir: str, t_initial_c: Optional[float] = None,
#                   ground_initial_c: Optional[float] = None,
#                   ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                   shadow_samples: int = 3, workers: int = 1) -> tuple:
#     """Solve every component in the bundle for one weather folder.
#     Returns (face_df, building_df, faces)."""
#     wx = load_weather_folder(weather_dir)
#     records = records_from_weather(wx)
#     frames, bframes, faces_out = [], [], []
#     if bundle.building_faces:
#         fdf, bdf = _simulate(bundle.building_faces, bundle.capacitances, records,
#                              ground_albedo=ground_albedo, t_initial_c=t_initial_c,
#                              ray_tracer=bundle.tracer, shadow_samples=shadow_samples,
#                              wall_settings=bundle.wall_settings, workers=workers)
#         frames.append(fdf)
#         bframes.append(bdf)
#         faces_out += bundle.building_faces
#     for cls in LAYER_CLASSES:
#         faces = bundle.layer_faces.get(cls)
#         if not faces:
#             continue
#         gs = bundle.layer_settings[cls]
#         cap = gs.density * gs.specific_heat * gs.depth_m * sum(f.area for f in faces)
#         init = records[0]["T_amb_C"] if cls == "tree" else ground_initial_c
#         ldf, _ = _simulate(faces, {cls: cap}, records, ground_albedo=ground_albedo,
#                            t_initial_c=init, ray_tracer=bundle.tracer,
#                            shadow_samples=shadow_samples, wall_settings=gs.slab_settings(),
#                            surface_group=cls, wetness=gs.wetness,
#                            back_follows_ambient=gs.back_follows_ambient, workers=workers)
#         _tag_layer_columns(ldf)
#         frames.append(ldf)
#         faces_out += faces
#     if not frames:
#         raise ValueError("no solvable components in this scene")
#     face_df = pd.concat(frames, ignore_index=True, sort=False)
#     building_df = (pd.concat(bframes, ignore_index=True) if bframes else
#                    pd.DataFrame(columns=["building_id", "hour", "T_amb_C", "T_eq_C",
#                                          "h_eff_WK", "tau_hr", "T_building_C"]))
#     return face_df, building_df, faces_out


# def run_analysis_from_obj(buildings_obj_path=None, weather_dir: Optional[str] = None,
#                           trees_obj_path: Optional[str] = None, consider_shadows: bool = True,
#                           t_initial_c: Optional[float] = None, shadow_samples: int = 3,
#                           wall_settings: Optional[WallSettings] = None,
#                           ground_obj_path: Optional[str] = None,
#                           ground_initial_c: Optional[float] = None,
#                           ground_settings: Optional[GroundSettings] = None,
#                           pathway_obj_path: Optional[str] = None,
#                           lawn_obj_path: Optional[str] = None,
#                           pond_obj_path: Optional[str] = None,
#                           scene=None, workers: int = 1) -> tuple:
#     """Convenience wrapper: prepare + solve one weather folder.
#     Returns (face_df, building_df, faces)."""
#     if shadow_samples not in (1, 3):
#         raise ValueError("shadow_samples must be 1 or 3")
#     bundle = prepare_obj_bundle(
#         buildings_obj_path, ground_obj_path, trees_obj_path, pathway_obj_path, lawn_obj_path,
#         pond_obj_path, scene=_scene_from(scene) if scene is not None else None,
#         wall_settings=wall_settings,
#         fixed_layer_settings={"ground": ground_settings} if ground_settings else None,
#         consider_shadows=consider_shadows)
#     return solve_obj_day(bundle, weather_dir, t_initial_c=t_initial_c,
#                          ground_initial_c=ground_initial_c, shadow_samples=shadow_samples,
#                          workers=workers)


# def export_vtk_series_from_analysis(analysis: tuple, out_dir: str) -> str:
#     """Write one legacy-VTK file per hour (+ scene.pvd) with per-cell CELL_DATA
#     for every solved component. T_C aliases T_surface_C. surface_type_id:
#     0 wall, 1 roof, 2 floor, 3 ceiling, 5 tree, 6 ground, 7 pathway, 8 lawn,
#     9 pond. Tree/pond/lawn cells also set is_tree/is_pond/is_lawn flags."""
#     face_df, building_df, faces = analysis
#     os.makedirs(out_dir, exist_ok=True)
#     polygons = [np.asarray(f.vertices, float) for f in faces]
#     ids = [f.face_id for f in faces]
#     building_ids = dict.fromkeys(f.building_id for f in faces if f.surface_type not in LAYER_CLASSES)
#     bmap = {bid: i for i, bid in enumerate(building_ids)}
#     pmap = {pid: i for i, pid in enumerate(dict.fromkeys(f.parent_face_id for f in faces))}
#     static = {
#         "surface_type_id": ([_SURFACE_TYPE_ID.get(f.surface_type, -1) for f in faces], "int"),
#         "is_building": ([int(f.surface_type not in LAYER_CLASSES) for f in faces], "int"),
#         "building_index": ([bmap.get(f.building_id, -1) for f in faces], "int"),
#         "parent_face_index": ([pmap[f.parent_face_id] for f in faces], "int"),
#     }
#     for cls in LAYER_CLASSES:
#         static[f"is_{cls}"] = ([int(f.surface_type == cls) for f in faces], "int")
#     fields = ["T_surface_C", "T_solair_C", "I_direct_Wm2", "I_total_Wm2",
#               "sunlit_fraction", "shadow_fraction", "shadowed", "sunlit",
#               "h_conv_Wm2K", "q_solar_absorbed_Wm2", "q_conv_out_Wm2", "q_longwave_out_Wm2",
#               "q_evap_out_Wm2", "T_inner_C", "T_wall_mean_C", "q_conduction_in_Wm2",
#               "q_indoor_out_Wm2", "q_storage_Wm2", "q_outer_node_storage_Wm2",
#               "stored_heat_Jm2", "stored_heat_J", "wall_capacity_Jm2K", "wall_capacity_JK",
#               "wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K", "T_back_surface_C",
#               "T_slab_mean_C", "q_back_out_Wm2", "slab_capacity_Jm2K", "slab_capacity_JK",
#               "slab_thickness_m", "T_back_boundary_C", "h_back_Wm2K", "ground_depth_m",
#               "T_ground_bottom_C", "T_deep_ground_C", "h_deep_ground_Wm2K",
#               "q_deep_ground_out_Wm2", "T_ground_mean_C", "ground_capacity_Jm2K",
#               "ground_capacity_JK"]
#     fields = [name for name in fields if name in face_df.columns]
#     entries = []
#     for h, frame in face_df.groupby("hour", sort=True):
#         amb_c = float(frame["T_amb_C"].iloc[0])
#         frame = frame.set_index("face_id").reindex(ids)
#         scalars = dict(static)
#         for name in fields:
#             fill = amb_c if name.startswith("T_") else 0
#             dtype = "int" if name in ("shadowed", "sunlit") else "float"
#             values = frame[name].fillna(fill).to_numpy(dtype=int if dtype == "int" else float)
#             scalars[name] = (values, dtype)
#         scalars["T_C"] = scalars["T_surface_C"]
#         fname = f"frame_{int(round(h * 100)):04d}.vtk"
#         _write_vtk_polydata(os.path.join(out_dir, fname), polygons, scalars)
#         entries.append((float(h), fname))
#     pvd_path = os.path.join(out_dir, "scene.pvd")
#     with open(pvd_path, "w") as f:
#         f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
#         for t, fname in entries:
#             f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
#         f.write('</Collection>\n</VTKFile>\n')
#     log(f"[vtk export] Wrote {len(entries)} frame(s) + scene.pvd to {out_dir}")
#     return pvd_path


# def export_vtk_series_from_obj(buildings_obj_path, weather_dir: str, out_dir: str,
#                                trees_obj_path: Optional[str] = None, consider_shadows: bool = True,
#                                shadow_samples: int = 3, analysis=None,
#                                wall_settings: Optional[WallSettings] = None,
#                                t_initial_c: Optional[float] = None,
#                                ground_obj_path: Optional[str] = None,
#                                ground_settings: Optional[GroundSettings] = None,
#                                ground_initial_c: Optional[float] = None,
#                                pathway_obj_path: Optional[str] = None,
#                                lawn_obj_path: Optional[str] = None,
#                                pond_obj_path: Optional[str] = None, scene=None) -> str:
#     """Backward-compatible wrapper: run (unless `analysis` is given) and export."""
#     if analysis is None:
#         analysis = run_analysis_from_obj(
#             buildings_obj_path, weather_dir, trees_obj_path=trees_obj_path,
#             consider_shadows=consider_shadows, t_initial_c=t_initial_c,
#             shadow_samples=shadow_samples, wall_settings=wall_settings,
#             ground_obj_path=ground_obj_path, ground_initial_c=ground_initial_c,
#             ground_settings=ground_settings, pathway_obj_path=pathway_obj_path,
#             lawn_obj_path=lawn_obj_path, pond_obj_path=pond_obj_path, scene=scene)
#     return export_vtk_series_from_analysis(analysis, out_dir)


# # ---- fast legacy-VTK writer (falls back to the slow one for mixed polygons) --

# VTK_BINARY = False      # set by --vtk_binary: big-endian legacy BINARY files (much faster/smaller)


# def _write_vtk_polydata_binary(path, arr, cell_scalars) -> None:
#     n_poly, n_v, _ = arr.shape
#     conn = np.concatenate([np.full((n_poly, 1), n_v), np.arange(n_poly * n_v).reshape(n_poly, n_v)],
#                           axis=1).astype(">i4")
#     with open(path, "wb") as f:
#         f.write(b"# vtk DataFile Version 3.0\nsolar_thermal_model export\nBINARY\nDATASET POLYDATA\n")
#         f.write(f"POINTS {n_poly * n_v} float\n".encode())
#         f.write(arr.reshape(-1, 3).astype(">f4").tobytes())
#         f.write(f"\nPOLYGONS {n_poly} {n_poly * (n_v + 1)}\n".encode())
#         f.write(conn.tobytes())
#         f.write(f"\nCELL_DATA {n_poly}\n".encode())
#         for name, (values, dtype) in cell_scalars.items():
#             f.write(f"SCALARS {name} {dtype} 1\nLOOKUP_TABLE default\n".encode())
#             if dtype == "int":
#                 f.write(np.asarray(values).astype(">i4").tobytes())
#             else:
#                 f.write(np.nan_to_num(np.asarray(values, dtype=float)).astype(">f4").tobytes())
#             f.write(b"\n")


# def _write_vtk_polydata(path: str, polygons: list, cell_scalars: dict) -> None:
#     try:
#         arr = np.asarray(polygons, dtype=float)
#         if arr.ndim != 3 or arr.shape[2] != 3:
#             raise ValueError("not a uniform polygon array")
#     except (ValueError, TypeError):
#         return _write_vtk_polydata_slow(path, polygons, cell_scalars)
#     if VTK_BINARY:
#         return _write_vtk_polydata_binary(path, arr, cell_scalars)
#     n_poly, n_v, _ = arr.shape
#     conn = np.concatenate([np.full((n_poly, 1), n_v), np.arange(n_poly * n_v).reshape(n_poly, n_v)], axis=1)
#     with open(path, "w") as f:
#         f.write("# vtk DataFile Version 3.0\nsolar_thermal_model export\nASCII\nDATASET POLYDATA\n")
#         f.write(f"POINTS {n_poly * n_v} float\n")
#         np.savetxt(f, arr.reshape(-1, 3), fmt="%.4f")
#         f.write(f"POLYGONS {n_poly} {n_poly * (n_v + 1)}\n")
#         np.savetxt(f, conn, fmt="%d")
#         f.write(f"CELL_DATA {n_poly}\n")
#         for name, (values, dtype) in cell_scalars.items():
#             f.write(f"SCALARS {name} {dtype} 1\nLOOKUP_TABLE default\n")
#             if dtype == "int":
#                 np.savetxt(f, np.asarray(values).astype(np.int64), fmt="%d")
#             else:
#                 np.savetxt(f, np.nan_to_num(np.asarray(values, dtype=float)), fmt="%.4f")


# def class_summary(face_df: pd.DataFrame) -> pd.DataFrame:
#     """Area-weighted hourly means per surface_group / surface_type."""
#     cols = ["T_surface_C", "I_total_Wm2", "sunlit_fraction", "q_evap_out_Wm2"]
#     d = face_df[["surface_group", "surface_type", "hour", "area_m2"] + cols].copy()
#     for c in cols:
#         d[c] = d[c] * d["area_m2"]
#     keys = ["surface_group", "surface_type", "hour"]
#     g = d.groupby(keys, as_index=False).sum(numeric_only=True)
#     for c in cols:
#         g[c] = g[c] / g["area_m2"]
#     mm = face_df.groupby(keys)["T_surface_C"].agg(["min", "max"]).reset_index()
#     g = g.merge(mm, on=keys)
#     return g.rename(columns={"T_surface_C": "T_surface_mean_C", "I_total_Wm2": "I_total_mean_Wm2",
#                              "sunlit_fraction": "sunlit_fraction_mean",
#                              "q_evap_out_Wm2": "q_evap_out_mean_Wm2",
#                              "min": "T_surface_min_C", "max": "T_surface_max_C"})


# # ==========================================================================
# # 10. Plots
# # ==========================================================================

# def plot_results(face_df: pd.DataFrame, building_df: pd.DataFrame, out_prefix: str,
#                   building_id: Optional[str] = None):
#     import matplotlib.pyplot as plt

#     bldg_ids = building_df["building_id"].unique()
#     single = building_id if building_id is not None else (bldg_ids[0] if len(bldg_ids) == 1 else None)

#     face_sub = face_df if single is None else face_df[face_df["building_id"] == single]
#     bldg_sub = building_df if single is None else building_df[building_df["building_id"] == single]
#     # Plot area-weighted parent walls, avoiding one curve per mesh triangle.
#     if "parent_face_id" in face_sub and face_sub.parent_face_id.nunique() < face_sub.face_id.nunique():
#         keys = ["building_id", "parent_face_id", "hour"]
#         values = ["I_total_Wm2", "T_solair_C", "T_surface_C"]
#         weighted = face_sub[keys + ["area_m2"]].copy()
#         for name in values:
#             weighted[name] = face_sub[name] * face_sub.area_m2
#         grouped = weighted.groupby(keys, as_index=False).sum()
#         for name in values:
#             grouped[name] /= grouped.area_m2
#         face_sub = grouped.rename(columns={"parent_face_id": "face_id"})

#     fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

#     # --- Panel 1: incident solar flux per face vs hour (shows exposure windows) ---
#     ax = axes[0]
#     for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
#         sub = sub.sort_values("hour")
#         lab = fid if single is not None else f"{bid}:{fid}"
#         ax.plot(sub["hour"], sub["I_total_Wm2"], marker="o", ms=3, label=lab)
#     ax.set_ylabel("Incident solar flux, W/m$^2$")
#     title = "Per-face solar exposure through the day (sunlit windows)"
#     ax.set_title(title if single is None else f"{title} -- {single}")
#     ax.legend(fontsize=7, ncol=2)
#     ax.grid(alpha=0.3)

#     # --- Panel 2: surface temperature + ambient + legacy bulk estimate ---
#     ax = axes[1]
#     for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
#         sub = sub.sort_values("hour")
#         lab = f"{fid} (surface)" if single is not None else f"{bid}:{fid} (surface)"
#         ax.plot(sub["hour"], sub["T_surface_C"], marker="o", ms=3, label=lab)
#     ax.plot(bldg_sub.groupby("hour")["T_amb_C"].first().index,
#             bldg_sub.groupby("hour")["T_amb_C"].first().values,
#             "k--", lw=2, label="T_amb (air)")
#     for bid, sub in bldg_sub.groupby("building_id"):
#         sub = sub.sort_values("hour")
#         lab = "T_building (lumped, RC)" if single is not None else f"T_building {bid} (RC)"
#         ax.plot(sub["hour"], sub["T_building_C"], lw=2.5, label=lab)
#     ax.set_xlabel("Hour of day (local standard time)")
#     ax.set_ylabel("Temperature, \u00b0C")
#     ax.set_title("Surface temperature per face vs. ambient vs. building bulk temperature")
#     ax.legend(fontsize=7, ncol=2)
#     ax.grid(alpha=0.3)

#     fig.tight_layout()
#     fig.savefig(f"{out_prefix}_plot.png", dpi=150)
#     print(f"Wrote {out_prefix}_plot.png")


# # ==========================================================================
# # 11. Script entry point
# # ==========================================================================

# _SCENE_KEY_NORMS = {"buildings", "site", "settings"} | {k for ks in _CLASS_KEYS.values() for k in ks}
# _NAME_PATTERNS = {"ground": r"ground|terrain", "tree": r"tree", "pathway": r"pathway|walkway|road",
#                   "lawn": r"lawn|grass", "pond": r"pond|water"}


# def _expand_paths(values) -> list:
#     """Flatten CLI path lists; expand quoted glob patterns that the shell did not."""
#     out = []
#     for v in values or []:
#         if any(c in v for c in "*?[") and not os.path.exists(v):
#             out.extend(sorted(glob.glob(v)))
#         else:
#             out.append(v)
#     return out


# def load_scenes(paths) -> tuple:
#     """Load every scene-like JSON among `paths`; merge component lists by id.
#     Returns (scene_dict_or_None, [files used])."""
#     scenes, used = [], []
#     for p in _expand_paths(paths):
#         if not os.path.isfile(p):
#             warn(f"json: not a file, skipping: {p}")
#             continue
#         try:
#             d = load_scene(p)
#         except Exception as exc:
#             warn(f"json: cannot parse {p} ({exc}); skipping")
#             continue
#         if not isinstance(d, dict) or not any(_norm_key(k) in _SCENE_KEY_NORMS for k in d):
#             log(f"[input] json: {p} does not look like a scene file; ignored")
#             continue
#         scenes.append(d)
#         used.append(p)
#     if not scenes:
#         return None, []
#     merged = dict(scenes[0])
#     for other in scenes[1:]:
#         for key, val in other.items():
#             if isinstance(val, list):
#                 mine = list(merged.get(key) or [])
#                 seen = {x.get("id") for x in mine if isinstance(x, dict) and x.get("id") is not None}
#                 for x in val:
#                     if not (isinstance(x, dict) and x.get("id") in seen):
#                         mine.append(x)
#                 merged[key] = mine
#             elif key not in merged:
#                 merged[key] = val
#     return merged, used


# def classify_obj_inputs(args) -> tuple:
#     """Sort the OBJ arguments. `--buildings_obj *.obj` typically expands to every
#     OBJ in the folder, so files that are given explicitly for another class (or
#     whose name says ground/tree/pathway/lawn/pond) are removed from the
#     buildings list; with several building meshes left, 'refined' ones win.
#     Returns (building_obj_list, {class: path or None})."""
#     explicit = {"ground": args.ground_obj, "tree": args.trees_obj, "pathway": args.pathway_obj,
#                 "lawn": args.lawn_obj, "pond": args.pond_obj}
#     explicit_real = {os.path.realpath(p) for p in explicit.values() if p}
#     remaining = []
#     for p in _expand_paths(args.buildings_obj):
#         if os.path.realpath(p) in explicit_real:
#             continue
#         stem = os.path.basename(p).lower()
#         owner = next((c for c, pat in _NAME_PATTERNS.items() if re.search(pat, stem)), None)
#         if owner is None:
#             remaining.append(p)
#         elif not explicit[owner]:
#             explicit[owner] = p
#             log(f"[input] {p}: name says '{owner}'; using it as the {owner} surface")
#         else:
#             log(f"[input] {p}: ignored in --buildings_obj (a {owner} file was given explicitly)")
#     named = [p for p in remaining if "building" in os.path.basename(p).lower()]
#     pool = named or remaining
#     refined = [p for p in pool if "refined" in os.path.basename(p).lower()]
#     chosen = refined or pool
#     if len(chosen) > 1:
#         log(f"[input] several building meshes found, merging: {', '.join(chosen)}")
#     return chosen, explicit


# def _day_tags(dirs: list) -> list:
#     seen, out = {}, []
#     for d in dirs:
#         t = os.path.basename(os.path.normpath(d)) or "day"
#         n = seen.get(t, 0)
#         seen[t] = n + 1
#         out.append(t if n == 0 else f"{t}_{n}")
#     return out


# _FORK_STATE = {}


# def _day_worker(job):
#     return _FORK_STATE["fn"](*job)


# def run_days(day_fn, jobs: list, workers: int) -> list:
#     """jobs = [(weather_dir, tag)]. Several days + workers>1: one forked process
#     per day (geometry/BVH are inherited, not rebuilt). Otherwise days run in
#     order and `workers` threads trace shadow rays inside each day."""
#     if len(jobs) > 1 and workers > 1 and "fork" in multiprocessing.get_all_start_methods():
#         _FORK_STATE["fn"] = lambda wd, tag: day_fn(wd, tag, 1)
#         results = []
#         try:
#             with concurrent.futures.ProcessPoolExecutor(
#                     max_workers=min(workers, len(jobs)),
#                     mp_context=multiprocessing.get_context("fork")) as ex:
#                 futures = [ex.submit(_day_worker, j) for j in jobs]
#                 for j, fut in zip(jobs, futures):
#                     try:
#                         results.append(fut.result())
#                     except Exception as exc:
#                         results.append((j[1] or j[0], False, f"worker failed: {exc!r}"))
#         finally:
#             _FORK_STATE.clear()
#         return results
#     return [day_fn(wd, tag, workers) for wd, tag in jobs]


# def _out_prefix(args, tag) -> str:
#     return os.path.join(args.output_dir, f"{args.out_prefix}_{tag}" if tag else args.out_prefix)


# def _log_summary(label: str, face_df: pd.DataFrame) -> None:
#     s = class_summary(face_df)
#     for (grp, typ), sub in s.groupby(["surface_group", "surface_type"]):
#         hot, cold = sub.loc[sub["T_surface_mean_C"].idxmax()], sub.loc[sub["T_surface_mean_C"].idxmin()]
#         log(f"[day {label}] {grp}/{typ}: area {sub['area_m2'].iloc[0]:.1f} m2, mean T_surface "
#             f"max {hot['T_surface_mean_C']:.1f} C at {hot['hour']:g} h, "
#             f"min {cold['T_surface_mean_C']:.1f} C at {cold['hour']:g} h")


# COMPACT_FACE_COLUMNS = [
#     "building_id", "face_id", "surface_type", "surface_group", "area_m2", "hour", "T_amb_C",
#     "sunlit_fraction", "I_total_Wm2", "T_surface_C", "T_solair_C", "h_conv_Wm2K",
#     "q_solar_absorbed_Wm2", "q_conv_out_Wm2", "q_longwave_out_Wm2", "q_evap_out_Wm2",
#     "q_conduction_in_Wm2", "q_storage_Wm2"]
# _CSV_STATE = {}


# def _csv_part(job):
#     part, lo, hi, header = job
#     _CSV_STATE["df"].iloc[lo:hi].to_csv(part, index=False, header=header)
#     return part


# def write_csv_parallel(df: pd.DataFrame, path: str, workers: int = 1) -> None:
#     """DataFrame.to_csv is single-threaded and dominates the run time for millions
#     of rows. With workers > 1 the rows are split into slices that forked processes
#     write at the same time; the slices are then joined byte-for-byte. The result
#     is identical to a plain to_csv."""
#     n = len(df)
#     if workers <= 1 or n < 200_000 or "fork" not in multiprocessing.get_all_start_methods():
#         df.to_csv(path, index=False)
#         return
#     bounds = np.linspace(0, n, workers * 4 + 1).astype(int)
#     jobs = [(f"{path}.part{i}", int(bounds[i]), int(bounds[i + 1]), i == 0)
#             for i in range(len(bounds) - 1)]
#     _CSV_STATE["df"] = df
#     try:
#         with concurrent.futures.ProcessPoolExecutor(
#                 max_workers=workers, mp_context=multiprocessing.get_context("fork")) as ex:
#             list(ex.map(_csv_part, jobs))
#         with open(path, "wb") as out:
#             for job in jobs:
#                 with open(job[0], "rb") as part:
#                     shutil.copyfileobj(part, out, 1 << 24)
#     finally:
#         _CSV_STATE.clear()
#         for job in jobs:
#             try:
#                 os.remove(job[0])
#             except OSError:
#                 pass


# def _write_day_csvs(prefix: str, face_df, building_df, workers: int = 1, faces_csv: str = "full") -> None:
#     t0 = time.time()
#     written = []
#     if faces_csv != "none":
#         out = face_df if faces_csv == "full" else face_df[
#             [c for c in COMPACT_FACE_COLUMNS if c in face_df.columns]]
#         write_csv_parallel(out, f"{prefix}_faces_hourly.csv", workers)
#         written.append(f"{prefix}_faces_hourly.csv")
#     building_df.to_csv(f"{prefix}_building_hourly.csv", index=False)
#     class_summary(face_df).to_csv(f"{prefix}_class_summary.csv", index=False)
#     written += [f"{prefix}_building_hourly.csv", f"{prefix}_class_summary.csv"]
#     log(f"Wrote {', '.join(written)} ({time.time() - t0:.0f} s)")


# def make_obj_day_fn(bundle: ObjBundle, args):
#     def day_fn(weather_dir, tag, threads):
#         label = tag or os.path.basename(os.path.normpath(weather_dir))
#         t0 = time.time()
#         try:
#             log(f"[day {label}] start: {weather_dir} (threads={threads})")
#             analysis = solve_obj_day(
#                 bundle, weather_dir, t_initial_c=args.initial_temp_c,
#                 ground_initial_c=args.ground_initial_temp_c, ground_albedo=args.ground_albedo
#                 if args.ground_albedo is not None else DEFAULT_GROUND_ALBEDO,
#                 shadow_samples=args.shadow_samples, workers=threads)
#             face_df, building_df, faces = analysis
#             log(f"[day {label}] solved {len(face_df)} cell-hours in {time.time() - t0:.0f} s")
#             prefix = _out_prefix(args, tag)
#             _write_day_csvs(prefix, face_df, building_df, threads, args.faces_csv)
#             _log_summary(label, face_df)
#             log(f"[day {label}] CSV written at {time.time() - t0:.0f} s")
#             if args.vtk_out is not None:
#                 export_vtk_series_from_analysis(
#                     analysis, os.path.join(args.vtk_out, tag) if tag else args.vtk_out)
#             if args.plot and len(building_df):
#                 try:
#                     plot_results(face_df, building_df, prefix)
#                 except Exception as exc:
#                     warn(f"plot skipped: {exc}")
#             return label, True, f"done in {time.time() - t0:.0f} s"
#         except Exception as exc:
#             log(f"[day {label}] FAILED: {exc!r}\n{traceback.format_exc()}")
#             return label, False, repr(exc)
#     return day_fn


# def make_json_day_fn(scene: dict, args, wall_settings):
#     def day_fn(weather_dir, tag, threads):
#         label = tag or (os.path.basename(os.path.normpath(weather_dir)) if weather_dir else "clear-sky")
#         t0 = time.time()
#         try:
#             log(f"[day {label}] start (JSON pipeline)")
#             if weather_dir is not None:
#                 if args.vtk_out is not None:
#                     export_vtk_series(scene, weather_dir,
#                                       os.path.join(args.vtk_out, tag) if tag else args.vtk_out,
#                                       consider_shadows=not args.no_shadows,
#                                       wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
#                 face_df, building_df, faces = run_analysis_from_folder(
#                     scene, weather_dir, consider_shadows=not args.no_shadows,
#                     wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
#             else:
#                 if args.vtk_out is not None:
#                     warn("--vtk_out needs --weather_dir/--weather_root; VTK skipped")
#                 face_df, building_df, faces = run_analysis(
#                     scene, date(args.year, args.month, args.day), load_weather(args.weather_csv),
#                     utc_offset_hr=args.utc_offset, consider_shadows=not args.no_shadows,
#                     wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
#             prefix = _out_prefix(args, tag)
#             _write_day_csvs(prefix, face_df, building_df, threads, args.faces_csv)
#             _log_summary(label, face_df)
#             if args.plot:
#                 plot_results(face_df, building_df, prefix)
#             return label, True, f"done in {time.time() - t0:.0f} s"
#         except Exception as exc:
#             log(f"[day {label}] FAILED: {exc!r}\n{traceback.format_exc()}")
#             return label, False, repr(exc)
#     return day_fn


# def build_arg_parser():
#     import argparse
#     p = argparse.ArgumentParser(
#         description="Solar, conduction and storage model for buildings, ground, pathways, "
#                     "lawns, ponds and trees. Every input is optional; whatever is present "
#                     "and valid is solved.")
#     p.add_argument("--json", nargs="*", default=None,
#                    help="Scene JSON file(s) (globs fine). Used for material properties, and "
#                         "as the geometry source when no OBJ is given. Default: building_scene.json")
#     p.add_argument("--buildings_obj", nargs="*", default=None,
#                    help="Building OBJ(s). A shell glob like ../triSurface/*.obj is fine: files "
#                         "given for other classes (or named ground/tree/pathway/lawn/pond) are "
#                         "sorted out automatically.")
#     p.add_argument("--ground_obj", default=None)
#     p.add_argument("--trees_obj", default=None,
#                    help="Tree OBJ: opaque shadow casters AND solved canopy surfaces.")
#     p.add_argument("--pathway_obj", default=None)
#     p.add_argument("--lawn_obj", default=None)
#     p.add_argument("--pond_obj", default=None)
#     p.add_argument("--shadow_samples", type=int, choices=(1, 3), default=3,
#                    help="Rays per OBJ triangle: 3 (default) estimates partial shade; 1 uses the centroid.")
#     p.add_argument("--weather_csv", default=None,
#                    help="CSV hour,T_amb_C,wind_speed_ms[,...] (clear-sky path, JSON geometry only)")
#     p.add_argument("--weather_dir", nargs="*", default=None,
#                    help="One or more weatherData/MMDD folders (a glob like weatherData/*/. is fine); "
#                         "each is solved as a separate day.")
#     p.add_argument("--weather_root", default=None,
#                    help="weatherData/ root; combined with --month/--day to find MMDD")
#     p.add_argument("--month", type=int, default=6)
#     p.add_argument("--day", type=int, default=21)
#     p.add_argument("--year", type=int, default=2026)
#     p.add_argument("--utc_offset", type=float, default=0.0)
#     p.add_argument("--out_prefix", default="results")
#     p.add_argument("--output_dir", default=".", help="Directory for CSV/PNG outputs.")
#     p.add_argument("--vtk_out", default=None,
#                    help="Directory for the ParaView time series (a sub-folder per day when "
#                         "several weather days are given).")
#     p.add_argument("--vtk_ascii", action="store_true",
#                    help="Write ASCII legacy VTK instead of the default binary (about 100x slower "
#                         "and 3x larger; ParaView reads both).")
#     p.add_argument("--vtk_binary", action="store_true", help=argparse.SUPPRESS)  # now the default
#     p.add_argument("--faces_csv", choices=("full", "compact", "none"), default="full",
#                    help="Per-cell hourly CSV: all 56 columns (default), 18 key columns, or skip it "
#                         "(the VTK files carry the same fields). With --workers > 1 it is written in parallel.")
#     p.add_argument("--workers", type=int, default=1,
#                    help="Several weather days: that many days run in parallel (processes). "
#                         "A single day: that many threads trace shadow rays.")
#     p.add_argument("--plot", action="store_true")
#     p.add_argument("--no_shadows", action="store_true", help="Skip shadow casting (faster).")
#     p.add_argument("--no_overlap_removal", action="store_true",
#                    help="Keep ground cells that lie under lawn/pond/pathway cells.")
#     p.add_argument("--wall_thickness_m", type=float, default=0.23)
#     p.add_argument("--wall_nodes", type=int, default=5)
#     p.add_argument("--indoor_temp_c", type=float, default=24.0)
#     p.add_argument("--indoor_h", type=float, default=8.0)
#     p.add_argument("--initial_temp_c", type=float, default=None,
#                    help="Initial uniform building slab temperature; default --indoor_temp_c.")
#     # Overrides. Left unset (None) so JSON / built-in defaults apply.
#     p.add_argument("--ground_depth_m", type=float, default=None)
#     p.add_argument("--ground_nodes", type=int, default=None)
#     p.add_argument("--deep_ground_temp_c", type=float, default=None,
#                    help="Fixed deep temperature below ground/pathway/lawn/pond layers (default 24).")
#     p.add_argument("--ground_deep_h", type=float, default=None)
#     p.add_argument("--ground_initial_temp_c", type=float, default=None,
#                    help="Initial ground/pathway/lawn/pond temperature; default is the deep temperature.")
#     p.add_argument("--ground_conductivity", type=float, default=None)
#     p.add_argument("--ground_density", type=float, default=None)
#     p.add_argument("--ground_specific_heat", type=float, default=None)
#     p.add_argument("--ground_albedo", type=float, default=None)
#     p.add_argument("--ground_emissivity", type=float, default=None)
#     p.add_argument("--lawn_wetness", type=float, default=None,
#                    help="Lawn evaporation as a fraction of free-water evaporation (default 0.25).")
#     p.add_argument("--tree_wetness", type=float, default=None,
#                    help="Canopy transpiration fraction (default 0.30).")
#     p.add_argument("--pond_depth_m", type=float, default=None, help="Water depth (default 0.5).")
#     p.add_argument("--pond_mixing_factor", type=float, default=None,
#                    help="Pond effective conductivity = factor x 0.6 W/m/K, representing wind and "
#                         "convective mixing (default 30; 1 = still water, which heats only a thin skin).")
#     return p


# def main(argv=None) -> int:
#     parser = build_arg_parser()
#     args = parser.parse_args(argv)
#     args.workers = max(1, args.workers)
#     global VTK_BINARY
#     VTK_BINARY = not args.vtk_ascii
#     if args.pond_mixing_factor is not None and not (math.isfinite(args.pond_mixing_factor)
#                                                    and args.pond_mixing_factor > 0):
#         parser.error("--pond_mixing_factor must be a positive number")
#     os.makedirs(args.output_dir, exist_ok=True)
#     try:
#         wall_settings = WallSettings(args.wall_thickness_m, args.wall_nodes,
#                                      args.indoor_temp_c, args.indoor_h)
#     except ValueError as error:
#         parser.error(str(error))

#     # ---- inputs -----------------------------------------------------------
#     json_args = args.json if args.json else (["building_scene.json"]
#                                              if os.path.isfile("building_scene.json") else [])
#     scene, scene_files = load_scenes(json_args)
#     log(f"[input] scene JSON: {', '.join(scene_files) if scene_files else 'none'}")
#     building_objs, class_objs = classify_obj_inputs(args)

#     weather_dirs, seen = [], set()
#     for d in _expand_paths(args.weather_dir):
#         rp = os.path.realpath(d)
#         if os.path.isdir(d) and rp not in seen:
#             seen.add(rp)
#             weather_dirs.append(d)
#         elif not os.path.isdir(d):
#             warn(f"weather_dir: not a directory, skipping: {d}")
#     if not weather_dirs and args.weather_root:
#         try:
#             weather_dirs = [resolve_weather_folder(args.weather_root, args.month, args.day)]
#         except FileNotFoundError as exc:
#             warn(str(exc))
#     log(f"[input] weather days: {len(weather_dirs)}  workers: {args.workers}")

#     overrides = {
#         "ground": dict(depth_m=args.ground_depth_m, nodes=args.ground_nodes,
#                        deep_h_wm2k=args.ground_deep_h, thermal_conductivity=args.ground_conductivity,
#                        density=args.ground_density, specific_heat=args.ground_specific_heat,
#                        albedo=args.ground_albedo, emissivity=args.ground_emissivity,
#                        deep_temperature_c=args.deep_ground_temp_c),
#         "pathway": dict(deep_temperature_c=args.deep_ground_temp_c),
#         "lawn": dict(deep_temperature_c=args.deep_ground_temp_c, wetness=args.lawn_wetness),
#         "pond": dict(deep_temperature_c=args.deep_ground_temp_c, depth_m=args.pond_depth_m,
#                      mixing_factor=args.pond_mixing_factor),
#         "tree": dict(wetness=args.tree_wetness),
#     }

#     # ---- choose pipeline --------------------------------------------------
#     day_fn = None
#     jobs = []
#     any_obj = bool(building_objs) or any(class_objs.values())
#     if any_obj:
#         if not weather_dirs:
#             log("ERROR: the OBJ pipeline needs real weather data: pass --weather_dir or --weather_root")
#             return 2
#         bundle = prepare_obj_bundle(
#             building_objs or None, class_objs["ground"], class_objs["tree"], class_objs["pathway"],
#             class_objs["lawn"], class_objs["pond"], scene=scene, wall_settings=wall_settings,
#             layer_overrides=overrides, consider_shadows=not args.no_shadows,
#             remove_overlaps=not args.no_overlap_removal)
#         if bundle.n_cells:
#             log(f"[input] OBJ pipeline: {bundle.n_cells} cells in total")
#             day_fn = make_obj_day_fn(bundle, args)
#         else:
#             warn("no valid OBJ geometry found")
#     if day_fn is None:
#         if scene is None or not scene.get("buildings"):
#             log("ERROR: nothing to solve: no valid OBJ files and no buildings in a scene JSON")
#             return 2
#         log("[input] JSON pipeline (buildings from the scene JSON; shadows from buildings and "
#             "tree spheres; lawns/ponds/ground need OBJ files)")
#         day_fn = make_json_day_fn(scene, args, wall_settings)
#     if weather_dirs:
#         tags = _day_tags(weather_dirs) if len(weather_dirs) > 1 else [None]
#         jobs = list(zip(weather_dirs, tags))
#     else:
#         jobs = [(None, None)]

#     results = run_days(day_fn, jobs, args.workers)
#     failed = [r for r in results if not r[1]]
#     for label, ok, msg in results:
#         log(f"[result] {label}: {'OK' if ok else 'FAILED'} - {msg}")
#     return 1 if failed and len(failed) == len(results) else 0


# if __name__ == "__main__":
#     sys.exit(main())



#Subrat: Works for buildings trees ponds lawns and ground but writing is very slow. Newer one above writes vtk in binary making it much faster. This one is kept for reference and for debugging.

# """
# Run it as:
# python3 solar_thermal_model.py --vtk_out . --weather_dir weatherData/*/.   --buildings_obj ../momentum/constant/triSurface/buildings_refined.obj  --trees_obj ../momentum/constant/triSurface/trees.obj   --pathway_obj ../momentum/constant/triSurface/pathways.obj   --lawn_obj ../momentum/constant/triSurface/lawns.obj   --pond_obj ../momentum/constant/triSurface/ponds.obj   --json *.json --output_dir . --workers 2> log.solar_thermal_model 2>&1

# Analytical solar-exposure + convective/radiative surface-temperature model
# for buildings exported from the scene JSON (SNU-SPH / urban-microclimate
# style export: buildings[].faces[] with vertices, normal, surfaceType,
# materialProperties.thermalProperties; optionally trees[] for canopy shading).

# No CFD, and no full (multi-bounce) ray tracing -- everything below is
# geometry plus visibility rays toward the sun at each weather time:

#   1. SOLAR GEOMETRY  (Cooper's declination + standard hour-angle equations)
#      -> sun altitude & azimuth for any date / time / lat-lon.

#   2. CLEAR-SKY IRRADIANCE  (Hottel, 1976 atmospheric-transmittance model)
#      -> direct-normal (I_DN) and diffuse-horizontal (I_dH) irradiance from
#         the solar altitude alone -- used only when you DON'T have a real
#         weather-data export (see `run_analysis`). If you DO have one (a
#         `weatherData/MMDD/` folder with Tambient, windSpeed, IDN, Idif,
#         sunPosVector, ...), use `run_analysis_from_folder` instead: it
#         reads the real sun-position vector and real DNI/diffuse straight
#         from those files and skips this model entirely -- more accurate,
#         and removes any ambiguity about axis conventions since the sun
#         vector is already given in the same (East, North, Up) axes as the
#         building JSON's face normals.

#   3. PER-FACE SUNLIT TEST  -- cos(theta) = dot(face_normal, sun_unit_vector).
#      A face receives direct beam radiation only when cos(theta) > 0 AND the
#      sun is above the horizon (self-orientation), AND (new) a visibility ray
#      from the face toward the sun is not blocked by another building or tree
#      in the scene (mutual shading -- see Section 5, "Shadow occlusion").

#   4. CONVECTION -- classic linear wind correlation h_c = a + b*V (McAdams-
#      type, W/m^2K), applied per face using your wind-speed time series.

#   5. SURFACE TEMPERATURE: each exposed building or requested ground triangle
#      is a 0.23 m slab with five nodes through its thickness, heat capacity
#      rho*cp*L per unit area, and
#      conductance k/dx between adjacent nodes. A matrix exponential advances
#      conduction/storage exactly for the spatially discretized system, holding
#      weather and ambient-linearized longwave coefficients constant between
#      records. A building inner face exchanges heat with fixed 24 C indoor air;
#      a ground back face exchanges heat with fixed deep soil. These settings,
#      materials, and initial temperatures are configurable.
#      T_solair_C remains an equivalent driving temperature for the bulk model.

#   6. WHOLE-BUILDING TEMPERATURE -- single-node RC (lumped capacitance,
#      C = rho*V*cp) network per building, driven by that building's own
#      faces' sol-air temperatures. Because C and the effective conductance
#      h_eff are taken as constant within each hour, the ODE
#      C dT/dt = h_eff*(T_eq - T) has the exact analytical solution

#          T(t+dt) = T_eq + (T(t) - T_eq) * exp(-dt / tau),   tau = C/h_eff

#      so the whole march-forward is a closed-form exponential step, not a
#      numerical ODE integration. Each building in the scene gets its own
#      node (its own C, its own faces) -- they are only coupled through the
#      shadows they cast on each other, not thermally.

# Assumptions you should sanity-check against your own scene / site
# (flagged again inline where they are used):
#   - Scene axes are treated as ENU: +x = East, +y = North, +z = Up. If your
#     tool defines "north" differently, set `north_offset_deg` below.
#   - Longwave emissivity is not in the JSON -> defaulted to 0.9 (typical
#     concrete/masonry). Change `DEFAULT_EMISSIVITY` if you know better.
#   - Ground/surroundings albedo for the reflected-radiation term is not in
#      the JSON -> defaulted to 0.2. Change `DEFAULT_GROUND_ALBEDO`.
#   - The optional OBJ ground calculation uses the case soil defaults k=1.5
#     W/m/K, rho=1150 kg/m3 and cp=650 J/kg/K. Its active depth is 0.23 m and
#     its default lower conductance k/L connects it to fixed 24 C deep soil.
#   - The floor face (surfaceType == "floor") is excluded from solar +
#     outdoor convection (it's in contact with the ground, not outdoor air).
#     Toggle `INCLUDE_FLOOR` if you want to add a simple ground-conduction
#     term instead. A "ceiling" face (surfaceType == "ceiling") is *always*
#     excluded the same way, with no toggle -- in multi-floor exports (one
#     JSON "building" per story, stacked in z) this tags the boundary between
#     two stacked stories, which never touches outdoor air regardless of
#     settings. Only "wall" and "roof" faces are treated as always outdoor-
#     exposed.
#   - Two of the face normals in the source JSON point the "wrong" way for
#     what they geometrically are (roof normal given as (0,0,-1), i.e.
#     inward, floor as (0,0,1), also inward). This script auto-corrects
#     normal sign using `surfaceType` so the physics is right regardless of
#     what the exporter wrote. Also, wall face "area" was exported as 0 in
#     this file -- areas are recomputed from the vertices, never trusted
#     from the JSON.
#   - The OBJ path retains individual mesh triangles, with three visibility
#     samples per triangle by default, accelerated by a triangle BVH. The JSON
#     path still tests one centroid per polygon. Only the direct
#     beam is blockable; the diffuse-sky and ground-reflected components
#     still use the unobstructed sky-view-factor formulas (a standard
#     simplification -- properly reducing diffuse view factors for nearby
#     obstructions would need a more expensive hemispherical integration).
#   - In the JSON path, tree canopies are approximated as a bounding sphere for shadow
#     purposes (a common simplification in urban solar-access studies), not
#     actual leaf/branch geometry. The JSON field names used to build that
#     sphere are guessed from common conventions (see `build_tree_occluder`)
#     since no example tree entry was available when this was written --
#     check `_TREE_*_KEYS` against your real tree schema and adjust if needed.
#     The OBJ path traces the actual tree triangles, treating them as opaque.

# MULTI-SURFACE OBJ MODE (buildings, ground, pathway, lawn, pond, trees)
#   Every component is optional; whatever has valid triangles is solved and the
#   rest is skipped with a log line. Geometry comes from the OBJ files, material
#   properties from the scene JSON when present (else the documented defaults;
#   the log lists the source of every property). Ground-like classes use the
#   same transient slab solver as the walls. Ponds, lawns and tree canopies
#   additionally lose latent heat by evaporation/transpiration (Lewis analogy,
#   linearised about ambient so the exact matrix-exponential update still
#   applies), and tree canopies exchange heat with ambient air on their back
#   side. Ground cells under building roofs or under pond/pathway/lawn cells
#   are removed so nothing is solved twice. Several weather days can be given:
#   each becomes its own set of outputs, run in parallel with --workers.
# """

# import concurrent.futures
# import glob
# import json
# import math
# import multiprocessing
# import os
# import re
# import sys
# import time
# import traceback
# from dataclasses import dataclass, field
# from datetime import date
# from typing import Optional

# import numpy as np
# import pandas as pd

# # --------------------------------------------------------------------------
# # Tunable defaults (not present in the JSON -- see docstring above)
# # --------------------------------------------------------------------------
# DEFAULT_EMISSIVITY = 0.90
# DEFAULT_GROUND_ALBEDO = 0.20
# STEFAN_BOLTZMANN = 5.670374e-8          # W/m^2/K^4
# SOLAR_CONSTANT = 1367.0                 # W/m^2
# INCLUDE_FLOOR = False                   # floor treated as adiabatic (ground contact)


# # ==========================================================================
# # 1. Geometry: load scene, fix normals, compute real areas
# # ==========================================================================

# @dataclass
# class Face:
#     building_id: str
#     face_id: str
#     surface_type: str
#     normal: np.ndarray          # outward unit normal, corrected
#     area: float                 # m^2, computed from vertices (JSON area ignored)
#     tilt_deg: float             # 0 = horizontal facing up, 90 = vertical, 180 = facing down
#     center: np.ndarray
#     vertices: np.ndarray        # (N,3) planar polygon loop, as given (for geometry export)
#     thermal_conductivity: float
#     density: float
#     specific_heat: float
#     albedo: float
#     emissivity: float = DEFAULT_EMISSIVITY
#     parent_face_id: str = ""    # logical wall/roof; OBJ physics stays per triangle


# def _polygon_area_3d(verts: np.ndarray) -> float:
#     """Area of a planar polygon in 3D via Newell's method (cross-product sum)."""
#     n = len(verts)
#     normal_sum = np.zeros(3)
#     for i in range(n):
#         v1 = verts[i]
#         v2 = verts[(i + 1) % n]
#         normal_sum += np.cross(v1, v2)
#     return 0.5 * np.linalg.norm(normal_sum)


# def load_scene(json_path: str) -> dict:
#     with open(json_path, "r") as f:
#         return json.load(f)


# def extract_faces(scene: dict) -> list:
#     """Flatten every building's faces into a list of corrected Face objects."""
#     faces = []
#     for b in scene.get("buildings", []):
#         bid = b.get("id", "building")
#         for f in b.get("faces", []):
#             verts = np.array([[v["x"], v["y"], v["z"]] for v in f["vertices"]])
#             area = _polygon_area_3d(verts)          # never trust JSON "area"
#             n = np.array([f["normal"]["x"], f["normal"]["y"], f["normal"]["z"]], dtype=float)
#             n = n / (np.linalg.norm(n) + 1e-12)

#             stype = f.get("surfaceType", "wall")
#             # --- fix inward-pointing roof/floor normals (see module docstring) ---
#             if stype == "roof" and n[2] < 0:
#                 n = -n
#             if stype == "floor" and n[2] > 0:
#                 n = -n

#             tilt_deg = math.degrees(math.acos(np.clip(n[2], -1.0, 1.0)))
#             c = f.get("center", {"x": 0, "y": 0, "z": 0})
#             center = np.array([c["x"], c["y"], c["z"]])

#             def _prop(name, default):
#                 value = _lookup_property(f, name)
#                 return default if value is None else value
#             faces.append(Face(
#                 building_id=bid,
#                 face_id=f["id"],
#                 surface_type=stype,
#                 normal=n,
#                 area=area,
#                 tilt_deg=tilt_deg,
#                 center=center,
#                 vertices=verts,
#                 thermal_conductivity=_prop("thermal_conductivity", DEFAULT_MATERIAL["thermal_conductivity"]),
#                 density=_prop("density", DEFAULT_MATERIAL["density"]),
#                 specific_heat=_prop("specific_heat", DEFAULT_MATERIAL["specific_heat"]),
#                 albedo=_prop("albedo", DEFAULT_ALBEDO_ROOF if stype == "roof" else DEFAULT_ALBEDO_WALL),
#                 emissivity=_prop("emissivity", DEFAULT_EMISSIVITY),
#             ))
#     return faces


# def building_volume(scene: dict, building_id: Optional[str] = None) -> float:
#     for b in scene.get("buildings", []):
#         if building_id is None or b.get("id") == building_id:
#             geo = b.get("geometry", {})
#             if "volume" in geo:
#                 return geo["volume"]
#             return geo.get("footprintArea", 0) * b.get("height", 0)
#     return 0.0


# def site_location(scene: dict) -> tuple:
#     """Returns (lat_deg, lon_deg) -- prefers shadowSettings, falls back to site centroid."""
#     ss = scene.get("settings", {}).get("shadowSettings")
#     if ss and "latitude" in ss:
#         return ss["latitude"], ss["longitude"]
#     coords = scene["site"]["properties"]["originalCoordinates"]
#     lat = float(np.mean([c["lat"] for c in coords]))
#     lon = float(np.mean([c["lng"] for c in coords]))
#     return lat, lon


# # ==========================================================================
# # 2. Solar geometry
# # ==========================================================================

# def solar_position(lat_deg: float, lon_deg: float, day_of_year: int,
#                     local_std_hour: float, utc_offset_hr: float = 0.0) -> tuple:
#     """
#     Standard solar-position equations (Cooper's declination + hour angle).
#     Returns (altitude_deg, azimuth_deg); azimuth measured clockwise from
#     true North (0=N, 90=E, 180=S, 270=W). altitude < 0 => sun below horizon.
#     """
#     lat = math.radians(lat_deg)
#     n = day_of_year

#     decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0)))

#     # Equation of time (minutes), Duffie & Beckman form
#     B = math.radians(360.0 * (n - 81) / 364.0)
#     eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

#     std_meridian = 15.0 * utc_offset_hr
#     time_corr_min = 4.0 * (lon_deg - std_meridian) + eot
#     solar_time = local_std_hour + time_corr_min / 60.0

#     hour_angle = math.radians(15.0 * (solar_time - 12.0))

#     sin_alt = (math.sin(lat) * math.sin(decl) +
#                math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
#     altitude = math.asin(np.clip(sin_alt, -1.0, 1.0))

#     # Azimuth, measured clockwise from true North (standard NOAA-style form):
#     #   cos(A) = (sin(decl) - sin(lat)*sin(alt)) / (cos(lat)*cos(alt))
#     # acos() only returns 0-180 deg, so mirror into 180-360 for the
#     # afternoon (hour_angle > 0) to get the correct NE-at-sunrise /
#     # NW-at-sunset quadrant.
#     cos_alt = max(math.cos(altitude), 1e-9)
#     cos_A = (math.sin(decl) - math.sin(lat) * math.sin(altitude)) / (math.cos(lat) * cos_alt)
#     az_raw = math.degrees(math.acos(np.clip(cos_A, -1.0, 1.0)))
#     azimuth = az_raw if hour_angle <= 0 else 360.0 - az_raw

#     return math.degrees(altitude), azimuth


# def sun_unit_vector(altitude_deg: float, azimuth_deg: float,
#                      north_offset_deg: float = 0.0) -> np.ndarray:
#     """
#     Sun direction as a unit vector in scene coordinates, assuming
#     +x = East, +y = North, +z = Up (rotate with north_offset_deg if your
#     scene's "north" is not aligned with +y).
#     """
#     alt = math.radians(altitude_deg)
#     az = math.radians(azimuth_deg - north_offset_deg)
#     east = math.cos(alt) * math.sin(az)
#     north = math.cos(alt) * math.cos(az)
#     up = math.sin(alt)
#     return np.array([east, north, up])


# # ==========================================================================
# # 3. Clear-sky irradiance (Hottel, 1976)
# # ==========================================================================

# def hottel_clear_sky(altitude_deg: float, site_altitude_km: float = 0.0,
#                       r0: float = 1.0, r1: float = 1.0, rk: float = 1.0) -> tuple:
#     """
#     Hottel's (1976) atmospheric-transmittance clear-sky model.
#     r0, r1, rk are the climate-type correction factors from Hottel's paper
#     (defaults =1.0, i.e. the reference "mid-latitude summer" atmosphere --
#     nudge by a few % for your own climate/visibility if you know it).
#     Returns (I_DN, I_dH) in W/m^2 -- direct-normal and diffuse-horizontal.
#     Returns (0, 0) when the sun is below the horizon.
#     """
#     if altitude_deg <= 0:
#         return 0.0, 0.0
#     A = site_altitude_km
#     sin_b = math.sin(math.radians(altitude_deg))

#     a0 = r0 * (0.4237 - 0.00821 * (6.0 - A) ** 2)
#     a1 = r1 * (0.5055 + 0.00595 * (6.5 - A) ** 2)
#     k = rk * (0.2711 + 0.01858 * (2.5 - A) ** 2)

#     tau_b = a0 + a1 * math.exp(-k / sin_b)
#     tau_d = 0.2710 - 0.2939 * tau_b   # Liu-Jordan diffuse-transmittance approx

#     i_dn = SOLAR_CONSTANT * tau_b
#     i_dh = SOLAR_CONSTANT * tau_d * sin_b
#     return max(i_dn, 0.0), max(i_dh, 0.0)


# def sky_temperature(t_amb_k: float) -> float:
#     """Swinbank (1963) clear-sky effective sky temperature, in Kelvin."""
#     return 0.0552 * t_amb_k ** 1.5


# # ==========================================================================
# # 4. Per-face irradiance & sol-air temperature
# # ==========================================================================

# def face_irradiance(face: Face, sun_vec: np.ndarray, altitude_deg: float,
#                      i_dn: float, i_dh: float,
#                      ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                      external_shadow: bool = False) -> dict:
#     """
#     Direct + diffuse (isotropic sky) + ground-reflected irradiance on one face.
#     `external_shadow=True` means a ray/occlusion test (Section 5) found this
#     face blocked by another building or tree this hour -- the direct beam is
#     then zeroed regardless of the face's own orientation toward the sun.
#     """
#     cos_theta = float(np.dot(face.normal, sun_vec))
#     self_lit = (altitude_deg > 0) and (cos_theta > 0)
#     sunlit = self_lit and not external_shadow

#     i_direct = i_dn * cos_theta if sunlit else 0.0

#     tilt = math.radians(face.tilt_deg)
#     view_sky = (1 + math.cos(tilt)) / 2.0     # 1 for roof, 0.5 for wall, 0 for floor
#     view_ground = 1.0 - view_sky
#     i_diffuse = i_dh * view_sky

#     i_gh = i_dn * math.sin(math.radians(max(altitude_deg, 0.0))) + i_dh
#     i_reflected = ground_albedo * i_gh * view_ground

#     return dict(sunlit=sunlit, self_lit=self_lit, shadowed=external_shadow,
#                 cos_theta=cos_theta, view_sky=view_sky,
#                 i_direct=i_direct, i_diffuse=i_diffuse, i_reflected=i_reflected,
#                 i_total=i_direct + i_diffuse + i_reflected)


# def convective_h(wind_speed_ms: float, a: float = 5.7, b: float = 3.8) -> float:
#     """McAdams-type linear wind correlation (W/m^2K)."""
#     return a + b * max(wind_speed_ms, 0.0)


# def sol_air_temperature(t_amb_c: float, i_total: float, h_c: float,
#                          albedo: float, view_sky: float,
#                          emissivity: float = DEFAULT_EMISSIVITY) -> float:
#     """
#     ASHRAE sol-air temperature: the fictitious outdoor air temperature that
#     would give the same solar + convective + long-wave heat gain as the
#     real conditions, so a face just "sees" T_sol_air via plain convection.
#     """
#     alpha = 1.0 - albedo
#     t_amb_k = t_amb_c + 273.15
#     t_sky_k = sky_temperature(t_amb_k)
#     # net long-wave LOSS to the sky, scaled by how much sky this face can see
#     q_lw_loss = view_sky * emissivity * STEFAN_BOLTZMANN * (t_amb_k ** 4 - t_sky_k ** 4)
#     return t_amb_c + (alpha * i_total - q_lw_loss) / h_c


# def surface_temperature(t_amb_c, i_total, h_c, albedo, view_sky,
#                         emissivity=DEFAULT_EMISSIVITY):
#     """Local steady exterior temperature (C), with no wall conduction/storage.

#     Solve alpha*I = h_c*(Ts-Ta) + eps*sigma*(Ts^4 - Fsky*Tsky^4
#                                                 - (1-Fsky)*Ta^4).
#     Ground/surroundings are at ambient temperature; sky uses Swinbank.
#     Supports scalar or broadcastable array inputs. Unlike sol-air temperature,
#     outgoing longwave radiation is evaluated at the actual surface temperature.
#     """
#     ta = np.asarray(t_amb_c) + 273.15
#     absorbed = (1.0 - np.asarray(albedo)) * np.asarray(i_total)
#     eps_sigma = np.asarray(emissivity) * STEFAN_BOLTZMANN
#     background = np.asarray(view_sky) * sky_temperature(ta)**4 + (1-np.asarray(view_sky))*ta**4
#     ts = np.asarray(ta + absorbed / h_c)
#     for _ in range(30):
#         residual = h_c*(ts-ta) + eps_sigma*(ts**4-background) - absorbed
#         step = residual / (h_c + 4*eps_sigma*ts**3)
#         ts = ts - step
#         if np.max(np.abs(step), initial=0.0) < 1e-8:
#             break
#     return ts - 273.15


# @dataclass(frozen=True)
# class WallSettings:
#     """Uniform slab settings for exposed walls and roofs; SI units except C."""
#     thickness_m: float = 0.23
#     nodes: int = 5
#     indoor_temperature_c: float = 24.0
#     indoor_h_wm2k: float = 8.0

#     def __post_init__(self):
#         if not np.isfinite(self.thickness_m) or self.thickness_m <= 0:
#             raise ValueError("wall thickness must be finite and positive")
#         if not isinstance(self.nodes, int) or self.nodes < 2:
#             raise ValueError("wall nodes must be an integer >= 2")
#         if not np.isfinite(self.indoor_temperature_c) or self.indoor_temperature_c <= -273.15:
#             raise ValueError("indoor temperature must be finite and above absolute zero")
#         if not np.isfinite(self.indoor_h_wm2k) or self.indoor_h_wm2k < 0:
#             raise ValueError("indoor heat-transfer coefficient must be finite and nonnegative")


# @dataclass(frozen=True)
# class GroundSettings:
#     """Thermal properties and lower boundary for exposed ground triangles.

#     The active soil layer uses the same nodal slab solver as a wall. Its lower
#     face exchanges heat with fixed deep-soil temperature through
#     ``deep_h_wm2k``. When that value is omitted, k/depth is used, representing
#     conduction through one additional layer of the same thickness.
#     """
#     depth_m: float = 0.23
#     nodes: int = 5
#     deep_temperature_c: float = 24.0
#     deep_h_wm2k: Optional[float] = None
#     thermal_conductivity: float = 1.5
#     density: float = 1150.0
#     specific_heat: float = 650.0
#     albedo: float = DEFAULT_GROUND_ALBEDO
#     emissivity: float = 0.95
#     wetness: float = 0.0                  # fraction of potential evaporation (0 dry .. 1 water)
#     back_follows_ambient: bool = False    # back face exchanges with ambient air (tree canopy)

#     def __post_init__(self):
#         if not np.isfinite(self.depth_m) or self.depth_m <= 0:
#             raise ValueError("ground depth must be finite and positive")
#         if not np.isfinite(self.wetness) or not 0 <= self.wetness <= 1:
#             raise ValueError("wetness must be finite and between 0 and 1")
#         WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
#                      self.effective_deep_h_wm2k)
#         material = np.array([self.thermal_conductivity, self.density,
#                              self.specific_heat], dtype=float)
#         if np.any(~np.isfinite(material)) or np.any(material <= 0):
#             raise ValueError("ground conductivity, density and heat capacity must be positive and finite")
#         if not np.isfinite(self.albedo) or not 0 <= self.albedo <= 1:
#             raise ValueError("ground albedo must be finite and between 0 and 1")
#         if not np.isfinite(self.emissivity) or not 0 <= self.emissivity <= 1:
#             raise ValueError("ground emissivity must be finite and between 0 and 1")

#     @property
#     def effective_deep_h_wm2k(self) -> float:
#         return (self.thermal_conductivity / self.depth_m
#                 if self.deep_h_wm2k is None else self.deep_h_wm2k)

#     def slab_settings(self) -> WallSettings:
#         return WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
#                             self.effective_deep_h_wm2k)


# class SlabConduction:
#     """Independent 1-D slabs, with an exact exponential update in time.

#     Boundary nodes lie on the actual surfaces and have half control volumes.
#     Interior nodes have full control volumes. Per-area nodal capacities sum
#     to rho*cp*L; adjacent conductance is k/dx. No lateral cell coupling.
#     Grouping equal materials lets all their triangles share a small matrix.
#     """

#     def __init__(self, conductivity, density, specific_heat, emissivity,
#                  initial_c, settings=None):
#         self.settings = settings or WallSettings()
#         properties = np.column_stack(np.broadcast_arrays(
#             conductivity, density, specific_heat, emissivity))
#         if np.any(~np.isfinite(properties)) or np.any(properties[:, :3] <= 0):
#             raise ValueError("wall conductivity, density and heat capacity must be positive and finite")
#         self.dx = self.settings.thickness_m / (self.settings.nodes-1)
#         widths = np.full(self.settings.nodes, self.dx)
#         widths[[0, -1]] *= 0.5
#         self.capacity = properties[:, 1, None]*properties[:, 2, None]*widths
#         self.conductance = properties[:, 0] / self.dx
#         self.temperatures = np.full((len(properties), self.settings.nodes), initial_c, dtype=float)
#         if not np.isfinite(initial_c) or initial_c <= -273.15:
#             raise ValueError("initial wall temperature must be finite and above absolute zero")
#         self.initial_c = initial_c
#         self.back_h = self.settings.indoor_h_wm2k
#         self.back_temperature = self.settings.indoor_temperature_c
#         _, inverse = np.unique(properties, axis=0, return_inverse=True)
#         self.groups = [np.flatnonzero(inverse == i) for i in range(inverse.max()+1)]

#     def set_back(self, h, temperature):
#         """Replace the back-face coefficient / temperature (time-varying back
#         boundary, e.g. ambient air behind a tree canopy). Used by advance()
#         and diagnostics() until changed again."""
#         self.back_h = float(h)
#         self.back_temperature = float(temperature)

#     def advance(self, dt_seconds, external_h, external_source):
#         """C*Tdot = -K*T + b; exterior input is source - h*T[0].

#         Input coefficients are held constant for dt. Transform with sqrt(C)
#         to a symmetric matrix, then use its eigendecomposition to evaluate
#         exp(A*dt) and the integrated forcing. No SciPy dependency or explicit
#         time stepping; zero eigenvalues are handled for insulated slabs.
#         """
#         if not np.isfinite(dt_seconds) or dt_seconds < 0:
#             raise ValueError("wall time step must be finite and nonnegative")
#         if dt_seconds == 0:
#             return
#         n = self.settings.nodes
#         external_h = np.broadcast_to(external_h, len(self.temperatures))
#         external_source = np.broadcast_to(external_source, len(self.temperatures))
#         for ids in self.groups:
#             # Radiation coefficient depends on ambient and emissivity only,
#             # so it is common to a material group within a weather interval.
#             if not np.allclose(external_h[ids], external_h[ids[0]], rtol=0, atol=1e-12):
#                 raise ValueError("external h must be equal within each material group")
#             c = self.capacity[ids[0]]
#             root_c = np.sqrt(c)
#             g = self.conductance[ids[0]]
#             K = np.diag(np.full(n, 2*g)) + np.diag(np.full(n-1, -g), 1) + np.diag(np.full(n-1, -g), -1)
#             K[0,0] = g + external_h[ids[0]]
#             K[-1,-1] = g + self.back_h
#             A = -K / root_c[:, None] / root_c[None, :]
#             eigenvalues, eigenvectors = np.linalg.eigh(A)
#             exponent = np.exp(eigenvalues*dt_seconds)
#             phi = np.full(n, dt_seconds, dtype=float)
#             np.divide(np.expm1(eigenvalues*dt_seconds), eigenvalues,
#                       out=phi, where=np.abs(eigenvalues) > 1e-15)
#             b = np.zeros((len(ids), n))
#             b[:,0] = external_source[ids]
#             b[:,-1] = self.back_h*self.back_temperature
#             modes = (self.temperatures[ids]*root_c) @ eigenvectors
#             forcing = (b/root_c) @ eigenvectors
#             self.temperatures[ids] = ((modes*exponent + forcing*phi) @ eigenvectors.T) / root_c

#     def diagnostics(self, external_h, external_source):
#         """Instantaneous conductive/storage fluxes at the current output time."""
#         t = self.temperatures
#         q_outer = external_source-external_h*t[:,0]
#         q_inner = self.back_h*(t[:,-1]-self.back_temperature)
#         q_conduction = self.conductance*(t[:,0]-t[:,1])
#         t_mean = np.sum(self.capacity*t,axis=1)/self.capacity.sum(axis=1)
#         capacity = self.capacity.sum(axis=1)
#         return dict(T_inner_C=t[:,-1].copy(), T_back_surface_C=t[:,-1].copy(),
#                     T_wall_mean_C=t_mean, T_slab_mean_C=t_mean,
#                     q_conduction_in_Wm2=q_conduction, q_indoor_out_Wm2=q_inner,
#                     q_back_out_Wm2=q_inner,
#                     q_storage_Wm2=q_outer-q_inner,
#                     q_outer_node_storage_Wm2=q_outer-q_conduction,
#                     stored_heat_Jm2=np.sum(self.capacity*(t-self.initial_c),axis=1),
#                     wall_capacity_Jm2K=capacity, slab_capacity_Jm2K=capacity)


# # ==========================================================================
# # 5. Shadow occlusion -- cheap two-phase ray tracing
# #
# #    Phase 1 (broad, O(1) per occluder): does the ray toward the sun even
# #             pass through the occluder's axis-aligned bounding box?
# #    Phase 2 (narrow, exact): only for occluders that survive phase 1 --
# #             Moller-Trumbore ray-triangle tests against the occluder's
# #             actual wall/roof triangles (buildings), or a closed-form
# #             ray-sphere test (trees, canopy approximated as a sphere).
# #
# #    This is the standard bounding-volume-culling idea from ray tracing,
# #    kept deliberately simple (no BVH/kd-tree): for scenes with up to a
# #    few hundred buildings this keeps the great majority of rays from ever
# #    touching a triangle, which is normally the expensive part.
# # ==========================================================================

# @dataclass
# class Occluder:
#     kind: str                                   # "building" or "tree"
#     source_id: str
#     aabb_min: np.ndarray
#     aabb_max: np.ndarray
#     triangles: list = field(default_factory=list)     # [(p0,p1,p2), ...] for buildings
#     sphere_center: Optional[np.ndarray] = None          # for trees
#     sphere_radius: Optional[float] = None


# def _triangulate_quad(v0, v1, v2, v3):
#     return [(v0, v1, v2), (v0, v2, v3)]


# def _fan_triangulate(verts):
#     """Fan triangulation from the first vertex. Exact for convex polygons
#     (true for simple rectangular footprints); an approximation for
#     concave ones -- flag this if you have complex (e.g. OSM-derived,
#     L-shaped) footprints and it matters for your results."""
#     return [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]


# def build_building_occluder(building_json: dict) -> Optional[Occluder]:
#     """
#     Triangulates one building's wall + roof faces for use as an occluder of
#     OTHER buildings/faces. The floor is skipped -- sitting at ground level,
#     it cannot shadow anything above it, so leaving it out is a free
#     reduction in triangle count with no effect on the result.
#     """
#     tris, all_pts = [], []
#     for f in building_json.get("faces", []):
#         if f.get("surfaceType") == "floor":
#             continue
#         verts = [np.array([v["x"], v["y"], v["z"]]) for v in f["vertices"]]
#         all_pts.extend(verts)
#         if len(verts) == 4:
#             tris.extend(_triangulate_quad(*verts))
#         elif len(verts) == 3:
#             tris.append(tuple(verts))
#         elif len(verts) > 4:
#             tris.extend(_fan_triangulate(verts))
#     if not tris:
#         return None
#     pts = np.array(all_pts)
#     return Occluder(kind="building", source_id=building_json.get("id", "building"),
#                      aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tris)


# # Field-name guesses for tree entries -- NOT verified against a real export
# # (none was available when this was written). Check these against your
# # actual tree JSON and adjust if the names differ.
# _TREE_RADIUS_KEYS = ["crownRadius", "canopyRadius", "radius"]
# _TREE_DIAMETER_KEYS = ["crownDiameter", "canopyDiameter", "diameter"]
# _TREE_HEIGHT_KEYS = ["height", "totalHeight", "crownHeight"]
# _TREE_TRUNK_KEYS = ["trunkHeight", "trunk_height"]


# def resolve_tree_geometry(tree_json: dict) -> Optional[tuple]:
#     """
#     Shared field-lookup used both for the shadow-occluder sphere and for the
#     fuller trunk+crown mesh in the ParaView export. Checks a nested
#     "dimensions" sub-object first (more precise, seen in real exports) and
#     falls back to top-level fields. Returns
#     (pos [3], trunk_height, crown_height, crown_radius) or None if no
#     recognisable position/size fields are found.
#     """
#     pos = None
#     if "position" in tree_json:
#         p = tree_json["position"]
#         pos = np.array([p.get("x", 0), p.get("y", 0), p.get("z", 0)])
#     elif all(k in tree_json for k in ("x", "y", "z")):
#         pos = np.array([tree_json["x"], tree_json["y"], tree_json["z"]])
#     if pos is None:
#         return None

#     dims = tree_json.get("dimensions", {})

#     def find(keys, diameter_keys=()):
#         for k in keys:
#             if k in dims:
#                 return dims[k]
#         for k in keys:
#             if k in tree_json:
#                 return tree_json[k]
#         for k in diameter_keys:
#             if k in dims:
#                 return dims[k] / 2.0
#         for k in diameter_keys:
#             if k in tree_json:
#                 return tree_json[k] / 2.0
#         return None

#     radius = find(_TREE_RADIUS_KEYS, _TREE_DIAMETER_KEYS)
#     height = find(_TREE_HEIGHT_KEYS)
#     trunk = find(_TREE_TRUNK_KEYS)
#     if trunk is None:
#         trunk = 0.0
#     if radius is None and height is None:
#         return None
#     if radius is None:
#         radius = 0.3 * height             # rough canopy-width/height fallback ratio
#     if height is None:
#         height = trunk + 2.5 * radius
#     crown_height = max(height - trunk, 0.1)
#     return pos, trunk, crown_height, radius


# def build_tree_occluder(tree_json: dict) -> Optional[Occluder]:
#     """
#     Approximates a tree canopy as a single bounding sphere for shadow
#     purposes -- a standard simplification in urban solar-access studies
#     (real foliage isn't meshed anyway). Returns None if no recognisable
#     position/size fields are found, rather than guessing at a made-up size.
#     """
#     geo = resolve_tree_geometry(tree_json)
#     if geo is None:
#         return None
#     pos, trunk, crown_height, radius = geo
#     canopy_center = pos + np.array([0.0, 0.0, trunk + crown_height / 2.0])
#     return Occluder(kind="tree", source_id=tree_json.get("id", "tree"),
#                      aabb_min=canopy_center - radius, aabb_max=canopy_center + radius,
#                      sphere_center=canopy_center, sphere_radius=radius)


# def build_occluders_by_building(scene: dict) -> dict:
#     """
#     Returns {building_id: [Occluder, ...]} -- for each building, the list of
#     every OTHER building's triangulated mesh plus every tree's bounding
#     sphere (a building is never tested against its own geometry).
#     """
#     buildings = scene.get("buildings", [])
#     building_occs = {}
#     for b in buildings:
#         occ = build_building_occluder(b)
#         if occ is not None:
#             building_occs[b.get("id", "building")] = occ

#     tree_occs = []
#     skipped = 0
#     for t in scene.get("trees", []):
#         occ = build_tree_occluder(t)
#         if occ is not None:
#             tree_occs.append(occ)
#         else:
#             skipped += 1
#     if skipped:
#         print(f"[shadows] Skipped {skipped} tree(s) in this scene -- no recognised "
#               f"position/size fields. Check _TREE_*_KEYS in solar_thermal_model.py "
#               f"against your tree JSON schema.")

#     by_building = {}
#     for bid in building_occs:
#         others = [o for oid, o in building_occs.items() if oid != bid]
#         by_building[bid] = others + tree_occs
#     return by_building


# def _ray_aabb_hit(origin: np.ndarray, direction: np.ndarray,
#                    box_min: np.ndarray, box_max: np.ndarray) -> bool:
#     """Standard slab test for ray-AABB intersection, t > 0 only."""
#     t_min, t_max = 0.0, np.inf
#     for i in range(3):
#         d = direction[i]
#         if abs(d) < 1e-12:
#             if origin[i] < box_min[i] or origin[i] > box_max[i]:
#                 return False
#             continue
#         t1 = (box_min[i] - origin[i]) / d
#         t2 = (box_max[i] - origin[i]) / d
#         if t1 > t2:
#             t1, t2 = t2, t1
#         t_min = max(t_min, t1)
#         t_max = min(t_max, t2)
#         if t_min > t_max:
#             return False
#     return True


# def _ray_triangle_hit(origin: np.ndarray, direction: np.ndarray,
#                        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
#                        eps: float = 1e-9) -> bool:
#     """Moller-Trumbore ray-triangle intersection; only t > eps counts (so the
#     face's own near-coplanar geometry can never register as a self-hit)."""
#     edge1 = v1 - v0
#     edge2 = v2 - v0
#     h = np.cross(direction, edge2)
#     a = np.dot(edge1, h)
#     if -eps < a < eps:
#         return False
#     f = 1.0 / a
#     s = origin - v0
#     u = f * np.dot(s, h)
#     if u < 0.0 or u > 1.0:
#         return False
#     q = np.cross(s, edge1)
#     v = f * np.dot(direction, q)
#     if v < 0.0 or u + v > 1.0:
#         return False
#     t = f * np.dot(edge2, q)
#     return t > eps


# def _ray_sphere_hit(origin: np.ndarray, direction: np.ndarray,
#                      center: np.ndarray, radius: float, eps: float = 1e-9) -> bool:
#     oc = origin - center
#     b = np.dot(oc, direction)
#     c = np.dot(oc, oc) - radius * radius
#     disc = b * b - c
#     if disc < 0:
#         return False
#     sq = math.sqrt(disc)
#     t1, t2 = -b - sq, -b + sq
#     return t1 > eps or t2 > eps


# def is_shadowed(point: np.ndarray, sun_vec: np.ndarray, occluders: list) -> bool:
#     """
#     True if the ray from `point` toward the sun is blocked by any occluder
#     before reaching it. AABB broad-phase first (cheap); triangle/sphere
#     narrow-phase only for occluders the ray could plausibly hit.
#     """
#     origin = point  # small offset added by the caller along the face normal
#     for occ in occluders:
#         if not _ray_aabb_hit(origin, sun_vec, occ.aabb_min, occ.aabb_max):
#             continue
#         if occ.kind == "tree":
#             if _ray_sphere_hit(origin, sun_vec, occ.sphere_center, occ.sphere_radius):
#                 return True
#         else:
#             for v0, v1, v2 in occ.triangles:
#                 if _ray_triangle_hit(origin, sun_vec, v0, v1, v2):
#                     return True
#     return False


# class TriangleRayTracer:
#     """NumPy packet ray tracing over a median-split triangle BVH.

#     Rays in a packet share the sun direction. Bounds prune whole triangle
#     groups; leaves test at most 16 triangles at once. Both triangle sides
#     occlude, and positive hit distance plus an outward offset avoid self hits.
#     No optional ray-tracing libraries are required.
#     """

#     def __init__(self, triangles, leaf_size=16):
#         self.triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
#         self.nodes = []
#         if not len(self.triangles):
#             return
#         lo = self.triangles.min(axis=1)
#         hi = self.triangles.max(axis=1)
#         centers = (lo + hi) / 2

#         def build(ids):
#             idx = len(self.nodes)
#             bounds = (lo[ids].min(axis=0), hi[ids].max(axis=0))
#             self.nodes.append(None)
#             if len(ids) <= leaf_size:
#                 self.nodes[idx] = (*bounds, ids, -1, -1)
#             else:
#                 axis = np.argmax(np.ptp(centers[ids], axis=0))
#                 mid = len(ids) // 2
#                 ids = ids[np.argpartition(centers[ids, axis], mid)]
#                 left, right = build(ids[:mid]), build(ids[mid:])
#                 self.nodes[idx] = (*bounds, None, left, right)
#             return idx

#         build(np.arange(len(self.triangles)))

#     def blocked(self, origins, direction, chunk_size=8192):
#         origins = np.asarray(origins, dtype=float).reshape(-1, 3)
#         direction = np.asarray(direction, dtype=float)
#         direction = direction / np.linalg.norm(direction)
#         blocked = np.zeros(len(origins), dtype=bool)
#         if not self.nodes:
#             return blocked
#         v0 = self.triangles[:, 0]
#         e1 = self.triangles[:, 1] - v0
#         e2 = self.triangles[:, 2] - v0
#         cross = np.cross(direction, e2)
#         det = np.einsum('ij,ij->i', e1, cross)
#         valid = np.abs(det) > 1e-10
#         inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
#         for start in range(0, len(origins), chunk_size):
#             stack = [(0, np.arange(start, min(start+chunk_size, len(origins))))]
#             while stack:
#                 node, rays = stack.pop()
#                 rays = rays[~blocked[rays]]
#                 if not len(rays):
#                     continue
#                 lo, hi, ids, left, right = self.nodes[node]
#                 points = origins[rays]
#                 near = np.zeros(len(rays))
#                 far = np.full(len(rays), np.inf)
#                 for axis in range(3):
#                     if abs(direction[axis]) < 1e-12:
#                         outside = (points[:, axis] < lo[axis]-1e-9) | (points[:, axis] > hi[axis]+1e-9)
#                         far[outside] = -np.inf
#                     else:
#                         t1 = (lo[axis]-1e-9-points[:, axis]) / direction[axis]
#                         t2 = (hi[axis]+1e-9-points[:, axis]) / direction[axis]
#                         near = np.maximum(near, np.minimum(t1, t2))
#                         far = np.minimum(far, np.maximum(t1, t2))
#                 rays = rays[far >= near]
#                 if not len(rays):
#                     continue
#                 if ids is None:
#                     stack.extend([(left, rays), (right, rays)])
#                     continue
#                 ids = ids[valid[ids]]
#                 if not len(ids):
#                     continue
#                 delta = origins[rays, None, :] - v0[ids]
#                 u = np.einsum('rtj,tj->rt', delta, cross[ids]) * inv[ids]
#                 q = np.cross(delta, e1[ids])
#                 v = np.einsum('rtj,j->rt', q, direction) * inv[ids]
#                 distance = np.einsum('rtj,tj->rt', q, e2[ids]) * inv[ids]
#                 hit = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (distance > 1e-7)
#                 blocked[rays[np.any(hit, axis=1)]] = True
#         return blocked


# def triangle_sunlit_fraction(faces, sun_vec, ray_tracer, samples=3):
#     """Visible area estimate using one centroid or three equal-weight samples.

#     Resolution is the input mesh. Three interior barycentric samples provide
#     fractional direct-beam exposure for triangles crossed by a shadow edge.
#     """
#     if samples not in (1, 3):
#         raise ValueError("shadow_samples must be 1 or 3")
#     normals = np.array([f.normal for f in faces])
#     eligible = np.flatnonzero(normals @ sun_vec > 0) if sun_vec[2] > 0 else np.array([], dtype=int)
#     fractions = np.zeros(len(faces))
#     if not len(eligible):
#         return fractions
#     vertices = np.array([faces[i].vertices for i in eligible])
#     weights = np.array([[1/3, 1/3, 1/3]]) if samples == 1 else np.array([
#         [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
#     origins = np.einsum('sj,tjk->tsk', weights, vertices)
#     origins += normals[eligible, None, :] * _SHADOW_RAY_OFFSET
#     hits = ray_tracer.blocked(origins.reshape(-1, 3), sun_vec).reshape(-1, samples)
#     fractions[eligible] = 1 - hits.mean(axis=1)
#     return fractions


# # ==========================================================================
# # 6. Weather input (plug your real data in here)
# # ==========================================================================

# def load_weather(csv_path: Optional[str] = None, n_hours: int = 24,
#                   t_mean_c: float = 24.0, t_amp_c: float = 6.0,
#                   wind_mean_ms: float = 3.0) -> pd.DataFrame:
#     """
#     If csv_path is given, expects columns: hour, T_amb_C, wind_speed_ms
#     (optionally GHI_Wm2, DNI_Wm2, DHI_Wm2 to override the clear-sky model
#     with your own measured/forecast radiation).

#     If csv_path is None, a synthetic diurnal profile is generated as a
#     PLACEHOLDER only (sinusoidal T_amb, constant wind) so the pipeline is
#     runnable end-to-end -- swap this out for your real weather data.
#     """
#     if csv_path:
#         df = pd.read_csv(csv_path)
#         return df

#     hours = np.arange(n_hours)
#     # min T_amb near 05:00, max near 15:00 -- simple diurnal sine, placeholder only
#     t_amb = t_mean_c - t_amp_c * np.cos(2 * np.pi * (hours - 15) / 24.0)
#     wind = np.full(n_hours, wind_mean_ms)
#     return pd.DataFrame({"hour": hours, "T_amb_C": t_amb, "wind_speed_ms": wind})


# # --------------------------------------------------------------------------
# # Real weather-data-folder loader: OpenFOAM-style time-table export
# # (weatherData/MMDD/{Tambient, windSpeed, IDN, Idif, sunPosVector, ...})
# #
# # Each file looks like:
# #   // comment
# #   (
# #   ( <time_s> <scalar> )                    <- scalar field
# #   ( <time_s> ( <vx> <vy> <vz> ) )          <- vector field
# #   ...
# #   )
# # --------------------------------------------------------------------------

# _VEC_RE = re.compile(
#     r'\(\s*(-?[0-9.eE+-]+)\s+\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)\s*\)'
# )
# _SCALAR_RE = re.compile(r'\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)')


# def parse_foam_field(path: str) -> tuple:
#     """Parses one OpenFOAM-style time-table file. Returns (times_s, values),
#     values is a 1D array for scalar fields or an (N,3) array for vector fields."""
#     with open(path) as fh:
#         content = re.sub(r'//.*', '', fh.read())   # strip comments

#     vec = _VEC_RE.findall(content)
#     if vec:
#         times = np.array([float(m[0]) for m in vec])
#         values = np.array([[float(m[1]), float(m[2]), float(m[3])] for m in vec])
#         return times, values

#     sca = _SCALAR_RE.findall(content)
#     times = np.array([float(m[0]) for m in sca])
#     values = np.array([float(m[1]) for m in sca])
#     return times, values


# def resolve_weather_folder(root: str, month: int, day: int) -> str:
#     """weatherData/<root> + MMDD (e.g. month=12, day=11 -> '1211')."""
#     mmdd = f"{month:02d}{day:02d}"
#     candidate = os.path.join(root, mmdd)
#     if not os.path.isdir(candidate):
#         raise FileNotFoundError(f"No such weather folder: {candidate}")
#     return candidate


# def load_weather_folder(weather_dir: str) -> dict:
#     """
#     Reads every field file in a `weatherData/MMDD/` folder and returns a
#     dict of hourly arrays ready for `run_analysis_from_folder`:
#         hour, T_amb_C, wind_speed_ms, IDN_Wm2, Idif_Wm2,
#         sun_vec (N,3) [East, North, Up unit vector, straight from the file],
#         altitude_deg  [from solarElevation if present, else asin(sun_vec_z)]
#     Any other fields present (cloudCover, relativeHumidity, atmosphericPressure,
#     wetBulbTemperature, wambient, wPonds, windVelocity, ...) are passed through
#     unchanged under their own file name, in case you want them for other
#     calculations (e.g. evaporative cooling from the ponds).
#     """
#     files = {f: os.path.join(weather_dir, f) for f in os.listdir(weather_dir)
#               if os.path.isfile(os.path.join(weather_dir, f))}

#     parsed = {name: parse_foam_field(path) for name, path in files.items()}
#     # all fields share the same time base -- use whichever field is present first
#     times_s = next(iter(parsed.values()))[0]
#     hours = times_s / 3600.0

#     def get(name, default=None):
#         return parsed[name][1] if name in parsed else default

#     t_amb_k = get("Tambient")
#     wx = dict(
#         hour=hours,
#         T_amb_C=(t_amb_k - 273.15) if t_amb_k is not None else None,
#         wind_speed_ms=get("windSpeed"),
#         IDN_Wm2=get("IDN", np.zeros_like(hours)),
#         Idif_Wm2=get("Idif", np.zeros_like(hours)),
#     )

#     if "sunPosVector" in parsed:
#         wx["sun_vec"] = parsed["sunPosVector"][1]                 # (N,3) already East/North/Up
#         lengths = np.linalg.norm(wx["sun_vec"], axis=1)
#         if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
#             raise ValueError("sunPosVector must contain finite, nonzero vectors")
#         wx["sun_vec"] = wx["sun_vec"] / lengths[:, None]
#         wx["altitude_deg"] = np.degrees(np.arcsin(np.clip(wx["sun_vec"][:, 2], -1, 1)))
#     elif "solarElevation" in parsed and "solarAzimuth" in parsed:
#         alt = parsed["solarElevation"][1]
#         az = parsed["solarAzimuth"][1]
#         wx["altitude_deg"] = alt
#         wx["sun_vec"] = np.array([sun_unit_vector(a, z) for a, z in zip(alt, az)])
#     else:
#         raise ValueError("Weather folder has neither sunPosVector nor "
#                           "solarElevation+solarAzimuth -- cannot get sun direction.")

#     if wx["T_amb_C"] is None:
#         raise ValueError("Weather folder is missing a 'Tambient' file.")
#     if wx["wind_speed_ms"] is None:
#         raise ValueError("Weather folder is missing a 'windSpeed' file.")

#     # keep any extra fields around too, unmodified
#     for name, (t, v) in parsed.items():
#         if name not in ("Tambient", "windSpeed", "IDN", "Idif", "sunPosVector",
#                          "solarElevation", "solarAzimuth"):
#             wx[name] = v

#     return wx


# # ==========================================================================
# # 7. Shared per-hour physics loop + per-building RC response
# #    (used by BOTH the clear-sky/solar-formula path and the real
# #    weather-data-folder path below)
# # ==========================================================================

# _SHADOW_RAY_OFFSET = 1e-3   # m, nudge the ray origin off the face surface

# def _simulate(faces: list, capacitance_by_building: dict, records: list,
#               ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#               occluders_by_building: Optional[dict] = None,
#               t_initial_c: Optional[float] = None,
#               ray_tracer=None, shadow_samples: int = 3,
#               wall_settings: Optional[WallSettings] = None,
#               surface_group: str = "building",
#               wetness: float = 0.0, back_follows_ambient: bool = False,
#               workers: int = 1) -> tuple:
#     """Per-cell solar/thermal balance plus the separate legacy bulk RC estimate.

#     OBJ cells use packet rays; JSON faces retain their centroid visibility test.
#     Output temperatures are states at the labelled timestamps. The first
#     output is the initial condition; every later record supplies the constant
#     boundary condition over the interval ending at that timestamp. This keeps
#     each output's shadow/solar fields aligned with its thermal state.
#     """
#     excluded = {"ceiling"} | (set() if INCLUDE_FLOOR else {"floor"})
#     exposed = [f for f in faces if f.surface_type not in excluded]
#     if not exposed or not records:
#         raise ValueError("Simulation requires exposed faces and weather records")
#     bids = list(capacitance_by_building)
#     bindex = {bid: i for i, bid in enumerate(bids)}
#     indices = np.array([bindex[f.building_id] for f in exposed])
#     normals = np.array([f.normal for f in exposed])
#     areas = np.array([f.area for f in exposed])
#     albedo = np.array([f.albedo for f in exposed])
#     emissivity = np.array([f.emissivity for f in exposed])
#     view_sky = (1 + normals[:, 2]) / 2
#     static = dict(building_id=[f.building_id for f in exposed],
#                   face_id=[f.face_id for f in exposed],
#                   parent_face_id=[f.parent_face_id or f.face_id for f in exposed],
#                   surface_type=[f.surface_type for f in exposed],
#                   surface_group=surface_group, area_m2=areas,
#                   tilt_deg=[f.tilt_deg for f in exposed])
#     frames, bulk_rows = [], []
#     bulk_T = np.full(len(bids), t_initial_c if t_initial_c is not None else records[0]["T_amb_C"], dtype=float)
#     caps = np.array([capacitance_by_building[b] for b in bids])
#     wall_settings = wall_settings or WallSettings()
#     wall_initial_c = (t_initial_c if t_initial_c is not None
#                       else wall_settings.indoor_temperature_c)
#     slab = SlabConduction([f.thermal_conductivity for f in exposed],
#                           [f.density for f in exposed], [f.specific_heat for f in exposed],
#                           emissivity, wall_initial_c, wall_settings)
#     # Sunlit fractions depend only on geometry and sun position, not on the
#     # thermal state, so all hours are ray-traced up front (optionally threaded).
#     shadow_fractions = {}
#     if ray_tracer is not None:
#         sun_vecs = [np.asarray(r["sun_vec"], float) / np.linalg.norm(r["sun_vec"]) for r in records]
#         todo = [i for i, r in enumerate(records) if r["altitude_deg"] > 0]
#         verts_arr = np.array([f.vertices for f in exposed], dtype=float)
#         print(f"[shadows] {surface_group}: tracing {len(exposed)} cells x {len(todo)} hours "
#               f"(threads={max(1, int(workers))})", flush=True)

#         def _shadow_job(i):
#             return i, sunlit_fractions_from_arrays(normals, verts_arr, sun_vecs[i],
#                                                    ray_tracer, shadow_samples)
#         if workers > 1 and len(todo) > 1:
#             with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as pool:
#                 shadow_fractions = dict(pool.map(_shadow_job, todo))
#         else:
#             shadow_fractions = dict(_shadow_job(i) for i in todo)
#     for ri, rec in enumerate(records):
#         t_amb = rec["T_amb_C"]
#         sun = np.asarray(rec["sun_vec"], dtype=float)
#         sun = sun / np.linalg.norm(sun)
#         alt = rec["altitude_deg"]
#         h_c = convective_h(rec["wind_speed_ms"])
#         cos_theta = normals @ sun
#         self_lit = (alt > 0) & (cos_theta > 0)
#         fraction = self_lit.astype(float)
#         if ray_tracer is not None and alt > 0:
#             fraction = shadow_fractions[ri]
#         elif occluders_by_building:
#             for i in np.flatnonzero(self_lit):
#                 f = exposed[i]
#                 fraction[i] = not is_shadowed(f.center + f.normal*_SHADOW_RAY_OFFSET,
#                                               sun, occluders_by_building.get(f.building_id, []))
#         shadow_fraction = np.where(self_lit, 1-fraction, 0)
#         direct = max(rec["i_dn"], 0) * np.maximum(cos_theta, 0) * fraction
#         diffuse = max(rec["i_dh"], 0) * view_sky
#         ghi = max(rec["i_dn"], 0) * max(sun[2], 0) + max(rec["i_dh"], 0)
#         reflected = ground_albedo * ghi * (1-view_sky)
#         total = direct + diffuse + reflected
#         solair = sol_air_temperature(t_amb, total, h_c, albedo, view_sky, emissivity)
#         q_solar = (1-albedo)*total
#         ta_k = t_amb + 273.15
#         # Tangent linearization at ambient, frozen for this weather interval:
#         # q_lw(Ts) ~= q_lw(Ta) + h_rad*(Ts-Ta). This permits the exact
#         # matrix-exponential wall update; full Ts^4 radiation is nonlinear.
#         q_lw_ambient = emissivity*STEFAN_BOLTZMANN*view_sky*(ta_k**4-sky_temperature(ta_k)**4)
#         h_rad = 4*emissivity*STEFAN_BOLTZMANN*ta_k**3
#         # Evaporation / transpiration, same tangent linearisation at ambient:
#         # q_evap(Ts) ~= e0 + e_slope*(Ts-Ta); coef = wetness*L*h_c/cp_air (Lewis).
#         e0 = e_slope = 0.0
#         if wetness > 0:
#             p_atm = rec.get("p_atm", 101325.0)
#             w_amb = rec.get("w_amb")
#             if w_amb is None:
#                 w_amb = _humidity_ratio(t_amb, rec.get("rh", 0.5), p_atm)
#             ws, dws = _saturation_humidity_ratio(t_amb, p_atm)
#             coef = wetness*LATENT_HEAT_VAPORIZATION*h_c/CP_AIR
#             e0, e_slope = coef*(ws-w_amb), coef*dws
#         external_h = h_c+h_rad+e_slope
#         external_source = q_solar-q_lw_ambient-e0 + external_h*t_amb
#         if back_follows_ambient:
#             slab.set_back(h_c, t_amb)

#         if ri > 0:
#             dt = (rec["hour"] - records[ri-1]["hour"])*3600
#             if dt <= 0:
#                 raise ValueError("Weather times must be strictly increasing")
#             slab.advance(dt, external_h, external_source)

#         ts = slab.temperatures[:,0].copy()
#         q_conv = h_c*(ts-t_amb)
#         q_evap = e0 + e_slope*(ts-t_amb)
#         q_lw = q_lw_ambient + h_rad*(ts-t_amb)
#         wall_fields = slab.diagnostics(external_h, external_source)
#         wall_fields["stored_heat_J"] = wall_fields["stored_heat_Jm2"]*areas
#         wall_fields["wall_capacity_JK"] = wall_fields["wall_capacity_Jm2K"]*areas
#         wall_fields["slab_capacity_JK"] = wall_fields["slab_capacity_Jm2K"]*areas
#         frames.append(pd.DataFrame(dict(static, hour=rec["hour"], solar_altitude_deg=alt,
#             self_lit=self_lit, shadowed=shadow_fraction > 0, sunlit=fraction > 0,
#             sunlit_fraction=fraction, shadow_fraction=shadow_fraction, cos_theta=cos_theta,
#             I_direct_Wm2=direct, I_diffuse_Wm2=diffuse, I_reflected_Wm2=reflected,
#             I_total_Wm2=total, h_conv_Wm2K=h_c, T_amb_C=t_amb,
#             T_solair_C=solair, T_surface_C=ts, q_solar_absorbed_Wm2=q_solar,
#             q_conv_out_Wm2=q_conv, q_longwave_out_Wm2=q_lw, q_evap_out_Wm2=q_evap,
#             wall_thickness_m=wall_settings.thickness_m,
#             slab_thickness_m=wall_settings.thickness_m,
#             T_indoor_air_C=slab.back_temperature,
#             T_back_boundary_C=slab.back_temperature,
#             h_indoor_Wm2K=slab.back_h,
#             h_back_Wm2K=slab.back_h, **wall_fields)))
#         ah = np.bincount(indices, weights=areas*h_c, minlength=len(bids))
#         aht = np.bincount(indices, weights=areas*h_c*solair, minlength=len(bids))
#         teq = np.divide(aht, ah, out=np.full(len(bids), t_amb, dtype=float), where=ah > 0)
#         tau = np.divide(caps, ah, out=np.full(len(bids), np.inf), where=ah > 0)
#         for bi, bid in enumerate(bids):
#             bulk_rows.append(dict(building_id=bid, hour=rec["hour"], T_amb_C=t_amb,
#                 T_eq_C=teq[bi], h_eff_WK=ah[bi], tau_hr=tau[bi]/3600,
#                 T_building_C=bulk_T[bi]))
#         if ri+1 < len(records):
#             dt_bulk = (records[ri+1]["hour"] - rec["hour"])*3600
#             if dt_bulk <= 0:
#                 raise ValueError("Weather times must be strictly increasing")
#             decay = np.exp(-np.divide(
#                 dt_bulk, tau, out=np.full_like(tau, np.inf), where=tau > 0))
#             bulk_T = teq + (bulk_T-teq)*decay
#         if ray_tracer is not None:
#             print(f"[simulation] hour {rec['hour']:g}: {len(exposed)} surface cells", flush=True)
#     return pd.concat(frames, ignore_index=True), pd.DataFrame(bulk_rows)


# def compute_capacitances(scene: dict, faces: list) -> dict:
#     """{building_id: C [J/K]} for every building in the scene, C = rho*V*cp."""
#     caps = {}
#     faces_by_bid = {}
#     for f in faces:
#         faces_by_bid.setdefault(f.building_id, f)
#     for bid, f0 in faces_by_bid.items():
#         vol = building_volume(scene, bid)
#         caps[bid] = f0.density * vol * f0.specific_heat
#     return caps


# def run_analysis(json_path: str, day: date, weather: pd.DataFrame,
#                   utc_offset_hr: float = 0.0, north_offset_deg: float = 0.0,
#                   site_altitude_km: float = 0.0,
#                   consider_shadows: bool = True,
#                   t_initial_c: Optional[float] = None,
#                   wall_settings: Optional[WallSettings] = None) -> tuple:
#     """
#     CLEAR-SKY / SOLAR-FORMULA path: use this when you do NOT have a real
#     weather-data export, only a date + location + basic T_amb/wind. Solar
#     position and irradiance are both modelled analytically (see functions
#     `solar_position` and `hottel_clear_sky` above).
#     Returns (face_hourly_df, building_hourly_df, faces).
#     """
#     scene = _scene_from(json_path)
#     faces = extract_faces(scene)
#     lat, lon = site_location(scene)
#     capacitances = compute_capacitances(scene, faces)
#     occluders = build_occluders_by_building(scene) if consider_shadows else None
#     day_of_year = day.timetuple().tm_yday

#     records = []
#     for _, row in weather.iterrows():
#         hr = row["hour"]
#         alt, az = solar_position(lat, lon, day_of_year, hr, utc_offset_hr)
#         sun_vec = sun_unit_vector(alt, az, north_offset_deg)
#         if "GHI_Wm2" in row and not pd.isna(row.get("DNI_Wm2", np.nan)):
#             i_dn, i_dh = row["DNI_Wm2"], row["DHI_Wm2"]
#         else:
#             i_dn, i_dh = hottel_clear_sky(alt, site_altitude_km)
#         records.append(dict(hour=hr, T_amb_C=row["T_amb_C"], wind_speed_ms=row["wind_speed_ms"],
#                              sun_vec=sun_vec, altitude_deg=alt, i_dn=i_dn, i_dh=i_dh))

#     face_df, building_df = _simulate(faces, capacitances, records,
#                                       occluders_by_building=occluders, t_initial_c=t_initial_c,
#                                       wall_settings=wall_settings)
#     return face_df, building_df, faces


# def run_analysis_from_folder(json_path: str, weather_dir: str,
#                               ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                               consider_shadows: bool = True,
#                               t_initial_c: Optional[float] = None,
#                   wall_settings: Optional[WallSettings] = None) -> tuple:
#     """
#     REAL WEATHER-DATA path (preferred whenever you have it): reads Tambient,
#     windSpeed, IDN, Idif and sunPosVector straight from an exported
#     `weatherData/MMDD/` folder (see `load_weather_folder`). No solar-position
#     formula and no clear-sky model are used here at all -- the sun vector and
#     irradiance are exactly what your own pipeline computed, in the same
#     (x=East, y=North, z=Up) axes as the building JSON's face normals, so
#     there is no coordinate-convention guessing.
#     Returns (face_hourly_df, building_hourly_df, faces).
#     """
#     scene = _scene_from(json_path)
#     faces = extract_faces(scene)
#     capacitances = compute_capacitances(scene, faces)
#     occluders = build_occluders_by_building(scene) if consider_shadows else None
#     wx = load_weather_folder(weather_dir)

#     records = []
#     n = len(wx["hour"])
#     for i in range(n):
#         records.append(dict(
#             hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
#             sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
#             i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i],
#         ))

#     face_df, building_df = _simulate(faces, capacitances, records, ground_albedo=ground_albedo,
#                                       occluders_by_building=occluders, t_initial_c=t_initial_c,
#                                       wall_settings=wall_settings)
#     return face_df, building_df, faces


# # ==========================================================================
# # 8. ParaView export -- writes the same results as a time series of legacy
# #    VTK PolyData files (plus a .pvd collection), so you can open one file
# #    in ParaView, scrub an "hour of day" timeline, and see the building
# #    (colored by sol-air temperature, shadowed/sunlit status, or irradiance)
# #    together with the trees, with no plugins and no `pip install vtk` --
# #    this writes plain-text legacy VTK, which ParaView reads natively.
# # ==========================================================================

# _SURFACE_TYPE_ID = {"wall": 0, "roof": 1, "floor": 2, "ceiling": 3, "tree": 5, "ground": 6,
#                     "pathway": 7, "lawn": 8, "pond": 9}
# TREE_TRUNK_TYPE_ID = 4
# TREE_CROWN_TYPE_ID = 5


# def _ring_polygon(cx: float, cy: float, z: float, radius: float, n: int) -> list:
#     return [(cx + radius * math.cos(2 * math.pi * i / n),
#               cy + radius * math.sin(2 * math.pi * i / n), z) for i in range(n)]


# def _cylinder_polys(cx: float, cy: float, z0: float, z1: float, radius: float, n: int = 8) -> list:
#     """Side quads + a bottom cap (top is hidden inside the tree crown)."""
#     ring0 = _ring_polygon(cx, cy, z0, radius, n)
#     ring1 = _ring_polygon(cx, cy, z1, radius, n)
#     polys = []
#     for i in range(n):
#         j = (i + 1) % n
#         polys.append([ring0[i], ring0[j], ring1[j], ring1[i]])
#     polys.append(list(reversed(ring0)))   # bottom cap, wound outward-facing-down
#     return polys


# def _cone_polys(cx: float, cy: float, base_z: float, apex_z: float, radius: float, n: int = 8) -> list:
#     """Side triangles (apex at top) + a base cap."""
#     ring = _ring_polygon(cx, cy, base_z, radius, n)
#     apex = (cx, cy, apex_z)
#     polys = [[ring[i], ring[(i + 1) % n], apex] for i in range(n)]
#     polys.append(list(ring))              # base cap
#     return polys


# def _tree_mesh(tree_json: dict, trunk_radius_ratio: float = 0.15) -> tuple:
#     """Returns (polygons, surface_type_ids) for one tree's trunk + crown, or
#     (None, None) if the tree's geometry couldn't be resolved (see
#     resolve_tree_geometry)."""
#     geo = resolve_tree_geometry(tree_json)
#     if geo is None:
#         return None, None
#     pos, trunk_h, crown_h, crown_r = geo
#     trunk_r = max(crown_r * trunk_radius_ratio, 0.1)
#     trunk_polys = _cylinder_polys(pos[0], pos[1], pos[2], pos[2] + trunk_h, trunk_r)
#     crown_polys = _cone_polys(pos[0], pos[1], pos[2] + trunk_h, pos[2] + trunk_h + crown_h, crown_r)
#     polys = trunk_polys + crown_polys
#     type_ids = [TREE_TRUNK_TYPE_ID] * len(trunk_polys) + [TREE_CROWN_TYPE_ID] * len(crown_polys)
#     return polys, type_ids


# def _write_vtk_polydata_slow(path: str, polygons: list, cell_scalars: dict) -> None:
#     """
#     polygons: list of polygons, each a list of (x,y,z) tuples (a planar loop).
#         Points are NOT shared between polygons -- every polygon gets its own
#         copy of its corner points. That is deliberate: colouring is by CELL
#         data (one value per face), and un-shared points guarantee ParaView
#         renders each face as a single flat colour instead of interpolating
#         across a shared edge with a differently-valued neighbour.
#     cell_scalars: {name: (values, "float"|"int")}, one value per polygon,
#         same length and order as `polygons`.
#     """
#     points, polygon_indices = [], []
#     idx = 0
#     for poly in polygons:
#         n = len(poly)
#         polygon_indices.append(list(range(idx, idx + n)))
#         points.extend(poly)
#         idx += n

#     with open(path, "w") as f:
#         f.write("# vtk DataFile Version 3.0\n")
#         f.write("solar_thermal_model export\n")
#         f.write("ASCII\n")
#         f.write("DATASET POLYDATA\n")
#         f.write(f"POINTS {len(points)} float\n")
#         for p in points:
#             f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
#         total_size = sum(len(idxs) + 1 for idxs in polygon_indices)
#         f.write(f"POLYGONS {len(polygon_indices)} {total_size}\n")
#         for idxs in polygon_indices:
#             f.write(f"{len(idxs)} " + " ".join(str(i) for i in idxs) + "\n")
#         f.write(f"CELL_DATA {len(polygons)}\n")
#         for name, (values, dtype) in cell_scalars.items():
#             f.write(f"SCALARS {name} {dtype} 1\n")
#             f.write("LOOKUP_TABLE default\n")
#             fmt = "{}\n" if dtype == "int" else "{:.4f}\n"
#             for v in values:
#                 f.write(fmt.format(v))


# def export_vtk_series(json_path: str, weather_dir: str, out_dir: str,
#                        consider_shadows: bool = True,
#                        wall_settings: Optional[WallSettings] = None,
#                        t_initial_c: Optional[float] = None) -> str:
#     """
#     Runs the real-weather-data analysis (see `run_analysis_from_folder`) and
#     writes one legacy-VTK PolyData file per hour plus a `scene.pvd` ParaView
#     collection tying them together as a time series. Open `scene.pvd` in
#     ParaView, colour by any of the per-face arrays below, and scrub the time
#     slider to watch temperature and shadowing move through the day together.

#     Per-face (cell) data written on every building face:
#         T_C            transient outer surface temperature, deg C
#         I_total_Wm2    total incident solar irradiance, W/m^2
#         shadowed       1 if ray-blocked by another building/tree this hour
#         sunlit         1 if actually receiving direct beam this hour
#                         (self_lit AND NOT shadowed)
#         surface_type_id  0=wall 1=roof 2=floor 3=ceiling 4=tree trunk 5=tree crown
#         building_index  which building/floor this face belongs to (-1 for trees)
#         is_building     1 for building faces, 0 for tree geometry

#     Floor/ceiling faces are excluded from the thermal simulation itself (see
#     the module docstring) but ARE still drawn here, with T_C set to that
#     hour's ambient air temperature, so the building reads as a solid, complete
#     shape rather than having visible holes -- filter on surface_type_id in
#     ParaView if you want only the faces that were actually simulated.
#     Trees are static (their geometry doesn't change hour to hour) but are
#     re-written into every frame file so each one opens as a complete scene on
#     its own; T_C for tree geometry is likewise just that hour's ambient
#     temperature (canopy temperature isn't modelled).
#     """
#     scene = _scene_from(json_path)
#     faces = extract_faces(scene)
#     face_df, building_df, _ = run_analysis_from_folder(
#         json_path, weather_dir, consider_shadows=consider_shadows,
#         wall_settings=wall_settings, t_initial_c=t_initial_c)
#     os.makedirs(out_dir, exist_ok=True)

#     tree_polys, tree_type_ids = [], []
#     n_tree_geo_ok, n_tree_geo_skipped = 0, 0
#     for t in scene.get("trees", []):
#         polys, type_ids = _tree_mesh(t)
#         if polys is None:
#             n_tree_geo_skipped += 1
#             continue
#         n_tree_geo_ok += 1
#         tree_polys.extend(polys)
#         tree_type_ids.extend(type_ids)
#     if n_tree_geo_skipped:
#         print(f"[vtk export] Skipped {n_tree_geo_skipped} tree(s) with unrecognised geometry fields.")

#     building_ids_sorted = sorted({f.building_id for f in faces})
#     building_index_map = {bid: i for i, bid in enumerate(building_ids_sorted)}

#     face_lookup = {}
#     for _, r in face_df.iterrows():
#         face_lookup[(r["building_id"], r["face_id"], r["hour"])] = r

#     hours = sorted(face_df["hour"].unique())
#     pvd_entries = []
#     for h in hours:
#         amb_rows = building_df[building_df["hour"] == h]
#         amb_c = float(amb_rows["T_amb_C"].iloc[0]) if len(amb_rows) else 0.0

#         polys, T_C, shadowed, sunlit, I_tot, surf_id, bidx, is_bldg = [], [], [], [], [], [], [], []
#         for f in faces:
#             row = face_lookup.get((f.building_id, f.face_id, h))
#             polys.append([tuple(v) for v in f.vertices])
#             if row is None:                          # floor/ceiling: not simulated
#                 T_C.append(amb_c); shadowed.append(0); sunlit.append(0); I_tot.append(0.0)
#             else:
#                 T_C.append(float(row["T_surface_C"])); shadowed.append(int(bool(row["shadowed"])))
#                 sunlit.append(int(bool(row["sunlit"]))); I_tot.append(float(row["I_total_Wm2"]))
#             surf_id.append(_SURFACE_TYPE_ID.get(f.surface_type, -1))
#             bidx.append(building_index_map[f.building_id])
#             is_bldg.append(1)

#         polys.extend(tree_polys)
#         T_C.extend([amb_c] * len(tree_polys))
#         shadowed.extend([0] * len(tree_polys))
#         sunlit.extend([0] * len(tree_polys))
#         I_tot.extend([0.0] * len(tree_polys))
#         surf_id.extend(tree_type_ids)
#         bidx.extend([-1] * len(tree_polys))
#         is_bldg.extend([0] * len(tree_polys))

#         fname = f"frame_{int(round(h * 100)):04d}.vtk"   # e.g. hour 13.5 -> frame_1350.vtk
#         _write_vtk_polydata(os.path.join(out_dir, fname), polys, {
#             "T_C": (T_C, "float"),
#             "I_total_Wm2": (I_tot, "float"),
#             "shadowed": (shadowed, "int"),
#             "sunlit": (sunlit, "int"),
#             "surface_type_id": (surf_id, "int"),
#             "building_index": (bidx, "int"),
#             "is_building": (is_bldg, "int"),
#         })
#         pvd_entries.append((float(h), fname))

#     pvd_path = os.path.join(out_dir, "scene.pvd")
#     with open(pvd_path, "w") as f:
#         f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
#         for t, fname in pvd_entries:
#             f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
#         f.write('</Collection>\n</VTKFile>\n')

#     print(f"[vtk export] Wrote {len(pvd_entries)} frame(s) + scene.pvd to {out_dir} "
#           f"({n_tree_geo_ok} tree(s) meshed).")
#     return pvd_path


# # ==========================================================================
# # 9. OBJ pipeline: retain input triangles for partial-wall shadows.
# #    Coplanar grouping is used only for parent IDs and bulk-volume estimates.
# #    OBJ does not carry thermal materials: defaults below assume concrete.
# #    Spatial resolution is limited by the input mesh; use a finer OBJ for
# #    sharper shadow edges. Trees are opaque meshes, without leaf transmission.
# # ==========================================================================

# DEFAULT_MATERIAL = dict(thermal_conductivity=2.3, density=2400, specific_heat=880)
# DEFAULT_ALBEDO_WALL = 0.3
# DEFAULT_ALBEDO_ROOF = 0.4
# ROOF_NORMAL_Z_THRESHOLD = 0.7   # |normal.z| above this -> "roof", else "wall"


# def _obj_vertex_index(token: str, n_verts_so_far: int) -> int:
#     """
#     Parses one OBJ face-vertex token in any of the four standard forms
#     (`v`, `v/vt`, `v/vt/vn`, `v//vn`) and returns a 0-based vertex index.
#     Handles negative (relative-to-current-end) indices too, per the OBJ
#     spec.
#     """
#     idx = int(token.split("/")[0])
#     return (n_verts_so_far + idx) if idx < 0 else (idx - 1)


# def load_obj_mesh(path: str) -> tuple:
#     """
#     OBJ parser for the buildings_refined.obj / trees.obj this pipeline
#     consumes. json_to_obj.py's own write_obj() only ever emits plain
#     `v x y z` / `f i j k` lines, but real pipelines often pass the file
#     through something else afterwards (normal generation, mesh repair,
#     etc.) that adds `vn`/`vt` references -- seen in practice as
#     `f 1//1 2//1 3//1` style faces. This is intentionally more permissive
#     than that one exact format: it accepts all four standard face-vertex
#     forms (`v`, `v/vt`, `v/vt/vn`, `v//vn`), skips `vn`/`vt` lines outright
#     (this pipeline recomputes every normal from triangle geometry anyway,
#     never reads one from the file), and fan-triangulates any face with more
#     than 3 vertices rather than assuming triangles-only.
#     Returns (vertices [N,3] float array, triangles [list of (i,j,k) 0-based
#     int tuples]).
#     """
#     verts, tris = [], []
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("v "):
#                 p = line.split()
#                 verts.append((float(p[1]), float(p[2]), float(p[3])))
#             elif line.startswith("f "):
#                 tokens = line.split()[1:]
#                 idxs = [_obj_vertex_index(tok, len(verts)) for tok in tokens]
#                 for t in range(1, len(idxs) - 1):     # fan triangulation (no-op for triangles)
#                     tris.append((idxs[0], idxs[t], idxs[t + 1]))
#     return np.array(verts, dtype=float), tris


# def _triangle_normal_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple:
#     n = np.cross(v1 - v0, v2 - v0)
#     area = float(np.linalg.norm(n)) / 2.0
#     if area < 1e-12:
#         return None, 0.0
#     return n / (2.0 * area), area


# def group_coplanar(vertices: np.ndarray, triangles: list,
#                     normal_decimals: int = 3, offset_decimals: int = 2) -> dict:
#     """
#     Merges triangles sharing the same (rounded outward normal, rounded plane
#     offset) into "logical faces". Exact for flat, regularly-subdivided quads
#     (this OBJ format's building geometry): every wall segment along one
#     straight facade -- including across every stacked storey of a
#     multi-floor tower, since they're all perfectly coplanar -- collapses
#     back into ONE logical wall face.
#     Returns {plane_key: {"triangles": [(i,j,k), ...], "normal", "area",
#                           "center", "z_min", "z_max"}}.
#     """
#     groups = {}
#     for (i, j, k) in triangles:
#         v0, v1, v2 = vertices[i], vertices[j], vertices[k]
#         n, area = _triangle_normal_area(v0, v1, v2)
#         if n is None:
#             continue
#         d = float(np.dot(n, v0))
#         key = (round(float(n[0]), normal_decimals), round(float(n[1]), normal_decimals),
#                round(float(n[2]), normal_decimals), round(d, offset_decimals))
#         g = groups.setdefault(key, dict(triangles=[], normal_sum=np.zeros(3), area=0.0,
#                                           centroid_sum=np.zeros(3), z_min=np.inf, z_max=-np.inf))
#         centroid = (v0 + v1 + v2) / 3.0
#         g["triangles"].append((i, j, k))
#         g["normal_sum"] += n * area
#         g["area"] += area
#         g["centroid_sum"] += centroid * area
#         g["z_min"] = min(g["z_min"], float(v0[2]), float(v1[2]), float(v2[2]))
#         g["z_max"] = max(g["z_max"], float(v0[2]), float(v1[2]), float(v2[2]))

#     out = {}
#     for key, g in groups.items():
#         if g["area"] <= 0:
#             continue
#         normal = g["normal_sum"] / (np.linalg.norm(g["normal_sum"]) + 1e-12)
#         center = g["centroid_sum"] / g["area"]
#         out[key] = dict(triangles=g["triangles"], normal=normal, area=g["area"],
#                           center=center, z_min=g["z_min"], z_max=g["z_max"])
#     return out


# def _mesh_to_occluder(vertices: np.ndarray, triangles: list, source_id: str) -> Optional[Occluder]:
#     if not triangles:
#         return None
#     tri_pts = [(vertices[i], vertices[j], vertices[k]) for (i, j, k) in triangles]
#     idx = sorted({v for tri in triangles for v in tri})
#     pts = vertices[idx]
#     return Occluder(kind="building", source_id=source_id,
#                      aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tri_pts)


# def _position_key(p, decimals: int = 4) -> tuple:
#     return (round(float(p[0]), decimals), round(float(p[1]), decimals), round(float(p[2]), decimals))


# def _obj_components_by_shared_position(vertices: np.ndarray, triangles: list,
#                                          decimals: int = 4) -> list:
#     """
#     Union-find over vertex POSITIONS, not indices -- this is the piece that
#     was missing before. json_to_obj.py does not share vertex indices across
#     different original JSON faces, even when they are geometrically
#     touching (e.g. two wall quads meeting at a shared vertical edge), so an
#     index-based union-find (see _obj_connected_components, used for trees)
#     cannot tell "two touching walls of the same building" apart from "two
#     unrelated triangles that happen to have unrelated index numbers" -- it
#     under-merges everything into one-triangle-ish fragments. Position-based
#     matching (round each vertex to `decimals` and treat equal rounded
#     positions as the same point) is what actually reconstructs "which
#     triangles are physically part of the same connected building".
#     Returns a list of triangle-index lists, one per physically distinct
#     building.
#     """
#     pos_to_id: dict = {}
#     vertex_logical_id = np.empty(len(vertices), dtype=np.int64)
#     for v_idx in range(len(vertices)):
#         key = _position_key(vertices[v_idx], decimals)
#         lid = pos_to_id.get(key)
#         if lid is None:
#             lid = len(pos_to_id)
#             pos_to_id[key] = lid
#         vertex_logical_id[v_idx] = lid

#     parent = list(range(len(pos_to_id)))

#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra != rb:
#             parent[ra] = rb

#     for (i, j, k) in triangles:
#         li, lj, lk = int(vertex_logical_id[i]), int(vertex_logical_id[j]), int(vertex_logical_id[k])
#         union(li, lj)
#         union(lj, lk)

#     comp_triangles: dict = {}
#     for t_idx, (i, j, k) in enumerate(triangles):
#         root = find(int(vertex_logical_id[i]))
#         comp_triangles.setdefault(root, []).append(t_idx)
#     return list(comp_triangles.values())


# def _obj_connected_components(vertices: np.ndarray, triangles: list) -> list:
#     """
#     Union-find over shared vertex INDICES. This groups triangles generated
#     together as one contiguous primitive (e.g. one tree's trunk cylinder) --
#     json_to_obj.py DOES share vertex indices within a single primitive's own
#     subdivision, so this works there, but NOT across different original
#     JSON faces (see _obj_components_by_shared_position for that -- used for
#     buildings, where genuine cross-face adjacency detection matters).
#     Verified on a real trees.obj: 49 trees x (1 trunk + 1 crown, each its
#     own component) = 98 components.
#     Returns a list of triangle-index lists, one per component.
#     """
#     parent = list(range(len(vertices)))

#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra != rb:
#             parent[ra] = rb

#     for (i, j, k) in triangles:
#         union(i, j)
#         union(j, k)

#     comp_triangles = {}
#     for t_idx, (i, j, k) in enumerate(triangles):
#         comp_triangles.setdefault(find(i), []).append(t_idx)
#     return list(comp_triangles.values())


# def build_tree_occluders_from_obj(trees_obj_path: str) -> list:
#     """Loads trees.obj and returns one Occluder per connected primitive
#     (trunk/crown separately) -- more, smaller occluders only help the
#     broad-phase AABB culling, so there's no reason to merge them further."""
#     if not trees_obj_path or not os.path.isfile(trees_obj_path):
#         return []
#     V, F = load_obj_mesh(trees_obj_path)
#     if not F:
#         return []
#     occluders = []
#     for c_idx, tri_indices in enumerate(_obj_connected_components(V, F)):
#         occ = _mesh_to_occluder(V, [F[t] for t in tri_indices], f"tree_part_{c_idx}")
#         if occ is not None:
#             occluders.append(occ)
#     return occluders


# # --------------------------------------------------------------------------
# # 9b. Multi-surface OBJ pipeline: buildings, ground, pathway, lawn, pond, trees
# #
# #     Every component is optional. A component is solved when its OBJ holds
# #     valid triangles (or, for trees only, when the JSON describes them);
# #     anything missing or unusable is reported in the log and skipped.
# #     Geometry comes from the OBJ files; material properties come from the
# #     scene JSON when it has them, otherwise from the documented defaults in
# #     LAYER_DEFAULTS / DEFAULT_MATERIAL (the log says which, per property).
# # --------------------------------------------------------------------------

# LATENT_HEAT_VAPORIZATION = 2.45e6      # J/kg
# CP_AIR = 1006.0                        # J/kg/K
# LAYER_CLASSES = ("ground", "pathway", "lawn", "pond", "tree")
# LAYER_PRIORITY = ("pond", "pathway", "lawn", "ground")   # top-most first


# def log(message: str) -> None:
#     print(message, flush=True)


# def warn(message: str) -> None:
#     print(f"[warning] {message}", flush=True)


# def _scene_from(scene_or_path):
#     """Accept either a parsed scene dict or a path to a scene JSON."""
#     return scene_or_path if isinstance(scene_or_path, dict) else load_scene(scene_or_path)


# # ---- material properties from the scene JSON -----------------------------

# def _norm_key(key) -> str:
#     return re.sub(r"[^a-z0-9]", "", str(key).lower())


# _PROPERTY_ALIASES = {
#     "thermal_conductivity": ("thermalconductivity", "conductivity"),
#     "density": ("density", "rho"),
#     "specific_heat": ("specificheat", "cp"),
#     "albedo": ("albedo", "reflectivity"),
#     "emissivity": ("emissivity", "emittance"),
#     "depth_m": ("depthm", "depth", "thicknessm", "thickness"),
#     "wetness": ("wetness", "evaporationfraction", "moistureavailability"),
#     "mixing_factor": ("mixingfactor", "mixingmultiplier"),
# }
# _CONTAINER_KEYS = ("thermalproperties", "thermalprops")
# _MATERIAL_KEYS = ("materialproperties", "material", "properties")
# _CLASS_KEYS = {
#     "ground": ("ground", "terrain", "soil", "groundproperties"),
#     "pathway": ("pathways", "pathway", "paths", "path", "roads", "road",
#                 "pavements", "pavement", "walkways"),
#     "lawn": ("lawns", "lawn", "grass", "grasslands", "greenareas"),
#     "pond": ("ponds", "pond", "water", "waterbodies"),
#     "tree": ("trees", "tree"),
# }


# def _as_float(value) -> Optional[float]:
#     if isinstance(value, bool):
#         return None
#     try:
#         x = float(value)
#     except (TypeError, ValueError):
#         return None
#     return x if math.isfinite(x) else None


# def _thermal_containers(item) -> list:
#     """Nested 'thermalProperties' dicts anywhere (shallowly) inside an item."""
#     found = []

#     def walk(obj, depth):
#         if depth > 4:
#             return
#         if isinstance(obj, dict):
#             for k, v in obj.items():
#                 if _norm_key(k) in _CONTAINER_KEYS and isinstance(v, dict):
#                     found.append(v)
#             for v in obj.values():
#                 walk(v, depth + 1)
#         elif isinstance(obj, list):
#             for v in obj[:3]:
#                 walk(v, depth + 1)

#     walk(item, 0)
#     return found


# def _lookup_property(item, prop: str) -> Optional[float]:
#     """Look up a numeric property by alias. Search order: thermalProperties
#     containers, materialProperties-like dicts, then the item's own top level.
#     A bare top-level 'density' is ignored (too ambiguous, e.g. tree density)."""
#     if not isinstance(item, dict):
#         return None
#     dicts = _thermal_containers(item)
#     for k, v in item.items():
#         if _norm_key(k) in _MATERIAL_KEYS and isinstance(v, dict):
#             dicts.append(v)
#     if prop != "density":
#         dicts.append(item)
#     aliases = _PROPERTY_ALIASES[prop]
#     for d in dicts:
#         norm = {_norm_key(k): v for k, v in d.items()}
#         for alias in aliases:
#             x = _as_float(norm.get(alias))
#             if x is not None:
#                 return x
#     return None


# def _scene_items(scene: dict, cls: str) -> list:
#     wanted = _CLASS_KEYS[cls]
#     containers = [scene]
#     if isinstance(scene.get("settings"), dict):
#         containers.append(scene["settings"])
#     items = []
#     for c in containers:
#         for k, v in c.items():
#             if _norm_key(k) in wanted:
#                 if isinstance(v, list):
#                     items.extend(x for x in v if isinstance(x, dict))
#                 elif isinstance(v, dict):
#                     items.append(v)
#     return items


# def json_layer_properties(scene: Optional[dict], cls: str) -> dict:
#     """Mean of each property over all JSON items of this class."""
#     if not isinstance(scene, dict):
#         return {}
#     items = _scene_items(scene, cls)
#     out = {}
#     for prop in _PROPERTY_ALIASES:
#         vals = [v for v in (_lookup_property(it, prop) for it in items) if v is not None]
#         if vals:
#             out[prop] = float(np.mean(vals))
#     return out


# # Documented defaults. Change these (or give the property in the JSON / CLI).
# _LAYER_BASE = dict(depth_m=0.23, nodes=5, deep_temperature_c=24.0, deep_h_wm2k=None,
#                    thermal_conductivity=1.5, density=1150.0, specific_heat=650.0,
#                    albedo=DEFAULT_GROUND_ALBEDO, emissivity=0.95, wetness=0.0,
#                    back_follows_ambient=False,
#                    mixing_factor=1.0)   # multiplies conductivity (turbulent mixing, water only)
# LAYER_DEFAULTS = {
#     "ground": {},
#     # paving (concrete/asphalt-like)
#     "pathway": dict(thermal_conductivity=1.4, density=2100.0, specific_heat=880.0,
#                     albedo=0.25, emissivity=0.92, depth_m=0.15),
#     # turf over the same soil as the ground; wetness = fraction of potential
#     # evaporation (0 = dry surface, 1 = free water surface)
#     "lawn": dict(albedo=0.25, emissivity=0.97, wetness=0.25),
#     # shallow water with free-surface evaporation. Still water conducts heat
#     # poorly (k = 0.6), so a real pond is mixed by wind and convection; this is
#     # represented by an effective conductivity k_eff = mixing_factor * k.
#     # mixing_factor = 1 gives still water (surface skin heats strongly).
#     "pond": dict(depth_m=0.5, thermal_conductivity=0.6, density=997.0,
#                  specific_heat=4182.0, albedo=0.06, emissivity=0.96, wetness=1.0,
#                  mixing_factor=30.0),
#     # thin canopy skin, back side exchanging with ambient air, partial transpiration
#     "tree": dict(depth_m=0.002, nodes=2, thermal_conductivity=0.15, density=700.0,
#                  specific_heat=3500.0, albedo=0.20, emissivity=0.96, wetness=0.30,
#                  back_follows_ambient=True),
# }


# def resolve_layer_settings(cls: str, scene: Optional[dict] = None,
#                            overrides: Optional[dict] = None) -> tuple:
#     """Precedence: built-in default < class default < scene JSON < CLI override.
#     Returns (GroundSettings, provenance {property: source})."""
#     base = dict(_LAYER_BASE)
#     prov = {k: "default" for k in base}
#     for k, v in LAYER_DEFAULTS.get(cls, {}).items():
#         base[k] = v
#         prov[k] = f"default:{cls}"
#     json_vals = json_layer_properties(scene, cls)
#     cli_vals = {k: v for k, v in (overrides or {}).items() if v is not None}

#     def build(use_json):
#         d, p = dict(base), dict(prov)
#         if use_json:
#             for k, v in json_vals.items():
#                 d[k], p[k] = v, "json"
#         for k, v in cli_vals.items():
#             d[k], p[k] = v, "cli"
#         d["wetness"] = min(max(float(d["wetness"]), 0.0), 1.0)
#         d["nodes"] = int(d["nodes"])
#         mixing = float(d.pop("mixing_factor"))
#         p.pop("mixing_factor", None)
#         if not math.isfinite(mixing) or mixing <= 0:
#             raise ValueError("mixing_factor must be finite and positive")
#         if mixing != 1.0:
#             d["thermal_conductivity"] = float(d["thermal_conductivity"]) * mixing
#             p["thermal_conductivity"] = f"{p['thermal_conductivity']} x{mixing:g} mixing"
#         return GroundSettings(**d), p

#     try:
#         return build(True)
#     except ValueError as exc:
#         warn(f"{cls}: invalid property in JSON ({exc}); falling back to defaults")
#         return build(False)


# def describe_layer_settings(cls: str, gs: GroundSettings, prov: dict) -> str:
#     parts = []
#     for key, label, fmt in (("thermal_conductivity", "k", "{:.3g}"), ("density", "rho", "{:.0f}"),
#                             ("specific_heat", "cp", "{:.0f}"), ("albedo", "albedo", "{:.2f}"),
#                             ("emissivity", "eps", "{:.2f}"), ("depth_m", "depth_m", "{:.3g}"),
#                             ("wetness", "wetness", "{:.2f}")):
#         parts.append(f"{label}={fmt.format(getattr(gs, key))}[{prov.get(key, 'given')}]")
#     return f"[properties] {cls}: " + " ".join(parts)


# class BuildingMaterials:
#     """Wall/roof properties for OBJ buildings taken from the scene JSON.

#     JSON faces are averaged (area-weighted) per surface type. An OBJ building
#     is matched to JSON building(s) whose XY bounding box (+1 m) contains its
#     centre; unmatched OBJ buildings get the scene-wide wall/roof means.
#     Missing properties fall back to DEFAULT_MATERIAL / DEFAULT_ALBEDO_*.
#     """
#     _KEYS = ("thermal_conductivity", "density", "specific_heat", "albedo", "emissivity")

#     def __init__(self, scene: Optional[dict] = None):
#         self.defaults = {
#             "wall": dict(DEFAULT_MATERIAL, albedo=DEFAULT_ALBEDO_WALL, emissivity=DEFAULT_EMISSIVITY),
#             "roof": dict(DEFAULT_MATERIAL, albedo=DEFAULT_ALBEDO_ROOF, emissivity=DEFAULT_EMISSIVITY)}
#         self.overall = {k: dict(v) for k, v in self.defaults.items()}
#         self.provenance = {"wall": "default", "roof": "default"}
#         self.buildings = []
#         self.n_matched = 0
#         self.n_unmatched = 0
#         if isinstance(scene, dict):
#             try:
#                 self._read_scene(scene)
#             except Exception as exc:                      # never let JSON quirks stop the run
#                 warn(f"buildings: could not read materials from JSON ({exc}); using defaults")

#     def _merge(self, base: dict, samples: list) -> dict:
#         out = dict(base)
#         for k in self._KEYS:
#             vals = [(a, f[k]) for a, f in samples if k in f]
#             if vals:
#                 out[k] = float(np.average([v for _, v in vals], weights=[a for a, _ in vals]))
#         if (any(not math.isfinite(out[k]) or out[k] <= 0 for k in self._KEYS[:3])
#                 or not 0 <= out["albedo"] <= 1 or not 0 <= out["emissivity"] <= 1):
#             warn("buildings: out-of-range material values in JSON ignored")
#             return dict(base)
#         return out

#     def _read_scene(self, scene: dict):
#         per_building, overall = [], {"wall": [], "roof": []}
#         for bi, b in enumerate(scene.get("buildings", []) or []):
#             if not isinstance(b, dict):
#                 continue
#             per, pts = {"wall": [], "roof": []}, []
#             for f in b.get("faces", []) or []:
#                 try:
#                     verts = np.array([[v["x"], v["y"], v["z"]] for v in f["vertices"]], float)
#                 except Exception:
#                     continue
#                 if len(verts) < 3 or not np.isfinite(verts).all():
#                     continue
#                 pts.append(verts)
#                 kind = {"roof": "roof", "wall": "wall"}.get(str(f.get("surfaceType", "wall")).lower())
#                 if kind is None:
#                     continue
#                 found = {k: _lookup_property(f, k) for k in self._KEYS}
#                 found = {k: v for k, v in found.items() if v is not None}
#                 if found:
#                     sample = (max(_polygon_area_3d(verts), 1e-9), found)
#                     per[kind].append(sample)
#                     overall[kind].append(sample)
#             if pts:
#                 allp = np.vstack(pts)
#                 per_building.append((str(b.get("id", f"building_{bi}")),
#                                      allp.min(axis=0)[:2], allp.max(axis=0)[:2], per))
#         for kind in ("wall", "roof"):
#             if overall[kind]:
#                 self.overall[kind] = self._merge(self.defaults[kind], overall[kind])
#                 self.provenance[kind] = "json"
#         for bid, lo, hi, per in per_building:
#             entry = {kind: self._merge(self.overall[kind], per[kind]) for kind in ("wall", "roof")}
#             self.buildings.append((bid, lo, hi, entry))

#     def for_component(self, center_xy) -> dict:
#         c = np.asarray(center_xy, float)
#         hits = [e for _, lo, hi, e in self.buildings
#                 if np.all(c >= lo - 1.0) and np.all(c <= hi + 1.0)]
#         if not hits:
#             self.n_unmatched += 1
#             return self.overall
#         self.n_matched += 1
#         if len(hits) == 1:
#             return hits[0]
#         return {kind: {k: float(np.mean([h[kind][k] for h in hits])) for k in self._KEYS}
#                 for kind in ("wall", "roof")}

#     def describe(self) -> str:
#         parts = []
#         for kind in ("wall", "roof"):
#             m = self.overall[kind]
#             parts.append(f"{kind}: k={m['thermal_conductivity']:.3g} rho={m['density']:.0f} "
#                          f"cp={m['specific_heat']:.0f} albedo={m['albedo']:.2f} "
#                          f"eps={m['emissivity']:.2f} [{self.provenance[kind]}]")
#         return ("[properties] buildings: " + "; ".join(parts) +
#                 f"; OBJ buildings matched to JSON: {self.n_matched}, using scene mean: {self.n_unmatched}")


# # ---- OBJ loading with validation -------------------------------------------

# def _clean_mesh(V, F, label: str):
#     """Drop invalid / degenerate triangles. Returns (V, F ndarray) or None."""
#     V = np.asarray(V, dtype=float)
#     F = np.asarray(F, dtype=np.int64).reshape(-1, 3)
#     n0 = len(F)
#     if V.ndim != 2 or len(V) == 0 or n0 == 0:
#         warn(f"{label}: no vertices/faces")
#         return None
#     F = F[((F >= 0) & (F < len(V))).all(axis=1)]
#     F = F[np.isfinite(V)[F].all(axis=(1, 2))] if len(F) else F
#     if len(F):
#         tri = V[F]
#         area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
#         F = F[area > 1e-12]
#     if n0 - len(F):
#         warn(f"{label}: dropped {n0 - len(F)} invalid/degenerate triangle(s) of {n0}")
#     if not len(F):
#         warn(f"{label}: no valid triangles left")
#         return None
#     return V, F


# def load_obj_checked(path, label: str):
#     """Load one OBJ (or a list, merged). Returns (V, F) or None; never raises."""
#     if not path:
#         return None
#     paths = [path] if isinstance(path, str) else list(path)
#     meshes = []
#     for p in paths:
#         if not os.path.isfile(p):
#             warn(f"{label}: file not found, skipping: {p}")
#             continue
#         try:
#             V, F = load_obj_mesh(p)
#         except Exception as exc:
#             warn(f"{label}: could not parse {p} ({exc})")
#             continue
#         mesh = _clean_mesh(V, F, f"{label} {os.path.basename(p)}")
#         if mesh is not None:
#             meshes.append(mesh)
#     if not meshes:
#         return None
#     if len(meshes) == 1:
#         return meshes[0]
#     offset, Vs, Fs = 0, [], []
#     for V, F in meshes:
#         Vs.append(V)
#         Fs.append(F + offset)
#         offset += len(V)
#     return np.vstack(Vs), np.vstack(Fs)


# def _json_trees_mesh(scene: Optional[dict]):
#     """Fallback tree geometry (trunk + cone crown) built from JSON tree entries."""
#     if not isinstance(scene, dict):
#         return None
#     polys = []
#     for t in scene.get("trees", []) or []:
#         if not isinstance(t, dict):
#             continue
#         try:
#             p, _ = _tree_mesh(t)
#         except Exception:
#             continue
#         if p:
#             polys.extend(p)
#     if not polys:
#         return None
#     V, F = [], []
#     for poly in polys:
#         base = len(V)
#         V.extend(poly)
#         F.extend((base, base + i, base + i + 1) for i in range(1, len(poly) - 1))
#     return _clean_mesh(V, F, "trees (from JSON)")


# # ---- XY coverage test (remove ground cells hidden under overlay surfaces) --

# class XYTriangleIndex:
#     """Uniform-grid index of triangles projected on the XY plane."""

#     def __init__(self, tri_xyz):
#         t = np.asarray(tri_xyz, float)[:, :, :2]
#         self.t = t
#         lo, hi = t.min(axis=1), t.max(axis=1)
#         span = (hi - lo).max(axis=1)
#         extent = float((hi.max(axis=0) - lo.min(axis=0)).max())
#         self.cell = max(float(np.percentile(span, 90)), extent / 200.0, 1e-6)
#         self.origin = lo.min(axis=0)
#         i0 = np.floor((lo - self.origin) / self.cell).astype(int)
#         i1 = np.floor((hi - self.origin) / self.cell).astype(int)
#         self.grid = {}
#         for k in range(len(t)):
#             for ix in range(i0[k, 0], i1[k, 0] + 1):
#                 for iy in range(i0[k, 1], i1[k, 1] + 1):
#                     self.grid.setdefault((ix, iy), []).append(k)

#     def covers(self, pts_xy) -> np.ndarray:
#         pts = np.asarray(pts_xy, float).reshape(-1, 2)
#         out = np.zeros(len(pts), dtype=bool)
#         cells = np.floor((pts - self.origin) / self.cell).astype(int)
#         keys, inv = np.unique(cells, axis=0, return_inverse=True)
#         inv = inv.ravel()
#         order = np.argsort(inv, kind="stable")
#         bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
#         for ki, (cx, cy) in enumerate(keys):
#             tris = self.grid.get((int(cx), int(cy)))
#             if not tris:
#                 continue
#             idx = order[bounds[ki]:bounds[ki + 1]]
#             tt = self.t[tris]
#             a = tt[:, 0]
#             v0, v1 = tt[:, 1] - a, tt[:, 2] - a
#             den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
#             ok = np.abs(den) > 1e-14
#             den = np.where(ok, den, 1.0)
#             d = pts[idx][:, None, :] - a[None]
#             u = (d[..., 0] * v1[:, 1] - v1[:, 0] * d[..., 1]) / den
#             v = (v0[:, 0] * d[..., 1] - d[..., 0] * v0[:, 1]) / den
#             inside = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
#             out[idx] = inside.any(axis=1)
#         return out


# # ---- building faces from OBJ ----------------------------------------------

# def build_faces_from_obj(buildings_obj_path=None, materials: Optional[BuildingMaterials] = None,
#                          mesh=None) -> tuple:
#     """Recover buildings, retaining every nondegenerate triangle as a cell.

#     Coplanar groups provide parent wall IDs and approximate bulk volumes only;
#     visibility, irradiance, temperature and VTK values belong to each triangle.
#     Wall/roof materials come from `materials` (JSON-derived when available).
#     Returns (faces, capacitances, face_triangles, occluders); the last is kept
#     for API compatibility and is empty (rays use the BVH built from the mesh).
#     """
#     if mesh is None:
#         mesh = load_obj_checked(buildings_obj_path, "buildings")
#     if mesh is None:
#         return [], {}, {}, {}
#     V, Farr = mesh
#     materials = materials or BuildingMaterials()
#     F = [tuple(t) for t in Farr.tolist()]
#     faces, face_triangles, capacitances = [], {}, {}
#     for b_idx, tri_indices in enumerate(_obj_components_by_shared_position(V, F)):
#         bid = f"building_{b_idx:02d}"
#         comp_tris = [F[t] for t in tri_indices]
#         groups = group_coplanar(V, comp_tris)
#         if not groups:
#             continue
#         z_min = min(g["z_min"] for g in groups.values())
#         z_max = max(g["z_max"] for g in groups.values())
#         pts = V[np.unique(np.asarray(comp_tris).ravel())]
#         mats = materials.for_component(0.5 * (pts.min(axis=0) + pts.max(axis=0))[:2])
#         wm = mats["wall"]
#         roof_area = sum(g["area"] * abs(g["normal"][2]) for g in groups.values()
#                         if abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
#                         and g["center"][2] > z_min + 1e-6)
#         capacitances[bid] = wm["density"] * roof_area * (z_max - z_min) * wm["specific_heat"]
#         for gi, g in enumerate(groups.values()):
#             horizontal = abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
#             bottom = horizontal and g["z_max"] <= z_min + 1e-6
#             stype = "floor" if bottom else ("roof" if horizontal else "wall")
#             normal = g["normal"].copy()
#             if (stype == "roof" and normal[2] < 0) or (stype == "floor" and normal[2] > 0):
#                 normal = -normal
#             tilt = math.degrees(math.acos(np.clip(normal[2], -1, 1)))
#             parent = f"{bid}_face_{gi:03d}_{stype}"
#             m = mats["roof" if stype == "roof" else "wall"]
#             for ti, tri in enumerate(g["triangles"]):
#                 verts = V[list(tri)]
#                 _, area = _triangle_normal_area(*verts)
#                 fid = f"{parent}_tri_{ti:06d}"
#                 faces.append(Face(
#                     building_id=bid, face_id=fid, parent_face_id=parent, surface_type=stype,
#                     normal=normal, area=area, tilt_deg=tilt, center=verts.mean(axis=0),
#                     vertices=verts, thermal_conductivity=m["thermal_conductivity"],
#                     density=m["density"], specific_heat=m["specific_heat"],
#                     albedo=m["albedo"], emissivity=m["emissivity"]))
#                 face_triangles[fid] = [tri]
#     return faces, capacitances, face_triangles, {}


# # ---- ground / pathway / lawn / pond / tree cells ---------------------------

# def _layer_geometry(V, F, orient: str):
#     """Per-triangle vertices, unit normals and areas. orient='up' flips normals
#     to point upward; orient='outward' flips them away from the centre of each
#     connected piece (closed tree proxies)."""
#     tri = V[F].copy()
#     n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
#     norm = np.linalg.norm(n, axis=1)
#     area = 0.5 * norm
#     nrm = n / norm[:, None]
#     if orient == "up":
#         flip = nrm[:, 2] < 0
#     else:
#         flip = np.zeros(len(F), dtype=bool)
#         centroids = tri.mean(axis=1)
#         for comp in _obj_components_by_shared_position(V, F):
#             idx = np.asarray(comp)
#             centre = tri[idx].reshape(-1, 3).mean(axis=0)
#             flip[idx] = np.einsum("ij,ij->i", nrm[idx], centroids[idx] - centre) < 0
#     tri[flip] = tri[flip][:, [0, 2, 1]]
#     nrm[flip] *= -1
#     return tri, nrm, area


# def build_layer_faces(cls: str, V, F, settings: GroundSettings, building_tracer=None,
#                       higher_index: Optional[XYTriangleIndex] = None,
#                       orient: str = "up") -> list:
#     """One Face per mesh triangle. Cells under a building roof (vertical ray)
#     and cells whose centroid lies under a higher-priority surface (e.g. ground
#     under a pond) are removed so no point is solved twice."""
#     tri, nrm, area = _layer_geometry(V, F, orient)
#     cen = tri.mean(axis=1)
#     keep = np.ones(len(tri), dtype=bool)
#     n_under = n_cov = 0
#     if building_tracer is not None and cls != "tree":
#         blocked = building_tracer.blocked(cen + np.array([0.0, 0.0, _SHADOW_RAY_OFFSET]),
#                                           np.array([0.0, 0.0, 1.0]))
#         n_under = int(blocked.sum())
#         keep &= ~blocked
#     if higher_index is not None and cls != "tree":
#         covered = higher_index.covers(cen[:, :2])
#         n_cov = int((covered & keep).sum())
#         keep &= ~covered
#     faces = []
#     for ti in np.flatnonzero(keep):
#         faces.append(Face(
#             building_id=cls, face_id=f"{cls}_tri_{ti:06d}", parent_face_id=f"{cls}_surface",
#             surface_type=cls, normal=nrm[ti], area=float(area[ti]),
#             tilt_deg=math.degrees(math.acos(np.clip(nrm[ti][2], -1, 1))),
#             center=cen[ti], vertices=tri[ti],
#             thermal_conductivity=settings.thermal_conductivity, density=settings.density,
#             specific_heat=settings.specific_heat, albedo=settings.albedo,
#             emissivity=settings.emissivity))
#     log(f"[input] {cls}: {len(faces)} cells solved "
#         f"({n_under} under buildings and {n_cov} under other surfaces removed), "
#         f"area {sum(f.area for f in faces):.1f} m2")
#     return faces


# @dataclass
# class ObjBundle:
#     building_faces: list
#     capacitances: dict
#     layer_faces: dict
#     layer_settings: dict
#     tracer: object
#     wall_settings: WallSettings

#     @property
#     def n_cells(self) -> int:
#         return len(self.building_faces) + sum(len(v) for v in self.layer_faces.values())

#     def all_faces(self) -> list:
#         out = list(self.building_faces)
#         for cls in LAYER_CLASSES:
#             out += self.layer_faces.get(cls, [])
#         return out


# def prepare_obj_bundle(buildings_obj=None, ground_obj=None, trees_obj=None, pathway_obj=None,
#                        lawn_obj=None, pond_obj=None, scene: Optional[dict] = None,
#                        wall_settings: Optional[WallSettings] = None,
#                        layer_overrides: Optional[dict] = None,
#                        fixed_layer_settings: Optional[dict] = None,
#                        consider_shadows: bool = True, remove_overlaps: bool = True) -> ObjBundle:
#     """Load every available component once (geometry, materials, BVH). The
#     result is reused for every weather day."""
#     wall_settings = wall_settings or WallSettings()
#     layer_overrides = layer_overrides or {}
#     fixed_layer_settings = fixed_layer_settings or {}
#     paths = dict(ground=ground_obj, pathway=pathway_obj, lawn=lawn_obj, pond=pond_obj, tree=trees_obj)
#     meshes = {}
#     for cls, p in paths.items():
#         mesh = load_obj_checked(p, f"{cls} OBJ") if p else None
#         if mesh is None and cls == "tree":
#             mesh = _json_trees_mesh(scene)
#             if mesh is not None:
#                 log("[input] tree: no usable trees OBJ, using tree geometry from the JSON")
#         if mesh is not None:
#             meshes[cls] = mesh
#             log(f"[input] {cls}: {len(mesh[1])} valid triangles")
#         elif p:
#             warn(f"{cls}: nothing solved ({p})")
#     bmesh = load_obj_checked(buildings_obj, "buildings OBJ") if buildings_obj else None

#     materials = BuildingMaterials(scene)
#     building_faces, capacitances = [], {}
#     if bmesh is not None:
#         building_faces, capacitances, _, _ = build_faces_from_obj(mesh=bmesh, materials=materials)
#         n_b = len(capacitances)
#         log(f"[input] buildings: {n_b} building(s), {len(building_faces)} cells")
#         log(materials.describe())
#     elif buildings_obj:
#         warn("buildings: nothing solved")

#     b_tris = bmesh[0][bmesh[1]] if bmesh is not None else np.empty((0, 3, 3))
#     building_tracer = TriangleRayTracer(b_tris) if len(b_tris) else None
#     tracer = None
#     if consider_shadows:
#         tree_tris = meshes["tree"][0][meshes["tree"][1]] if "tree" in meshes else np.empty((0, 3, 3))
#         if len(tree_tris):
#             tracer = TriangleRayTracer(np.concatenate([b_tris, tree_tris]) if len(b_tris) else tree_tris)
#         else:
#             tracer = building_tracer
#         if tracer is not None:
#             log(f"[shadows] occluders: {len(b_tris)} building + {len(tree_tris)} tree triangles")
#         else:
#             log("[shadows] no occluders present: only self-orientation (cos) shading applies")

#     tri_cache = {cls: m[0][m[1]] for cls, m in meshes.items()}
#     layer_faces, layer_settings = {}, {}
#     for cls in LAYER_CLASSES:
#         if cls not in meshes:
#             continue
#         if cls in fixed_layer_settings:
#             gs, prov = fixed_layer_settings[cls], {}
#         else:
#             gs, prov = resolve_layer_settings(cls, scene, layer_overrides.get(cls))
#         log(describe_layer_settings(cls, gs, prov))
#         higher = None
#         if remove_overlaps and cls in LAYER_PRIORITY:
#             above = [tri_cache[c] for c in LAYER_PRIORITY[:LAYER_PRIORITY.index(cls)] if c in tri_cache]
#             if above:
#                 higher = XYTriangleIndex(np.concatenate(above))
#         V, F = meshes[cls]
#         faces = build_layer_faces(cls, V, F, gs, building_tracer, higher,
#                                   "outward" if cls == "tree" else "up")
#         if faces:
#             layer_faces[cls], layer_settings[cls] = faces, gs
#         else:
#             warn(f"{cls}: no cells left after filtering; skipped")
#     return ObjBundle(building_faces, capacitances, layer_faces, layer_settings, tracer, wall_settings)


# # ---- weather -> per-record dicts (adds humidity / pressure when available) --

# def _series(values, n):
#     if values is None:
#         return None
#     arr = np.asarray(values, dtype=float)
#     return arr if arr.ndim == 1 and len(arr) == n else None


# def records_from_weather(wx: dict) -> list:
#     n = len(wx["hour"])
#     p = _series(wx.get("atmosphericPressure"), n)
#     if p is not None:                       # accept Pa, hPa or kPa
#         med = float(np.nanmedian(p))
#         p = p * (1000.0 if med < 200 else 100.0 if med < 2000 else 1.0)
#     w_amb = _series(wx.get("wambient"), n)
#     rh = _series(wx.get("relativeHumidity"), n)
#     if rh is not None and float(np.nanmax(rh)) > 1.5:
#         rh = rh / 100.0
#     records = []
#     for i in range(n):
#         rec = dict(hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
#                    sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
#                    i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i])
#         if p is not None and np.isfinite(p[i]):
#             rec["p_atm"] = float(p[i])
#         if w_amb is not None and np.isfinite(w_amb[i]) and 0 <= w_amb[i] < 0.1:
#             rec["w_amb"] = float(w_amb[i])
#         if rh is not None and np.isfinite(rh[i]):
#             rec["rh"] = float(np.clip(rh[i], 0.0, 1.0))
#         records.append(rec)
#     return records


# def _saturation_pressure_pa(t_c: float) -> float:
#     return 610.94 * math.exp(17.625 * t_c / (t_c + 243.04))


# def _humidity_ratio(t_c: float, rh: float, p_pa: float) -> float:
#     pv = min(rh * _saturation_pressure_pa(t_c), 0.99 * p_pa)
#     return 0.622 * pv / (p_pa - pv)


# def _saturation_humidity_ratio(t_c: float, p_pa: float) -> tuple:
#     """(w_sat, d w_sat / dT) at temperature t_c."""
#     es = min(_saturation_pressure_pa(t_c), 0.99 * p_pa)
#     des = es * 17.625 * 243.04 / (t_c + 243.04) ** 2
#     return 0.622 * es / (p_pa - es), 0.622 * p_pa * des / (p_pa - es) ** 2


# def sunlit_fractions_from_arrays(normals, verts, sun_vec, ray_tracer, samples=3):
#     """Array version of triangle_sunlit_fraction (no per-call Python loops)."""
#     fractions = np.zeros(len(normals))
#     if sun_vec[2] <= 0:
#         return fractions
#     eligible = np.flatnonzero(normals @ sun_vec > 0)
#     if not len(eligible):
#         return fractions
#     weights = np.array([[1/3, 1/3, 1/3]]) if samples == 1 else np.array([
#         [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
#     origins = np.einsum('sj,tjk->tsk', weights, verts[eligible])
#     origins += normals[eligible, None, :] * _SHADOW_RAY_OFFSET
#     hits = ray_tracer.blocked(origins.reshape(-1, 3), sun_vec).reshape(-1, samples)
#     fractions[eligible] = 1 - hits.mean(axis=1)
#     return fractions


# def _tag_layer_columns(df: pd.DataFrame) -> None:
#     """Non-building classes: expose the slab fields under the ground_* names
#     (they hold the sub-surface layer of whatever class `surface_group` says)
#     and blank the building-only columns."""
#     df["ground_depth_m"] = df["slab_thickness_m"]
#     df["T_ground_bottom_C"] = df["T_back_surface_C"]
#     df["T_deep_ground_C"] = df["T_back_boundary_C"]
#     df["h_deep_ground_Wm2K"] = df["h_back_Wm2K"]
#     df["q_deep_ground_out_Wm2"] = df["q_back_out_Wm2"]
#     df["T_ground_mean_C"] = df["T_slab_mean_C"]
#     df["ground_capacity_Jm2K"] = df["slab_capacity_Jm2K"]
#     df["ground_capacity_JK"] = df["slab_capacity_JK"]
#     building_only = ["wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K", "T_inner_C",
#                      "T_wall_mean_C", "q_indoor_out_Wm2", "wall_capacity_Jm2K", "wall_capacity_JK"]
#     df[building_only] = np.nan


# def solve_obj_day(bundle: ObjBundle, weather_dir: str, t_initial_c: Optional[float] = None,
#                   ground_initial_c: Optional[float] = None,
#                   ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                   shadow_samples: int = 3, workers: int = 1) -> tuple:
#     """Solve every component in the bundle for one weather folder.
#     Returns (face_df, building_df, faces)."""
#     wx = load_weather_folder(weather_dir)
#     records = records_from_weather(wx)
#     frames, bframes, faces_out = [], [], []
#     if bundle.building_faces:
#         fdf, bdf = _simulate(bundle.building_faces, bundle.capacitances, records,
#                              ground_albedo=ground_albedo, t_initial_c=t_initial_c,
#                              ray_tracer=bundle.tracer, shadow_samples=shadow_samples,
#                              wall_settings=bundle.wall_settings, workers=workers)
#         frames.append(fdf)
#         bframes.append(bdf)
#         faces_out += bundle.building_faces
#     for cls in LAYER_CLASSES:
#         faces = bundle.layer_faces.get(cls)
#         if not faces:
#             continue
#         gs = bundle.layer_settings[cls]
#         cap = gs.density * gs.specific_heat * gs.depth_m * sum(f.area for f in faces)
#         init = records[0]["T_amb_C"] if cls == "tree" else ground_initial_c
#         ldf, _ = _simulate(faces, {cls: cap}, records, ground_albedo=ground_albedo,
#                            t_initial_c=init, ray_tracer=bundle.tracer,
#                            shadow_samples=shadow_samples, wall_settings=gs.slab_settings(),
#                            surface_group=cls, wetness=gs.wetness,
#                            back_follows_ambient=gs.back_follows_ambient, workers=workers)
#         _tag_layer_columns(ldf)
#         frames.append(ldf)
#         faces_out += faces
#     if not frames:
#         raise ValueError("no solvable components in this scene")
#     face_df = pd.concat(frames, ignore_index=True, sort=False)
#     building_df = (pd.concat(bframes, ignore_index=True) if bframes else
#                    pd.DataFrame(columns=["building_id", "hour", "T_amb_C", "T_eq_C",
#                                          "h_eff_WK", "tau_hr", "T_building_C"]))
#     return face_df, building_df, faces_out


# def run_analysis_from_obj(buildings_obj_path=None, weather_dir: Optional[str] = None,
#                           trees_obj_path: Optional[str] = None, consider_shadows: bool = True,
#                           t_initial_c: Optional[float] = None, shadow_samples: int = 3,
#                           wall_settings: Optional[WallSettings] = None,
#                           ground_obj_path: Optional[str] = None,
#                           ground_initial_c: Optional[float] = None,
#                           ground_settings: Optional[GroundSettings] = None,
#                           pathway_obj_path: Optional[str] = None,
#                           lawn_obj_path: Optional[str] = None,
#                           pond_obj_path: Optional[str] = None,
#                           scene=None, workers: int = 1) -> tuple:
#     """Convenience wrapper: prepare + solve one weather folder.
#     Returns (face_df, building_df, faces)."""
#     if shadow_samples not in (1, 3):
#         raise ValueError("shadow_samples must be 1 or 3")
#     bundle = prepare_obj_bundle(
#         buildings_obj_path, ground_obj_path, trees_obj_path, pathway_obj_path, lawn_obj_path,
#         pond_obj_path, scene=_scene_from(scene) if scene is not None else None,
#         wall_settings=wall_settings,
#         fixed_layer_settings={"ground": ground_settings} if ground_settings else None,
#         consider_shadows=consider_shadows)
#     return solve_obj_day(bundle, weather_dir, t_initial_c=t_initial_c,
#                          ground_initial_c=ground_initial_c, shadow_samples=shadow_samples,
#                          workers=workers)


# def export_vtk_series_from_analysis(analysis: tuple, out_dir: str) -> str:
#     """Write one legacy-VTK file per hour (+ scene.pvd) with per-cell CELL_DATA
#     for every solved component. T_C aliases T_surface_C. surface_type_id:
#     0 wall, 1 roof, 2 floor, 3 ceiling, 5 tree, 6 ground, 7 pathway, 8 lawn,
#     9 pond. Tree/pond/lawn cells also set is_tree/is_pond/is_lawn flags."""
#     face_df, building_df, faces = analysis
#     os.makedirs(out_dir, exist_ok=True)
#     polygons = [np.asarray(f.vertices, float) for f in faces]
#     ids = [f.face_id for f in faces]
#     building_ids = dict.fromkeys(f.building_id for f in faces if f.surface_type not in LAYER_CLASSES)
#     bmap = {bid: i for i, bid in enumerate(building_ids)}
#     pmap = {pid: i for i, pid in enumerate(dict.fromkeys(f.parent_face_id for f in faces))}
#     static = {
#         "surface_type_id": ([_SURFACE_TYPE_ID.get(f.surface_type, -1) for f in faces], "int"),
#         "is_building": ([int(f.surface_type not in LAYER_CLASSES) for f in faces], "int"),
#         "building_index": ([bmap.get(f.building_id, -1) for f in faces], "int"),
#         "parent_face_index": ([pmap[f.parent_face_id] for f in faces], "int"),
#     }
#     for cls in LAYER_CLASSES:
#         static[f"is_{cls}"] = ([int(f.surface_type == cls) for f in faces], "int")
#     fields = ["T_surface_C", "T_solair_C", "I_direct_Wm2", "I_total_Wm2",
#               "sunlit_fraction", "shadow_fraction", "shadowed", "sunlit",
#               "h_conv_Wm2K", "q_solar_absorbed_Wm2", "q_conv_out_Wm2", "q_longwave_out_Wm2",
#               "q_evap_out_Wm2", "T_inner_C", "T_wall_mean_C", "q_conduction_in_Wm2",
#               "q_indoor_out_Wm2", "q_storage_Wm2", "q_outer_node_storage_Wm2",
#               "stored_heat_Jm2", "stored_heat_J", "wall_capacity_Jm2K", "wall_capacity_JK",
#               "wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K", "T_back_surface_C",
#               "T_slab_mean_C", "q_back_out_Wm2", "slab_capacity_Jm2K", "slab_capacity_JK",
#               "slab_thickness_m", "T_back_boundary_C", "h_back_Wm2K", "ground_depth_m",
#               "T_ground_bottom_C", "T_deep_ground_C", "h_deep_ground_Wm2K",
#               "q_deep_ground_out_Wm2", "T_ground_mean_C", "ground_capacity_Jm2K",
#               "ground_capacity_JK"]
#     fields = [name for name in fields if name in face_df.columns]
#     entries = []
#     for h, frame in face_df.groupby("hour", sort=True):
#         amb_c = float(frame["T_amb_C"].iloc[0])
#         frame = frame.set_index("face_id").reindex(ids)
#         scalars = dict(static)
#         for name in fields:
#             fill = amb_c if name.startswith("T_") else 0
#             dtype = "int" if name in ("shadowed", "sunlit") else "float"
#             values = frame[name].fillna(fill).to_numpy(dtype=int if dtype == "int" else float)
#             scalars[name] = (values, dtype)
#         scalars["T_C"] = scalars["T_surface_C"]
#         fname = f"frame_{int(round(h * 100)):04d}.vtk"
#         _write_vtk_polydata(os.path.join(out_dir, fname), polygons, scalars)
#         entries.append((float(h), fname))
#     pvd_path = os.path.join(out_dir, "scene.pvd")
#     with open(pvd_path, "w") as f:
#         f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
#         for t, fname in entries:
#             f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
#         f.write('</Collection>\n</VTKFile>\n')
#     log(f"[vtk export] Wrote {len(entries)} frame(s) + scene.pvd to {out_dir}")
#     return pvd_path


# def export_vtk_series_from_obj(buildings_obj_path, weather_dir: str, out_dir: str,
#                                trees_obj_path: Optional[str] = None, consider_shadows: bool = True,
#                                shadow_samples: int = 3, analysis=None,
#                                wall_settings: Optional[WallSettings] = None,
#                                t_initial_c: Optional[float] = None,
#                                ground_obj_path: Optional[str] = None,
#                                ground_settings: Optional[GroundSettings] = None,
#                                ground_initial_c: Optional[float] = None,
#                                pathway_obj_path: Optional[str] = None,
#                                lawn_obj_path: Optional[str] = None,
#                                pond_obj_path: Optional[str] = None, scene=None) -> str:
#     """Backward-compatible wrapper: run (unless `analysis` is given) and export."""
#     if analysis is None:
#         analysis = run_analysis_from_obj(
#             buildings_obj_path, weather_dir, trees_obj_path=trees_obj_path,
#             consider_shadows=consider_shadows, t_initial_c=t_initial_c,
#             shadow_samples=shadow_samples, wall_settings=wall_settings,
#             ground_obj_path=ground_obj_path, ground_initial_c=ground_initial_c,
#             ground_settings=ground_settings, pathway_obj_path=pathway_obj_path,
#             lawn_obj_path=lawn_obj_path, pond_obj_path=pond_obj_path, scene=scene)
#     return export_vtk_series_from_analysis(analysis, out_dir)


# # ---- fast legacy-VTK writer (falls back to the slow one for mixed polygons) --

# VTK_BINARY = False      # set by --vtk_binary: big-endian legacy BINARY files (much faster/smaller)


# def _write_vtk_polydata_binary(path, arr, cell_scalars) -> None:
#     n_poly, n_v, _ = arr.shape
#     conn = np.concatenate([np.full((n_poly, 1), n_v), np.arange(n_poly * n_v).reshape(n_poly, n_v)],
#                           axis=1).astype(">i4")
#     with open(path, "wb") as f:
#         f.write(b"# vtk DataFile Version 3.0\nsolar_thermal_model export\nBINARY\nDATASET POLYDATA\n")
#         f.write(f"POINTS {n_poly * n_v} float\n".encode())
#         f.write(arr.reshape(-1, 3).astype(">f4").tobytes())
#         f.write(f"\nPOLYGONS {n_poly} {n_poly * (n_v + 1)}\n".encode())
#         f.write(conn.tobytes())
#         f.write(f"\nCELL_DATA {n_poly}\n".encode())
#         for name, (values, dtype) in cell_scalars.items():
#             f.write(f"SCALARS {name} {dtype} 1\nLOOKUP_TABLE default\n".encode())
#             if dtype == "int":
#                 f.write(np.asarray(values).astype(">i4").tobytes())
#             else:
#                 f.write(np.nan_to_num(np.asarray(values, dtype=float)).astype(">f4").tobytes())
#             f.write(b"\n")


# def _write_vtk_polydata(path: str, polygons: list, cell_scalars: dict) -> None:
#     try:
#         arr = np.asarray(polygons, dtype=float)
#         if arr.ndim != 3 or arr.shape[2] != 3:
#             raise ValueError("not a uniform polygon array")
#     except (ValueError, TypeError):
#         return _write_vtk_polydata_slow(path, polygons, cell_scalars)
#     if VTK_BINARY:
#         return _write_vtk_polydata_binary(path, arr, cell_scalars)
#     n_poly, n_v, _ = arr.shape
#     conn = np.concatenate([np.full((n_poly, 1), n_v), np.arange(n_poly * n_v).reshape(n_poly, n_v)], axis=1)
#     with open(path, "w") as f:
#         f.write("# vtk DataFile Version 3.0\nsolar_thermal_model export\nASCII\nDATASET POLYDATA\n")
#         f.write(f"POINTS {n_poly * n_v} float\n")
#         np.savetxt(f, arr.reshape(-1, 3), fmt="%.4f")
#         f.write(f"POLYGONS {n_poly} {n_poly * (n_v + 1)}\n")
#         np.savetxt(f, conn, fmt="%d")
#         f.write(f"CELL_DATA {n_poly}\n")
#         for name, (values, dtype) in cell_scalars.items():
#             f.write(f"SCALARS {name} {dtype} 1\nLOOKUP_TABLE default\n")
#             if dtype == "int":
#                 np.savetxt(f, np.asarray(values).astype(np.int64), fmt="%d")
#             else:
#                 np.savetxt(f, np.nan_to_num(np.asarray(values, dtype=float)), fmt="%.4f")


# def class_summary(face_df: pd.DataFrame) -> pd.DataFrame:
#     """Area-weighted hourly means per surface_group / surface_type."""
#     cols = ["T_surface_C", "I_total_Wm2", "sunlit_fraction", "q_evap_out_Wm2"]
#     d = face_df[["surface_group", "surface_type", "hour", "area_m2"] + cols].copy()
#     for c in cols:
#         d[c] = d[c] * d["area_m2"]
#     keys = ["surface_group", "surface_type", "hour"]
#     g = d.groupby(keys, as_index=False).sum(numeric_only=True)
#     for c in cols:
#         g[c] = g[c] / g["area_m2"]
#     mm = face_df.groupby(keys)["T_surface_C"].agg(["min", "max"]).reset_index()
#     g = g.merge(mm, on=keys)
#     return g.rename(columns={"T_surface_C": "T_surface_mean_C", "I_total_Wm2": "I_total_mean_Wm2",
#                              "sunlit_fraction": "sunlit_fraction_mean",
#                              "q_evap_out_Wm2": "q_evap_out_mean_Wm2",
#                              "min": "T_surface_min_C", "max": "T_surface_max_C"})


# # ==========================================================================
# # 10. Plots
# # ==========================================================================

# def plot_results(face_df: pd.DataFrame, building_df: pd.DataFrame, out_prefix: str,
#                   building_id: Optional[str] = None):
#     import matplotlib.pyplot as plt

#     bldg_ids = building_df["building_id"].unique()
#     single = building_id if building_id is not None else (bldg_ids[0] if len(bldg_ids) == 1 else None)

#     face_sub = face_df if single is None else face_df[face_df["building_id"] == single]
#     bldg_sub = building_df if single is None else building_df[building_df["building_id"] == single]
#     # Plot area-weighted parent walls, avoiding one curve per mesh triangle.
#     if "parent_face_id" in face_sub and face_sub.parent_face_id.nunique() < face_sub.face_id.nunique():
#         keys = ["building_id", "parent_face_id", "hour"]
#         values = ["I_total_Wm2", "T_solair_C", "T_surface_C"]
#         weighted = face_sub[keys + ["area_m2"]].copy()
#         for name in values:
#             weighted[name] = face_sub[name] * face_sub.area_m2
#         grouped = weighted.groupby(keys, as_index=False).sum()
#         for name in values:
#             grouped[name] /= grouped.area_m2
#         face_sub = grouped.rename(columns={"parent_face_id": "face_id"})

#     fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

#     # --- Panel 1: incident solar flux per face vs hour (shows exposure windows) ---
#     ax = axes[0]
#     for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
#         sub = sub.sort_values("hour")
#         lab = fid if single is not None else f"{bid}:{fid}"
#         ax.plot(sub["hour"], sub["I_total_Wm2"], marker="o", ms=3, label=lab)
#     ax.set_ylabel("Incident solar flux, W/m$^2$")
#     title = "Per-face solar exposure through the day (sunlit windows)"
#     ax.set_title(title if single is None else f"{title} -- {single}")
#     ax.legend(fontsize=7, ncol=2)
#     ax.grid(alpha=0.3)

#     # --- Panel 2: surface temperature + ambient + legacy bulk estimate ---
#     ax = axes[1]
#     for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
#         sub = sub.sort_values("hour")
#         lab = f"{fid} (surface)" if single is not None else f"{bid}:{fid} (surface)"
#         ax.plot(sub["hour"], sub["T_surface_C"], marker="o", ms=3, label=lab)
#     ax.plot(bldg_sub.groupby("hour")["T_amb_C"].first().index,
#             bldg_sub.groupby("hour")["T_amb_C"].first().values,
#             "k--", lw=2, label="T_amb (air)")
#     for bid, sub in bldg_sub.groupby("building_id"):
#         sub = sub.sort_values("hour")
#         lab = "T_building (lumped, RC)" if single is not None else f"T_building {bid} (RC)"
#         ax.plot(sub["hour"], sub["T_building_C"], lw=2.5, label=lab)
#     ax.set_xlabel("Hour of day (local standard time)")
#     ax.set_ylabel("Temperature, \u00b0C")
#     ax.set_title("Surface temperature per face vs. ambient vs. building bulk temperature")
#     ax.legend(fontsize=7, ncol=2)
#     ax.grid(alpha=0.3)

#     fig.tight_layout()
#     fig.savefig(f"{out_prefix}_plot.png", dpi=150)
#     print(f"Wrote {out_prefix}_plot.png")


# # ==========================================================================
# # 11. Script entry point
# # ==========================================================================

# _SCENE_KEY_NORMS = {"buildings", "site", "settings"} | {k for ks in _CLASS_KEYS.values() for k in ks}
# _NAME_PATTERNS = {"ground": r"ground|terrain", "tree": r"tree", "pathway": r"pathway|walkway|road",
#                   "lawn": r"lawn|grass", "pond": r"pond|water"}


# def _expand_paths(values) -> list:
#     """Flatten CLI path lists; expand quoted glob patterns that the shell did not."""
#     out = []
#     for v in values or []:
#         if any(c in v for c in "*?[") and not os.path.exists(v):
#             out.extend(sorted(glob.glob(v)))
#         else:
#             out.append(v)
#     return out


# def load_scenes(paths) -> tuple:
#     """Load every scene-like JSON among `paths`; merge component lists by id.
#     Returns (scene_dict_or_None, [files used])."""
#     scenes, used = [], []
#     for p in _expand_paths(paths):
#         if not os.path.isfile(p):
#             warn(f"json: not a file, skipping: {p}")
#             continue
#         try:
#             d = load_scene(p)
#         except Exception as exc:
#             warn(f"json: cannot parse {p} ({exc}); skipping")
#             continue
#         if not isinstance(d, dict) or not any(_norm_key(k) in _SCENE_KEY_NORMS for k in d):
#             log(f"[input] json: {p} does not look like a scene file; ignored")
#             continue
#         scenes.append(d)
#         used.append(p)
#     if not scenes:
#         return None, []
#     merged = dict(scenes[0])
#     for other in scenes[1:]:
#         for key, val in other.items():
#             if isinstance(val, list):
#                 mine = list(merged.get(key) or [])
#                 seen = {x.get("id") for x in mine if isinstance(x, dict) and x.get("id") is not None}
#                 for x in val:
#                     if not (isinstance(x, dict) and x.get("id") in seen):
#                         mine.append(x)
#                 merged[key] = mine
#             elif key not in merged:
#                 merged[key] = val
#     return merged, used


# def classify_obj_inputs(args) -> tuple:
#     """Sort the OBJ arguments. `--buildings_obj *.obj` typically expands to every
#     OBJ in the folder, so files that are given explicitly for another class (or
#     whose name says ground/tree/pathway/lawn/pond) are removed from the
#     buildings list; with several building meshes left, 'refined' ones win.
#     Returns (building_obj_list, {class: path or None})."""
#     explicit = {"ground": args.ground_obj, "tree": args.trees_obj, "pathway": args.pathway_obj,
#                 "lawn": args.lawn_obj, "pond": args.pond_obj}
#     explicit_real = {os.path.realpath(p) for p in explicit.values() if p}
#     remaining = []
#     for p in _expand_paths(args.buildings_obj):
#         if os.path.realpath(p) in explicit_real:
#             continue
#         stem = os.path.basename(p).lower()
#         owner = next((c for c, pat in _NAME_PATTERNS.items() if re.search(pat, stem)), None)
#         if owner is None:
#             remaining.append(p)
#         elif not explicit[owner]:
#             explicit[owner] = p
#             log(f"[input] {p}: name says '{owner}'; using it as the {owner} surface")
#         else:
#             log(f"[input] {p}: ignored in --buildings_obj (a {owner} file was given explicitly)")
#     named = [p for p in remaining if "building" in os.path.basename(p).lower()]
#     pool = named or remaining
#     refined = [p for p in pool if "refined" in os.path.basename(p).lower()]
#     chosen = refined or pool
#     if len(chosen) > 1:
#         log(f"[input] several building meshes found, merging: {', '.join(chosen)}")
#     return chosen, explicit


# def _day_tags(dirs: list) -> list:
#     seen, out = {}, []
#     for d in dirs:
#         t = os.path.basename(os.path.normpath(d)) or "day"
#         n = seen.get(t, 0)
#         seen[t] = n + 1
#         out.append(t if n == 0 else f"{t}_{n}")
#     return out


# _FORK_STATE = {}


# def _day_worker(job):
#     return _FORK_STATE["fn"](*job)


# def run_days(day_fn, jobs: list, workers: int) -> list:
#     """jobs = [(weather_dir, tag)]. Several days + workers>1: one forked process
#     per day (geometry/BVH are inherited, not rebuilt). Otherwise days run in
#     order and `workers` threads trace shadow rays inside each day."""
#     if len(jobs) > 1 and workers > 1 and "fork" in multiprocessing.get_all_start_methods():
#         _FORK_STATE["fn"] = lambda wd, tag: day_fn(wd, tag, 1)
#         results = []
#         try:
#             with concurrent.futures.ProcessPoolExecutor(
#                     max_workers=min(workers, len(jobs)),
#                     mp_context=multiprocessing.get_context("fork")) as ex:
#                 futures = [ex.submit(_day_worker, j) for j in jobs]
#                 for j, fut in zip(jobs, futures):
#                     try:
#                         results.append(fut.result())
#                     except Exception as exc:
#                         results.append((j[1] or j[0], False, f"worker failed: {exc!r}"))
#         finally:
#             _FORK_STATE.clear()
#         return results
#     return [day_fn(wd, tag, workers) for wd, tag in jobs]


# def _out_prefix(args, tag) -> str:
#     return os.path.join(args.output_dir, f"{args.out_prefix}_{tag}" if tag else args.out_prefix)


# def _log_summary(label: str, face_df: pd.DataFrame) -> None:
#     s = class_summary(face_df)
#     for (grp, typ), sub in s.groupby(["surface_group", "surface_type"]):
#         hot, cold = sub.loc[sub["T_surface_mean_C"].idxmax()], sub.loc[sub["T_surface_mean_C"].idxmin()]
#         log(f"[day {label}] {grp}/{typ}: area {sub['area_m2'].iloc[0]:.1f} m2, mean T_surface "
#             f"max {hot['T_surface_mean_C']:.1f} C at {hot['hour']:g} h, "
#             f"min {cold['T_surface_mean_C']:.1f} C at {cold['hour']:g} h")


# def _write_day_csvs(prefix: str, face_df, building_df) -> None:
#     face_df.to_csv(f"{prefix}_faces_hourly.csv", index=False)
#     building_df.to_csv(f"{prefix}_building_hourly.csv", index=False)
#     class_summary(face_df).to_csv(f"{prefix}_class_summary.csv", index=False)
#     log(f"Wrote {prefix}_faces_hourly.csv, {prefix}_building_hourly.csv, {prefix}_class_summary.csv")


# def make_obj_day_fn(bundle: ObjBundle, args):
#     def day_fn(weather_dir, tag, threads):
#         label = tag or os.path.basename(os.path.normpath(weather_dir))
#         t0 = time.time()
#         try:
#             log(f"[day {label}] start: {weather_dir} (threads={threads})")
#             analysis = solve_obj_day(
#                 bundle, weather_dir, t_initial_c=args.initial_temp_c,
#                 ground_initial_c=args.ground_initial_temp_c, ground_albedo=args.ground_albedo
#                 if args.ground_albedo is not None else DEFAULT_GROUND_ALBEDO,
#                 shadow_samples=args.shadow_samples, workers=threads)
#             face_df, building_df, faces = analysis
#             log(f"[day {label}] solved {len(face_df)} cell-hours in {time.time() - t0:.0f} s")
#             prefix = _out_prefix(args, tag)
#             _write_day_csvs(prefix, face_df, building_df)
#             _log_summary(label, face_df)
#             log(f"[day {label}] CSV written at {time.time() - t0:.0f} s")
#             if args.vtk_out is not None:
#                 export_vtk_series_from_analysis(
#                     analysis, os.path.join(args.vtk_out, tag) if tag else args.vtk_out)
#             if args.plot and len(building_df):
#                 try:
#                     plot_results(face_df, building_df, prefix)
#                 except Exception as exc:
#                     warn(f"plot skipped: {exc}")
#             return label, True, f"done in {time.time() - t0:.0f} s"
#         except Exception as exc:
#             log(f"[day {label}] FAILED: {exc!r}\n{traceback.format_exc()}")
#             return label, False, repr(exc)
#     return day_fn


# def make_json_day_fn(scene: dict, args, wall_settings):
#     def day_fn(weather_dir, tag, threads):
#         label = tag or (os.path.basename(os.path.normpath(weather_dir)) if weather_dir else "clear-sky")
#         t0 = time.time()
#         try:
#             log(f"[day {label}] start (JSON pipeline)")
#             if weather_dir is not None:
#                 if args.vtk_out is not None:
#                     export_vtk_series(scene, weather_dir,
#                                       os.path.join(args.vtk_out, tag) if tag else args.vtk_out,
#                                       consider_shadows=not args.no_shadows,
#                                       wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
#                 face_df, building_df, faces = run_analysis_from_folder(
#                     scene, weather_dir, consider_shadows=not args.no_shadows,
#                     wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
#             else:
#                 if args.vtk_out is not None:
#                     warn("--vtk_out needs --weather_dir/--weather_root; VTK skipped")
#                 face_df, building_df, faces = run_analysis(
#                     scene, date(args.year, args.month, args.day), load_weather(args.weather_csv),
#                     utc_offset_hr=args.utc_offset, consider_shadows=not args.no_shadows,
#                     wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
#             prefix = _out_prefix(args, tag)
#             _write_day_csvs(prefix, face_df, building_df)
#             _log_summary(label, face_df)
#             if args.plot:
#                 plot_results(face_df, building_df, prefix)
#             return label, True, f"done in {time.time() - t0:.0f} s"
#         except Exception as exc:
#             log(f"[day {label}] FAILED: {exc!r}\n{traceback.format_exc()}")
#             return label, False, repr(exc)
#     return day_fn


# def build_arg_parser():
#     import argparse
#     p = argparse.ArgumentParser(
#         description="Solar, conduction and storage model for buildings, ground, pathways, "
#                     "lawns, ponds and trees. Every input is optional; whatever is present "
#                     "and valid is solved.")
#     p.add_argument("--json", nargs="*", default=None,
#                    help="Scene JSON file(s) (globs fine). Used for material properties, and "
#                         "as the geometry source when no OBJ is given. Default: building_scene.json")
#     p.add_argument("--buildings_obj", nargs="*", default=None,
#                    help="Building OBJ(s). A shell glob like ../triSurface/*.obj is fine: files "
#                         "given for other classes (or named ground/tree/pathway/lawn/pond) are "
#                         "sorted out automatically.")
#     p.add_argument("--ground_obj", default=None)
#     p.add_argument("--trees_obj", default=None,
#                    help="Tree OBJ: opaque shadow casters AND solved canopy surfaces.")
#     p.add_argument("--pathway_obj", default=None)
#     p.add_argument("--lawn_obj", default=None)
#     p.add_argument("--pond_obj", default=None)
#     p.add_argument("--shadow_samples", type=int, choices=(1, 3), default=3,
#                    help="Rays per OBJ triangle: 3 (default) estimates partial shade; 1 uses the centroid.")
#     p.add_argument("--weather_csv", default=None,
#                    help="CSV hour,T_amb_C,wind_speed_ms[,...] (clear-sky path, JSON geometry only)")
#     p.add_argument("--weather_dir", nargs="*", default=None,
#                    help="One or more weatherData/MMDD folders (a glob like weatherData/*/. is fine); "
#                         "each is solved as a separate day.")
#     p.add_argument("--weather_root", default=None,
#                    help="weatherData/ root; combined with --month/--day to find MMDD")
#     p.add_argument("--month", type=int, default=6)
#     p.add_argument("--day", type=int, default=21)
#     p.add_argument("--year", type=int, default=2026)
#     p.add_argument("--utc_offset", type=float, default=0.0)
#     p.add_argument("--out_prefix", default="results")
#     p.add_argument("--output_dir", default=".", help="Directory for CSV/PNG outputs.")
#     p.add_argument("--vtk_out", default=None,
#                    help="Directory for the ParaView time series (a sub-folder per day when "
#                         "several weather days are given).")
#     p.add_argument("--vtk_binary", action="store_true",
#                    help="Write binary legacy VTK (several times faster and smaller; ParaView reads it "
#                         "natively). Default is ASCII.")
#     p.add_argument("--workers", type=int, default=1,
#                    help="Several weather days: that many days run in parallel (processes). "
#                         "A single day: that many threads trace shadow rays.")
#     p.add_argument("--plot", action="store_true")
#     p.add_argument("--no_shadows", action="store_true", help="Skip shadow casting (faster).")
#     p.add_argument("--no_overlap_removal", action="store_true",
#                    help="Keep ground cells that lie under lawn/pond/pathway cells.")
#     p.add_argument("--wall_thickness_m", type=float, default=0.23)
#     p.add_argument("--wall_nodes", type=int, default=5)
#     p.add_argument("--indoor_temp_c", type=float, default=24.0)
#     p.add_argument("--indoor_h", type=float, default=8.0)
#     p.add_argument("--initial_temp_c", type=float, default=None,
#                    help="Initial uniform building slab temperature; default --indoor_temp_c.")
#     # Overrides. Left unset (None) so JSON / built-in defaults apply.
#     p.add_argument("--ground_depth_m", type=float, default=None)
#     p.add_argument("--ground_nodes", type=int, default=None)
#     p.add_argument("--deep_ground_temp_c", type=float, default=None,
#                    help="Fixed deep temperature below ground/pathway/lawn/pond layers (default 24).")
#     p.add_argument("--ground_deep_h", type=float, default=None)
#     p.add_argument("--ground_initial_temp_c", type=float, default=None,
#                    help="Initial ground/pathway/lawn/pond temperature; default is the deep temperature.")
#     p.add_argument("--ground_conductivity", type=float, default=None)
#     p.add_argument("--ground_density", type=float, default=None)
#     p.add_argument("--ground_specific_heat", type=float, default=None)
#     p.add_argument("--ground_albedo", type=float, default=None)
#     p.add_argument("--ground_emissivity", type=float, default=None)
#     p.add_argument("--lawn_wetness", type=float, default=None,
#                    help="Lawn evaporation as a fraction of free-water evaporation (default 0.25).")
#     p.add_argument("--tree_wetness", type=float, default=None,
#                    help="Canopy transpiration fraction (default 0.30).")
#     p.add_argument("--pond_depth_m", type=float, default=None, help="Water depth (default 0.5).")
#     p.add_argument("--pond_mixing_factor", type=float, default=None,
#                    help="Pond effective conductivity = factor x 0.6 W/m/K, representing wind and "
#                         "convective mixing (default 30; 1 = still water, which heats only a thin skin).")
#     return p


# def main(argv=None) -> int:
#     parser = build_arg_parser()
#     args = parser.parse_args(argv)
#     args.workers = max(1, args.workers)
#     global VTK_BINARY
#     VTK_BINARY = args.vtk_binary
#     if args.pond_mixing_factor is not None and not (math.isfinite(args.pond_mixing_factor)
#                                                    and args.pond_mixing_factor > 0):
#         parser.error("--pond_mixing_factor must be a positive number")
#     os.makedirs(args.output_dir, exist_ok=True)
#     try:
#         wall_settings = WallSettings(args.wall_thickness_m, args.wall_nodes,
#                                      args.indoor_temp_c, args.indoor_h)
#     except ValueError as error:
#         parser.error(str(error))

#     # ---- inputs -----------------------------------------------------------
#     json_args = args.json if args.json else (["building_scene.json"]
#                                              if os.path.isfile("building_scene.json") else [])
#     scene, scene_files = load_scenes(json_args)
#     log(f"[input] scene JSON: {', '.join(scene_files) if scene_files else 'none'}")
#     building_objs, class_objs = classify_obj_inputs(args)

#     weather_dirs, seen = [], set()
#     for d in _expand_paths(args.weather_dir):
#         rp = os.path.realpath(d)
#         if os.path.isdir(d) and rp not in seen:
#             seen.add(rp)
#             weather_dirs.append(d)
#         elif not os.path.isdir(d):
#             warn(f"weather_dir: not a directory, skipping: {d}")
#     if not weather_dirs and args.weather_root:
#         try:
#             weather_dirs = [resolve_weather_folder(args.weather_root, args.month, args.day)]
#         except FileNotFoundError as exc:
#             warn(str(exc))
#     log(f"[input] weather days: {len(weather_dirs)}  workers: {args.workers}")

#     overrides = {
#         "ground": dict(depth_m=args.ground_depth_m, nodes=args.ground_nodes,
#                        deep_h_wm2k=args.ground_deep_h, thermal_conductivity=args.ground_conductivity,
#                        density=args.ground_density, specific_heat=args.ground_specific_heat,
#                        albedo=args.ground_albedo, emissivity=args.ground_emissivity,
#                        deep_temperature_c=args.deep_ground_temp_c),
#         "pathway": dict(deep_temperature_c=args.deep_ground_temp_c),
#         "lawn": dict(deep_temperature_c=args.deep_ground_temp_c, wetness=args.lawn_wetness),
#         "pond": dict(deep_temperature_c=args.deep_ground_temp_c, depth_m=args.pond_depth_m,
#                      mixing_factor=args.pond_mixing_factor),
#         "tree": dict(wetness=args.tree_wetness),
#     }

#     # ---- choose pipeline --------------------------------------------------
#     day_fn = None
#     jobs = []
#     any_obj = bool(building_objs) or any(class_objs.values())
#     if any_obj:
#         if not weather_dirs:
#             log("ERROR: the OBJ pipeline needs real weather data: pass --weather_dir or --weather_root")
#             return 2
#         bundle = prepare_obj_bundle(
#             building_objs or None, class_objs["ground"], class_objs["tree"], class_objs["pathway"],
#             class_objs["lawn"], class_objs["pond"], scene=scene, wall_settings=wall_settings,
#             layer_overrides=overrides, consider_shadows=not args.no_shadows,
#             remove_overlaps=not args.no_overlap_removal)
#         if bundle.n_cells:
#             log(f"[input] OBJ pipeline: {bundle.n_cells} cells in total")
#             day_fn = make_obj_day_fn(bundle, args)
#         else:
#             warn("no valid OBJ geometry found")
#     if day_fn is None:
#         if scene is None or not scene.get("buildings"):
#             log("ERROR: nothing to solve: no valid OBJ files and no buildings in a scene JSON")
#             return 2
#         log("[input] JSON pipeline (buildings from the scene JSON; shadows from buildings and "
#             "tree spheres; lawns/ponds/ground need OBJ files)")
#         day_fn = make_json_day_fn(scene, args, wall_settings)
#     if weather_dirs:
#         tags = _day_tags(weather_dirs) if len(weather_dirs) > 1 else [None]
#         jobs = list(zip(weather_dirs, tags))
#     else:
#         jobs = [(None, None)]

#     results = run_days(day_fn, jobs, args.workers)
#     failed = [r for r in results if not r[1]]
#     for label, ok, msg in results:
#         log(f"[result] {label}: {'OK' if ok else 'FAILED'} - {msg}")
#     return 1 if failed and len(failed) == len(results) else 0


# if __name__ == "__main__":
#     sys.exit(main())








#Subrat: Below code works fine for buildings only, considering that trees are considered if they are a part of the obj. The material properties are considered from the json file directly.

# """
# Analytical solar-exposure + convective/radiative surface-temperature model
# for buildings exported from the scene JSON (SNU-SPH / urban-microclimate
# style export: buildings[].faces[] with vertices, normal, surfaceType,
# materialProperties.thermalProperties; optionally trees[] for canopy shading).

# No CFD, and no full (multi-bounce) ray tracing -- everything below is
# geometry plus visibility rays toward the sun at each weather time:

#   1. SOLAR GEOMETRY  (Cooper's declination + standard hour-angle equations)
#      -> sun altitude & azimuth for any date / time / lat-lon.

#   2. CLEAR-SKY IRRADIANCE  (Hottel, 1976 atmospheric-transmittance model)
#      -> direct-normal (I_DN) and diffuse-horizontal (I_dH) irradiance from
#         the solar altitude alone -- used only when you DON'T have a real
#         weather-data export (see `run_analysis`). If you DO have one (a
#         `weatherData/MMDD/` folder with Tambient, windSpeed, IDN, Idif,
#         sunPosVector, ...), use `run_analysis_from_folder` instead: it
#         reads the real sun-position vector and real DNI/diffuse straight
#         from those files and skips this model entirely -- more accurate,
#         and removes any ambiguity about axis conventions since the sun
#         vector is already given in the same (East, North, Up) axes as the
#         building JSON's face normals.

#   3. PER-FACE SUNLIT TEST  -- cos(theta) = dot(face_normal, sun_unit_vector).
#      A face receives direct beam radiation only when cos(theta) > 0 AND the
#      sun is above the horizon (self-orientation), AND (new) a visibility ray
#      from the face toward the sun is not blocked by another building or tree
#      in the scene (mutual shading -- see Section 5, "Shadow occlusion").

#   4. CONVECTION -- classic linear wind correlation h_c = a + b*V (McAdams-
#      type, W/m^2K), applied per face using your wind-speed time series.

#   5. SURFACE TEMPERATURE: each exposed building or requested ground triangle
#      is a 0.23 m slab with five nodes through its thickness, heat capacity
#      rho*cp*L per unit area, and
#      conductance k/dx between adjacent nodes. A matrix exponential advances
#      conduction/storage exactly for the spatially discretized system, holding
#      weather and ambient-linearized longwave coefficients constant between
#      records. A building inner face exchanges heat with fixed 24 C indoor air;
#      a ground back face exchanges heat with fixed deep soil. These settings,
#      materials, and initial temperatures are configurable.
#      T_solair_C remains an equivalent driving temperature for the bulk model.

#   6. WHOLE-BUILDING TEMPERATURE -- single-node RC (lumped capacitance,
#      C = rho*V*cp) network per building, driven by that building's own
#      faces' sol-air temperatures. Because C and the effective conductance
#      h_eff are taken as constant within each hour, the ODE
#      C dT/dt = h_eff*(T_eq - T) has the exact analytical solution

#          T(t+dt) = T_eq + (T(t) - T_eq) * exp(-dt / tau),   tau = C/h_eff

#      so the whole march-forward is a closed-form exponential step, not a
#      numerical ODE integration. Each building in the scene gets its own
#      node (its own C, its own faces) -- they are only coupled through the
#      shadows they cast on each other, not thermally.

# Assumptions you should sanity-check against your own scene / site
# (flagged again inline where they are used):
#   - Scene axes are treated as ENU: +x = East, +y = North, +z = Up. If your
#     tool defines "north" differently, set `north_offset_deg` below.
#   - Longwave emissivity is not in the JSON -> defaulted to 0.9 (typical
#     concrete/masonry). Change `DEFAULT_EMISSIVITY` if you know better.
#   - Ground/surroundings albedo for the reflected-radiation term is not in
#      the JSON -> defaulted to 0.2. Change `DEFAULT_GROUND_ALBEDO`.
#   - The optional OBJ ground calculation uses the case soil defaults k=1.5
#     W/m/K, rho=1150 kg/m3 and cp=650 J/kg/K. Its active depth is 0.23 m and
#     its default lower conductance k/L connects it to fixed 24 C deep soil.
#   - The floor face (surfaceType == "floor") is excluded from solar +
#     outdoor convection (it's in contact with the ground, not outdoor air).
#     Toggle `INCLUDE_FLOOR` if you want to add a simple ground-conduction
#     term instead. A "ceiling" face (surfaceType == "ceiling") is *always*
#     excluded the same way, with no toggle -- in multi-floor exports (one
#     JSON "building" per story, stacked in z) this tags the boundary between
#     two stacked stories, which never touches outdoor air regardless of
#     settings. Only "wall" and "roof" faces are treated as always outdoor-
#     exposed.
#   - Two of the face normals in the source JSON point the "wrong" way for
#     what they geometrically are (roof normal given as (0,0,-1), i.e.
#     inward, floor as (0,0,1), also inward). This script auto-corrects
#     normal sign using `surfaceType` so the physics is right regardless of
#     what the exporter wrote. Also, wall face "area" was exported as 0 in
#     this file -- areas are recomputed from the vertices, never trusted
#     from the JSON.
#   - The OBJ path retains individual mesh triangles, with three visibility
#     samples per triangle by default, accelerated by a triangle BVH. The JSON
#     path still tests one centroid per polygon. Only the direct
#     beam is blockable; the diffuse-sky and ground-reflected components
#     still use the unobstructed sky-view-factor formulas (a standard
#     simplification -- properly reducing diffuse view factors for nearby
#     obstructions would need a more expensive hemispherical integration).
#   - In the JSON path, tree canopies are approximated as a bounding sphere for shadow
#     purposes (a common simplification in urban solar-access studies), not
#     actual leaf/branch geometry. The JSON field names used to build that
#     sphere are guessed from common conventions (see `build_tree_occluder`)
#     since no example tree entry was available when this was written --
#     check `_TREE_*_KEYS` against your real tree schema and adjust if needed.
#     The OBJ path traces the actual tree triangles, treating them as opaque.
# """

# import json
# import math
# import os
# import re
# from dataclasses import dataclass, field
# from datetime import date
# from typing import Optional

# import numpy as np
# import pandas as pd

# # --------------------------------------------------------------------------
# # Tunable defaults (not present in the JSON -- see docstring above)
# # --------------------------------------------------------------------------
# DEFAULT_EMISSIVITY = 0.90
# DEFAULT_GROUND_ALBEDO = 0.20
# STEFAN_BOLTZMANN = 5.670374e-8          # W/m^2/K^4
# SOLAR_CONSTANT = 1367.0                 # W/m^2
# INCLUDE_FLOOR = False                   # floor treated as adiabatic (ground contact)


# # ==========================================================================
# # 1. Geometry: load scene, fix normals, compute real areas
# # ==========================================================================

# @dataclass
# class Face:
#     building_id: str
#     face_id: str
#     surface_type: str
#     normal: np.ndarray          # outward unit normal, corrected
#     area: float                 # m^2, computed from vertices (JSON area ignored)
#     tilt_deg: float             # 0 = horizontal facing up, 90 = vertical, 180 = facing down
#     center: np.ndarray
#     vertices: np.ndarray        # (N,3) planar polygon loop, as given (for geometry export)
#     thermal_conductivity: float
#     density: float
#     specific_heat: float
#     albedo: float
#     emissivity: float = DEFAULT_EMISSIVITY
#     parent_face_id: str = ""    # logical wall/roof; OBJ physics stays per triangle


# def _polygon_area_3d(verts: np.ndarray) -> float:
#     """Area of a planar polygon in 3D via Newell's method (cross-product sum)."""
#     n = len(verts)
#     normal_sum = np.zeros(3)
#     for i in range(n):
#         v1 = verts[i]
#         v2 = verts[(i + 1) % n]
#         normal_sum += np.cross(v1, v2)
#     return 0.5 * np.linalg.norm(normal_sum)


# def load_scene(json_path: str) -> dict:
#     with open(json_path, "r") as f:
#         return json.load(f)


# def extract_faces(scene: dict) -> list:
#     """Flatten every building's faces into a list of corrected Face objects."""
#     faces = []
#     for b in scene.get("buildings", []):
#         bid = b.get("id", "building")
#         for f in b.get("faces", []):
#             verts = np.array([[v["x"], v["y"], v["z"]] for v in f["vertices"]])
#             area = _polygon_area_3d(verts)          # never trust JSON "area"
#             n = np.array([f["normal"]["x"], f["normal"]["y"], f["normal"]["z"]], dtype=float)
#             n = n / (np.linalg.norm(n) + 1e-12)

#             stype = f.get("surfaceType", "wall")
#             # --- fix inward-pointing roof/floor normals (see module docstring) ---
#             if stype == "roof" and n[2] < 0:
#                 n = -n
#             if stype == "floor" and n[2] > 0:
#                 n = -n

#             tilt_deg = math.degrees(math.acos(np.clip(n[2], -1.0, 1.0)))
#             c = f.get("center", {"x": 0, "y": 0, "z": 0})
#             center = np.array([c["x"], c["y"], c["z"]])

#             tp = f["materialProperties"]["thermalProperties"]
#             faces.append(Face(
#                 building_id=bid,
#                 face_id=f["id"],
#                 surface_type=stype,
#                 normal=n,
#                 area=area,
#                 tilt_deg=tilt_deg,
#                 center=center,
#                 vertices=verts,
#                 thermal_conductivity=tp["thermal_conductivity"],
#                 density=tp["density"],
#                 specific_heat=tp["specific_heat"],
#                 albedo=tp["albedo"],
#             ))
#     return faces


# def building_volume(scene: dict, building_id: Optional[str] = None) -> float:
#     for b in scene.get("buildings", []):
#         if building_id is None or b.get("id") == building_id:
#             geo = b.get("geometry", {})
#             if "volume" in geo:
#                 return geo["volume"]
#             return geo.get("footprintArea", 0) * b.get("height", 0)
#     return 0.0


# def site_location(scene: dict) -> tuple:
#     """Returns (lat_deg, lon_deg) -- prefers shadowSettings, falls back to site centroid."""
#     ss = scene.get("settings", {}).get("shadowSettings")
#     if ss and "latitude" in ss:
#         return ss["latitude"], ss["longitude"]
#     coords = scene["site"]["properties"]["originalCoordinates"]
#     lat = float(np.mean([c["lat"] for c in coords]))
#     lon = float(np.mean([c["lng"] for c in coords]))
#     return lat, lon


# # ==========================================================================
# # 2. Solar geometry
# # ==========================================================================

# def solar_position(lat_deg: float, lon_deg: float, day_of_year: int,
#                     local_std_hour: float, utc_offset_hr: float = 0.0) -> tuple:
#     """
#     Standard solar-position equations (Cooper's declination + hour angle).
#     Returns (altitude_deg, azimuth_deg); azimuth measured clockwise from
#     true North (0=N, 90=E, 180=S, 270=W). altitude < 0 => sun below horizon.
#     """
#     lat = math.radians(lat_deg)
#     n = day_of_year

#     decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0)))

#     # Equation of time (minutes), Duffie & Beckman form
#     B = math.radians(360.0 * (n - 81) / 364.0)
#     eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

#     std_meridian = 15.0 * utc_offset_hr
#     time_corr_min = 4.0 * (lon_deg - std_meridian) + eot
#     solar_time = local_std_hour + time_corr_min / 60.0

#     hour_angle = math.radians(15.0 * (solar_time - 12.0))

#     sin_alt = (math.sin(lat) * math.sin(decl) +
#                math.cos(lat) * math.cos(decl) * math.cos(hour_angle))
#     altitude = math.asin(np.clip(sin_alt, -1.0, 1.0))

#     # Azimuth, measured clockwise from true North (standard NOAA-style form):
#     #   cos(A) = (sin(decl) - sin(lat)*sin(alt)) / (cos(lat)*cos(alt))
#     # acos() only returns 0-180 deg, so mirror into 180-360 for the
#     # afternoon (hour_angle > 0) to get the correct NE-at-sunrise /
#     # NW-at-sunset quadrant.
#     cos_alt = max(math.cos(altitude), 1e-9)
#     cos_A = (math.sin(decl) - math.sin(lat) * math.sin(altitude)) / (math.cos(lat) * cos_alt)
#     az_raw = math.degrees(math.acos(np.clip(cos_A, -1.0, 1.0)))
#     azimuth = az_raw if hour_angle <= 0 else 360.0 - az_raw

#     return math.degrees(altitude), azimuth


# def sun_unit_vector(altitude_deg: float, azimuth_deg: float,
#                      north_offset_deg: float = 0.0) -> np.ndarray:
#     """
#     Sun direction as a unit vector in scene coordinates, assuming
#     +x = East, +y = North, +z = Up (rotate with north_offset_deg if your
#     scene's "north" is not aligned with +y).
#     """
#     alt = math.radians(altitude_deg)
#     az = math.radians(azimuth_deg - north_offset_deg)
#     east = math.cos(alt) * math.sin(az)
#     north = math.cos(alt) * math.cos(az)
#     up = math.sin(alt)
#     return np.array([east, north, up])


# # ==========================================================================
# # 3. Clear-sky irradiance (Hottel, 1976)
# # ==========================================================================

# def hottel_clear_sky(altitude_deg: float, site_altitude_km: float = 0.0,
#                       r0: float = 1.0, r1: float = 1.0, rk: float = 1.0) -> tuple:
#     """
#     Hottel's (1976) atmospheric-transmittance clear-sky model.
#     r0, r1, rk are the climate-type correction factors from Hottel's paper
#     (defaults =1.0, i.e. the reference "mid-latitude summer" atmosphere --
#     nudge by a few % for your own climate/visibility if you know it).
#     Returns (I_DN, I_dH) in W/m^2 -- direct-normal and diffuse-horizontal.
#     Returns (0, 0) when the sun is below the horizon.
#     """
#     if altitude_deg <= 0:
#         return 0.0, 0.0
#     A = site_altitude_km
#     sin_b = math.sin(math.radians(altitude_deg))

#     a0 = r0 * (0.4237 - 0.00821 * (6.0 - A) ** 2)
#     a1 = r1 * (0.5055 + 0.00595 * (6.5 - A) ** 2)
#     k = rk * (0.2711 + 0.01858 * (2.5 - A) ** 2)

#     tau_b = a0 + a1 * math.exp(-k / sin_b)
#     tau_d = 0.2710 - 0.2939 * tau_b   # Liu-Jordan diffuse-transmittance approx

#     i_dn = SOLAR_CONSTANT * tau_b
#     i_dh = SOLAR_CONSTANT * tau_d * sin_b
#     return max(i_dn, 0.0), max(i_dh, 0.0)


# def sky_temperature(t_amb_k: float) -> float:
#     """Swinbank (1963) clear-sky effective sky temperature, in Kelvin."""
#     return 0.0552 * t_amb_k ** 1.5


# # ==========================================================================
# # 4. Per-face irradiance & sol-air temperature
# # ==========================================================================

# def face_irradiance(face: Face, sun_vec: np.ndarray, altitude_deg: float,
#                      i_dn: float, i_dh: float,
#                      ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                      external_shadow: bool = False) -> dict:
#     """
#     Direct + diffuse (isotropic sky) + ground-reflected irradiance on one face.
#     `external_shadow=True` means a ray/occlusion test (Section 5) found this
#     face blocked by another building or tree this hour -- the direct beam is
#     then zeroed regardless of the face's own orientation toward the sun.
#     """
#     cos_theta = float(np.dot(face.normal, sun_vec))
#     self_lit = (altitude_deg > 0) and (cos_theta > 0)
#     sunlit = self_lit and not external_shadow

#     i_direct = i_dn * cos_theta if sunlit else 0.0

#     tilt = math.radians(face.tilt_deg)
#     view_sky = (1 + math.cos(tilt)) / 2.0     # 1 for roof, 0.5 for wall, 0 for floor
#     view_ground = 1.0 - view_sky
#     i_diffuse = i_dh * view_sky

#     i_gh = i_dn * math.sin(math.radians(max(altitude_deg, 0.0))) + i_dh
#     i_reflected = ground_albedo * i_gh * view_ground

#     return dict(sunlit=sunlit, self_lit=self_lit, shadowed=external_shadow,
#                 cos_theta=cos_theta, view_sky=view_sky,
#                 i_direct=i_direct, i_diffuse=i_diffuse, i_reflected=i_reflected,
#                 i_total=i_direct + i_diffuse + i_reflected)


# def convective_h(wind_speed_ms: float, a: float = 5.7, b: float = 3.8) -> float:
#     """McAdams-type linear wind correlation (W/m^2K)."""
#     return a + b * max(wind_speed_ms, 0.0)


# def sol_air_temperature(t_amb_c: float, i_total: float, h_c: float,
#                          albedo: float, view_sky: float,
#                          emissivity: float = DEFAULT_EMISSIVITY) -> float:
#     """
#     ASHRAE sol-air temperature: the fictitious outdoor air temperature that
#     would give the same solar + convective + long-wave heat gain as the
#     real conditions, so a face just "sees" T_sol_air via plain convection.
#     """
#     alpha = 1.0 - albedo
#     t_amb_k = t_amb_c + 273.15
#     t_sky_k = sky_temperature(t_amb_k)
#     # net long-wave LOSS to the sky, scaled by how much sky this face can see
#     q_lw_loss = view_sky * emissivity * STEFAN_BOLTZMANN * (t_amb_k ** 4 - t_sky_k ** 4)
#     return t_amb_c + (alpha * i_total - q_lw_loss) / h_c


# def surface_temperature(t_amb_c, i_total, h_c, albedo, view_sky,
#                         emissivity=DEFAULT_EMISSIVITY):
#     """Local steady exterior temperature (C), with no wall conduction/storage.

#     Solve alpha*I = h_c*(Ts-Ta) + eps*sigma*(Ts^4 - Fsky*Tsky^4
#                                                 - (1-Fsky)*Ta^4).
#     Ground/surroundings are at ambient temperature; sky uses Swinbank.
#     Supports scalar or broadcastable array inputs. Unlike sol-air temperature,
#     outgoing longwave radiation is evaluated at the actual surface temperature.
#     """
#     ta = np.asarray(t_amb_c) + 273.15
#     absorbed = (1.0 - np.asarray(albedo)) * np.asarray(i_total)
#     eps_sigma = np.asarray(emissivity) * STEFAN_BOLTZMANN
#     background = np.asarray(view_sky) * sky_temperature(ta)**4 + (1-np.asarray(view_sky))*ta**4
#     ts = np.asarray(ta + absorbed / h_c)
#     for _ in range(30):
#         residual = h_c*(ts-ta) + eps_sigma*(ts**4-background) - absorbed
#         step = residual / (h_c + 4*eps_sigma*ts**3)
#         ts = ts - step
#         if np.max(np.abs(step), initial=0.0) < 1e-8:
#             break
#     return ts - 273.15


# @dataclass(frozen=True)
# class WallSettings:
#     """Uniform slab settings for exposed walls and roofs; SI units except C."""
#     thickness_m: float = 0.23
#     nodes: int = 5
#     indoor_temperature_c: float = 24.0
#     indoor_h_wm2k: float = 8.0

#     def __post_init__(self):
#         if not np.isfinite(self.thickness_m) or self.thickness_m <= 0:
#             raise ValueError("wall thickness must be finite and positive")
#         if not isinstance(self.nodes, int) or self.nodes < 2:
#             raise ValueError("wall nodes must be an integer >= 2")
#         if not np.isfinite(self.indoor_temperature_c) or self.indoor_temperature_c <= -273.15:
#             raise ValueError("indoor temperature must be finite and above absolute zero")
#         if not np.isfinite(self.indoor_h_wm2k) or self.indoor_h_wm2k < 0:
#             raise ValueError("indoor heat-transfer coefficient must be finite and nonnegative")


# @dataclass(frozen=True)
# class GroundSettings:
#     """Thermal properties and lower boundary for exposed ground triangles.

#     The active soil layer uses the same nodal slab solver as a wall. Its lower
#     face exchanges heat with fixed deep-soil temperature through
#     ``deep_h_wm2k``. When that value is omitted, k/depth is used, representing
#     conduction through one additional layer of the same thickness.
#     """
#     depth_m: float = 0.23
#     nodes: int = 5
#     deep_temperature_c: float = 24.0
#     deep_h_wm2k: Optional[float] = None
#     thermal_conductivity: float = 1.5
#     density: float = 1150.0
#     specific_heat: float = 650.0
#     albedo: float = DEFAULT_GROUND_ALBEDO
#     emissivity: float = 0.95

#     def __post_init__(self):
#         if not np.isfinite(self.depth_m) or self.depth_m <= 0:
#             raise ValueError("ground depth must be finite and positive")
#         WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
#                      self.effective_deep_h_wm2k)
#         material = np.array([self.thermal_conductivity, self.density,
#                              self.specific_heat], dtype=float)
#         if np.any(~np.isfinite(material)) or np.any(material <= 0):
#             raise ValueError("ground conductivity, density and heat capacity must be positive and finite")
#         if not np.isfinite(self.albedo) or not 0 <= self.albedo <= 1:
#             raise ValueError("ground albedo must be finite and between 0 and 1")
#         if not np.isfinite(self.emissivity) or not 0 <= self.emissivity <= 1:
#             raise ValueError("ground emissivity must be finite and between 0 and 1")

#     @property
#     def effective_deep_h_wm2k(self) -> float:
#         return (self.thermal_conductivity / self.depth_m
#                 if self.deep_h_wm2k is None else self.deep_h_wm2k)

#     def slab_settings(self) -> WallSettings:
#         return WallSettings(self.depth_m, self.nodes, self.deep_temperature_c,
#                             self.effective_deep_h_wm2k)


# class SlabConduction:
#     """Independent 1-D slabs, with an exact exponential update in time.

#     Boundary nodes lie on the actual surfaces and have half control volumes.
#     Interior nodes have full control volumes. Per-area nodal capacities sum
#     to rho*cp*L; adjacent conductance is k/dx. No lateral cell coupling.
#     Grouping equal materials lets all their triangles share a small matrix.
#     """

#     def __init__(self, conductivity, density, specific_heat, emissivity,
#                  initial_c, settings=None):
#         self.settings = settings or WallSettings()
#         properties = np.column_stack(np.broadcast_arrays(
#             conductivity, density, specific_heat, emissivity))
#         if np.any(~np.isfinite(properties)) or np.any(properties[:, :3] <= 0):
#             raise ValueError("wall conductivity, density and heat capacity must be positive and finite")
#         self.dx = self.settings.thickness_m / (self.settings.nodes-1)
#         widths = np.full(self.settings.nodes, self.dx)
#         widths[[0, -1]] *= 0.5
#         self.capacity = properties[:, 1, None]*properties[:, 2, None]*widths
#         self.conductance = properties[:, 0] / self.dx
#         self.temperatures = np.full((len(properties), self.settings.nodes), initial_c, dtype=float)
#         if not np.isfinite(initial_c) or initial_c <= -273.15:
#             raise ValueError("initial wall temperature must be finite and above absolute zero")
#         self.initial_c = initial_c
#         _, inverse = np.unique(properties, axis=0, return_inverse=True)
#         self.groups = [np.flatnonzero(inverse == i) for i in range(inverse.max()+1)]

#     def advance(self, dt_seconds, external_h, external_source):
#         """C*Tdot = -K*T + b; exterior input is source - h*T[0].

#         Input coefficients are held constant for dt. Transform with sqrt(C)
#         to a symmetric matrix, then use its eigendecomposition to evaluate
#         exp(A*dt) and the integrated forcing. No SciPy dependency or explicit
#         time stepping; zero eigenvalues are handled for insulated slabs.
#         """
#         if not np.isfinite(dt_seconds) or dt_seconds < 0:
#             raise ValueError("wall time step must be finite and nonnegative")
#         if dt_seconds == 0:
#             return
#         n = self.settings.nodes
#         external_h = np.broadcast_to(external_h, len(self.temperatures))
#         external_source = np.broadcast_to(external_source, len(self.temperatures))
#         for ids in self.groups:
#             # Radiation coefficient depends on ambient and emissivity only,
#             # so it is common to a material group within a weather interval.
#             if not np.allclose(external_h[ids], external_h[ids[0]], rtol=0, atol=1e-12):
#                 raise ValueError("external h must be equal within each material group")
#             c = self.capacity[ids[0]]
#             root_c = np.sqrt(c)
#             g = self.conductance[ids[0]]
#             K = np.diag(np.full(n, 2*g)) + np.diag(np.full(n-1, -g), 1) + np.diag(np.full(n-1, -g), -1)
#             K[0,0] = g + external_h[ids[0]]
#             K[-1,-1] = g + self.settings.indoor_h_wm2k
#             A = -K / root_c[:, None] / root_c[None, :]
#             eigenvalues, eigenvectors = np.linalg.eigh(A)
#             exponent = np.exp(eigenvalues*dt_seconds)
#             phi = np.full(n, dt_seconds, dtype=float)
#             np.divide(np.expm1(eigenvalues*dt_seconds), eigenvalues,
#                       out=phi, where=np.abs(eigenvalues) > 1e-15)
#             b = np.zeros((len(ids), n))
#             b[:,0] = external_source[ids]
#             b[:,-1] = self.settings.indoor_h_wm2k*self.settings.indoor_temperature_c
#             modes = (self.temperatures[ids]*root_c) @ eigenvectors
#             forcing = (b/root_c) @ eigenvectors
#             self.temperatures[ids] = ((modes*exponent + forcing*phi) @ eigenvectors.T) / root_c

#     def diagnostics(self, external_h, external_source):
#         """Instantaneous conductive/storage fluxes at the current output time."""
#         t = self.temperatures
#         q_outer = external_source-external_h*t[:,0]
#         q_inner = self.settings.indoor_h_wm2k*(t[:,-1]-self.settings.indoor_temperature_c)
#         q_conduction = self.conductance*(t[:,0]-t[:,1])
#         t_mean = np.sum(self.capacity*t,axis=1)/self.capacity.sum(axis=1)
#         capacity = self.capacity.sum(axis=1)
#         return dict(T_inner_C=t[:,-1].copy(), T_back_surface_C=t[:,-1].copy(),
#                     T_wall_mean_C=t_mean, T_slab_mean_C=t_mean,
#                     q_conduction_in_Wm2=q_conduction, q_indoor_out_Wm2=q_inner,
#                     q_back_out_Wm2=q_inner,
#                     q_storage_Wm2=q_outer-q_inner,
#                     q_outer_node_storage_Wm2=q_outer-q_conduction,
#                     stored_heat_Jm2=np.sum(self.capacity*(t-self.initial_c),axis=1),
#                     wall_capacity_Jm2K=capacity, slab_capacity_Jm2K=capacity)


# # ==========================================================================
# # 5. Shadow occlusion -- cheap two-phase ray tracing
# #
# #    Phase 1 (broad, O(1) per occluder): does the ray toward the sun even
# #             pass through the occluder's axis-aligned bounding box?
# #    Phase 2 (narrow, exact): only for occluders that survive phase 1 --
# #             Moller-Trumbore ray-triangle tests against the occluder's
# #             actual wall/roof triangles (buildings), or a closed-form
# #             ray-sphere test (trees, canopy approximated as a sphere).
# #
# #    This is the standard bounding-volume-culling idea from ray tracing,
# #    kept deliberately simple (no BVH/kd-tree): for scenes with up to a
# #    few hundred buildings this keeps the great majority of rays from ever
# #    touching a triangle, which is normally the expensive part.
# # ==========================================================================

# @dataclass
# class Occluder:
#     kind: str                                   # "building" or "tree"
#     source_id: str
#     aabb_min: np.ndarray
#     aabb_max: np.ndarray
#     triangles: list = field(default_factory=list)     # [(p0,p1,p2), ...] for buildings
#     sphere_center: Optional[np.ndarray] = None          # for trees
#     sphere_radius: Optional[float] = None


# def _triangulate_quad(v0, v1, v2, v3):
#     return [(v0, v1, v2), (v0, v2, v3)]


# def _fan_triangulate(verts):
#     """Fan triangulation from the first vertex. Exact for convex polygons
#     (true for simple rectangular footprints); an approximation for
#     concave ones -- flag this if you have complex (e.g. OSM-derived,
#     L-shaped) footprints and it matters for your results."""
#     return [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]


# def build_building_occluder(building_json: dict) -> Optional[Occluder]:
#     """
#     Triangulates one building's wall + roof faces for use as an occluder of
#     OTHER buildings/faces. The floor is skipped -- sitting at ground level,
#     it cannot shadow anything above it, so leaving it out is a free
#     reduction in triangle count with no effect on the result.
#     """
#     tris, all_pts = [], []
#     for f in building_json.get("faces", []):
#         if f.get("surfaceType") == "floor":
#             continue
#         verts = [np.array([v["x"], v["y"], v["z"]]) for v in f["vertices"]]
#         all_pts.extend(verts)
#         if len(verts) == 4:
#             tris.extend(_triangulate_quad(*verts))
#         elif len(verts) == 3:
#             tris.append(tuple(verts))
#         elif len(verts) > 4:
#             tris.extend(_fan_triangulate(verts))
#     if not tris:
#         return None
#     pts = np.array(all_pts)
#     return Occluder(kind="building", source_id=building_json.get("id", "building"),
#                      aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tris)


# # Field-name guesses for tree entries -- NOT verified against a real export
# # (none was available when this was written). Check these against your
# # actual tree JSON and adjust if the names differ.
# _TREE_RADIUS_KEYS = ["crownRadius", "canopyRadius", "radius"]
# _TREE_DIAMETER_KEYS = ["crownDiameter", "canopyDiameter", "diameter"]
# _TREE_HEIGHT_KEYS = ["height", "totalHeight", "crownHeight"]
# _TREE_TRUNK_KEYS = ["trunkHeight", "trunk_height"]


# def resolve_tree_geometry(tree_json: dict) -> Optional[tuple]:
#     """
#     Shared field-lookup used both for the shadow-occluder sphere and for the
#     fuller trunk+crown mesh in the ParaView export. Checks a nested
#     "dimensions" sub-object first (more precise, seen in real exports) and
#     falls back to top-level fields. Returns
#     (pos [3], trunk_height, crown_height, crown_radius) or None if no
#     recognisable position/size fields are found.
#     """
#     pos = None
#     if "position" in tree_json:
#         p = tree_json["position"]
#         pos = np.array([p.get("x", 0), p.get("y", 0), p.get("z", 0)])
#     elif all(k in tree_json for k in ("x", "y", "z")):
#         pos = np.array([tree_json["x"], tree_json["y"], tree_json["z"]])
#     if pos is None:
#         return None

#     dims = tree_json.get("dimensions", {})

#     def find(keys, diameter_keys=()):
#         for k in keys:
#             if k in dims:
#                 return dims[k]
#         for k in keys:
#             if k in tree_json:
#                 return tree_json[k]
#         for k in diameter_keys:
#             if k in dims:
#                 return dims[k] / 2.0
#         for k in diameter_keys:
#             if k in tree_json:
#                 return tree_json[k] / 2.0
#         return None

#     radius = find(_TREE_RADIUS_KEYS, _TREE_DIAMETER_KEYS)
#     height = find(_TREE_HEIGHT_KEYS)
#     trunk = find(_TREE_TRUNK_KEYS)
#     if trunk is None:
#         trunk = 0.0
#     if radius is None and height is None:
#         return None
#     if radius is None:
#         radius = 0.3 * height             # rough canopy-width/height fallback ratio
#     if height is None:
#         height = trunk + 2.5 * radius
#     crown_height = max(height - trunk, 0.1)
#     return pos, trunk, crown_height, radius


# def build_tree_occluder(tree_json: dict) -> Optional[Occluder]:
#     """
#     Approximates a tree canopy as a single bounding sphere for shadow
#     purposes -- a standard simplification in urban solar-access studies
#     (real foliage isn't meshed anyway). Returns None if no recognisable
#     position/size fields are found, rather than guessing at a made-up size.
#     """
#     geo = resolve_tree_geometry(tree_json)
#     if geo is None:
#         return None
#     pos, trunk, crown_height, radius = geo
#     canopy_center = pos + np.array([0.0, 0.0, trunk + crown_height / 2.0])
#     return Occluder(kind="tree", source_id=tree_json.get("id", "tree"),
#                      aabb_min=canopy_center - radius, aabb_max=canopy_center + radius,
#                      sphere_center=canopy_center, sphere_radius=radius)


# def build_occluders_by_building(scene: dict) -> dict:
#     """
#     Returns {building_id: [Occluder, ...]} -- for each building, the list of
#     every OTHER building's triangulated mesh plus every tree's bounding
#     sphere (a building is never tested against its own geometry).
#     """
#     buildings = scene.get("buildings", [])
#     building_occs = {}
#     for b in buildings:
#         occ = build_building_occluder(b)
#         if occ is not None:
#             building_occs[b.get("id", "building")] = occ

#     tree_occs = []
#     skipped = 0
#     for t in scene.get("trees", []):
#         occ = build_tree_occluder(t)
#         if occ is not None:
#             tree_occs.append(occ)
#         else:
#             skipped += 1
#     if skipped:
#         print(f"[shadows] Skipped {skipped} tree(s) in this scene -- no recognised "
#               f"position/size fields. Check _TREE_*_KEYS in solar_thermal_model.py "
#               f"against your tree JSON schema.")

#     by_building = {}
#     for bid in building_occs:
#         others = [o for oid, o in building_occs.items() if oid != bid]
#         by_building[bid] = others + tree_occs
#     return by_building


# def _ray_aabb_hit(origin: np.ndarray, direction: np.ndarray,
#                    box_min: np.ndarray, box_max: np.ndarray) -> bool:
#     """Standard slab test for ray-AABB intersection, t > 0 only."""
#     t_min, t_max = 0.0, np.inf
#     for i in range(3):
#         d = direction[i]
#         if abs(d) < 1e-12:
#             if origin[i] < box_min[i] or origin[i] > box_max[i]:
#                 return False
#             continue
#         t1 = (box_min[i] - origin[i]) / d
#         t2 = (box_max[i] - origin[i]) / d
#         if t1 > t2:
#             t1, t2 = t2, t1
#         t_min = max(t_min, t1)
#         t_max = min(t_max, t2)
#         if t_min > t_max:
#             return False
#     return True


# def _ray_triangle_hit(origin: np.ndarray, direction: np.ndarray,
#                        v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
#                        eps: float = 1e-9) -> bool:
#     """Moller-Trumbore ray-triangle intersection; only t > eps counts (so the
#     face's own near-coplanar geometry can never register as a self-hit)."""
#     edge1 = v1 - v0
#     edge2 = v2 - v0
#     h = np.cross(direction, edge2)
#     a = np.dot(edge1, h)
#     if -eps < a < eps:
#         return False
#     f = 1.0 / a
#     s = origin - v0
#     u = f * np.dot(s, h)
#     if u < 0.0 or u > 1.0:
#         return False
#     q = np.cross(s, edge1)
#     v = f * np.dot(direction, q)
#     if v < 0.0 or u + v > 1.0:
#         return False
#     t = f * np.dot(edge2, q)
#     return t > eps


# def _ray_sphere_hit(origin: np.ndarray, direction: np.ndarray,
#                      center: np.ndarray, radius: float, eps: float = 1e-9) -> bool:
#     oc = origin - center
#     b = np.dot(oc, direction)
#     c = np.dot(oc, oc) - radius * radius
#     disc = b * b - c
#     if disc < 0:
#         return False
#     sq = math.sqrt(disc)
#     t1, t2 = -b - sq, -b + sq
#     return t1 > eps or t2 > eps


# def is_shadowed(point: np.ndarray, sun_vec: np.ndarray, occluders: list) -> bool:
#     """
#     True if the ray from `point` toward the sun is blocked by any occluder
#     before reaching it. AABB broad-phase first (cheap); triangle/sphere
#     narrow-phase only for occluders the ray could plausibly hit.
#     """
#     origin = point  # small offset added by the caller along the face normal
#     for occ in occluders:
#         if not _ray_aabb_hit(origin, sun_vec, occ.aabb_min, occ.aabb_max):
#             continue
#         if occ.kind == "tree":
#             if _ray_sphere_hit(origin, sun_vec, occ.sphere_center, occ.sphere_radius):
#                 return True
#         else:
#             for v0, v1, v2 in occ.triangles:
#                 if _ray_triangle_hit(origin, sun_vec, v0, v1, v2):
#                     return True
#     return False


# class TriangleRayTracer:
#     """NumPy packet ray tracing over a median-split triangle BVH.

#     Rays in a packet share the sun direction. Bounds prune whole triangle
#     groups; leaves test at most 16 triangles at once. Both triangle sides
#     occlude, and positive hit distance plus an outward offset avoid self hits.
#     No optional ray-tracing libraries are required.
#     """

#     def __init__(self, triangles, leaf_size=16):
#         self.triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
#         self.nodes = []
#         if not len(self.triangles):
#             return
#         lo = self.triangles.min(axis=1)
#         hi = self.triangles.max(axis=1)
#         centers = (lo + hi) / 2

#         def build(ids):
#             idx = len(self.nodes)
#             bounds = (lo[ids].min(axis=0), hi[ids].max(axis=0))
#             self.nodes.append(None)
#             if len(ids) <= leaf_size:
#                 self.nodes[idx] = (*bounds, ids, -1, -1)
#             else:
#                 axis = np.argmax(np.ptp(centers[ids], axis=0))
#                 mid = len(ids) // 2
#                 ids = ids[np.argpartition(centers[ids, axis], mid)]
#                 left, right = build(ids[:mid]), build(ids[mid:])
#                 self.nodes[idx] = (*bounds, None, left, right)
#             return idx

#         build(np.arange(len(self.triangles)))

#     def blocked(self, origins, direction, chunk_size=8192):
#         origins = np.asarray(origins, dtype=float).reshape(-1, 3)
#         direction = np.asarray(direction, dtype=float)
#         direction = direction / np.linalg.norm(direction)
#         blocked = np.zeros(len(origins), dtype=bool)
#         if not self.nodes:
#             return blocked
#         v0 = self.triangles[:, 0]
#         e1 = self.triangles[:, 1] - v0
#         e2 = self.triangles[:, 2] - v0
#         cross = np.cross(direction, e2)
#         det = np.einsum('ij,ij->i', e1, cross)
#         valid = np.abs(det) > 1e-10
#         inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
#         for start in range(0, len(origins), chunk_size):
#             stack = [(0, np.arange(start, min(start+chunk_size, len(origins))))]
#             while stack:
#                 node, rays = stack.pop()
#                 rays = rays[~blocked[rays]]
#                 if not len(rays):
#                     continue
#                 lo, hi, ids, left, right = self.nodes[node]
#                 points = origins[rays]
#                 near = np.zeros(len(rays))
#                 far = np.full(len(rays), np.inf)
#                 for axis in range(3):
#                     if abs(direction[axis]) < 1e-12:
#                         outside = (points[:, axis] < lo[axis]-1e-9) | (points[:, axis] > hi[axis]+1e-9)
#                         far[outside] = -np.inf
#                     else:
#                         t1 = (lo[axis]-1e-9-points[:, axis]) / direction[axis]
#                         t2 = (hi[axis]+1e-9-points[:, axis]) / direction[axis]
#                         near = np.maximum(near, np.minimum(t1, t2))
#                         far = np.minimum(far, np.maximum(t1, t2))
#                 rays = rays[far >= near]
#                 if not len(rays):
#                     continue
#                 if ids is None:
#                     stack.extend([(left, rays), (right, rays)])
#                     continue
#                 ids = ids[valid[ids]]
#                 if not len(ids):
#                     continue
#                 delta = origins[rays, None, :] - v0[ids]
#                 u = np.einsum('rtj,tj->rt', delta, cross[ids]) * inv[ids]
#                 q = np.cross(delta, e1[ids])
#                 v = np.einsum('rtj,j->rt', q, direction) * inv[ids]
#                 distance = np.einsum('rtj,tj->rt', q, e2[ids]) * inv[ids]
#                 hit = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (distance > 1e-7)
#                 blocked[rays[np.any(hit, axis=1)]] = True
#         return blocked


# def triangle_sunlit_fraction(faces, sun_vec, ray_tracer, samples=3):
#     """Visible area estimate using one centroid or three equal-weight samples.

#     Resolution is the input mesh. Three interior barycentric samples provide
#     fractional direct-beam exposure for triangles crossed by a shadow edge.
#     """
#     if samples not in (1, 3):
#         raise ValueError("shadow_samples must be 1 or 3")
#     normals = np.array([f.normal for f in faces])
#     eligible = np.flatnonzero(normals @ sun_vec > 0) if sun_vec[2] > 0 else np.array([], dtype=int)
#     fractions = np.zeros(len(faces))
#     if not len(eligible):
#         return fractions
#     vertices = np.array([faces[i].vertices for i in eligible])
#     weights = np.array([[1/3, 1/3, 1/3]]) if samples == 1 else np.array([
#         [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
#     origins = np.einsum('sj,tjk->tsk', weights, vertices)
#     origins += normals[eligible, None, :] * _SHADOW_RAY_OFFSET
#     hits = ray_tracer.blocked(origins.reshape(-1, 3), sun_vec).reshape(-1, samples)
#     fractions[eligible] = 1 - hits.mean(axis=1)
#     return fractions


# # ==========================================================================
# # 6. Weather input (plug your real data in here)
# # ==========================================================================

# def load_weather(csv_path: Optional[str] = None, n_hours: int = 24,
#                   t_mean_c: float = 24.0, t_amp_c: float = 6.0,
#                   wind_mean_ms: float = 3.0) -> pd.DataFrame:
#     """
#     If csv_path is given, expects columns: hour, T_amb_C, wind_speed_ms
#     (optionally GHI_Wm2, DNI_Wm2, DHI_Wm2 to override the clear-sky model
#     with your own measured/forecast radiation).

#     If csv_path is None, a synthetic diurnal profile is generated as a
#     PLACEHOLDER only (sinusoidal T_amb, constant wind) so the pipeline is
#     runnable end-to-end -- swap this out for your real weather data.
#     """
#     if csv_path:
#         df = pd.read_csv(csv_path)
#         return df

#     hours = np.arange(n_hours)
#     # min T_amb near 05:00, max near 15:00 -- simple diurnal sine, placeholder only
#     t_amb = t_mean_c - t_amp_c * np.cos(2 * np.pi * (hours - 15) / 24.0)
#     wind = np.full(n_hours, wind_mean_ms)
#     return pd.DataFrame({"hour": hours, "T_amb_C": t_amb, "wind_speed_ms": wind})


# # --------------------------------------------------------------------------
# # Real weather-data-folder loader: OpenFOAM-style time-table export
# # (weatherData/MMDD/{Tambient, windSpeed, IDN, Idif, sunPosVector, ...})
# #
# # Each file looks like:
# #   // comment
# #   (
# #   ( <time_s> <scalar> )                    <- scalar field
# #   ( <time_s> ( <vx> <vy> <vz> ) )          <- vector field
# #   ...
# #   )
# # --------------------------------------------------------------------------

# _VEC_RE = re.compile(
#     r'\(\s*(-?[0-9.eE+-]+)\s+\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)\s*\)'
# )
# _SCALAR_RE = re.compile(r'\(\s*(-?[0-9.eE+-]+)\s+(-?[0-9.eE+-]+)\s*\)')


# def parse_foam_field(path: str) -> tuple:
#     """Parses one OpenFOAM-style time-table file. Returns (times_s, values),
#     values is a 1D array for scalar fields or an (N,3) array for vector fields."""
#     with open(path) as fh:
#         content = re.sub(r'//.*', '', fh.read())   # strip comments

#     vec = _VEC_RE.findall(content)
#     if vec:
#         times = np.array([float(m[0]) for m in vec])
#         values = np.array([[float(m[1]), float(m[2]), float(m[3])] for m in vec])
#         return times, values

#     sca = _SCALAR_RE.findall(content)
#     times = np.array([float(m[0]) for m in sca])
#     values = np.array([float(m[1]) for m in sca])
#     return times, values


# def resolve_weather_folder(root: str, month: int, day: int) -> str:
#     """weatherData/<root> + MMDD (e.g. month=12, day=11 -> '1211')."""
#     mmdd = f"{month:02d}{day:02d}"
#     candidate = os.path.join(root, mmdd)
#     if not os.path.isdir(candidate):
#         raise FileNotFoundError(f"No such weather folder: {candidate}")
#     return candidate


# def load_weather_folder(weather_dir: str) -> dict:
#     """
#     Reads every field file in a `weatherData/MMDD/` folder and returns a
#     dict of hourly arrays ready for `run_analysis_from_folder`:
#         hour, T_amb_C, wind_speed_ms, IDN_Wm2, Idif_Wm2,
#         sun_vec (N,3) [East, North, Up unit vector, straight from the file],
#         altitude_deg  [from solarElevation if present, else asin(sun_vec_z)]
#     Any other fields present (cloudCover, relativeHumidity, atmosphericPressure,
#     wetBulbTemperature, wambient, wPonds, windVelocity, ...) are passed through
#     unchanged under their own file name, in case you want them for other
#     calculations (e.g. evaporative cooling from the ponds).
#     """
#     files = {f: os.path.join(weather_dir, f) for f in os.listdir(weather_dir)
#               if os.path.isfile(os.path.join(weather_dir, f))}

#     parsed = {name: parse_foam_field(path) for name, path in files.items()}
#     # all fields share the same time base -- use whichever field is present first
#     times_s = next(iter(parsed.values()))[0]
#     hours = times_s / 3600.0

#     def get(name, default=None):
#         return parsed[name][1] if name in parsed else default

#     t_amb_k = get("Tambient")
#     wx = dict(
#         hour=hours,
#         T_amb_C=(t_amb_k - 273.15) if t_amb_k is not None else None,
#         wind_speed_ms=get("windSpeed"),
#         IDN_Wm2=get("IDN", np.zeros_like(hours)),
#         Idif_Wm2=get("Idif", np.zeros_like(hours)),
#     )

#     if "sunPosVector" in parsed:
#         wx["sun_vec"] = parsed["sunPosVector"][1]                 # (N,3) already East/North/Up
#         lengths = np.linalg.norm(wx["sun_vec"], axis=1)
#         if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
#             raise ValueError("sunPosVector must contain finite, nonzero vectors")
#         wx["sun_vec"] = wx["sun_vec"] / lengths[:, None]
#         wx["altitude_deg"] = np.degrees(np.arcsin(np.clip(wx["sun_vec"][:, 2], -1, 1)))
#     elif "solarElevation" in parsed and "solarAzimuth" in parsed:
#         alt = parsed["solarElevation"][1]
#         az = parsed["solarAzimuth"][1]
#         wx["altitude_deg"] = alt
#         wx["sun_vec"] = np.array([sun_unit_vector(a, z) for a, z in zip(alt, az)])
#     else:
#         raise ValueError("Weather folder has neither sunPosVector nor "
#                           "solarElevation+solarAzimuth -- cannot get sun direction.")

#     if wx["T_amb_C"] is None:
#         raise ValueError("Weather folder is missing a 'Tambient' file.")
#     if wx["wind_speed_ms"] is None:
#         raise ValueError("Weather folder is missing a 'windSpeed' file.")

#     # keep any extra fields around too, unmodified
#     for name, (t, v) in parsed.items():
#         if name not in ("Tambient", "windSpeed", "IDN", "Idif", "sunPosVector",
#                          "solarElevation", "solarAzimuth"):
#             wx[name] = v

#     return wx


# # ==========================================================================
# # 7. Shared per-hour physics loop + per-building RC response
# #    (used by BOTH the clear-sky/solar-formula path and the real
# #    weather-data-folder path below)
# # ==========================================================================

# _SHADOW_RAY_OFFSET = 1e-3   # m, nudge the ray origin off the face surface

# def _simulate(faces: list, capacitance_by_building: dict, records: list,
#               ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#               occluders_by_building: Optional[dict] = None,
#               t_initial_c: Optional[float] = None,
#               ray_tracer=None, shadow_samples: int = 3,
#               wall_settings: Optional[WallSettings] = None,
#               surface_group: str = "building") -> tuple:
#     """Per-cell solar/thermal balance plus the separate legacy bulk RC estimate.

#     OBJ cells use packet rays; JSON faces retain their centroid visibility test.
#     Output temperatures are states at the labelled timestamps. The first
#     output is the initial condition; every later record supplies the constant
#     boundary condition over the interval ending at that timestamp. This keeps
#     each output's shadow/solar fields aligned with its thermal state.
#     """
#     excluded = {"ceiling"} | (set() if INCLUDE_FLOOR else {"floor"})
#     exposed = [f for f in faces if f.surface_type not in excluded]
#     if not exposed or not records:
#         raise ValueError("Simulation requires exposed faces and weather records")
#     bids = list(capacitance_by_building)
#     bindex = {bid: i for i, bid in enumerate(bids)}
#     indices = np.array([bindex[f.building_id] for f in exposed])
#     normals = np.array([f.normal for f in exposed])
#     areas = np.array([f.area for f in exposed])
#     albedo = np.array([f.albedo for f in exposed])
#     emissivity = np.array([f.emissivity for f in exposed])
#     view_sky = (1 + normals[:, 2]) / 2
#     static = dict(building_id=[f.building_id for f in exposed],
#                   face_id=[f.face_id for f in exposed],
#                   parent_face_id=[f.parent_face_id or f.face_id for f in exposed],
#                   surface_type=[f.surface_type for f in exposed],
#                   surface_group=surface_group, area_m2=areas,
#                   tilt_deg=[f.tilt_deg for f in exposed])
#     frames, bulk_rows = [], []
#     bulk_T = np.full(len(bids), t_initial_c if t_initial_c is not None else records[0]["T_amb_C"], dtype=float)
#     caps = np.array([capacitance_by_building[b] for b in bids])
#     wall_settings = wall_settings or WallSettings()
#     wall_initial_c = (t_initial_c if t_initial_c is not None
#                       else wall_settings.indoor_temperature_c)
#     slab = SlabConduction([f.thermal_conductivity for f in exposed],
#                           [f.density for f in exposed], [f.specific_heat for f in exposed],
#                           emissivity, wall_initial_c, wall_settings)
#     for ri, rec in enumerate(records):
#         t_amb = rec["T_amb_C"]
#         sun = np.asarray(rec["sun_vec"], dtype=float)
#         sun = sun / np.linalg.norm(sun)
#         alt = rec["altitude_deg"]
#         h_c = convective_h(rec["wind_speed_ms"])
#         cos_theta = normals @ sun
#         self_lit = (alt > 0) & (cos_theta > 0)
#         fraction = self_lit.astype(float)
#         if ray_tracer is not None and alt > 0:
#             fraction = triangle_sunlit_fraction(exposed, sun, ray_tracer, shadow_samples)
#         elif occluders_by_building:
#             for i in np.flatnonzero(self_lit):
#                 f = exposed[i]
#                 fraction[i] = not is_shadowed(f.center + f.normal*_SHADOW_RAY_OFFSET,
#                                               sun, occluders_by_building.get(f.building_id, []))
#         shadow_fraction = np.where(self_lit, 1-fraction, 0)
#         direct = max(rec["i_dn"], 0) * np.maximum(cos_theta, 0) * fraction
#         diffuse = max(rec["i_dh"], 0) * view_sky
#         ghi = max(rec["i_dn"], 0) * max(sun[2], 0) + max(rec["i_dh"], 0)
#         reflected = ground_albedo * ghi * (1-view_sky)
#         total = direct + diffuse + reflected
#         solair = sol_air_temperature(t_amb, total, h_c, albedo, view_sky, emissivity)
#         q_solar = (1-albedo)*total
#         ta_k = t_amb + 273.15
#         # Tangent linearization at ambient, frozen for this weather interval:
#         # q_lw(Ts) ~= q_lw(Ta) + h_rad*(Ts-Ta). This permits the exact
#         # matrix-exponential wall update; full Ts^4 radiation is nonlinear.
#         q_lw_ambient = emissivity*STEFAN_BOLTZMANN*view_sky*(ta_k**4-sky_temperature(ta_k)**4)
#         h_rad = 4*emissivity*STEFAN_BOLTZMANN*ta_k**3
#         external_h = h_c+h_rad
#         external_source = q_solar-q_lw_ambient + external_h*t_amb

#         if ri > 0:
#             dt = (rec["hour"] - records[ri-1]["hour"])*3600
#             if dt <= 0:
#                 raise ValueError("Weather times must be strictly increasing")
#             slab.advance(dt, external_h, external_source)

#         ts = slab.temperatures[:,0].copy()
#         q_conv = h_c*(ts-t_amb)
#         q_lw = q_lw_ambient + h_rad*(ts-t_amb)
#         wall_fields = slab.diagnostics(external_h, external_source)
#         wall_fields["stored_heat_J"] = wall_fields["stored_heat_Jm2"]*areas
#         wall_fields["wall_capacity_JK"] = wall_fields["wall_capacity_Jm2K"]*areas
#         wall_fields["slab_capacity_JK"] = wall_fields["slab_capacity_Jm2K"]*areas
#         frames.append(pd.DataFrame(dict(static, hour=rec["hour"], solar_altitude_deg=alt,
#             self_lit=self_lit, shadowed=shadow_fraction > 0, sunlit=fraction > 0,
#             sunlit_fraction=fraction, shadow_fraction=shadow_fraction, cos_theta=cos_theta,
#             I_direct_Wm2=direct, I_diffuse_Wm2=diffuse, I_reflected_Wm2=reflected,
#             I_total_Wm2=total, h_conv_Wm2K=h_c, T_amb_C=t_amb,
#             T_solair_C=solair, T_surface_C=ts, q_solar_absorbed_Wm2=q_solar,
#             q_conv_out_Wm2=q_conv, q_longwave_out_Wm2=q_lw,
#             wall_thickness_m=wall_settings.thickness_m,
#             slab_thickness_m=wall_settings.thickness_m,
#             T_indoor_air_C=wall_settings.indoor_temperature_c,
#             T_back_boundary_C=wall_settings.indoor_temperature_c,
#             h_indoor_Wm2K=wall_settings.indoor_h_wm2k,
#             h_back_Wm2K=wall_settings.indoor_h_wm2k, **wall_fields)))
#         ah = np.bincount(indices, weights=areas*h_c, minlength=len(bids))
#         aht = np.bincount(indices, weights=areas*h_c*solair, minlength=len(bids))
#         teq = np.divide(aht, ah, out=np.full(len(bids), t_amb, dtype=float), where=ah > 0)
#         tau = np.divide(caps, ah, out=np.full(len(bids), np.inf), where=ah > 0)
#         for bi, bid in enumerate(bids):
#             bulk_rows.append(dict(building_id=bid, hour=rec["hour"], T_amb_C=t_amb,
#                 T_eq_C=teq[bi], h_eff_WK=ah[bi], tau_hr=tau[bi]/3600,
#                 T_building_C=bulk_T[bi]))
#         if ri+1 < len(records):
#             dt_bulk = (records[ri+1]["hour"] - rec["hour"])*3600
#             if dt_bulk <= 0:
#                 raise ValueError("Weather times must be strictly increasing")
#             decay = np.exp(-np.divide(
#                 dt_bulk, tau, out=np.full_like(tau, np.inf), where=tau > 0))
#             bulk_T = teq + (bulk_T-teq)*decay
#         if ray_tracer is not None:
#             print(f"[simulation] hour {rec['hour']:g}: {len(exposed)} surface cells", flush=True)
#     return pd.concat(frames, ignore_index=True), pd.DataFrame(bulk_rows)


# def compute_capacitances(scene: dict, faces: list) -> dict:
#     """{building_id: C [J/K]} for every building in the scene, C = rho*V*cp."""
#     caps = {}
#     faces_by_bid = {}
#     for f in faces:
#         faces_by_bid.setdefault(f.building_id, f)
#     for bid, f0 in faces_by_bid.items():
#         vol = building_volume(scene, bid)
#         caps[bid] = f0.density * vol * f0.specific_heat
#     return caps


# def run_analysis(json_path: str, day: date, weather: pd.DataFrame,
#                   utc_offset_hr: float = 0.0, north_offset_deg: float = 0.0,
#                   site_altitude_km: float = 0.0,
#                   consider_shadows: bool = True,
#                   t_initial_c: Optional[float] = None,
#                   wall_settings: Optional[WallSettings] = None) -> tuple:
#     """
#     CLEAR-SKY / SOLAR-FORMULA path: use this when you do NOT have a real
#     weather-data export, only a date + location + basic T_amb/wind. Solar
#     position and irradiance are both modelled analytically (see functions
#     `solar_position` and `hottel_clear_sky` above).
#     Returns (face_hourly_df, building_hourly_df, faces).
#     """
#     scene = load_scene(json_path)
#     faces = extract_faces(scene)
#     lat, lon = site_location(scene)
#     capacitances = compute_capacitances(scene, faces)
#     occluders = build_occluders_by_building(scene) if consider_shadows else None
#     day_of_year = day.timetuple().tm_yday

#     records = []
#     for _, row in weather.iterrows():
#         hr = row["hour"]
#         alt, az = solar_position(lat, lon, day_of_year, hr, utc_offset_hr)
#         sun_vec = sun_unit_vector(alt, az, north_offset_deg)
#         if "GHI_Wm2" in row and not pd.isna(row.get("DNI_Wm2", np.nan)):
#             i_dn, i_dh = row["DNI_Wm2"], row["DHI_Wm2"]
#         else:
#             i_dn, i_dh = hottel_clear_sky(alt, site_altitude_km)
#         records.append(dict(hour=hr, T_amb_C=row["T_amb_C"], wind_speed_ms=row["wind_speed_ms"],
#                              sun_vec=sun_vec, altitude_deg=alt, i_dn=i_dn, i_dh=i_dh))

#     face_df, building_df = _simulate(faces, capacitances, records,
#                                       occluders_by_building=occluders, t_initial_c=t_initial_c,
#                                       wall_settings=wall_settings)
#     return face_df, building_df, faces


# def run_analysis_from_folder(json_path: str, weather_dir: str,
#                               ground_albedo: float = DEFAULT_GROUND_ALBEDO,
#                               consider_shadows: bool = True,
#                               t_initial_c: Optional[float] = None,
#                   wall_settings: Optional[WallSettings] = None) -> tuple:
#     """
#     REAL WEATHER-DATA path (preferred whenever you have it): reads Tambient,
#     windSpeed, IDN, Idif and sunPosVector straight from an exported
#     `weatherData/MMDD/` folder (see `load_weather_folder`). No solar-position
#     formula and no clear-sky model are used here at all -- the sun vector and
#     irradiance are exactly what your own pipeline computed, in the same
#     (x=East, y=North, z=Up) axes as the building JSON's face normals, so
#     there is no coordinate-convention guessing.
#     Returns (face_hourly_df, building_hourly_df, faces).
#     """
#     scene = load_scene(json_path)
#     faces = extract_faces(scene)
#     capacitances = compute_capacitances(scene, faces)
#     occluders = build_occluders_by_building(scene) if consider_shadows else None
#     wx = load_weather_folder(weather_dir)

#     records = []
#     n = len(wx["hour"])
#     for i in range(n):
#         records.append(dict(
#             hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
#             sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
#             i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i],
#         ))

#     face_df, building_df = _simulate(faces, capacitances, records, ground_albedo=ground_albedo,
#                                       occluders_by_building=occluders, t_initial_c=t_initial_c,
#                                       wall_settings=wall_settings)
#     return face_df, building_df, faces


# # ==========================================================================
# # 8. ParaView export -- writes the same results as a time series of legacy
# #    VTK PolyData files (plus a .pvd collection), so you can open one file
# #    in ParaView, scrub an "hour of day" timeline, and see the building
# #    (colored by sol-air temperature, shadowed/sunlit status, or irradiance)
# #    together with the trees, with no plugins and no `pip install vtk` --
# #    this writes plain-text legacy VTK, which ParaView reads natively.
# # ==========================================================================

# _SURFACE_TYPE_ID = {"wall": 0, "roof": 1, "floor": 2, "ceiling": 3, "ground": 6}
# TREE_TRUNK_TYPE_ID = 4
# TREE_CROWN_TYPE_ID = 5


# def _ring_polygon(cx: float, cy: float, z: float, radius: float, n: int) -> list:
#     return [(cx + radius * math.cos(2 * math.pi * i / n),
#               cy + radius * math.sin(2 * math.pi * i / n), z) for i in range(n)]


# def _cylinder_polys(cx: float, cy: float, z0: float, z1: float, radius: float, n: int = 8) -> list:
#     """Side quads + a bottom cap (top is hidden inside the tree crown)."""
#     ring0 = _ring_polygon(cx, cy, z0, radius, n)
#     ring1 = _ring_polygon(cx, cy, z1, radius, n)
#     polys = []
#     for i in range(n):
#         j = (i + 1) % n
#         polys.append([ring0[i], ring0[j], ring1[j], ring1[i]])
#     polys.append(list(reversed(ring0)))   # bottom cap, wound outward-facing-down
#     return polys


# def _cone_polys(cx: float, cy: float, base_z: float, apex_z: float, radius: float, n: int = 8) -> list:
#     """Side triangles (apex at top) + a base cap."""
#     ring = _ring_polygon(cx, cy, base_z, radius, n)
#     apex = (cx, cy, apex_z)
#     polys = [[ring[i], ring[(i + 1) % n], apex] for i in range(n)]
#     polys.append(list(ring))              # base cap
#     return polys


# def _tree_mesh(tree_json: dict, trunk_radius_ratio: float = 0.15) -> tuple:
#     """Returns (polygons, surface_type_ids) for one tree's trunk + crown, or
#     (None, None) if the tree's geometry couldn't be resolved (see
#     resolve_tree_geometry)."""
#     geo = resolve_tree_geometry(tree_json)
#     if geo is None:
#         return None, None
#     pos, trunk_h, crown_h, crown_r = geo
#     trunk_r = max(crown_r * trunk_radius_ratio, 0.1)
#     trunk_polys = _cylinder_polys(pos[0], pos[1], pos[2], pos[2] + trunk_h, trunk_r)
#     crown_polys = _cone_polys(pos[0], pos[1], pos[2] + trunk_h, pos[2] + trunk_h + crown_h, crown_r)
#     polys = trunk_polys + crown_polys
#     type_ids = [TREE_TRUNK_TYPE_ID] * len(trunk_polys) + [TREE_CROWN_TYPE_ID] * len(crown_polys)
#     return polys, type_ids


# def _write_vtk_polydata(path: str, polygons: list, cell_scalars: dict) -> None:
#     """
#     polygons: list of polygons, each a list of (x,y,z) tuples (a planar loop).
#         Points are NOT shared between polygons -- every polygon gets its own
#         copy of its corner points. That is deliberate: colouring is by CELL
#         data (one value per face), and un-shared points guarantee ParaView
#         renders each face as a single flat colour instead of interpolating
#         across a shared edge with a differently-valued neighbour.
#     cell_scalars: {name: (values, "float"|"int")}, one value per polygon,
#         same length and order as `polygons`.
#     """
#     points, polygon_indices = [], []
#     idx = 0
#     for poly in polygons:
#         n = len(poly)
#         polygon_indices.append(list(range(idx, idx + n)))
#         points.extend(poly)
#         idx += n

#     with open(path, "w") as f:
#         f.write("# vtk DataFile Version 3.0\n")
#         f.write("solar_thermal_model export\n")
#         f.write("ASCII\n")
#         f.write("DATASET POLYDATA\n")
#         f.write(f"POINTS {len(points)} float\n")
#         for p in points:
#             f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
#         total_size = sum(len(idxs) + 1 for idxs in polygon_indices)
#         f.write(f"POLYGONS {len(polygon_indices)} {total_size}\n")
#         for idxs in polygon_indices:
#             f.write(f"{len(idxs)} " + " ".join(str(i) for i in idxs) + "\n")
#         f.write(f"CELL_DATA {len(polygons)}\n")
#         for name, (values, dtype) in cell_scalars.items():
#             f.write(f"SCALARS {name} {dtype} 1\n")
#             f.write("LOOKUP_TABLE default\n")
#             fmt = "{}\n" if dtype == "int" else "{:.4f}\n"
#             for v in values:
#                 f.write(fmt.format(v))


# def export_vtk_series(json_path: str, weather_dir: str, out_dir: str,
#                        consider_shadows: bool = True,
#                        wall_settings: Optional[WallSettings] = None,
#                        t_initial_c: Optional[float] = None) -> str:
#     """
#     Runs the real-weather-data analysis (see `run_analysis_from_folder`) and
#     writes one legacy-VTK PolyData file per hour plus a `scene.pvd` ParaView
#     collection tying them together as a time series. Open `scene.pvd` in
#     ParaView, colour by any of the per-face arrays below, and scrub the time
#     slider to watch temperature and shadowing move through the day together.

#     Per-face (cell) data written on every building face:
#         T_C            transient outer surface temperature, deg C
#         I_total_Wm2    total incident solar irradiance, W/m^2
#         shadowed       1 if ray-blocked by another building/tree this hour
#         sunlit         1 if actually receiving direct beam this hour
#                         (self_lit AND NOT shadowed)
#         surface_type_id  0=wall 1=roof 2=floor 3=ceiling 4=tree trunk 5=tree crown
#         building_index  which building/floor this face belongs to (-1 for trees)
#         is_building     1 for building faces, 0 for tree geometry

#     Floor/ceiling faces are excluded from the thermal simulation itself (see
#     the module docstring) but ARE still drawn here, with T_C set to that
#     hour's ambient air temperature, so the building reads as a solid, complete
#     shape rather than having visible holes -- filter on surface_type_id in
#     ParaView if you want only the faces that were actually simulated.
#     Trees are static (their geometry doesn't change hour to hour) but are
#     re-written into every frame file so each one opens as a complete scene on
#     its own; T_C for tree geometry is likewise just that hour's ambient
#     temperature (canopy temperature isn't modelled).
#     """
#     scene = load_scene(json_path)
#     faces = extract_faces(scene)
#     face_df, building_df, _ = run_analysis_from_folder(
#         json_path, weather_dir, consider_shadows=consider_shadows,
#         wall_settings=wall_settings, t_initial_c=t_initial_c)
#     os.makedirs(out_dir, exist_ok=True)

#     tree_polys, tree_type_ids = [], []
#     n_tree_geo_ok, n_tree_geo_skipped = 0, 0
#     for t in scene.get("trees", []):
#         polys, type_ids = _tree_mesh(t)
#         if polys is None:
#             n_tree_geo_skipped += 1
#             continue
#         n_tree_geo_ok += 1
#         tree_polys.extend(polys)
#         tree_type_ids.extend(type_ids)
#     if n_tree_geo_skipped:
#         print(f"[vtk export] Skipped {n_tree_geo_skipped} tree(s) with unrecognised geometry fields.")

#     building_ids_sorted = sorted({f.building_id for f in faces})
#     building_index_map = {bid: i for i, bid in enumerate(building_ids_sorted)}

#     face_lookup = {}
#     for _, r in face_df.iterrows():
#         face_lookup[(r["building_id"], r["face_id"], r["hour"])] = r

#     hours = sorted(face_df["hour"].unique())
#     pvd_entries = []
#     for h in hours:
#         amb_rows = building_df[building_df["hour"] == h]
#         amb_c = float(amb_rows["T_amb_C"].iloc[0]) if len(amb_rows) else 0.0

#         polys, T_C, shadowed, sunlit, I_tot, surf_id, bidx, is_bldg = [], [], [], [], [], [], [], []
#         for f in faces:
#             row = face_lookup.get((f.building_id, f.face_id, h))
#             polys.append([tuple(v) for v in f.vertices])
#             if row is None:                          # floor/ceiling: not simulated
#                 T_C.append(amb_c); shadowed.append(0); sunlit.append(0); I_tot.append(0.0)
#             else:
#                 T_C.append(float(row["T_surface_C"])); shadowed.append(int(bool(row["shadowed"])))
#                 sunlit.append(int(bool(row["sunlit"]))); I_tot.append(float(row["I_total_Wm2"]))
#             surf_id.append(_SURFACE_TYPE_ID.get(f.surface_type, -1))
#             bidx.append(building_index_map[f.building_id])
#             is_bldg.append(1)

#         polys.extend(tree_polys)
#         T_C.extend([amb_c] * len(tree_polys))
#         shadowed.extend([0] * len(tree_polys))
#         sunlit.extend([0] * len(tree_polys))
#         I_tot.extend([0.0] * len(tree_polys))
#         surf_id.extend(tree_type_ids)
#         bidx.extend([-1] * len(tree_polys))
#         is_bldg.extend([0] * len(tree_polys))

#         fname = f"frame_{int(round(h * 100)):04d}.vtk"   # e.g. hour 13.5 -> frame_1350.vtk
#         _write_vtk_polydata(os.path.join(out_dir, fname), polys, {
#             "T_C": (T_C, "float"),
#             "I_total_Wm2": (I_tot, "float"),
#             "shadowed": (shadowed, "int"),
#             "sunlit": (sunlit, "int"),
#             "surface_type_id": (surf_id, "int"),
#             "building_index": (bidx, "int"),
#             "is_building": (is_bldg, "int"),
#         })
#         pvd_entries.append((float(h), fname))

#     pvd_path = os.path.join(out_dir, "scene.pvd")
#     with open(pvd_path, "w") as f:
#         f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
#         for t, fname in pvd_entries:
#             f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
#         f.write('</Collection>\n</VTKFile>\n')

#     print(f"[vtk export] Wrote {len(pvd_entries)} frame(s) + scene.pvd to {out_dir} "
#           f"({n_tree_geo_ok} tree(s) meshed).")
#     return pvd_path


# # ==========================================================================
# # 9. OBJ pipeline: retain input triangles for partial-wall shadows.
# #    Coplanar grouping is used only for parent IDs and bulk-volume estimates.
# #    OBJ does not carry thermal materials: defaults below assume concrete.
# #    Spatial resolution is limited by the input mesh; use a finer OBJ for
# #    sharper shadow edges. Trees are opaque meshes, without leaf transmission.
# # ==========================================================================

# DEFAULT_MATERIAL = dict(thermal_conductivity=2.3, density=2400, specific_heat=880)
# DEFAULT_ALBEDO_WALL = 0.3
# DEFAULT_ALBEDO_ROOF = 0.4
# ROOF_NORMAL_Z_THRESHOLD = 0.7   # |normal.z| above this -> "roof", else "wall"


# def _obj_vertex_index(token: str, n_verts_so_far: int) -> int:
#     """
#     Parses one OBJ face-vertex token in any of the four standard forms
#     (`v`, `v/vt`, `v/vt/vn`, `v//vn`) and returns a 0-based vertex index.
#     Handles negative (relative-to-current-end) indices too, per the OBJ
#     spec.
#     """
#     idx = int(token.split("/")[0])
#     return (n_verts_so_far + idx) if idx < 0 else (idx - 1)


# def load_obj_mesh(path: str) -> tuple:
#     """
#     OBJ parser for the buildings_refined.obj / trees.obj this pipeline
#     consumes. json_to_obj.py's own write_obj() only ever emits plain
#     `v x y z` / `f i j k` lines, but real pipelines often pass the file
#     through something else afterwards (normal generation, mesh repair,
#     etc.) that adds `vn`/`vt` references -- seen in practice as
#     `f 1//1 2//1 3//1` style faces. This is intentionally more permissive
#     than that one exact format: it accepts all four standard face-vertex
#     forms (`v`, `v/vt`, `v/vt/vn`, `v//vn`), skips `vn`/`vt` lines outright
#     (this pipeline recomputes every normal from triangle geometry anyway,
#     never reads one from the file), and fan-triangulates any face with more
#     than 3 vertices rather than assuming triangles-only.
#     Returns (vertices [N,3] float array, triangles [list of (i,j,k) 0-based
#     int tuples]).
#     """
#     verts, tris = [], []
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if line.startswith("v "):
#                 p = line.split()
#                 verts.append((float(p[1]), float(p[2]), float(p[3])))
#             elif line.startswith("f "):
#                 tokens = line.split()[1:]
#                 idxs = [_obj_vertex_index(tok, len(verts)) for tok in tokens]
#                 for t in range(1, len(idxs) - 1):     # fan triangulation (no-op for triangles)
#                     tris.append((idxs[0], idxs[t], idxs[t + 1]))
#     return np.array(verts, dtype=float), tris


# def _triangle_normal_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple:
#     n = np.cross(v1 - v0, v2 - v0)
#     area = float(np.linalg.norm(n)) / 2.0
#     if area < 1e-12:
#         return None, 0.0
#     return n / (2.0 * area), area


# def group_coplanar(vertices: np.ndarray, triangles: list,
#                     normal_decimals: int = 3, offset_decimals: int = 2) -> dict:
#     """
#     Merges triangles sharing the same (rounded outward normal, rounded plane
#     offset) into "logical faces". Exact for flat, regularly-subdivided quads
#     (this OBJ format's building geometry): every wall segment along one
#     straight facade -- including across every stacked storey of a
#     multi-floor tower, since they're all perfectly coplanar -- collapses
#     back into ONE logical wall face.
#     Returns {plane_key: {"triangles": [(i,j,k), ...], "normal", "area",
#                           "center", "z_min", "z_max"}}.
#     """
#     groups = {}
#     for (i, j, k) in triangles:
#         v0, v1, v2 = vertices[i], vertices[j], vertices[k]
#         n, area = _triangle_normal_area(v0, v1, v2)
#         if n is None:
#             continue
#         d = float(np.dot(n, v0))
#         key = (round(float(n[0]), normal_decimals), round(float(n[1]), normal_decimals),
#                round(float(n[2]), normal_decimals), round(d, offset_decimals))
#         g = groups.setdefault(key, dict(triangles=[], normal_sum=np.zeros(3), area=0.0,
#                                           centroid_sum=np.zeros(3), z_min=np.inf, z_max=-np.inf))
#         centroid = (v0 + v1 + v2) / 3.0
#         g["triangles"].append((i, j, k))
#         g["normal_sum"] += n * area
#         g["area"] += area
#         g["centroid_sum"] += centroid * area
#         g["z_min"] = min(g["z_min"], float(v0[2]), float(v1[2]), float(v2[2]))
#         g["z_max"] = max(g["z_max"], float(v0[2]), float(v1[2]), float(v2[2]))

#     out = {}
#     for key, g in groups.items():
#         if g["area"] <= 0:
#             continue
#         normal = g["normal_sum"] / (np.linalg.norm(g["normal_sum"]) + 1e-12)
#         center = g["centroid_sum"] / g["area"]
#         out[key] = dict(triangles=g["triangles"], normal=normal, area=g["area"],
#                           center=center, z_min=g["z_min"], z_max=g["z_max"])
#     return out


# def _mesh_to_occluder(vertices: np.ndarray, triangles: list, source_id: str) -> Optional[Occluder]:
#     if not triangles:
#         return None
#     tri_pts = [(vertices[i], vertices[j], vertices[k]) for (i, j, k) in triangles]
#     idx = sorted({v for tri in triangles for v in tri})
#     pts = vertices[idx]
#     return Occluder(kind="building", source_id=source_id,
#                      aabb_min=pts.min(axis=0), aabb_max=pts.max(axis=0), triangles=tri_pts)


# def _position_key(p, decimals: int = 4) -> tuple:
#     return (round(float(p[0]), decimals), round(float(p[1]), decimals), round(float(p[2]), decimals))


# def _obj_components_by_shared_position(vertices: np.ndarray, triangles: list,
#                                          decimals: int = 4) -> list:
#     """
#     Union-find over vertex POSITIONS, not indices -- this is the piece that
#     was missing before. json_to_obj.py does not share vertex indices across
#     different original JSON faces, even when they are geometrically
#     touching (e.g. two wall quads meeting at a shared vertical edge), so an
#     index-based union-find (see _obj_connected_components, used for trees)
#     cannot tell "two touching walls of the same building" apart from "two
#     unrelated triangles that happen to have unrelated index numbers" -- it
#     under-merges everything into one-triangle-ish fragments. Position-based
#     matching (round each vertex to `decimals` and treat equal rounded
#     positions as the same point) is what actually reconstructs "which
#     triangles are physically part of the same connected building".
#     Returns a list of triangle-index lists, one per physically distinct
#     building.
#     """
#     pos_to_id: dict = {}
#     vertex_logical_id = np.empty(len(vertices), dtype=np.int64)
#     for v_idx in range(len(vertices)):
#         key = _position_key(vertices[v_idx], decimals)
#         lid = pos_to_id.get(key)
#         if lid is None:
#             lid = len(pos_to_id)
#             pos_to_id[key] = lid
#         vertex_logical_id[v_idx] = lid

#     parent = list(range(len(pos_to_id)))

#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra != rb:
#             parent[ra] = rb

#     for (i, j, k) in triangles:
#         li, lj, lk = int(vertex_logical_id[i]), int(vertex_logical_id[j]), int(vertex_logical_id[k])
#         union(li, lj)
#         union(lj, lk)

#     comp_triangles: dict = {}
#     for t_idx, (i, j, k) in enumerate(triangles):
#         root = find(int(vertex_logical_id[i]))
#         comp_triangles.setdefault(root, []).append(t_idx)
#     return list(comp_triangles.values())


# def _obj_connected_components(vertices: np.ndarray, triangles: list) -> list:
#     """
#     Union-find over shared vertex INDICES. This groups triangles generated
#     together as one contiguous primitive (e.g. one tree's trunk cylinder) --
#     json_to_obj.py DOES share vertex indices within a single primitive's own
#     subdivision, so this works there, but NOT across different original
#     JSON faces (see _obj_components_by_shared_position for that -- used for
#     buildings, where genuine cross-face adjacency detection matters).
#     Verified on a real trees.obj: 49 trees x (1 trunk + 1 crown, each its
#     own component) = 98 components.
#     Returns a list of triangle-index lists, one per component.
#     """
#     parent = list(range(len(vertices)))

#     def find(x):
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(a, b):
#         ra, rb = find(a), find(b)
#         if ra != rb:
#             parent[ra] = rb

#     for (i, j, k) in triangles:
#         union(i, j)
#         union(j, k)

#     comp_triangles = {}
#     for t_idx, (i, j, k) in enumerate(triangles):
#         comp_triangles.setdefault(find(i), []).append(t_idx)
#     return list(comp_triangles.values())


# def build_tree_occluders_from_obj(trees_obj_path: str) -> list:
#     """Loads trees.obj and returns one Occluder per connected primitive
#     (trunk/crown separately) -- more, smaller occluders only help the
#     broad-phase AABB culling, so there's no reason to merge them further."""
#     if not trees_obj_path or not os.path.isfile(trees_obj_path):
#         return []
#     V, F = load_obj_mesh(trees_obj_path)
#     if not F:
#         return []
#     occluders = []
#     for c_idx, tri_indices in enumerate(_obj_connected_components(V, F)):
#         occ = _mesh_to_occluder(V, [F[t] for t in tri_indices], f"tree_part_{c_idx}")
#         if occ is not None:
#             occluders.append(occ)
#     return occluders


# def build_faces_from_obj(buildings_obj_path: str) -> tuple:
#     """Recover buildings, retaining every nondegenerate triangle as a cell.

#     Coplanar groups provide parent wall IDs and approximate bulk volumes only;
#     visibility, irradiance, temperature and VTK values belong to each triangle.
#     OBJ wall winding must point outward. Horizontal bottoms are excluded from
#     outdoor physics, and roof normals are corrected upward for legacy exports.
#     """
#     V, F = load_obj_mesh(buildings_obj_path)
#     components = _obj_components_by_shared_position(V, F)
#     faces, face_triangles = [], {}
#     capacitances, occluders = {}, {}
#     for b_idx, tri_indices in enumerate(components):
#         bid = f"building_{b_idx:02d}"
#         comp_tris = [F[t] for t in tri_indices]
#         groups = group_coplanar(V, comp_tris)
#         if not groups:
#             continue
#         z_min = min(g["z_min"] for g in groups.values())
#         z_max = max(g["z_max"] for g in groups.values())
#         roof_area = sum(g["area"] * abs(g["normal"][2]) for g in groups.values()
#                         if abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
#                         and g["center"][2] > z_min + 1e-6)
#         capacitances[bid] = (DEFAULT_MATERIAL["density"] * roof_area * (z_max-z_min)
#                              * DEFAULT_MATERIAL["specific_heat"])
#         occluders[bid] = _mesh_to_occluder(V, comp_tris, bid)
#         for gi, g in enumerate(groups.values()):
#             horizontal = abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
#             bottom = horizontal and g["z_max"] <= z_min+1e-6
#             stype = "floor" if bottom else ("roof" if horizontal else "wall")
#             normal = g["normal"].copy()
#             if (stype == "roof" and normal[2] < 0) or (stype == "floor" and normal[2] > 0):
#                 normal = -normal
#             tilt = math.degrees(math.acos(np.clip(normal[2], -1, 1)))
#             parent = f"{bid}_face_{gi:03d}_{stype}"
#             for ti, tri in enumerate(g["triangles"]):
#                 verts = V[list(tri)]
#                 _, area = _triangle_normal_area(*verts)
#                 fid = f"{parent}_tri_{ti:06d}"
#                 faces.append(Face(building_id=bid, face_id=fid, parent_face_id=parent,
#                     surface_type=stype, normal=normal, area=area, tilt_deg=tilt,
#                     center=verts.mean(axis=0), vertices=verts,
#                     **DEFAULT_MATERIAL,
#                     albedo=DEFAULT_ALBEDO_ROOF if stype == "roof" else DEFAULT_ALBEDO_WALL))
#                 face_triangles[fid] = [tri]
#     return faces, capacitances, face_triangles, occluders


# def build_ground_faces_from_obj(ground_obj_path: str,
#                                 building_ray_tracer=None,
#                                 ground_settings: Optional[GroundSettings] = None) -> list:
#     """Load exposed ground triangles and orient their normals upward.

#     A vertical ray removes cells beneath a building roof. This prevents soil
#     under a building footprint from receiving outdoor convection or appearing
#     as a second coincident thermal surface in the exported scene.
#     """
#     settings = ground_settings or GroundSettings()
#     vertices, triangles = load_obj_mesh(ground_obj_path)
#     candidates = []
#     for source_index, tri in enumerate(triangles):
#         verts = vertices[list(tri)]
#         normal, area = _triangle_normal_area(*verts)
#         if normal is None:
#             continue
#         if normal[2] < 0:
#             normal = -normal
#             verts = verts[[0, 2, 1]]
#         candidates.append((source_index, verts, normal, area))
#     if not candidates:
#         raise ValueError(f"Ground OBJ has no nondegenerate triangles: {ground_obj_path}")

#     keep = np.ones(len(candidates), dtype=bool)
#     if building_ray_tracer is not None:
#         centers = np.array([item[1].mean(axis=0) for item in candidates])
#         origins = centers + np.array([0.0, 0.0, _SHADOW_RAY_OFFSET])
#         keep = ~building_ray_tracer.blocked(origins, np.array([0.0, 0.0, 1.0]))

#     faces = []
#     for retain, (source_index, verts, normal, area) in zip(keep, candidates):
#         if not retain:
#             continue
#         fid = f"ground_tri_{source_index:06d}"
#         faces.append(Face(
#             building_id="ground", face_id=fid, parent_face_id="ground_surface",
#             surface_type="ground", normal=normal, area=area,
#             tilt_deg=math.degrees(math.acos(np.clip(normal[2], -1, 1))),
#             center=verts.mean(axis=0), vertices=verts,
#             thermal_conductivity=settings.thermal_conductivity,
#             density=settings.density, specific_heat=settings.specific_heat,
#             albedo=settings.albedo, emissivity=settings.emissivity))
#     print(f"[ground] Loaded {len(faces)} exposed triangles; "
#           f"removed {len(candidates)-len(faces)} beneath buildings", flush=True)
#     return faces


# def run_analysis_from_obj(buildings_obj_path: str, weather_dir: str,
#                             trees_obj_path: Optional[str] = None,
#                             consider_shadows: bool = True,
#                             t_initial_c: Optional[float] = None,
#                             shadow_samples: int = 3,
#                             wall_settings: Optional[WallSettings] = None,
#                             ground_obj_path: Optional[str] = None,
#                             ground_initial_c: Optional[float] = None,
#                             ground_settings: Optional[GroundSettings] = None) -> tuple:
#     """Per-triangle exposure and transient conducting slab temperatures from OBJ.

#     All building and tree triangles are opaque occluders, including other
#     parts of the receiver's own building. If ground_obj_path is supplied,
#     exposed ground triangles receive the same shadow and exterior balance plus
#     a transient soil slab. Returns (cell_df, building_df, faces).
#     """
#     if shadow_samples not in (1, 3):
#         raise ValueError("shadow_samples must be 1 or 3")
#     faces, capacitances, _, building_occluders = build_faces_from_obj(buildings_obj_path)
#     building_triangles = [tri for occ in building_occluders.values() if occ is not None
#                           for tri in occ.triangles]
#     building_tracer = TriangleRayTracer(building_triangles) if ground_obj_path else None
#     ground_faces = (build_ground_faces_from_obj(ground_obj_path, building_tracer, ground_settings)
#                     if ground_obj_path else [])
#     tracer = None
#     if consider_shadows:
#         triangles = list(building_triangles)
#         if trees_obj_path:
#             tv, tf = load_obj_mesh(trees_obj_path)
#             triangles.extend(tv[np.asarray(tf, dtype=int)].tolist())
#         print(f"[shadows] Building ray acceleration for {len(triangles)} triangles", flush=True)
#         tracer = TriangleRayTracer(triangles)
#     wx = load_weather_folder(weather_dir)
#     records = [dict(hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
#                       sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
#                       i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i])
#                for i in range(len(wx["hour"]))]
#     face_df, building_df = _simulate(faces, capacitances, records,
#         t_initial_c=t_initial_c, ray_tracer=tracer, shadow_samples=shadow_samples,
#         wall_settings=wall_settings)
#     if ground_faces:
#         settings = ground_settings or GroundSettings()
#         ground_capacitance = (settings.density * settings.specific_heat * settings.depth_m
#                               * sum(f.area for f in ground_faces))
#         ground_df, _ = _simulate(
#             ground_faces, {"ground": ground_capacitance}, records,
#             t_initial_c=ground_initial_c, ray_tracer=tracer,
#             shadow_samples=shadow_samples, wall_settings=settings.slab_settings(),
#             surface_group="ground")
#         ground_df["ground_depth_m"] = ground_df["slab_thickness_m"]
#         ground_df["T_ground_bottom_C"] = ground_df["T_back_surface_C"]
#         ground_df["T_deep_ground_C"] = ground_df["T_back_boundary_C"]
#         ground_df["h_deep_ground_Wm2K"] = ground_df["h_back_Wm2K"]
#         ground_df["q_deep_ground_out_Wm2"] = ground_df["q_back_out_Wm2"]
#         ground_df["T_ground_mean_C"] = ground_df["T_slab_mean_C"]
#         ground_df["ground_capacity_Jm2K"] = ground_df["slab_capacity_Jm2K"]
#         ground_df["ground_capacity_JK"] = ground_df["slab_capacity_JK"]
#         building_only = ["wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K",
#                          "T_inner_C", "T_wall_mean_C", "q_indoor_out_Wm2",
#                          "wall_capacity_Jm2K", "wall_capacity_JK"]
#         ground_df[building_only] = np.nan
#         face_df = pd.concat([face_df, ground_df], ignore_index=True, sort=False)
#         faces = faces + ground_faces
#     return face_df, building_df, faces


# def export_vtk_series_from_obj(buildings_obj_path: str, weather_dir: str, out_dir: str,
#                                  trees_obj_path: Optional[str] = None,
#                                  consider_shadows: bool = True,
#                                  shadow_samples: int = 3,
#                                  analysis=None,
#                                  wall_settings: Optional[WallSettings] = None,
#                                  t_initial_c: Optional[float] = None,
#                                  ground_obj_path: Optional[str] = None,
#                                  ground_settings: Optional[GroundSettings] = None,
#                                  ground_initial_c: Optional[float] = None) -> str:
#     """Export per-triangle CELL_DATA; optionally reuse an existing simulation.

#     T_C aliases T_surface_C (transient outer surface temperature), while
#     T_solair_C is retained separately. sunlit_fraction is 0..1; shadowed
#     means at least one sample was blocked on a sun-facing triangle.
#     Tree geometry is displayed at ambient temperature, not thermally solved.
#     Ground cells have is_ground=1 and surface_type_id=6.
#     """
#     if analysis is None:
#         analysis = run_analysis_from_obj(buildings_obj_path, weather_dir,
#             trees_obj_path=trees_obj_path, ground_obj_path=ground_obj_path,
#             consider_shadows=consider_shadows,
#             shadow_samples=shadow_samples, wall_settings=wall_settings,
#             t_initial_c=t_initial_c, ground_settings=ground_settings,
#             ground_initial_c=ground_initial_c)
#     face_df, building_df, faces = analysis
#     os.makedirs(out_dir, exist_ok=True)
#     polygons = [f.vertices.tolist() for f in faces]
#     tree_polys = []
#     if trees_obj_path:
#         tv, tf = load_obj_mesh(trees_obj_path)
#         tree_polys = tv[np.asarray(tf, dtype=int)].tolist()
#     polygons.extend(tree_polys)
#     nt = len(tree_polys)
#     ids = [f.face_id for f in faces]
#     building_ids = dict.fromkeys(f.building_id for f in faces if f.surface_type != "ground")
#     bmap = {bid: i for i, bid in enumerate(building_ids)}
#     pmap = {pid: i for i, pid in enumerate(dict.fromkeys(f.parent_face_id for f in faces))}
#     static = {
#         "surface_type_id": ([ _SURFACE_TYPE_ID.get(f.surface_type, -1) for f in faces] + [TREE_CROWN_TYPE_ID]*nt, "int"),
#         "is_building": ([int(f.surface_type != "ground") for f in faces] + [0]*nt, "int"),
#         "is_ground": ([int(f.surface_type == "ground") for f in faces] + [0]*nt, "int"),
#         "building_index": ([bmap.get(f.building_id, -1) for f in faces] + [-1]*nt, "int"),
#         "parent_face_index": ([pmap[f.parent_face_id] for f in faces] + [-1]*nt, "int"),
#     }
#     fields = ["T_surface_C", "T_solair_C", "I_direct_Wm2", "I_total_Wm2",
#               "sunlit_fraction", "shadow_fraction", "shadowed", "sunlit",
#               "h_conv_Wm2K", "q_solar_absorbed_Wm2", "q_conv_out_Wm2", "q_longwave_out_Wm2",
#               "T_inner_C", "T_wall_mean_C", "q_conduction_in_Wm2", "q_indoor_out_Wm2",
#               "q_storage_Wm2", "q_outer_node_storage_Wm2", "stored_heat_Jm2",
#               "stored_heat_J", "wall_capacity_Jm2K", "wall_capacity_JK",
#               "wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K",
#               "T_back_surface_C", "T_slab_mean_C", "q_back_out_Wm2",
#               "slab_capacity_Jm2K", "slab_capacity_JK", "slab_thickness_m",
#               "T_back_boundary_C", "h_back_Wm2K", "ground_depth_m",
#               "T_ground_bottom_C", "T_deep_ground_C", "h_deep_ground_Wm2K",
#               "q_deep_ground_out_Wm2", "T_ground_mean_C",
#               "ground_capacity_Jm2K", "ground_capacity_JK"]
#     fields = [name for name in fields if name in face_df.columns]
#     entries = []
#     for h, frame in face_df.groupby("hour", sort=True):
#         amb_c = float(frame["T_amb_C"].iloc[0])
#         frame = frame.set_index("face_id").reindex(ids)
#         scalars = dict(static)
#         for name in fields:
#             fill = amb_c if name.startswith("T_") else 0
#             dtype = "int" if name in ("shadowed", "sunlit") else "float"
#             values = frame[name].fillna(fill).to_numpy(dtype=int if dtype == "int" else float)
#             scalars[name] = (np.concatenate([values, np.full(nt, fill)]).astype(int if dtype == "int" else float), dtype)
#         scalars["T_C"] = scalars["T_surface_C"]
#         fname = f"frame_{int(round(h*100)):04d}.vtk"
#         _write_vtk_polydata(os.path.join(out_dir, fname), polygons, scalars)
#         entries.append((float(h), fname))
#     pvd_path = os.path.join(out_dir, "scene.pvd")
#     with open(pvd_path, "w") as f:
#         f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
#         for t, fname in entries:
#             f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
#         f.write('</Collection>\n</VTKFile>\n')
#     print(f"[vtk export] Wrote {len(entries)} frame(s) + scene.pvd to {out_dir}.", flush=True)
#     return pvd_path


# # ==========================================================================
# # 10. Plots
# # ==========================================================================

# def plot_results(face_df: pd.DataFrame, building_df: pd.DataFrame, out_prefix: str,
#                   building_id: Optional[str] = None):
#     import matplotlib.pyplot as plt

#     bldg_ids = building_df["building_id"].unique()
#     single = building_id if building_id is not None else (bldg_ids[0] if len(bldg_ids) == 1 else None)

#     face_sub = face_df if single is None else face_df[face_df["building_id"] == single]
#     bldg_sub = building_df if single is None else building_df[building_df["building_id"] == single]
#     # Plot area-weighted parent walls, avoiding one curve per mesh triangle.
#     if "parent_face_id" in face_sub and face_sub.parent_face_id.nunique() < face_sub.face_id.nunique():
#         keys = ["building_id", "parent_face_id", "hour"]
#         values = ["I_total_Wm2", "T_solair_C", "T_surface_C"]
#         weighted = face_sub[keys + ["area_m2"]].copy()
#         for name in values:
#             weighted[name] = face_sub[name] * face_sub.area_m2
#         grouped = weighted.groupby(keys, as_index=False).sum()
#         for name in values:
#             grouped[name] /= grouped.area_m2
#         face_sub = grouped.rename(columns={"parent_face_id": "face_id"})

#     fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

#     # --- Panel 1: incident solar flux per face vs hour (shows exposure windows) ---
#     ax = axes[0]
#     for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
#         sub = sub.sort_values("hour")
#         lab = fid if single is not None else f"{bid}:{fid}"
#         ax.plot(sub["hour"], sub["I_total_Wm2"], marker="o", ms=3, label=lab)
#     ax.set_ylabel("Incident solar flux, W/m$^2$")
#     title = "Per-face solar exposure through the day (sunlit windows)"
#     ax.set_title(title if single is None else f"{title} -- {single}")
#     ax.legend(fontsize=7, ncol=2)
#     ax.grid(alpha=0.3)

#     # --- Panel 2: surface temperature + ambient + legacy bulk estimate ---
#     ax = axes[1]
#     for (bid, fid), sub in face_sub.groupby(["building_id", "face_id"]):
#         sub = sub.sort_values("hour")
#         lab = f"{fid} (surface)" if single is not None else f"{bid}:{fid} (surface)"
#         ax.plot(sub["hour"], sub["T_surface_C"], marker="o", ms=3, label=lab)
#     ax.plot(bldg_sub.groupby("hour")["T_amb_C"].first().index,
#             bldg_sub.groupby("hour")["T_amb_C"].first().values,
#             "k--", lw=2, label="T_amb (air)")
#     for bid, sub in bldg_sub.groupby("building_id"):
#         sub = sub.sort_values("hour")
#         lab = "T_building (lumped, RC)" if single is not None else f"T_building {bid} (RC)"
#         ax.plot(sub["hour"], sub["T_building_C"], lw=2.5, label=lab)
#     ax.set_xlabel("Hour of day (local standard time)")
#     ax.set_ylabel("Temperature, \u00b0C")
#     ax.set_title("Surface temperature per face vs. ambient vs. building bulk temperature")
#     ax.legend(fontsize=7, ncol=2)
#     ax.grid(alpha=0.3)

#     fig.tight_layout()
#     fig.savefig(f"{out_prefix}_plot.png", dpi=150)
#     print(f"Wrote {out_prefix}_plot.png")


# # ==========================================================================
# # 11. Script entry point
# # ==========================================================================

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(
#         description="Analytical solar, conduction and storage model for buildings and ground")
#     parser.add_argument("--json", default="building_scene.json")
#     parser.add_argument("--buildings_obj", default=None,
#                          help="Use the OBJ-based pipeline instead of --json: path to a "
#                               "buildings_refined.obj from json_to_obj.py. Requires --weather_dir "
#                               "or --weather_root (see run_analysis_from_obj doc).")
#     parser.add_argument("--shadow_samples", type=int, choices=(1, 3), default=3,
#                          help="Rays per OBJ triangle: 3 (default) estimates partial shade; 1 uses the centroid.")
#     parser.add_argument("--trees_obj", default=None,
#                          help="Matching trees.obj from json_to_obj.py (optional, OBJ pipeline only).")
#     parser.add_argument("--ground_obj", default=None,
#                          help="Matching ground.obj; enables per-triangle ground shadow, "
#                               "surface-temperature, conduction and heat-storage calculations.")
#     parser.add_argument("--weather_csv", default=None,
#                          help="CSV with hour,T_amb_C,wind_speed_ms[,GHI_Wm2,DNI_Wm2,DHI_Wm2] "
#                               "(clear-sky/solar-formula path, --json only)")
#     parser.add_argument("--weather_dir", default=None,
#                          help="Path directly to a weatherData/MMDD folder (real-data path, preferred)")
#     parser.add_argument("--weather_root", default=None,
#                          help="Path to the weatherData/ root; combined with --month/--day to find MMDD")
#     parser.add_argument("--month", type=int, default=6)
#     parser.add_argument("--day", type=int, default=21)
#     parser.add_argument("--year", type=int, default=2026)
#     parser.add_argument("--utc_offset", type=float, default=0.0)
#     parser.add_argument("--out_prefix", default="results")
#     parser.add_argument("--plot", action="store_true")
#     parser.add_argument("--no_shadows", action="store_true",
#                          help="Skip mutual shadow-casting between buildings/trees (faster; "
#                               "fine if the scene has only one isolated building).")
#     parser.add_argument("--vtk_out", default=None,
#                          help="Directory to write a ParaView-ready VTK time series to (requires "
#                               "--weather_dir or --weather_root; see export_vtk_series doc).")
#     parser.add_argument("--wall_thickness_m", type=float, default=0.23)
#     parser.add_argument("--wall_nodes", type=int, default=5,
#                         help="Temperature nodes through each wall/roof thickness (>=2).")
#     parser.add_argument("--indoor_temp_c", type=float, default=24.0)
#     parser.add_argument("--indoor_h", type=float, default=8.0,
#                         help="Inner convection coefficient W/m2/K; 0 gives an insulated inner face.")
#     parser.add_argument("--initial_temp_c", type=float, default=None,
#                         help="Initial uniform slab temperature; default is --indoor_temp_c.")
#     parser.add_argument("--ground_depth_m", type=float, default=0.23,
#                         help="Depth of the transient upper-soil slab in metres.")
#     parser.add_argument("--ground_nodes", type=int, default=5,
#                         help="Temperature nodes through the upper-soil slab (>=2).")
#     parser.add_argument("--deep_ground_temp_c", type=float, default=24.0,
#                         help="Fixed temperature below the transient ground layer.")
#     parser.add_argument("--ground_deep_h", type=float, default=None,
#                         help="Lower soil conductance W/m2/K; default is ground k/depth.")
#     parser.add_argument("--ground_initial_temp_c", type=float, default=None,
#                         help="Initial uniform ground temperature; default is --deep_ground_temp_c.")
#     parser.add_argument("--ground_conductivity", type=float, default=1.5)
#     parser.add_argument("--ground_density", type=float, default=1150.0)
#     parser.add_argument("--ground_specific_heat", type=float, default=650.0)
#     parser.add_argument("--ground_albedo", type=float, default=DEFAULT_GROUND_ALBEDO)
#     parser.add_argument("--ground_emissivity", type=float, default=0.95)
#     args = parser.parse_args()
#     try:
#         wall_settings = WallSettings(args.wall_thickness_m, args.wall_nodes,
#                                      args.indoor_temp_c, args.indoor_h)
#         ground_settings = GroundSettings(
#             args.ground_depth_m, args.ground_nodes, args.deep_ground_temp_c,
#             args.ground_deep_h, args.ground_conductivity, args.ground_density,
#             args.ground_specific_heat, args.ground_albedo, args.ground_emissivity)
#     except ValueError as error:
#         parser.error(str(error))

#     weather_dir = args.weather_dir
#     if weather_dir is None and args.weather_root is not None:
#         weather_dir = resolve_weather_folder(args.weather_root, args.month, args.day)

#     if args.buildings_obj is not None:
#         if weather_dir is None:
#             raise SystemExit("--buildings_obj requires --weather_dir or --weather_root "
#                               "(the OBJ pipeline only supports the real-weather-data path).")
#         analysis = run_analysis_from_obj(
#             args.buildings_obj, weather_dir, trees_obj_path=args.trees_obj,
#             ground_obj_path=args.ground_obj,
#             consider_shadows=not args.no_shadows, shadow_samples=args.shadow_samples,
#             wall_settings=wall_settings, t_initial_c=args.initial_temp_c,
#             ground_settings=ground_settings, ground_initial_c=args.ground_initial_temp_c)
#         face_df, building_df, faces = analysis
#         if args.vtk_out is not None:
#             export_vtk_series_from_obj(args.buildings_obj, weather_dir, args.vtk_out,
#                 trees_obj_path=args.trees_obj, ground_obj_path=args.ground_obj,
#                 consider_shadows=not args.no_shadows,
#                 shadow_samples=args.shadow_samples, analysis=analysis)
#         face_df.to_csv(f"{args.out_prefix}_faces_hourly.csv", index=False)
#         building_df.to_csv(f"{args.out_prefix}_building_hourly.csv", index=False)
#         print(f"Wrote {args.out_prefix}_faces_hourly.csv and {args.out_prefix}_building_hourly.csv")
#         print(building_df[["hour", "T_amb_C", "T_eq_C", "T_building_C"]].round(2).to_string(index=False))
#         if args.plot:
#             plot_results(face_df, building_df, args.out_prefix)
#         raise SystemExit(0)

#     if args.vtk_out is not None:
#         if weather_dir is None:
#             raise SystemExit("--vtk_out requires --weather_dir or --weather_root "
#                               "(the ParaView export runs on the real-weather-data path).")
#         export_vtk_series(args.json, weather_dir, args.vtk_out, consider_shadows=not args.no_shadows,
#                           wall_settings=wall_settings, t_initial_c=args.initial_temp_c)

#     if weather_dir is not None:
#         face_df, building_df, faces = run_analysis_from_folder(
#             args.json, weather_dir, consider_shadows=not args.no_shadows,
#             wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
#     else:
#         weather = load_weather(args.weather_csv)
#         d = date(args.year, args.month, args.day)
#         face_df, building_df, faces = run_analysis(
#             args.json, d, weather, utc_offset_hr=args.utc_offset,
#             consider_shadows=not args.no_shadows, wall_settings=wall_settings,
#             t_initial_c=args.initial_temp_c)

#     face_df.to_csv(f"{args.out_prefix}_faces_hourly.csv", index=False)
#     building_df.to_csv(f"{args.out_prefix}_building_hourly.csv", index=False)
#     print(f"Wrote {args.out_prefix}_faces_hourly.csv and {args.out_prefix}_building_hourly.csv")
#     for bid, sub in building_df.groupby("building_id"):
#         print(f"\n-- {bid} --")
#         print(sub[["hour", "T_amb_C", "T_eq_C", "T_building_C"]].round(2).to_string(index=False))

#     if args.plot:
#         plot_results(face_df, building_df, args.out_prefix)
