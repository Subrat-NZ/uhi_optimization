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
"""

import json
import math
import os
import re
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

            tp = f["materialProperties"]["thermalProperties"]
            faces.append(Face(
                building_id=bid,
                face_id=f["id"],
                surface_type=stype,
                normal=n,
                area=area,
                tilt_deg=tilt_deg,
                center=center,
                vertices=verts,
                thermal_conductivity=tp["thermal_conductivity"],
                density=tp["density"],
                specific_heat=tp["specific_heat"],
                albedo=tp["albedo"],
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

    def __post_init__(self):
        if not np.isfinite(self.depth_m) or self.depth_m <= 0:
            raise ValueError("ground depth must be finite and positive")
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
    to rho*cp*L; adjacent conductance is k/dx. No lateral cell coupling.
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
        _, inverse = np.unique(properties, axis=0, return_inverse=True)
        self.groups = [np.flatnonzero(inverse == i) for i in range(inverse.max()+1)]

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
            K[-1,-1] = g + self.settings.indoor_h_wm2k
            A = -K / root_c[:, None] / root_c[None, :]
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            exponent = np.exp(eigenvalues*dt_seconds)
            phi = np.full(n, dt_seconds, dtype=float)
            np.divide(np.expm1(eigenvalues*dt_seconds), eigenvalues,
                      out=phi, where=np.abs(eigenvalues) > 1e-15)
            b = np.zeros((len(ids), n))
            b[:,0] = external_source[ids]
            b[:,-1] = self.settings.indoor_h_wm2k*self.settings.indoor_temperature_c
            modes = (self.temperatures[ids]*root_c) @ eigenvectors
            forcing = (b/root_c) @ eigenvectors
            self.temperatures[ids] = ((modes*exponent + forcing*phi) @ eigenvectors.T) / root_c

    def diagnostics(self, external_h, external_source):
        """Instantaneous conductive/storage fluxes at the current output time."""
        t = self.temperatures
        q_outer = external_source-external_h*t[:,0]
        q_inner = self.settings.indoor_h_wm2k*(t[:,-1]-self.settings.indoor_temperature_c)
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

def _simulate(faces: list, capacitance_by_building: dict, records: list,
              ground_albedo: float = DEFAULT_GROUND_ALBEDO,
              occluders_by_building: Optional[dict] = None,
              t_initial_c: Optional[float] = None,
              ray_tracer=None, shadow_samples: int = 3,
              wall_settings: Optional[WallSettings] = None,
              surface_group: str = "building") -> tuple:
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
            fraction = triangle_sunlit_fraction(exposed, sun, ray_tracer, shadow_samples)
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
        external_h = h_c+h_rad
        external_source = q_solar-q_lw_ambient + external_h*t_amb

        if ri > 0:
            dt = (rec["hour"] - records[ri-1]["hour"])*3600
            if dt <= 0:
                raise ValueError("Weather times must be strictly increasing")
            slab.advance(dt, external_h, external_source)

        ts = slab.temperatures[:,0].copy()
        q_conv = h_c*(ts-t_amb)
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
            q_conv_out_Wm2=q_conv, q_longwave_out_Wm2=q_lw,
            wall_thickness_m=wall_settings.thickness_m,
            slab_thickness_m=wall_settings.thickness_m,
            T_indoor_air_C=wall_settings.indoor_temperature_c,
            T_back_boundary_C=wall_settings.indoor_temperature_c,
            h_indoor_Wm2K=wall_settings.indoor_h_wm2k,
            h_back_Wm2K=wall_settings.indoor_h_wm2k, **wall_fields)))
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
    scene = load_scene(json_path)
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
    scene = load_scene(json_path)
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

_SURFACE_TYPE_ID = {"wall": 0, "roof": 1, "floor": 2, "ceiling": 3, "ground": 6}
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


def _write_vtk_polydata(path: str, polygons: list, cell_scalars: dict) -> None:
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
    scene = load_scene(json_path)
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


def build_faces_from_obj(buildings_obj_path: str) -> tuple:
    """Recover buildings, retaining every nondegenerate triangle as a cell.

    Coplanar groups provide parent wall IDs and approximate bulk volumes only;
    visibility, irradiance, temperature and VTK values belong to each triangle.
    OBJ wall winding must point outward. Horizontal bottoms are excluded from
    outdoor physics, and roof normals are corrected upward for legacy exports.
    """
    V, F = load_obj_mesh(buildings_obj_path)
    components = _obj_components_by_shared_position(V, F)
    faces, face_triangles = [], {}
    capacitances, occluders = {}, {}
    for b_idx, tri_indices in enumerate(components):
        bid = f"building_{b_idx:02d}"
        comp_tris = [F[t] for t in tri_indices]
        groups = group_coplanar(V, comp_tris)
        if not groups:
            continue
        z_min = min(g["z_min"] for g in groups.values())
        z_max = max(g["z_max"] for g in groups.values())
        roof_area = sum(g["area"] * abs(g["normal"][2]) for g in groups.values()
                        if abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
                        and g["center"][2] > z_min + 1e-6)
        capacitances[bid] = (DEFAULT_MATERIAL["density"] * roof_area * (z_max-z_min)
                             * DEFAULT_MATERIAL["specific_heat"])
        occluders[bid] = _mesh_to_occluder(V, comp_tris, bid)
        for gi, g in enumerate(groups.values()):
            horizontal = abs(g["normal"][2]) > ROOF_NORMAL_Z_THRESHOLD
            bottom = horizontal and g["z_max"] <= z_min+1e-6
            stype = "floor" if bottom else ("roof" if horizontal else "wall")
            normal = g["normal"].copy()
            if (stype == "roof" and normal[2] < 0) or (stype == "floor" and normal[2] > 0):
                normal = -normal
            tilt = math.degrees(math.acos(np.clip(normal[2], -1, 1)))
            parent = f"{bid}_face_{gi:03d}_{stype}"
            for ti, tri in enumerate(g["triangles"]):
                verts = V[list(tri)]
                _, area = _triangle_normal_area(*verts)
                fid = f"{parent}_tri_{ti:06d}"
                faces.append(Face(building_id=bid, face_id=fid, parent_face_id=parent,
                    surface_type=stype, normal=normal, area=area, tilt_deg=tilt,
                    center=verts.mean(axis=0), vertices=verts,
                    **DEFAULT_MATERIAL,
                    albedo=DEFAULT_ALBEDO_ROOF if stype == "roof" else DEFAULT_ALBEDO_WALL))
                face_triangles[fid] = [tri]
    return faces, capacitances, face_triangles, occluders


def build_ground_faces_from_obj(ground_obj_path: str,
                                building_ray_tracer=None,
                                ground_settings: Optional[GroundSettings] = None) -> list:
    """Load exposed ground triangles and orient their normals upward.

    A vertical ray removes cells beneath a building roof. This prevents soil
    under a building footprint from receiving outdoor convection or appearing
    as a second coincident thermal surface in the exported scene.
    """
    settings = ground_settings or GroundSettings()
    vertices, triangles = load_obj_mesh(ground_obj_path)
    candidates = []
    for source_index, tri in enumerate(triangles):
        verts = vertices[list(tri)]
        normal, area = _triangle_normal_area(*verts)
        if normal is None:
            continue
        if normal[2] < 0:
            normal = -normal
            verts = verts[[0, 2, 1]]
        candidates.append((source_index, verts, normal, area))
    if not candidates:
        raise ValueError(f"Ground OBJ has no nondegenerate triangles: {ground_obj_path}")

    keep = np.ones(len(candidates), dtype=bool)
    if building_ray_tracer is not None:
        centers = np.array([item[1].mean(axis=0) for item in candidates])
        origins = centers + np.array([0.0, 0.0, _SHADOW_RAY_OFFSET])
        keep = ~building_ray_tracer.blocked(origins, np.array([0.0, 0.0, 1.0]))

    faces = []
    for retain, (source_index, verts, normal, area) in zip(keep, candidates):
        if not retain:
            continue
        fid = f"ground_tri_{source_index:06d}"
        faces.append(Face(
            building_id="ground", face_id=fid, parent_face_id="ground_surface",
            surface_type="ground", normal=normal, area=area,
            tilt_deg=math.degrees(math.acos(np.clip(normal[2], -1, 1))),
            center=verts.mean(axis=0), vertices=verts,
            thermal_conductivity=settings.thermal_conductivity,
            density=settings.density, specific_heat=settings.specific_heat,
            albedo=settings.albedo, emissivity=settings.emissivity))
    print(f"[ground] Loaded {len(faces)} exposed triangles; "
          f"removed {len(candidates)-len(faces)} beneath buildings", flush=True)
    return faces


def run_analysis_from_obj(buildings_obj_path: str, weather_dir: str,
                            trees_obj_path: Optional[str] = None,
                            consider_shadows: bool = True,
                            t_initial_c: Optional[float] = None,
                            shadow_samples: int = 3,
                            wall_settings: Optional[WallSettings] = None,
                            ground_obj_path: Optional[str] = None,
                            ground_initial_c: Optional[float] = None,
                            ground_settings: Optional[GroundSettings] = None) -> tuple:
    """Per-triangle exposure and transient conducting slab temperatures from OBJ.

    All building and tree triangles are opaque occluders, including other
    parts of the receiver's own building. If ground_obj_path is supplied,
    exposed ground triangles receive the same shadow and exterior balance plus
    a transient soil slab. Returns (cell_df, building_df, faces).
    """
    if shadow_samples not in (1, 3):
        raise ValueError("shadow_samples must be 1 or 3")
    faces, capacitances, _, building_occluders = build_faces_from_obj(buildings_obj_path)
    building_triangles = [tri for occ in building_occluders.values() if occ is not None
                          for tri in occ.triangles]
    building_tracer = TriangleRayTracer(building_triangles) if ground_obj_path else None
    ground_faces = (build_ground_faces_from_obj(ground_obj_path, building_tracer, ground_settings)
                    if ground_obj_path else [])
    tracer = None
    if consider_shadows:
        triangles = list(building_triangles)
        if trees_obj_path:
            tv, tf = load_obj_mesh(trees_obj_path)
            triangles.extend(tv[np.asarray(tf, dtype=int)].tolist())
        print(f"[shadows] Building ray acceleration for {len(triangles)} triangles", flush=True)
        tracer = TriangleRayTracer(triangles)
    wx = load_weather_folder(weather_dir)
    records = [dict(hour=wx["hour"][i], T_amb_C=wx["T_amb_C"][i], wind_speed_ms=wx["wind_speed_ms"][i],
                      sun_vec=wx["sun_vec"][i], altitude_deg=wx["altitude_deg"][i],
                      i_dn=wx["IDN_Wm2"][i], i_dh=wx["Idif_Wm2"][i])
               for i in range(len(wx["hour"]))]
    face_df, building_df = _simulate(faces, capacitances, records,
        t_initial_c=t_initial_c, ray_tracer=tracer, shadow_samples=shadow_samples,
        wall_settings=wall_settings)
    if ground_faces:
        settings = ground_settings or GroundSettings()
        ground_capacitance = (settings.density * settings.specific_heat * settings.depth_m
                              * sum(f.area for f in ground_faces))
        ground_df, _ = _simulate(
            ground_faces, {"ground": ground_capacitance}, records,
            t_initial_c=ground_initial_c, ray_tracer=tracer,
            shadow_samples=shadow_samples, wall_settings=settings.slab_settings(),
            surface_group="ground")
        ground_df["ground_depth_m"] = ground_df["slab_thickness_m"]
        ground_df["T_ground_bottom_C"] = ground_df["T_back_surface_C"]
        ground_df["T_deep_ground_C"] = ground_df["T_back_boundary_C"]
        ground_df["h_deep_ground_Wm2K"] = ground_df["h_back_Wm2K"]
        ground_df["q_deep_ground_out_Wm2"] = ground_df["q_back_out_Wm2"]
        ground_df["T_ground_mean_C"] = ground_df["T_slab_mean_C"]
        ground_df["ground_capacity_Jm2K"] = ground_df["slab_capacity_Jm2K"]
        ground_df["ground_capacity_JK"] = ground_df["slab_capacity_JK"]
        building_only = ["wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K",
                         "T_inner_C", "T_wall_mean_C", "q_indoor_out_Wm2",
                         "wall_capacity_Jm2K", "wall_capacity_JK"]
        ground_df[building_only] = np.nan
        face_df = pd.concat([face_df, ground_df], ignore_index=True, sort=False)
        faces = faces + ground_faces
    return face_df, building_df, faces


def export_vtk_series_from_obj(buildings_obj_path: str, weather_dir: str, out_dir: str,
                                 trees_obj_path: Optional[str] = None,
                                 consider_shadows: bool = True,
                                 shadow_samples: int = 3,
                                 analysis=None,
                                 wall_settings: Optional[WallSettings] = None,
                                 t_initial_c: Optional[float] = None,
                                 ground_obj_path: Optional[str] = None,
                                 ground_settings: Optional[GroundSettings] = None,
                                 ground_initial_c: Optional[float] = None) -> str:
    """Export per-triangle CELL_DATA; optionally reuse an existing simulation.

    T_C aliases T_surface_C (transient outer surface temperature), while
    T_solair_C is retained separately. sunlit_fraction is 0..1; shadowed
    means at least one sample was blocked on a sun-facing triangle.
    Tree geometry is displayed at ambient temperature, not thermally solved.
    Ground cells have is_ground=1 and surface_type_id=6.
    """
    if analysis is None:
        analysis = run_analysis_from_obj(buildings_obj_path, weather_dir,
            trees_obj_path=trees_obj_path, ground_obj_path=ground_obj_path,
            consider_shadows=consider_shadows,
            shadow_samples=shadow_samples, wall_settings=wall_settings,
            t_initial_c=t_initial_c, ground_settings=ground_settings,
            ground_initial_c=ground_initial_c)
    face_df, building_df, faces = analysis
    os.makedirs(out_dir, exist_ok=True)
    polygons = [f.vertices.tolist() for f in faces]
    tree_polys = []
    if trees_obj_path:
        tv, tf = load_obj_mesh(trees_obj_path)
        tree_polys = tv[np.asarray(tf, dtype=int)].tolist()
    polygons.extend(tree_polys)
    nt = len(tree_polys)
    ids = [f.face_id for f in faces]
    building_ids = dict.fromkeys(f.building_id for f in faces if f.surface_type != "ground")
    bmap = {bid: i for i, bid in enumerate(building_ids)}
    pmap = {pid: i for i, pid in enumerate(dict.fromkeys(f.parent_face_id for f in faces))}
    static = {
        "surface_type_id": ([ _SURFACE_TYPE_ID.get(f.surface_type, -1) for f in faces] + [TREE_CROWN_TYPE_ID]*nt, "int"),
        "is_building": ([int(f.surface_type != "ground") for f in faces] + [0]*nt, "int"),
        "is_ground": ([int(f.surface_type == "ground") for f in faces] + [0]*nt, "int"),
        "building_index": ([bmap.get(f.building_id, -1) for f in faces] + [-1]*nt, "int"),
        "parent_face_index": ([pmap[f.parent_face_id] for f in faces] + [-1]*nt, "int"),
    }
    fields = ["T_surface_C", "T_solair_C", "I_direct_Wm2", "I_total_Wm2",
              "sunlit_fraction", "shadow_fraction", "shadowed", "sunlit",
              "h_conv_Wm2K", "q_solar_absorbed_Wm2", "q_conv_out_Wm2", "q_longwave_out_Wm2",
              "T_inner_C", "T_wall_mean_C", "q_conduction_in_Wm2", "q_indoor_out_Wm2",
              "q_storage_Wm2", "q_outer_node_storage_Wm2", "stored_heat_Jm2",
              "stored_heat_J", "wall_capacity_Jm2K", "wall_capacity_JK",
              "wall_thickness_m", "T_indoor_air_C", "h_indoor_Wm2K",
              "T_back_surface_C", "T_slab_mean_C", "q_back_out_Wm2",
              "slab_capacity_Jm2K", "slab_capacity_JK", "slab_thickness_m",
              "T_back_boundary_C", "h_back_Wm2K", "ground_depth_m",
              "T_ground_bottom_C", "T_deep_ground_C", "h_deep_ground_Wm2K",
              "q_deep_ground_out_Wm2", "T_ground_mean_C",
              "ground_capacity_Jm2K", "ground_capacity_JK"]
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
            scalars[name] = (np.concatenate([values, np.full(nt, fill)]).astype(int if dtype == "int" else float), dtype)
        scalars["T_C"] = scalars["T_surface_C"]
        fname = f"frame_{int(round(h*100)):04d}.vtk"
        _write_vtk_polydata(os.path.join(out_dir, fname), polygons, scalars)
        entries.append((float(h), fname))
    pvd_path = os.path.join(out_dir, "scene.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for t, fname in entries:
            f.write(f'<DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')
    print(f"[vtk export] Wrote {len(entries)} frame(s) + scene.pvd to {out_dir}.", flush=True)
    return pvd_path


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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Analytical solar, conduction and storage model for buildings and ground")
    parser.add_argument("--json", default="building_scene.json")
    parser.add_argument("--buildings_obj", default=None,
                         help="Use the OBJ-based pipeline instead of --json: path to a "
                              "buildings_refined.obj from json_to_obj.py. Requires --weather_dir "
                              "or --weather_root (see run_analysis_from_obj doc).")
    parser.add_argument("--shadow_samples", type=int, choices=(1, 3), default=3,
                         help="Rays per OBJ triangle: 3 (default) estimates partial shade; 1 uses the centroid.")
    parser.add_argument("--trees_obj", default=None,
                         help="Matching trees.obj from json_to_obj.py (optional, OBJ pipeline only).")
    parser.add_argument("--ground_obj", default=None,
                         help="Matching ground.obj; enables per-triangle ground shadow, "
                              "surface-temperature, conduction and heat-storage calculations.")
    parser.add_argument("--weather_csv", default=None,
                         help="CSV with hour,T_amb_C,wind_speed_ms[,GHI_Wm2,DNI_Wm2,DHI_Wm2] "
                              "(clear-sky/solar-formula path, --json only)")
    parser.add_argument("--weather_dir", default=None,
                         help="Path directly to a weatherData/MMDD folder (real-data path, preferred)")
    parser.add_argument("--weather_root", default=None,
                         help="Path to the weatherData/ root; combined with --month/--day to find MMDD")
    parser.add_argument("--month", type=int, default=6)
    parser.add_argument("--day", type=int, default=21)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--utc_offset", type=float, default=0.0)
    parser.add_argument("--out_prefix", default="results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no_shadows", action="store_true",
                         help="Skip mutual shadow-casting between buildings/trees (faster; "
                              "fine if the scene has only one isolated building).")
    parser.add_argument("--vtk_out", default=None,
                         help="Directory to write a ParaView-ready VTK time series to (requires "
                              "--weather_dir or --weather_root; see export_vtk_series doc).")
    parser.add_argument("--wall_thickness_m", type=float, default=0.23)
    parser.add_argument("--wall_nodes", type=int, default=5,
                        help="Temperature nodes through each wall/roof thickness (>=2).")
    parser.add_argument("--indoor_temp_c", type=float, default=24.0)
    parser.add_argument("--indoor_h", type=float, default=8.0,
                        help="Inner convection coefficient W/m2/K; 0 gives an insulated inner face.")
    parser.add_argument("--initial_temp_c", type=float, default=None,
                        help="Initial uniform slab temperature; default is --indoor_temp_c.")
    parser.add_argument("--ground_depth_m", type=float, default=0.23,
                        help="Depth of the transient upper-soil slab in metres.")
    parser.add_argument("--ground_nodes", type=int, default=5,
                        help="Temperature nodes through the upper-soil slab (>=2).")
    parser.add_argument("--deep_ground_temp_c", type=float, default=24.0,
                        help="Fixed temperature below the transient ground layer.")
    parser.add_argument("--ground_deep_h", type=float, default=None,
                        help="Lower soil conductance W/m2/K; default is ground k/depth.")
    parser.add_argument("--ground_initial_temp_c", type=float, default=None,
                        help="Initial uniform ground temperature; default is --deep_ground_temp_c.")
    parser.add_argument("--ground_conductivity", type=float, default=1.5)
    parser.add_argument("--ground_density", type=float, default=1150.0)
    parser.add_argument("--ground_specific_heat", type=float, default=650.0)
    parser.add_argument("--ground_albedo", type=float, default=DEFAULT_GROUND_ALBEDO)
    parser.add_argument("--ground_emissivity", type=float, default=0.95)
    args = parser.parse_args()
    try:
        wall_settings = WallSettings(args.wall_thickness_m, args.wall_nodes,
                                     args.indoor_temp_c, args.indoor_h)
        ground_settings = GroundSettings(
            args.ground_depth_m, args.ground_nodes, args.deep_ground_temp_c,
            args.ground_deep_h, args.ground_conductivity, args.ground_density,
            args.ground_specific_heat, args.ground_albedo, args.ground_emissivity)
    except ValueError as error:
        parser.error(str(error))

    weather_dir = args.weather_dir
    if weather_dir is None and args.weather_root is not None:
        weather_dir = resolve_weather_folder(args.weather_root, args.month, args.day)

    if args.buildings_obj is not None:
        if weather_dir is None:
            raise SystemExit("--buildings_obj requires --weather_dir or --weather_root "
                              "(the OBJ pipeline only supports the real-weather-data path).")
        analysis = run_analysis_from_obj(
            args.buildings_obj, weather_dir, trees_obj_path=args.trees_obj,
            ground_obj_path=args.ground_obj,
            consider_shadows=not args.no_shadows, shadow_samples=args.shadow_samples,
            wall_settings=wall_settings, t_initial_c=args.initial_temp_c,
            ground_settings=ground_settings, ground_initial_c=args.ground_initial_temp_c)
        face_df, building_df, faces = analysis
        if args.vtk_out is not None:
            export_vtk_series_from_obj(args.buildings_obj, weather_dir, args.vtk_out,
                trees_obj_path=args.trees_obj, ground_obj_path=args.ground_obj,
                consider_shadows=not args.no_shadows,
                shadow_samples=args.shadow_samples, analysis=analysis)
        face_df.to_csv(f"{args.out_prefix}_faces_hourly.csv", index=False)
        building_df.to_csv(f"{args.out_prefix}_building_hourly.csv", index=False)
        print(f"Wrote {args.out_prefix}_faces_hourly.csv and {args.out_prefix}_building_hourly.csv")
        print(building_df[["hour", "T_amb_C", "T_eq_C", "T_building_C"]].round(2).to_string(index=False))
        if args.plot:
            plot_results(face_df, building_df, args.out_prefix)
        raise SystemExit(0)

    if args.vtk_out is not None:
        if weather_dir is None:
            raise SystemExit("--vtk_out requires --weather_dir or --weather_root "
                              "(the ParaView export runs on the real-weather-data path).")
        export_vtk_series(args.json, weather_dir, args.vtk_out, consider_shadows=not args.no_shadows,
                          wall_settings=wall_settings, t_initial_c=args.initial_temp_c)

    if weather_dir is not None:
        face_df, building_df, faces = run_analysis_from_folder(
            args.json, weather_dir, consider_shadows=not args.no_shadows,
            wall_settings=wall_settings, t_initial_c=args.initial_temp_c)
    else:
        weather = load_weather(args.weather_csv)
        d = date(args.year, args.month, args.day)
        face_df, building_df, faces = run_analysis(
            args.json, d, weather, utc_offset_hr=args.utc_offset,
            consider_shadows=not args.no_shadows, wall_settings=wall_settings,
            t_initial_c=args.initial_temp_c)

    face_df.to_csv(f"{args.out_prefix}_faces_hourly.csv", index=False)
    building_df.to_csv(f"{args.out_prefix}_building_hourly.csv", index=False)
    print(f"Wrote {args.out_prefix}_faces_hourly.csv and {args.out_prefix}_building_hourly.csv")
    for bid, sub in building_df.groupby("building_id"):
        print(f"\n-- {bid} --")
        print(sub[["hour", "T_amb_C", "T_eq_C", "T_building_C"]].round(2).to_string(index=False))

    if args.plot:
        plot_results(face_df, building_df, args.out_prefix)
