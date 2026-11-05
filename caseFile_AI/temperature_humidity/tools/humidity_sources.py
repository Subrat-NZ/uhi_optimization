"""
humidity_sources.py  --  analytic water-vapour sources of ponds, lawns and trees

All fluxes are mass fluxes F [kg m-2 s-1] of vapour entering the air, evaluated from the weather of the target
hour (the reference air state Ta, q_a, p, wind) and written to 0/qFlux.  The convective coefficient is the one
of the thermal model, h_c = 5.7 + 3.8 u  [W m-2 K-1], turned into a mass-transfer conductance with the Lewis
analogy:   rho g_a = h_c / cp  ( g_a = 1/ra ).

POND   free water surface held at the wet-bulb temperature Tw of the weather file (prescribed):
           F = h_c (q_sat(Tw) - q_a) / cp
       (at Tw the sensible heat taken from the air, h_c (Ta - Tw), equals the latent heat, L F).

LAWN   always wet (hydrated: >= 5 mm of water in the top layer, checked against the hourly loss) -> evaporates
       at the potential rate, surface temperature from the wet-surface energy balance, solved exactly
       (bisection on  theta = Ts - Ta):
           Rni - h_r theta - h_g (Ts - T_deep) - h_c theta - L F = 0
           L F = (h_c / cp) L (q_sat(Ts) - q_a)         (F >= 0)
       Rni  isothermal net radiation of that ground triangle from the thermal model (albedo, shadows included)
       h_r  = 4 eps sigma Ta^3,   h_g = k / depth (conduction to the deep soil, as in the thermal model).

TREE   big-leaf Penman-Monteith with a Jarvis-type (Noilhan & Planton 1989) stomatal resistance, daytime only:
           L E = [Delta Rn + rho cp VPD / ra] / [Delta + gamma (1 + rc / ra)]      per m2 of projected crown
           ra = rho cp / h_c ,  rc = rs / LAI ,  rs = rs_min / (F1 F3 F4)
           F1 = (rs_min/rs_max + f)/(1 + f), f = 0.55 (Rg/RGL)(2/LAI);  F3 = max(F3_min, 1 - gD VPD[hPa]);
           F4 = max(F4_min, 1 - 0.0016 (298 - Ta)^2)
       Rn = (1 - albedo) Rg + eps (HIR - sigma Ta^4)  (zero at night -> no transpiration)
       The crown is a solid in the mesh, so the flux is spread over the crown surface:  F_surface = E * ratio,
       ratio = projected / surface area (1/4 for a sphere).
"""
import numpy as np

SIGMA = 5.670374419e-8


def e_sat_pa(t_c):
    return 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def q_sat(t_c, p):
    e = e_sat_pa(t_c)
    return 0.622 * e / (p - 0.378 * e)


def latent_heat(t_c):
    return 2.501e6 - 2370.0 * np.asarray(t_c, dtype=float)


def cp_air(q):
    return 1005.0 * (1.0 - q) + 1860.0 * q


def h_conv(wind):
    """Convective coefficient of solar_thermal_model.convective_h."""
    return 5.7 + 3.8 * wind


def pond_flux(t_water, ta, qa, p, hc):
    return np.maximum(hc * (q_sat(t_water, p) - qa) / cp_air(qa), 0.0)


def wet_surface(rni, ta, qa, p, hc, h_g, t_deep, eps=0.95, tol=1e-9):
    """Surface temperature (C) and vapour flux (kg m-2 s-1) of an always-wet surface. Arrays or scalars."""
    rni, hc, h_g, t_deep = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(rni, hc, h_g, t_deep))
    h_r = 4.0 * eps * SIGMA * (ta + 273.15) ** 3
    cp = cp_air(qa)

    def balance(theta):
        ts = ta + theta
        lat = latent_heat(ts) * np.maximum(hc * (q_sat(ts, p) - qa) / cp, 0.0)
        return rni - h_r * theta - h_g * (ts - t_deep) - hc * theta - lat

    lo = np.full(rni.shape, -60.0)
    hi = np.full(rni.shape, 60.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pos = balance(mid) > 0            # balance decreases with theta
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    theta = 0.5 * (lo + hi)
    ts = ta + theta
    flux = np.maximum(hc * (q_sat(ts, p) - qa) / cp, 0.0)
    return ts, flux, balance(theta)


def tree_transpiration(ta, qa, p, hc, rg, hir, sun_elevation_deg, lai=3.0, rs_min=150.0, rs_max=5000.0,
                       rgl=100.0, gD=0.025, f3_min=0.2, f4_min=0.1, albedo=0.20, eps=0.97,
                       crown_area_ratio=0.25, rho=None):
    """Transpiration of a tree crown. Returns dict(flux_surface, LE_projected, rs_leaf, rc, ra, Rn)."""
    ta_k = ta + 273.15
    rho = p / (287.05 * ta_k) if rho is None else rho
    cp = cp_air(qa)
    lam = latent_heat(ta)
    gamma = cp * p / (0.622 * lam)
    es = e_sat_pa(ta)
    ea = qa * p / (0.622 + 0.378 * qa)
    vpd = max(es - ea, 0.0)
    delta = (e_sat_pa(ta + 0.01) - e_sat_pa(ta - 0.01)) / 0.02          # Pa/K
    hir = hir if hir is not None and np.isfinite(hir) else SIGMA * ta_k ** 4 * 0.8
    rn = (1.0 - albedo) * rg + eps * (hir - SIGMA * ta_k ** 4)
    if sun_elevation_deg <= 0.0 or rg <= 0.0 or rn <= 0.0:
        return dict(flux_surface=0.0, LE_projected=0.0, rs_leaf=np.inf, rc=np.inf, ra=rho * cp / hc, Rn=float(rn))
    f = 0.55 * (rg / rgl) * (2.0 / lai)
    f1 = (rs_min / rs_max + f) / (1.0 + f)
    f3 = max(f3_min, 1.0 - gD * vpd / 100.0)
    f4 = max(f4_min, 1.0 - 0.0016 * (298.0 - ta_k) ** 2)
    rs = rs_min / (f1 * f3 * f4)
    rc = rs / lai
    ra = rho * cp / hc
    le = (delta * rn + rho * cp * vpd / ra) / (delta + gamma * (1.0 + rc / ra))
    le = max(le, 0.0)
    return dict(flux_surface=float(le / lam * crown_area_ratio), LE_projected=float(le), rs_leaf=float(rs),
                rc=float(rc), ra=float(ra), Rn=float(rn), F1=float(f1), F3=float(f3), F4=float(f4))
