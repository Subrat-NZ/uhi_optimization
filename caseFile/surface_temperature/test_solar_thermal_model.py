"""Regression checks for local shadows and the exterior heat balance."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import solar_thermal_model as model


def write_boxes(path, boxes, divisions=4):
    """Closed, outward-wound boxes with duplicated vertices across triangles."""
    lines = []
    count = 0
    for lower, upper in boxes:
        x, y, z = lower
        X, Y, Z = upper
        dx, dy, dz = np.subtract(upper, lower)
        sides = [((x,y,z),(0,0,dz),(0,dy,0)),
                 ((X,y,z),(0,dy,0),(0,0,dz)),
                 ((x,y,z),(dx,0,0),(0,0,dz)),
                 ((x,Y,z),(0,0,dz),(dx,0,0)),
                 ((x,y,Z),(dx,0,0),(0,dy,0)),
                 ((x,y,z),(0,dy,0),(dx,0,0))]
        for origin, u, v in sides:
            origin, u, v = map(np.array, (origin, u, v))
            for i in range(divisions):
                for j in range(divisions):
                    corners = [origin + u*a/divisions + v*b/divisions
                               for a,b in [(i,j),(i+1,j),(i+1,j+1),(i,j+1)]]
                    for indices in [(0,1,2),(0,2,3)]:
                        for k in indices:
                            lines.append('v ' + ' '.join(map(str,corners[k])))
                        lines.append(f'f {count+1} {count+2} {count+3}')
                        count += 3
    path.write_text('\n'.join(lines)+'\n')


def write_triangles(path, triangles):
    """Write independent, upward-wound OBJ triangles."""
    lines = []
    count = 0
    for tri in triangles:
        for point in tri:
            lines.append('v ' + ' '.join(map(str, point)))
        lines.append(f'f {count+1} {count+2} {count+3}')
        count += 3
    path.write_text('\n'.join(lines)+'\n')


class SolarThermalTests(unittest.TestCase):
    def test_slab_capacity_and_steady_resistance(self):
        settings = model.WallSettings()
        slab = model.SlabConduction([2.3], [2400], [880], [.9], 24, settings)
        self.assertAlmostEqual(slab.capacity.sum(), 2400 * 880 * .23)

        # Constant exterior air at 40 C, h=20; interior air at 24 C, h=8.
        slab.advance(30 * 86400, 20, np.array([20 * 40.]))
        flux = (40 - 24) / (1 / 20 + .23 / 2.3 + 1 / 8)
        outer = 40 - flux / 20
        inner = 24 + flux / 8
        np.testing.assert_allclose(
            slab.temperatures[0], np.linspace(outer, inner, 5), atol=1e-9)
        diagnostics = slab.diagnostics(20, np.array([800.]))
        np.testing.assert_allclose(
            diagnostics["q_conduction_in_Wm2"], flux, atol=1e-8)
        np.testing.assert_allclose(diagnostics["q_storage_Wm2"], 0, atol=1e-8)

    def test_analytical_update_semigroup_and_insulated_energy(self):
        settings = model.WallSettings(indoor_h_wm2k=0)
        full = model.SlabConduction([2.3], [2400], [880], [0], 20, settings)
        split = model.SlabConduction([2.3], [2400], [880], [0], 20, settings)
        full.advance(3600, 0, np.array([100.]))
        split.advance(900, 0, np.array([100.]))
        split.advance(2700, 0, np.array([100.]))
        np.testing.assert_allclose(full.temperatures, split.temperatures, atol=1e-10)
        diagnostics = full.diagnostics(0, np.array([100.]))
        np.testing.assert_allclose(
            diagnostics["stored_heat_Jm2"], 100 * 3600, atol=1e-6)
        np.testing.assert_allclose(diagnostics["q_storage_Wm2"], 100)

        # With no further input, conduction redistributes but conserves heat.
        before = full.temperatures.copy()
        full.advance(1800, 0, np.array([0.]))
        self.assertLess(full.temperatures[0, 0], before[0, 0])
        self.assertGreater(full.temperatures[0, -1], before[0, -1])
        np.testing.assert_allclose(
            full.diagnostics(0, np.array([0.]))["stored_heat_Jm2"],
            360000, atol=1e-6)

    def test_insulated_fourier_mode_and_spatial_convergence(self):
        errors = []
        for nodes in (5, 17):
            settings = model.WallSettings(nodes=nodes, indoor_h_wm2k=0)
            slab = model.SlabConduction([2.3], [2400], [880], [0], 25, settings)
            x = np.linspace(0, .23, nodes)
            slab.temperatures[0] = 25 + 5 * np.cos(np.pi * x / .23)
            slab.advance(3600, 0, np.array([0.]))
            alpha = 2.3 / (2400 * 880)
            exact = 25 + 5 * np.cos(np.pi * x / .23) * np.exp(
                -alpha * (np.pi / .23) ** 2 * 3600)
            errors.append(np.max(np.abs(slab.temperatures[0] - exact)))
        self.assertLess(errors[1], errors[0] / 10)
        self.assertLess(errors[1], .01)

    def test_invalid_wall_settings(self):
        for values in (dict(thickness_m=0), dict(nodes=1),
                       dict(indoor_h_wm2k=-1), dict(thickness_m=float("nan"))):
            with self.assertRaises(ValueError):
                model.WallSettings(**values)

        for values in (dict(depth_m=0), dict(nodes=1),
                       dict(deep_h_wm2k=-1), dict(albedo=1.1),
                       dict(thermal_conductivity=0)):
            with self.assertRaises(ValueError):
                model.GroundSettings(**values)

    def test_bvh_matches_brute_force_including_parallel_axes(self):
        rng = np.random.default_rng(1234)
        triangles = rng.uniform(-2, 2, (35,3,3))
        origins = rng.uniform(-3, 3, (80,3))
        tracer = model.TriangleRayTracer(triangles, leaf_size=4)
        for d in ([1,0,0], [0,0,1], [1,2,3], [-1,-2,.1]):
            direction = np.array(d)/np.linalg.norm(d)
            expected = [any(model._ray_triangle_hit(o,direction,*tri) for tri in triangles)
                        for o in origins]
            np.testing.assert_array_equal(tracer.blocked(origins,direction,chunk_size=17), expected)

    def test_partial_triangle_analytic_shadow_boundary(self):
        # Rays rise one metre per horizontal metre. Screen x=2, z<=3
        # shadows exactly z<1 on receiving plane x=0.
        tri = np.array([[0,0,0],[0,1,0],[0,0,2]], dtype=float)
        f = model.Face('receiver','cell','wall',np.array([1.,0,0]),1,90,
                       tri.mean(axis=0),tri,2.3,2400,880,.3)
        screen = np.array([[[2,-2,-5],[2,3,-5],[2,3,3]],
                           [[2,-2,-5],[2,3,3],[2,-2,3]]], dtype=float)
        fraction = model.triangle_sunlit_fraction([f],np.array([1.,0,1])/np.sqrt(2),
                                                  model.TriangleRayTracer(screen))
        self.assertAlmostEqual(fraction[0],1/3)

    def test_surface_energy_balance_and_wind_cooling(self):
        irradiance = np.array([0.,200.,800.])
        h = model.convective_h(2)
        ts = model.surface_temperature(25,irradiance,h,.3,.5)
        ta = 298.15
        lw = .9*model.STEFAN_BOLTZMANN*((ts+273.15)**4-.5*model.sky_temperature(ta)**4-.5*ta**4)
        np.testing.assert_allclose(.7*irradiance, h*(ts-25)+lw, atol=1e-7)
        self.assertTrue(np.all(np.diff(ts)>0))
        self.assertLess(model.surface_temperature(25,800,model.convective_h(6),.3,.5),ts[-1])
        self.assertAlmostEqual(float(model.surface_temperature(25,0,h,.3,0)),25)

    def test_obj_local_shadows_self_occlusion_and_vtk_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'boxes.obj'
            write_boxes(path,[((-1,0,0),(0,2,4)),((2,0,0),(3,2,3))])
            sun = np.array([1.,0,1])/np.sqrt(2)
            wx = dict(hour=np.array([12., 13.]), T_amb_C=[25., 25.],
                      wind_speed_ms=[2., 2.], sun_vec=[sun, sun],
                      altitude_deg=[45., 45.], IDN_Wm2=[800., 800.],
                      Idif_Wm2=[100., 100.])
            with patch.object(model,'load_weather_folder',return_value=wx):
                analysis = model.run_analysis_from_obj(
                    str(path), 'unused', t_initial_c=25)
                unshaded, _, _ = model.run_analysis_from_obj(
                    str(path), 'unused', consider_shadows=False, t_initial_c=25)
            df, bulk, faces = analysis
            self.assertEqual(len(bulk), 4)
            initial = df[df.hour == 12]
            np.testing.assert_allclose(initial.T_surface_C, 25)
            np.testing.assert_allclose(initial.stored_heat_Jm2, 0)
            df = df[df.hour == 13]
            self.assertTrue(all(len(f.vertices)==3 for f in faces))
            self.assertNotIn('floor',set(df.surface_type))
            receiver = [f.face_id for f in faces if np.allclose(f.normal,[1,0,0]) and abs(f.center[0])<1e-8]
            wall = df.set_index('face_id').loc[receiver]
            self.assertEqual(wall.parent_face_id.nunique(),1)
            self.assertTrue((wall.sunlit_fraction==0).any())
            self.assertTrue((wall.sunlit_fraction==1).any())
            self.assertGreater(wall.loc[wall.sunlit_fraction==1,'T_surface_C'].min(),
                               wall.loc[wall.sunlit_fraction==0,'T_surface_C'].max())
            self.assertTrue((unshaded.loc[unshaded.self_lit,'sunlit_fraction']==1).all())
            # Convex sun-facing roof of the front box must not shadow itself.
            front_roofs = [f.face_id for f in faces if f.surface_type=='roof' and f.center[0]>2]
            self.assertTrue((df.set_index('face_id').loc[front_roofs,'sunlit_fraction']==1).all())
            residual = (df.q_solar_absorbed_Wm2 - df.q_conv_out_Wm2
                        - df.q_longwave_out_Wm2 - df.q_indoor_out_Wm2
                        - df.q_storage_Wm2)
            self.assertLess(residual.abs().max(),1e-7)
            with patch.object(model,'run_analysis_from_obj',side_effect=AssertionError('Repeated simulation')):
                pvd = model.export_vtk_series_from_obj(str(path),'unused',directory,analysis=analysis)
            self.assertTrue(Path(pvd).exists())
            lines=(Path(directory)/'frame_1300.vtk').read_text().splitlines()
            n=len(faces)
            self.assertIn(f'CELL_DATA {n}',lines)
            def scalar(name):
                i=next(i for i,v in enumerate(lines) if v.startswith(f'SCALARS {name} '))
                return np.array(list(map(float,lines[i+2:i+2+n])))
            np.testing.assert_allclose(scalar('T_C'),scalar('T_surface_C'))
            expected=df.set_index('face_id').reindex([f.face_id for f in faces]).T_surface_C.fillna(25)
            np.testing.assert_allclose(scalar('T_C'),expected,atol=5e-5)
            self.assertGreater(np.ptp(scalar('sunlit_fraction')[[i for i,f in enumerate(faces) if f.face_id in receiver]]),0)

    def test_ground_shadow_storage_balance_and_vtk_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            buildings = Path(directory)/'building.obj'
            ground = Path(directory)/'ground.obj'
            write_boxes(buildings, [((0, 0, 0), (1, 1, 2))], divisions=1)
            write_triangles(ground, [
                # Removed because its centroid is beneath the building roof.
                ((.1, .1, 0), (.9, .1, 0), (.1, .9, 0)),
                # Building-shadow receiver west of the box for a +x,+z sun ray.
                ((-2, 0, 0), (-1, 0, 0), (-2, 1, 0)),
                # Unobstructed receiver, separated in y.
                ((-2, 3, 0), (-1, 3, 0), (-2, 4, 0)),
            ])
            sun = np.array([1., 0, 1])/np.sqrt(2)
            wx = dict(hour=np.array([12., 13.]), T_amb_C=[25., 25.],
                      wind_speed_ms=[2., 2.], sun_vec=[sun, sun],
                      altitude_deg=[45., 45.], IDN_Wm2=[800., 800.],
                      Idif_Wm2=[100., 100.])
            with patch.object(model, 'load_weather_folder', return_value=wx):
                analysis = model.run_analysis_from_obj(
                    str(buildings), 'unused', ground_obj_path=str(ground),
                    t_initial_c=25)
            df, bulk, faces = analysis
            ground_faces = [f for f in faces if f.surface_type == 'ground']
            self.assertEqual(len(ground_faces), 2)
            self.assertNotIn('ground', set(bulk.building_id))
            initial = df[(df.hour == 12) & (df.surface_group == 'ground')]
            np.testing.assert_allclose(initial.T_surface_C, 24)
            np.testing.assert_allclose(initial.stored_heat_Jm2, 0)
            ground_df = df[(df.hour == 13) & (df.surface_group == 'ground')]
            self.assertEqual(len(ground_df), 2)
            self.assertLess(ground_df.sunlit_fraction.min(), ground_df.sunlit_fraction.max())
            self.assertGreater(
                ground_df.loc[ground_df.sunlit_fraction.idxmax(), 'T_surface_C'],
                ground_df.loc[ground_df.sunlit_fraction.idxmin(), 'T_surface_C'])
            np.testing.assert_allclose(
                ground_df.slab_capacity_Jm2K, 1150 * 650 * .23)
            residual = (ground_df.q_solar_absorbed_Wm2 - ground_df.q_conv_out_Wm2
                        - ground_df.q_longwave_out_Wm2 - ground_df.q_back_out_Wm2
                        - ground_df.q_storage_Wm2)
            self.assertLess(residual.abs().max(), 1e-7)

            with patch.object(model, 'run_analysis_from_obj',
                              side_effect=AssertionError('Repeated simulation')):
                model.export_vtk_series_from_obj(
                    str(buildings), 'unused', directory, ground_obj_path=str(ground),
                    analysis=analysis)
            lines = (Path(directory)/'frame_1300.vtk').read_text().splitlines()
            n = len(faces)
            def scalar(name):
                i = next(i for i, value in enumerate(lines)
                         if value.startswith(f'SCALARS {name} '))
                return np.array(list(map(float, lines[i+2:i+2+n])))
            ground_mask = scalar('is_ground') == 1
            self.assertEqual(ground_mask.sum(), 2)
            self.assertTrue((scalar('surface_type_id')[ground_mask] == 6).all())
            self.assertTrue((scalar('is_building')[ground_mask] == 0).all())


if __name__ == '__main__':
    unittest.main()
