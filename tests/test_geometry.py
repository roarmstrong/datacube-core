from __future__ import absolute_import

import numpy as np
import osgeo
import pytest

try:
    import cPickle as pickle
except ImportError:
    import pickle

from datacube.utils import geometry


def test_pickleable():
    poly = geometry.polygon([(10, 20), (20, 20), (20, 10), (10, 20)], crs=geometry.CRS('EPSG:4326'))
    pickled = pickle.dumps(poly, pickle.HIGHEST_PROTOCOL)
    unpickled = pickle.loads(pickled)
    assert poly == unpickled


def test_geobox_simple():
    from affine import Affine
    t = geometry.GeoBox(4000, 4000,
                        Affine(0.00025, 0.0, 151.0, 0.0, -0.00025, -29.0),
                        geometry.CRS('EPSG:4326'))

    expect_lon = np.asarray([151.000125,  151.000375,  151.000625,  151.000875,  151.001125,
                             151.001375,  151.001625,  151.001875,  151.002125,  151.002375])

    expect_lat = np.asarray([-29.000125, -29.000375, -29.000625, -29.000875, -29.001125,
                             -29.001375, -29.001625, -29.001875, -29.002125, -29.002375])
    expect_resolution = np.asarray([-0.00025, 0.00025])

    assert t.coordinates['latitude'].values.shape == (4000,)
    assert t.coordinates['longitude'].values.shape == (4000,)

    np.testing.assert_almost_equal(t.resolution, expect_resolution)
    np.testing.assert_almost_equal(t.coords['latitude'].values[:10], expect_lat)
    np.testing.assert_almost_equal(t.coords['longitude'].values[:10], expect_lon)


def test_props():
    box1 = geometry.box(10, 10, 30, 30, crs=geometry.CRS('EPSG:4326'))
    assert box1
    assert box1.is_valid
    assert not box1.is_empty
    assert box1.area == 400.0
    assert box1.boundary.length == 80.0
    assert box1.centroid == geometry.point(20, 20, geometry.CRS('EPSG:4326'))

    triangle = geometry.polygon([(10, 20), (20, 20), (20, 10), (10, 20)], crs=geometry.CRS('EPSG:4326'))
    assert triangle.envelope == geometry.BoundingBox(10, 10, 20, 20)

    outer = next(iter(box1))
    assert outer.length == 80.0

    box1copy = geometry.box(10, 10, 30, 30, crs=geometry.CRS('EPSG:4326'))
    assert box1 == box1copy
    assert box1.convex_hull == box1copy  # NOTE: this might fail because of point order

    box2 = geometry.box(20, 10, 40, 30, crs=geometry.CRS('EPSG:4326'))
    assert box1 != box2


def test_tests():
    box1 = geometry.box(10, 10, 30, 30, crs=geometry.CRS('EPSG:4326'))
    box2 = geometry.box(20, 10, 40, 30, crs=geometry.CRS('EPSG:4326'))
    box3 = geometry.box(30, 10, 50, 30, crs=geometry.CRS('EPSG:4326'))
    box4 = geometry.box(40, 10, 60, 30, crs=geometry.CRS('EPSG:4326'))
    minibox = geometry.box(15, 15, 25, 25, crs=geometry.CRS('EPSG:4326'))

    assert not box1.touches(box2)
    assert box1.touches(box3)
    assert not box1.touches(box4)

    assert box1.intersects(box2)
    assert box1.intersects(box3)
    assert not box1.intersects(box4)

    assert not box1.crosses(box2)
    assert not box1.crosses(box3)
    assert not box1.crosses(box4)

    assert not box1.disjoint(box2)
    assert not box1.disjoint(box3)
    assert box1.disjoint(box4)

    assert box1.contains(minibox)
    assert not box1.contains(box2)
    assert not box1.contains(box3)
    assert not box1.contains(box4)

    assert minibox.within(box1)
    assert not box1.within(box2)
    assert not box1.within(box3)
    assert not box1.within(box4)


def test_ops():
    box1 = geometry.box(10, 10, 30, 30, crs=geometry.CRS('EPSG:4326'))
    box2 = geometry.box(20, 10, 40, 30, crs=geometry.CRS('EPSG:4326'))
    box4 = geometry.box(40, 10, 60, 30, crs=geometry.CRS('EPSG:4326'))

    union1 = box1.union(box2)
    assert union1.area == 600.0

    inter1 = box1.intersection(box2)
    assert bool(inter1)
    assert inter1.area == 200.0

    inter2 = box1.intersection(box4)
    assert not bool(inter2)
    assert inter2.is_empty
    # assert not inter2.is_valid  TODO: what's going on here?

    diff1 = box1.difference(box2)
    assert diff1.area == 200.0

    symdiff1 = box1.symmetric_difference(box2)
    assert symdiff1.area == 400.0


def test_unary_union():
    box1 = geometry.box(10, 10, 30, 30, crs=geometry.CRS('EPSG:4326'))
    box2 = geometry.box(20, 10, 40, 30, crs=geometry.CRS('EPSG:4326'))
    box3 = geometry.box(30, 10, 50, 30, crs=geometry.CRS('EPSG:4326'))
    box4 = geometry.box(40, 10, 60, 30, crs=geometry.CRS('EPSG:4326'))

    union0 = geometry.unary_union([box1])
    assert union0 == box1

    union1 = geometry.unary_union([box1, box4])
    assert union1.type == 'MultiPolygon'
    assert union1.area == 2.0 * box1.area

    union2 = geometry.unary_union([box1, box2])
    assert union2.type == 'Polygon'
    assert union2.area == 1.5 * box1.area

    union3 = geometry.unary_union([box1, box2, box3, box4])
    assert union3.type == 'Polygon'
    assert union3.area == 2.5 * box1.area

    union4 = geometry.unary_union([union1, box2, box3])
    assert union4.type == 'Polygon'
    assert union4.area == 2.5 * box1.area


def test_unary_intersection():
    box1 = geometry.box(10, 10, 30, 30, crs=geometry.CRS('EPSG:4326'))
    box2 = geometry.box(15, 10, 35, 30, crs=geometry.CRS('EPSG:4326'))
    box3 = geometry.box(20, 10, 40, 30, crs=geometry.CRS('EPSG:4326'))
    box4 = geometry.box(25, 10, 45, 30, crs=geometry.CRS('EPSG:4326'))
    box5 = geometry.box(30, 10, 50, 30, crs=geometry.CRS('EPSG:4326'))
    box6 = geometry.box(35, 10, 55, 30, crs=geometry.CRS('EPSG:4326'))

    inter1 = geometry.unary_intersection([box1])
    assert bool(inter1)
    assert inter1 == box1

    inter2 = geometry.unary_intersection([box1, box2])
    assert bool(inter2)
    assert inter2.area == 300.0

    inter3 = geometry.unary_intersection([box1, box2, box3])
    assert bool(inter3)
    assert inter3.area == 200.0

    inter4 = geometry.unary_intersection([box1, box2, box3, box4])
    assert bool(inter4)
    assert inter4.area == 100.0

    inter5 = geometry.unary_intersection([box1, box2, box3, box4, box5])
    assert bool(inter5)
    assert inter5.type == 'LineString'
    assert inter5.length == 20.0

    inter6 = geometry.unary_intersection([box1, box2, box3, box4, box5, box6])
    assert not bool(inter6)
    assert inter6.is_empty


class TestCRSEqualityComparisons(object):
    def test_sinusoidal_comparison(self):
        a = geometry.CRS("""PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",
                           DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],
                           PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],
                           PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],
                           PARAMETER["false_northing",0],UNIT["Meter",1]]""")
        b = geometry.CRS("""PROJCS["unnamed",GEOGCS["unnamed ellipse",
                           DATUM["unknown",SPHEROID["unnamed",6371007.181,0]],
                           PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Sinusoidal"],
                           PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],
                           PARAMETER["false_northing",0],UNIT["Meter",1]]""")
        c = geometry.CRS('+a=6371007.181 +b=6371007.181 +units=m +y_0=0 +proj=sinu +lon_0=0 +no_defs +x_0=0')
        assert a == b
        assert a == c
        assert b == c

        assert a != geometry.CRS('EPSG:4326')

    def test_grs80_comparison(self):
        a = geometry.CRS("""GEOGCS["GEOCENTRIC DATUM of AUSTRALIA",DATUM["GDA94",SPHEROID["GRS80",6378137,298.257222101]],
                            PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]""")
        b = geometry.CRS("""GEOGCS["GRS 1980(IUGG, 1980)",DATUM["unknown",SPHEROID["GRS80",6378137,298.257222101]],
                            PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]""")
        c = geometry.CRS('+proj=longlat +no_defs +ellps=GRS80')
        assert a == b
        assert a == c
        assert b == c

        assert a != geometry.CRS('EPSG:4326')

    def test_australian_albers_comparison(self):
        a = geometry.CRS("""PROJCS["GDA94_Australian_Albers",GEOGCS["GCS_GDA_1994",
                            DATUM["Geocentric_Datum_of_Australia_1994",SPHEROID["GRS_1980",6378137,298.257222101]],
                            PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],
                            PROJECTION["Albers_Conic_Equal_Area"],
                            PARAMETER["standard_parallel_1",-18],
                            PARAMETER["standard_parallel_2",-36],
                            PARAMETER["latitude_of_center",0],
                            PARAMETER["longitude_of_center",132],
                            PARAMETER["false_easting",0],
                            PARAMETER["false_northing",0],
                            UNIT["Meter",1]]""")
        b = geometry.CRS('EPSG:3577')

        assert a == b

        assert a != geometry.CRS('EPSG:4326')


def test_geobox():
    points_list = [
        [(148.2697, -35.20111), (149.31254, -35.20111), (149.31254, -36.331431), (148.2697, -36.331431)],
        [(148.2697, 35.20111), (149.31254, 35.20111), (149.31254, 36.331431), (148.2697, 36.331431)],
        [(-148.2697, 35.20111), (-149.31254, 35.20111), (-149.31254, 36.331431), (-148.2697, 36.331431)],
        [(-148.2697, -35.20111), (-149.31254, -35.20111), (-149.31254, -36.331431), (-148.2697, -36.331431),
         (148.2697, -35.20111)],
    ]
    for points in points_list:
        polygon = geometry.polygon(points, crs=geometry.CRS('EPSG:3577'))
        resolution = (-25, 25)
        geobox = geometry.GeoBox.from_geopolygon(polygon, resolution)

        assert abs(resolution[0]) > abs(geobox.extent.boundingbox.left - polygon.boundingbox.left)
        assert abs(resolution[0]) > abs(geobox.extent.boundingbox.right - polygon.boundingbox.right)
        assert abs(resolution[1]) > abs(geobox.extent.boundingbox.top - polygon.boundingbox.top)
        assert abs(resolution[1]) > abs(geobox.extent.boundingbox.bottom - polygon.boundingbox.bottom)


@pytest.mark.xfail(tuple(int(i) for i in osgeo.__version__.split('.')) < (2, 2),
                   reason='Fails under GDAL 2.1')
def test_wrap_dateline():
    sinus_crs = geometry.CRS("""PROJCS["unnamed",
                           GEOGCS["Unknown datum based upon the custom spheroid",
                           DATUM["Not specified (based on custom spheroid)", SPHEROID["Custom spheroid",6371007.181,0]],
                           PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],
                           PROJECTION["Sinusoidal"],
                           PARAMETER["longitude_of_center",0],
                           PARAMETER["false_easting",0],
                           PARAMETER["false_northing",0],
                           UNIT["Meter",1]]""")
    albers_crs = geometry.CRS('EPSG:3577')
    geog_crs = geometry.CRS('EPSG:4326')

    wrap = geometry.polygon([(12231455.716333, -5559752.598333),
                             (12231455.716333, -4447802.078667),
                             (13343406.236, -4447802.078667),
                             (13343406.236, -5559752.598333),
                             (12231455.716333, -5559752.598333)], crs=sinus_crs)
    wrapped = wrap.to_crs(geog_crs)
    assert wrapped.type == 'Polygon'
    wrapped = wrap.to_crs(geog_crs, wrapdateline=True)
    # assert wrapped.type == 'MultiPolygon' TODO: these cases are quite hard to implement.
    # hopefully GDAL's CutGeometryOnDateLineAndAddToMulti will be available through py API at some point

    wrap = geometry.polygon([(13343406.236, -5559752.598333),
                             (13343406.236, -4447802.078667),
                             (14455356.755667, -4447802.078667),
                             (14455356.755667, -5559752.598333),
                             (13343406.236, -5559752.598333)], crs=sinus_crs)
    wrapped = wrap.to_crs(geog_crs)
    assert wrapped.type == 'Polygon'
    wrapped = wrap.to_crs(geog_crs, wrapdateline=True)
    # assert wrapped.type == 'MultiPolygon' TODO: same as above

    wrap = geometry.polygon([(14455356.755667, -5559752.598333),
                             (14455356.755667, -4447802.078667),
                             (15567307.275333, -4447802.078667),
                             (15567307.275333, -5559752.598333),
                             (14455356.755667, -5559752.598333)], crs=sinus_crs)
    wrapped = wrap.to_crs(geog_crs)
    assert wrapped.type == 'Polygon'
    wrapped = wrap.to_crs(geog_crs, wrapdateline=True)
    # assert wrapped.type == 'MultiPolygon' TODO: same as above

    wrap = geometry.polygon([(3658653.1976781483, -4995675.379595791),
                             (4025493.916030875, -3947239.249752495),
                             (4912789.243100313, -4297237.125269571),
                             (4465089.861944263, -5313778.16975072),
                             (3658653.1976781483, -4995675.379595791)], crs=albers_crs)
    wrapped = wrap.to_crs(geog_crs)
    assert wrapped.type == 'Polygon'
    assert wrapped.intersects(geometry.line([(0, -90), (0, 90)], crs=geog_crs))
    wrapped = wrap.to_crs(geog_crs, wrapdateline=True)
    assert wrapped.type == 'MultiPolygon'
    assert not wrapped.intersects(geometry.line([(0, -90), (0, 90)], crs=geog_crs))


def test_3d_geometry_converted_to_2d_geometry():
    coordinates = [(115.8929714190001, -28.577007674999948, 0.0),
                   (115.90275429200005, -28.57698532699993, 0.0),
                   (115.90412631000004, -28.577577566999935, 0.0),
                   (115.90157040700001, -28.58521105999995, 0.0),
                   (115.89382838900008, -28.585473711999953, 0.0),
                   (115.8929714190001, -28.577007674999948, 0.0)]
    geom_3d = {'coordinates': [coordinates],
               'type': 'Polygon'}
    geom_2d = {'coordinates': [[(x, y) for x, y, z in coordinates]],
               'type': 'Polygon'}

    g_2d = geometry.Geometry(geom_2d)
    g_3d = geometry.Geometry(geom_3d)

    assert {2} == set(len(pt) for pt in g_3d.boundary.coords)  # All coordinates are 2D

    assert g_2d == g_3d  # 3D geometry has been converted to a 2D by dropping the Z axis


def test_3d_point_converted_to_2d_point():
    point = (-35.5029340, 145.9312455, 0.0)

    point_3d = {'coordinates': point,
                'type': 'Point'}
    point_2d = {'coordinates': (point.x, point.y),
                'type': 'Point'}

    p_2d = geometry.Geometry(point_2d)
    p_3d = geometry.Geometry(point_3d)

    assert len(p_3d.coords) == 2

    assert p_2d == p_3d