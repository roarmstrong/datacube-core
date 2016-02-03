# coding=utf-8
"""
Core classes used across modules.
"""
from __future__ import absolute_import, division

import logging
from collections import namedtuple
import os
from pathlib import Path

import numpy
import dateutil.parser
from affine import Affine
from osgeo import osr
from rasterio.coords import BoundingBox

from datacube import compat
from datacube.compat import parse_url

_LOG = logging.getLogger(__name__)

Range = namedtuple('Range', ('begin', 'end'))
Coordinate = namedtuple('Coordinate', ('dtype', 'begin', 'end', 'length', 'units'))
Variable = namedtuple('Variable', ('dtype', 'nodata', 'dimensions', 'units'))


def _uri_to_local_path(local_uri):
    """
    Platform dependent URI to Path
    :type local_uri: str
    :rtype: pathlib.Path

    For example on Unix:
    'file:///tmp/something.txt' -> '/tmp/something.txt'

    On Windows:
    'file:///C:/tmp/something.txt' -> 'C:\\tmp\\test.tmp'

    Only supports file:// schema URIs
    """
    if not local_uri:
        return None

    components = parse_url(local_uri)
    if components.scheme != 'file':
        raise ValueError('Only file URIs currently supported. Tried %r.' % components.scheme)

    path = _cross_platform_path(components.path)

    return Path(path)


def _cross_platform_path(path):
    if os.name == 'nt':
        return path[1:]
    else:
        return path


class DatasetMatcher(object):
    def __init__(self, metadata):
        # Match by exact metadata properties (a subset of the metadata doc)
        #: :type: dict
        self.metadata = metadata

    def __repr__(self):
        return "DatasetMatcher(metadata={!r})".format(self.metadata)


class StorageType(object):
    def __init__(self, name, description,
                 match, measurements,
                 location, filename_pattern, roi, definition, id_=None):
        # Which datasets to match.
        #: :type: DatasetMatcher
        self.match = match

        # A unique name for the storage type (specified by users)
        #: :type: str
        self.name = name

        # A human-readable, potentially multi-line, description for display on the UI.
        #: :type: str
        self.description = description

        # A dictionary of the measurements to store
        # (key is measurement id, value is a doc understood by the storage driver)
        #: :type: dict
        self.measurements = measurements

        # (Optional) Limited ROI for this storage type
        #: :type: dict
        self.roi = roi

        # The storage location where the storage units should be stored.
        # Defined in the users configuration file.
        #: :type: str
        self.location = location

        # Storage Unit filename pattern
        # TODO: define pattern expansion rules
        #: :type: str
        self.filename_pattern = filename_pattern

        self.definition = definition

        # Database primary key
        #: :type: int
        self.id_ = id_

    def local_uri_to_location_relative_path(self, uri):
        if not uri.startswith(self.location):
            raise ValueError('Not a local URI: %s', uri)
        return uri[len(self.location):]

    def resolve_location(self, offset):
        # We can't use urlparse.urljoin() because it takes a relative path, not a path inside the base.
        return '/'.join(s.strip('/') for s in (self.location, offset))

    @property
    def storage_pattern(self):
        return self.resolve_location(self.filename_pattern)

    @property
    def driver(self):
        return self.definition['driver']

    @property
    def crs(self):
        return str(self.definition['crs']).strip()

    @property
    def spatial_dimensions(self):
        """
        Latitude/Longitude or X/Y
        :rtype: tuple
        """
        sr = osr.SpatialReference(self.crs)
        if sr.IsGeographic():
            return 'longitude', 'latitude'
        elif sr.IsProjected():
            return 'x', 'y'

    @property
    def tile_size(self):
        """
        :return: tuple(x size, y size)
        """
        tile_size = self.definition['tile_size']
        return [tile_size[dim] for dim in self.spatial_dimensions]

    @property
    def resolution(self):
        """
        :return: tuple(x res, y res)
        """
        res = self.definition['resolution']
        return [res[dim] for dim in self.spatial_dimensions]

    @property
    def pixel_size(self):
        return [int(size / abs(res)) for size, res in zip(self.tile_size, self.resolution)]

    @property
    def chunking(self):
        chunks = self.definition['chunking']
        return [(dim, chunks[dim]) for dim in self.definition['dimension_order']]

    @property
    def filename_format(self):
        return self.definition['filename_format']

    def __repr__(self):
        return 'StorageType<name={!r}, id_={!r}>'.format(self.name, self.id_)


class StorageUnit(object):
    def __init__(self, dataset_ids, storage_type, descriptor, path, id_=None):
        #: :type: list[uuid.UUID]
        self.dataset_ids = dataset_ids

        #: :type: StorageType
        self.storage_type = storage_type

        # A descriptor for this segment. (parameters etc)
        # A 'document' understandable by the storage driver. Properties inside may be queried by users.
        #: :type: dict
        self.descriptor = descriptor

        # An offset from the location defined in the storage type.
        #: :type: pathlib.Path
        self.path = path

        # Database primary key
        #: :type: int
        self.id_ = id_

    @property
    def local_path(self):
        file_uri = self.storage_type.resolve_location(self.path)
        return _uri_to_local_path(file_uri)

    def coordinate_values(self, coord):
        if coord == 'time':
            pass
        return None

    @property
    def tile_index(self):
        try:
            return self.descriptor['tile_index']
        except KeyError:
            return [int(min(self.coordinates[dim].begin, self.coordinates[dim].end) // size)
                    for size, dim in zip(self.storage_type.tile_size, self.storage_type.spatial_dimensions)]

    @property
    def affine(self):
        tile_index = self.tile_index
        tile_res = self.storage_type.resolution
        tile_size = self.storage_type.tile_size
        x = (tile_index[0] + (1 if tile_res[0] < 0 else 0)) * tile_size[0]
        y = (tile_index[1] + (1 if tile_res[1] < 0 else 0)) * tile_size[1]
        return Affine(tile_res[0], 0.0, x, 0.0, tile_res[1], y)

    @property
    def geographic_extents(self):
        width, height = self.storage_type.pixel_size
        extents = [(0, 0), (0, height), (width, height), (width, 0)]
        self.affine.itransform(extents)
        epsg4326 = osr.SpatialReference()
        epsg4326.ImportFromEPSG(4326)
        crs = osr.SpatialReference(self.storage_type.crs)
        transform = osr.CoordinateTransformation(crs, epsg4326)
        return transform.TransformPoints(extents)

    @property
    def coordinates(self):
        #: :rtype: dict[str, Coordinate]
        return {name: Coordinate(dtype=numpy.dtype(attributes['dtype']),
                                 begin=attributes['begin'],
                                 end=attributes['end'],
                                 length=attributes['length'],
                                 units=attributes.get('units', None))
                for name, attributes in self.descriptor['coordinates'].items()}

    def __str__(self):
        return "StorageUnit <type={m.name}, path={path}>".format(path=self.path, m=self.storage_type)

    def __repr__(self):
        return "StorageUnit({!r}, {!r}, {!r}, {!r}, {!r})".format(self.dataset_ids, self.storage_type,
                                                                  self.descriptor, self.path, self.id_)


class Dataset(object):
    def __init__(self, collection, metadata_doc, local_uri):
        """
        A dataset on disk.

        :type collection: Collection
        :param metadata_doc: the document (typically a parsed json/yaml)
        :type metadata_doc: dict
        :param local_uri: A URI to access this dataset locally.
        :type local_uri: str
        """
        #: :type: Collection
        self.collection = collection

        self.metadata_type = collection.metadata_type if collection else None

        #: :type: dict
        self.metadata_doc = metadata_doc

        self.local_uri = local_uri

    @property
    def local_path(self):
        """
        A path to this dataset on the local filesystem (if available).

        :rtype: pathlib.Path
        """
        return _uri_to_local_path(self.local_uri)

    @property
    def id(self):
        return self.metadata_doc['id']

    @property
    def format(self):
        return self.metadata_doc['format']['name']

    @property
    def time(self):
        center_dt = self.metadata_doc['extent']['center_dt']
        if isinstance(center_dt, compat.string_types):
            center_dt = dateutil.parser.parse(center_dt)
        return center_dt

    @property
    def bounds(self):
        geo_ref_points = self.metadata_doc['grid_spatial']['projection']['geo_ref_points']
        return BoundingBox(geo_ref_points['ll']['x'], geo_ref_points['ll']['y'],
                           geo_ref_points['ur']['x'], geo_ref_points['ur']['y'])

    @property
    def crs(self):
        projection = self.metadata_doc['grid_spatial']['projection']

        crs = projection.get('spatial_reference', None)
        if crs:
            return str(crs)

        # TODO: really need CRS specified properly in agdc-metadata.yaml
        if projection['datum'] == 'GDA94':
            return 'EPSG:283' + str(abs(projection['zone']))

        if projection['datum'] == 'WGS84':
            if projection['zone'][-1] == 'S':
                return 'EPSG:327' + str(abs(int(projection['zone'][:-1])))
            else:
                return 'EPSG:326' + str(abs(int(projection['zone'][:-1])))

        raise RuntimeError('Cant figure out the projection: %s %s' % (projection['datum'], projection['zone']))

    def __str__(self):
        return "Dataset <id={id}>".format(id=self.id)

    def __repr__(self):
        return self.__str__()

    @property
    def metadata(self):
        return self.metadata_type.dataset_reader(self.metadata_doc)


class MetadataType(object):
    def __init__(self,
                 name,
                 dataset_offsets,
                 dataset_search_fields,
                 storage_unit_search_fields,
                 id_=None):
        self.name = name
        #: :type: DatasetOffsets
        self.dataset_offsets = dataset_offsets

        #: :type: dict[str, datacube.index.fields.Field]
        self.dataset_fields = dataset_search_fields
        #: :type: dict[str, datacube.index.fields.Field]
        self.storage_fields = storage_unit_search_fields

        self.id_ = id_

    def dataset_reader(self, dataset_doc):
        return _DocReader(self.dataset_offsets.__dict__, dataset_doc)


class Collection(object):
    def __init__(self,
                 name,
                 match,
                 metadata_type,
                 id_=None):
        """
        Collection of datasets & storage.

        :type metadata_type: MetadataType
        :type match: DatasetMatcher
        :type name: str
        """
        self.id_ = id_

        # Name of collection. Unique.
        self.name = name

        # Match datasets that should belong to this collection.
        self.match = match

        # All datasets in a collection must have the same metadata_type.
        self.metadata_type = metadata_type

    def __str__(self):
        return "Collection <id={id}, name={name}>".format(id=self.id_, name=self.name)

    def __repr__(self):
        return self.__str__()


class DatasetOffsets(object):
    """
    Where to find certain mandatory fields in dataset metadata.
    """

    def __init__(self,
                 uuid_field=None,
                 label_field=None,
                 creation_time_field=None,
                 measurements_dict=None,
                 sources=None):
        # UUID for a dataset. Always unique.
        #: :type: tuple[string]
        self.uuid_field = uuid_field or ('id',)

        # The dataset "label" is the logical identifier for a dataset.
        #
        # -> Multiple datasets may arrive with the same label, but only the 'latest' will be returned by default
        #    in searches.
        #
        # Use case: reprocessing a dataset.
        # -> When reprocessing a dataset, the new dataset should be produced with the same label as the old one.
        # -> Because you probably don't want both datasets returned from typical searches. (they are the same data)
        # -> When ingested, this reprocessed dataset will be the only one visible to typical searchers.
        # -> But the old dataset will still exist in the database for provenance & historical record.
        #       -> Existing higher-level/derived datasets will still link to the old dataset they were processed
        #          from, even if it's not the latest.
        #
        # An example label used by GA (called "dataset_ids" on historical systems):
        #      -> Eg. "LS7_ETM_SYS_P31_GALPGS01-002_114_73_20050107"
        #
        # But the collection owner can use any string to label their datasets.
        #: :type: tuple[string]
        self.label_field = label_field or ('label',)

        # datetime the dataset was processed/created.
        #: :type: tuple[string]
        self.creation_time_field = creation_time_field or ('creation_dt',)

        # Where to find a dict of measurements/bands in the dataset.
        #  -> Dict key is measurement/band id,
        #  -> Dict value is object with fields depending on the storage driver.
        #     (such as path to band file, offset within file etc.)
        #: :type: tuple[string]
        self.measurements_dict = measurements_dict or ('measurements',)

        # Where to find a dict of embedded source datasets
        #  -> The dict is of form: classifier->source_dataset_doc
        #  -> 'classifier' is how to classify/identify the relationship (usually the type of source it was eg. 'nbar').
        #      An arbitrary string, but you should be consistent between datasets (to query relationships).
        #: :type: tuple[string]
        self.sources = sources or ('lineage', 'source_datasets')


class GeoBox(object):
    """
    Defines a single Storage Unit, its CRS, location, resolution, and global attributes

    >>> from affine import Affine
    >>> wgs84 = osr.SpatialReference()
    >>> r = wgs84.ImportFromEPSG(4326)
    >>> t = GeoBox(4000, 4000, Affine(0.00025, 0.0, 151.0, 0.0, -0.00025, -29.0), wgs84.ExportToWkt())
    >>> t.coordinate_labels['latitude']
    array([-29.000125, -29.000375, -29.000625, ..., -29.999375, -29.999625,
           -29.999875])
    >>> t.coordinate_labels['longitude']
    array([ 151.000125,  151.000375,  151.000625, ...,  151.999375,
            151.999625,  151.999875])
    >>> t.geographic_extent
    [(151.0, -29.0), (151.0, -30.0), (152.0, -30.0), (152.0, -29.0)]


    :param crs_str: WKT representation of the coordinate reference system
    :type crs_str: str
    :param affine: Affine transformation defining the location of the storage unit
    :type affine: affine.Affine
    """
    def __init__(self, width, height, affine, crs_str):
        self.width = width
        self.height = height
        self.affine = affine
        self.crs_str = crs_str

    @classmethod
    def from_storage_type(cls, storage_type, tile_index):
        tile_size = storage_type.tile_size
        tile_res = storage_type.resolution
        return cls(crs_str=storage_type.crs,
                   affine=_get_tile_transform(tile_index, tile_size, tile_res),
                   width=int(tile_size[0] / abs(tile_res[0])),
                   height=int(tile_size[1] / abs(tile_res[1])))

    def __getitem__(self, item):
        indexes = [slice(index.start or 0, index.stop or size, index.step or 1)
                   for size, index in zip(self.shape, item)]
        for index in indexes:
            if index.step != 1:
                raise NotImplementedError('scaling not implemented, yet')

        affine = self.affine*Affine.translation(indexes[1].start, indexes[0].start)
        return GeoBox(width=indexes[1].stop - indexes[1].start,
                      height=indexes[0].stop - indexes[0].start,
                      affine=affine,
                      crs_str=self.crs_str)

    @property
    def shape(self):
        return self.height, self.width

    @property
    def crs(self):
        crs = osr.SpatialReference()
        crs.SetFromUserInput(self.crs_str)
        return crs

    @property
    def dimensions(self):
        crs = self.crs
        if crs.IsGeographic():
            return 'latitude', 'longitude'
        elif crs.IsProjected():
            return 'y', 'x'

    @property
    def coordinates(self):
        xs = numpy.array([0, self.width-1]) * self.affine.a + self.affine.c + self.affine.a / 2
        ys = numpy.array([0, self.height-1]) * self.affine.e + self.affine.f + self.affine.e / 2

        crs = self.crs
        if crs.IsGeographic():
            return {
                'latitude': Coordinate(ys.dtype, ys[0], ys[-1], self.height, 'degrees_north'),
                'longitude': Coordinate(xs.dtype, xs[0], xs[-1], self.width, 'degrees_east')
            }

        elif crs.IsProjected():
            units = crs.GetAttrValue('UNIT')
            return {
                'x': Coordinate(xs.dtype, xs[0], xs[-1], self.width, units),
                'y': Coordinate(ys.dtype, ys[0], ys[-1], self.height, units)
            }

    @property
    def coordinate_labels(self):
        xs = numpy.arange(self.width) * self.affine.a + self.affine.c + self.affine.a / 2
        ys = numpy.arange(self.height) * self.affine.e + self.affine.f + self.affine.e / 2

        crs = self.crs
        if crs.IsGeographic():
            return {
                'latitude': ys,
                'longitude': xs
            }
        elif crs.IsProjected():
            return {
                'x': xs,
                'y': ys
            }

    @property
    def geographic_extent(self):
        height, width = self.shape
        extents = [(0, 0), (0, height), (width, height), (width, 0)]
        self.affine.itransform(extents)
        if self.crs.IsGeographic():
            return extents

        epsg4326 = osr.SpatialReference()
        epsg4326.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(self.crs, epsg4326)
        return [p[:2] for p in transform.TransformPoints(extents)]

    @property
    def geographic_boundingbox(self):
        extents = self.geographic_extent
        return BoundingBox(left=min(lon for lon, lat in extents),
                           bottom=min(lat for lon, lat in extents),
                           right=max(lon for lon, lat in extents),
                           top=max(lat for lon, lat in extents))


def _get_tile_transform(tile_index, tile_size, tile_res):
    x = (tile_index[0] + (1 if tile_res[0] < 0 else 0)) * tile_size[0]
    y = (tile_index[1] + (1 if tile_res[1] < 0 else 0)) * tile_size[1]
    return Affine(tile_res[0], 0.0, x, 0.0, tile_res[1], y)


def _get_doc_offset(offset, document):
    """
    :type offset: list[str]
    :type document: dict

    >>> _get_doc_offset(['a'], {'a': 4})
    4
    >>> _get_doc_offset(['a', 'b'], {'a': {'b': 4}})
    4
    >>> _get_doc_offset(['a'], {})
    Traceback (most recent call last):
    ...
    KeyError: 'a'
    """
    value = document
    for key in offset:
        value = value[key]
    return value


def _set_doc_offset(offset, document, value):
    """
    :type offset: list[str]
    :type document: dict

    >>> doc = {'a': 4}
    >>> _set_doc_offset(['a'], doc, 5)
    >>> doc
    {'a': 5}
    >>> doc = {'a': {'b': 4}}
    >>> _set_doc_offset(['a', 'b'], doc, 'c')
    >>> doc
    {'a': {'b': 'c'}}
    """
    read_offset = offset[:-1]
    sub_doc = _get_doc_offset(read_offset, document)
    sub_doc[offset[-1]] = value


class _DocReader(object):
    def __init__(self, field_offsets, doc):
        """
        :type field_offsets: dict[str,list[str]]
        :type doc: dict
        >>> d = _DocReader({'lat': ['extent', 'lat']}, doc={'extent': {'lat': 4}})
        >>> d.lat
        4
        >>> d.lat = 5
        >>> d._doc
        {'extent': {'lat': 5}}
        >>> d.lon
        Traceback (most recent call last):
        ...
        ValueError: Unknown field 'lon'. Expected one of ['lat']
        """
        self.__dict__['_field_offsets'] = field_offsets
        self.__dict__['_doc'] = doc

    def __getattr__(self, name):
        offset = self._field_offsets.get(name)
        if offset is None:
            raise ValueError(
                'Unknown field %r. Expected one of %r' % (
                    name, list(self._field_offsets.keys())
                )
            )
        return _get_doc_offset(offset, self._doc)

    def __setattr__(self, name, val):
        offset = self._field_offsets.get(name)
        if offset is None:
            raise ValueError(
                'Unknown field %r. Expected one of %r' % (
                    name, list(self._field_offsets.keys())
                )
            )
        return _set_doc_offset(offset, self._doc, val)
