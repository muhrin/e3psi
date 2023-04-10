import mincepy
import uuid

from . import base
from . import graphs


class IrrepsObjHelper(mincepy.BaseHelper):
    TYPE = base.IrrepsObj
    TYPE_ID = uuid.UUID("f8cd9a74-07d4-4a5e-9ed0-1bddfdcb94a4")

    def yield_hashables(self, obj: base.IrrepsObj, hasher):
        yield from hasher.yield_hashables(obj.__dict__)

    def save_instance_state(self, obj: base.IrrepsObj, _saver):
        return obj.__dict__

    def load_instance_state(self, obj: base.IrrepsObj, saved_state, _loader):
        for key, value in saved_state.items():
            setattr(obj, key, value)


class AttrHelper(mincepy.TypeHelper):
    TYPE = base.Attr
    TYPE_ID = uuid.UUID("8a1832b6-0d11-4fe3-a7c2-5efada06b640")

    irreps = mincepy.field(attr="_irreps")


class OneSiteHelper(mincepy.TypeHelper):
    TYPE = graphs.OneSite
    TYPE_ID = uuid.UUID("ddbb8d73-1602-4afd-8dda-d231bebf1220")

    site = mincepy.field(attr="site")


class TwoSiteHelper(mincepy.TypeHelper):
    TYPE = graphs.TwoSite
    TYPE_ID = uuid.UUID("29f0bcb3-a3dc-43f1-b739-50bec72d4ccc")

    site1 = mincepy.field(attr="site1")
    site2 = mincepy.field(attr="site2")
    edge = mincepy.field(attr="edge", default=None)


HISTORIAN_TYPES = IrrepsObjHelper, AttrHelper, OneSiteHelper, TwoSiteHelper
