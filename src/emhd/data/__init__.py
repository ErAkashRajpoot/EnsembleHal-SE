from .codehalu import normalize_codehalu
from .cloudapibench import normalize_cloudapibench
from .defects4j import normalize_defects4j
from .normalize import FieldMap, normalize_jsonl
from .records import NormalizedRecord


def build_index(*args, **kwargs):
	from .indexer import build_index as _build_index

	return _build_index(*args, **kwargs)


def stratified_split(*args, **kwargs):
	from .splitter import stratified_split as _stratified_split

	return _stratified_split(*args, **kwargs)

__all__ = [
	"build_index",
	"FieldMap",
	"NormalizedRecord",
	"normalize_codehalu",
	"normalize_cloudapibench",
	"normalize_defects4j",
	"normalize_jsonl",
	"stratified_split",
]
