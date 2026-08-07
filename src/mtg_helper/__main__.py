from .cache_builder import CacheBuilder, CacheTypes

builder = CacheBuilder()
builder.build_index(CacheTypes.CARD_DATA)
builder.build_index(CacheTypes.RULINGS_DATA)
builder.build_index(CacheTypes.IMAGES)
