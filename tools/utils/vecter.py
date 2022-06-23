# reference: https://zhuanlan.zhihu.com/p/378918221

try:
    from osgeo import gdal, ogr, osr
except:
    import gdal
    import ogr
    import osr


def vector_translate(geojson_path: str,
                     wo_wkt: str,
                     g_type: str="POLYGON",
                     dim: str="XY") -> str:
	ogr.RegisterAll()
	gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
	data = ogr.Open(geojson_path)
	layer = data.GetLayer()
	spatial = layer.GetSpatialRef()
	layerName = layer.GetName()
	data.Destroy()
	dstSRS = osr.SpatialReference()
	dstSRS.ImportFromWkt(wo_wkt)
	ext = "." + geojson_path.split(".")[-1]
	save_path = geojson_path.replace(ext, ("_tmp" + ext))
	options = gdal.VectorTranslateOptions(
		srcSRS=spatial,
		dstSRS=dstSRS,
		reproject=True,
		layerName=layerName,
		geometryType=g_type,
		dim=dim
	)
	gdal.VectorTranslate(
		save_path,
		srcDS=geojson_path,
		options=options
	)
	return save_path