
import rasterio
from rasterio.transform import from_origin
import numpy as np
import rasterio
from rasterio.crs import CRS

# Test with a known EPSG code

class Write_georef_image:
    def __init__(self, top_left_lon , top_left_lat , pixel_size_x  , pixel_size_y ):
        self.top_left_lon = top_left_lon
        self.top_left_lat = top_left_lat
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y

    def to_image(self, image , output_path):
        # Example: Create a 3-band image with shape (bands, height, width)
        bands, height, width = image.shape 
        data = image   # shape: (3, 100, 100)


        # Create an affine transform (origin is top-left)
        transform = from_origin(west=self.top_left_lon, north=self.top_left_lat,
                                xsize=self.pixel_size_x, ysize=self.pixel_size_y)

        # Set CRS to WGS84
        crs = CRS.from_epsg(4326)


        # Write to GeoTIFF
        with rasterio.open(
            output_path ,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=bands,
            dtype='float32',
            crs=crs,
            transform=transform
        ) as dst:
            for i in range(bands):
                dst.write(data[i], i + 1)


