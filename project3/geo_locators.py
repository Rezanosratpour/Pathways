
from geopy.geocoders import Nominatim

def get_lat_lon(location_name):
    geolocator = Nominatim(user_agent="your_app_name_123456")  # use a unique string
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None