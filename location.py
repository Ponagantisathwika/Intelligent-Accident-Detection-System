import geocoder

def get_location():
    g = geocoder.ip('me')
    if g.ok:
        lat, lng = g.latlng
        maps_url = f"https://www.google.com/maps?q={lat},{lng}"
        return lat, lng, maps_url
    return None, None, "Location not available"
