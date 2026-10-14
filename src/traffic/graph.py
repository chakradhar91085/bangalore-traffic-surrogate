"""Road network graph: 16 roads in 8 Bangalore areas, linked by area membership and proximity."""
import numpy as np

# Approximate area centroids (lat, lon), entered by hand from public map data.
AREA_COORDS = {
    "Electronic City": (12.8452, 77.6602),
    "Hebbal": (13.0358, 77.5970),
    "Indiranagar": (12.9784, 77.6408),
    "Jayanagar": (12.9308, 77.5838),
    "Koramangala": (12.9352, 77.6245),
    "M.G. Road": (12.9756, 77.6068),
    "Whitefield": (12.9698, 77.7500),
    "Yeshwanthpur": (13.0285, 77.5400),
}
NEIGHBOUR_KM = 6.0


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (*a, *b))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def build_adjacency(road_areas):
    """road_areas: list of area names, one per road node. Returns (raw, normalised) float32 matrices."""
    n = len(road_areas)
    a = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same = road_areas[i] == road_areas[j]
            near = haversine_km(AREA_COORDS[road_areas[i]], AREA_COORDS[road_areas[j]]) <= NEIGHBOUR_KM
            if same or near:
                a[i, j] = 1.0
    d = a.sum(1)
    norm = a / np.sqrt(np.outer(d, d))
    return a, norm.astype(np.float32)
