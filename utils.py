import math
import pandas as pd


def gc_distance(pred, gt):
    latitude = pred[0]
    longitude = pred[1]
    latitude_GT = gt[0]
    longitude_GT = gt[1]

    R = 6371
    factor_rad = 0.01745329252
    longitude = factor_rad * longitude
    longitude_GT = factor_rad * longitude_GT
    latitude = factor_rad * latitude
    latitude_GT = factor_rad * latitude_GT
    delta_long = longitude_GT - longitude
    delta_lat = latitude_GT - latitude
    subterm0 = math.sin(delta_lat / 2)**2
    subterm1 = math.cos(latitude) * math.cos(latitude_GT)
    subterm2 = math.sin(delta_long / 2)**2
    subterm1 = subterm1 * subterm2
    a = subterm0 + subterm1
    c = 2 * math.asin(math.sqrt(a))
    gcd = R * c
    return gcd


def print_results(gc_dists):
    results = {}
    for p in gc_dists:
        results[p] = {'continent': 0, 'country': 0, 'region': 0, 'city': 0, 'street': 0}
        num_images = 0
        for img in gc_dists[p]:
            num_images += 1
            if gc_dists[p][img] <= 2500.0:
                results[p]['continent'] = results[p]['continent'] + 1
                if gc_dists[p][img] <= 750.0:
                    results[p]['country'] = results[p]['country'] + 1
                    if gc_dists[p][img] <= 200.0:
                        results[p]['region'] = results[p]['region'] + 1
                        if gc_dists[p][img] <= 25.0:
                            results[p]['city'] = results[p]['city'] + 1
                            if gc_dists[p][img] <= 1.0:
                                results[p]['street'] = results[p]['street'] + 1

        for thresh in results[p]:
            results[p][thresh] = f'{(100 * results[p][thresh] / num_images):.1f}'

    print(pd.DataFrame.from_dict(results, orient='index'))
