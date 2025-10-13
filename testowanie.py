
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from PIL import Image
import matplotlib.pyplot as plt
from utils.func import get_pictures, apply_hough_transformation, group_lines, create_reference_court, warp_points, plot_results, measure_error
from utils.lines import Line
from utils.corners import CourtFinder

mpl.rcParams['image.cmap'] = 'gray'

with open('test.config.json') as file:
    config = json.load(file)

train_pics = get_pictures('pics/compliant')['rgb']
test_df_rows = []

for i, (data, train_pic) in enumerate(zip(config['data'], train_pics)):
    path = Path(f"{config['testing_pics_dir']}/{config['commit']}")
    path.mkdir(exist_ok = True, parents = True)

    
    train_pic_hough_lines, line_endpoints = apply_hough_transformation(train_pic)
    line_objs = [Line.from_hough_line(line[0]) for line in line_endpoints]
    line_objs = [line for line in line_objs if line.xv is None]

    grouped_lines = group_lines(line_objs)


    train_pic_line_intersections = train_pic.copy()
    intersections = []
    for group1 in grouped_lines:
        for group2 in grouped_lines:
            intersection = group1.intersection(group2, train_pic_line_intersections)
            if intersection is not None:
                intersections.append(intersection)


    candidates = CourtFinder(intersections, train_pic)

    try:
        # closer_outer_baseline_point, closer_outer_netline_point, closer_outer_sideline -------
        closer_outer_baseline_intersecetion, closer_outer_netintersection, used_line = candidates.find_closer_outer_baseline_point()
        closer_outer_baseline_point = closer_outer_baseline_intersecetion.point
        closer_outer_netline_point = closer_outer_netintersection.point

        closer_outer_sideline = Line.from_points(closer_outer_baseline_point, closer_outer_netline_point)



        # further_outer_baseline_point, baseline  -------
        further_outer_baseline_intersection, last_local_line = candidates.find_further_outer_baseline_intersection(closer_outer_baseline_intersecetion, used_line, data['canny_thresh']['lower'] , data['canny_thresh']['upper'], offset=data['offset'])
        further_outer_baseline_point = further_outer_baseline_intersection.point
        
        baseline = Line.from_points(closer_outer_baseline_point, further_outer_baseline_point)



        # netline -----
        netline = candidates.find_netline(closer_outer_netline_point, baseline, data['max_line_gap'])


        # further_outer_sideline, further_outer_netline_point ---
        further_outer_sideline = candidates.find_further_doubles_sideline(further_outer_baseline_point, last_local_line, data['offset'], data['extra_offset'], data['bin_thresh'], data['surface_type'])
        further_outer_netline_point = further_outer_sideline.intersection(netline, train_pic).point




        # inner points
        closer_inner_baseline_point, further_inner_baseline_point = candidates.scan_endline(baseline, netline, closer_outer_baseline_point, further_outer_netline_point, closer_outer_netline_point, further_outer_baseline_point, data['bin_thresh_endline_scan']['baseline'], data['canny_thresh']['lower'], data['canny_thresh']['upper'], data['max_line_gap'],searching_line = 'base')


        closer_inner_netline_point, further_inner_netline_point = candidates.scan_endline(baseline, 
                                                                                netline, 
                                                                                closer_outer_baseline_point, 
                                                                                further_outer_netline_point, 
                                                                                closer_outer_netline_point, 
                                                                                further_outer_baseline_point, 
                                                                                data['bin_thresh_endline_scan']['netline'],
                                                                                data['canny_thresh']['lower'],
                                                                                data['canny_thresh']['upper'],
                                                                                data['max_line_gap'],
                                                                                searching_line = 'net',
                                                                                )
        
        closer_inner_sideline = Line.from_points(closer_inner_baseline_point, closer_inner_netline_point)
        further_inner_sideline = Line.from_points(further_inner_baseline_point, further_inner_netline_point)
        
        # net_service_point and centre_service_line
        net_service_point, centre_service_line = candidates.find_net_service_point_centre_service_line(closer_outer_baseline_point, 
                                                                                closer_outer_netline_point, 
                                                                                further_outer_baseline_point,
                                                                                further_outer_netline_point, 
                                                                                closer_inner_baseline_point,
                                                                                further_inner_baseline_point,
                                                                                closer_inner_netline_point,
                                                                                further_inner_netline_point,
                                                                                baseline, 
                                                                                netline,
                                                                                data['bin_thresh_centre_service_line'],
                                                                                data['canny_thresh']['lower'],
                                                                                data['canny_thresh']['upper'],
                                                                                data['max_line_gap_centre_service_line'],
                                                                                data['min_line_len_ratio'],
                                                                                data['hough_thresh'],
                                                                                )
        
        centre_service_point, further_service_point, closer_service_point = candidates.find_center(closer_outer_baseline_point, closer_outer_netline_point, further_outer_baseline_point, further_outer_netline_point, closer_inner_baseline_point,further_inner_baseline_point, closer_inner_netline_point, further_inner_netline_point, baseline, closer_inner_sideline,further_inner_sideline, centre_service_line, data['bin_thresh_centre_service_line'], data['canny_thresh']['lower'], data['canny_thresh']['upper'], data['max_line_gap_centre_service_line'], data['min_line_len_ratio'], data['hough_thresh'])
        
        service_line = Line.from_points(further_service_point, closer_service_point)

        dst_points = {
            'closer_outer_baseline_point': closer_outer_baseline_point,
            'closer_outer_netline_point': closer_outer_netline_point,
            'further_outer_baseline_point': further_outer_baseline_point,
            'further_outer_netline_point': further_outer_netline_point,
            'closer_inner_baseline_point': closer_inner_baseline_point,
            'further_inner_baseline_point': further_inner_baseline_point,
            'closer_inner_netline_point': closer_inner_netline_point, 
            'further_inner_netline_point': further_inner_netline_point,
            'net_service_point': net_service_point,
            'centre_service_point': centre_service_point, 
            'further_service_point': further_service_point, 
            'closer_service_point': closer_service_point
        }
        
        dst_lines = {
            'closer_outer_sideline': closer_outer_sideline,
            'baseline': baseline,
            'netline': netline,
            'further_outer_sideline': further_outer_sideline,
            'closer_inner_sideline': closer_inner_sideline,
            'further_inner_sideline': further_inner_sideline,
            'centre_service_line': centre_service_line,
            'service_line': service_line,
        }

        plot_results(train_pic, path, data['pic_name'], dst_lines, dst_points)

        ground_truth_points = {
            'closer_outer_baseline_point': data['ground_truth_points']['closer_outer_baseline_point'],
            'closer_outer_netline_point': data['ground_truth_points']['closer_outer_netline_point'],
            'further_outer_baseline_point': data['ground_truth_points']['further_outer_baseline_point'],
            'further_outer_netline_point': data['ground_truth_points']['further_outer_netline_point'],
            'closer_inner_baseline_point': data['ground_truth_points']['closer_inner_baseline_point'],
            'further_inner_baseline_point': data['ground_truth_points']['further_inner_baseline_point'],
            'closer_inner_netline_point': data['ground_truth_points']['closer_inner_netline_point'], 
            'further_inner_netline_point': data['ground_truth_points']['further_inner_netline_point'],
            'net_service_point': data['ground_truth_points']['net_service_point'],
            'centre_service_point': data['ground_truth_points']['centre_service_point'], 
            'further_service_point': data['ground_truth_points']['further_service_point'], 
            'closer_service_point': data['ground_truth_points']['closer_service_point']
        }

        errors = measure_error(train_pic, dst_points, ground_truth_points)

        # perspective transform
        ref_points, ref_img = create_reference_court()

        transformed_points = warp_points(ref_points, dst_points, train_pic, 
                                         ref_img, 
                                        'closer_outer_netline_point', 
                                        'closer_outer_baseline_point', 
                                        'further_outer_netline_point', 
                                        'further_outer_baseline_point')
        
        # print(transformed_points)

        

    except Exception as e:
        print(e)
        errors = dict.fromkeys(dst_points.keys(), None)

    finally:
        img_index_name = [i, data['pic_name']]
        distances = [errors[dist] for dist in errors]
        test_df_rows.append(img_index_name + distances)



pd.DataFrame(test_df_rows, columns = ['pic_index', 'pic_name'] + list(errors.keys())).to_csv(path / "test_df.csv", index=False)
